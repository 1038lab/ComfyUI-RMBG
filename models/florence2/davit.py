# Based on code from ComfyUI-Florence2 by Jukka Seppänen (kijai), MIT License
"""DaViT (Dual-Attention Vision Transformer) for Florence2. ComfyUI ops pattern."""

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from comfy.ldm.modules.attention import optimized_attention_for_device


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class PreNorm(nn.Module):
    """Pre-norm wrapper with residual. Attrs named norm/fn to match weight keys."""
    def __init__(self, norm, fn):
        super().__init__()
        self.norm = norm
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        shortcut = x
        if self.norm is not None:
            x, size = self.fn(self.norm(x), *args, **kwargs)
        else:
            x, size = self.fn(x, *args, **kwargs)
        return shortcut + x, size


class Mlp(nn.Module):
    """MLP with net.fc1/net.fc2 weight keys via OrderedDict Sequential."""
    def __init__(self, in_features, hidden_features=None, out_features=None, dtype=None, device=None, operations=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(OrderedDict([
            ("fc1", operations.Linear(in_features, hidden_features, dtype=dtype, device=device)),
            ("act", nn.GELU()),
            ("fc2", operations.Linear(hidden_features, out_features, dtype=dtype, device=device)),
        ]))

    def forward(self, x, size):
        return self.net(x), size


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, kernel_size, padding, stride, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.dw = operations.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias, dtype=dtype, device=device)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        x = self.dw(x.transpose(1, 2).view(B, C, H, W))
        size = (x.size(-2), x.size(-1))
        return x.flatten(2).transpose(1, 2), size


class ConvEmbed(nn.Module):
    """Patch embedding via Conv2d + optional LayerNorm."""
    def __init__(self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2, norm_layer_on=True, pre_norm=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.patch_size = patch_size
        self.pre_norm = pre_norm
        self.proj = operations.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, dtype=dtype, device=device)
        if norm_layer_on:
            self.norm = operations.LayerNorm(in_chans if pre_norm else embed_dim, dtype=dtype, device=device)
        else:
            self.norm = None

    def forward(self, x, size):
        H, W = size
        if len(x.size()) == 3:
            if self.norm and self.pre_norm:
                x = self.norm(x)
            B, _, C = x.shape
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.norm and not self.pre_norm:
            x = self.norm(x)
        return x, (H, W)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, batch_size, window_size, H, W):
    x = windows.view(batch_size, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, C)

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        optimized_attention = optimized_attention_for_device(x.device, small_input=True)
        x = optimized_attention(q, k, v, self.num_heads, skip_reshape=True)
        x = self.proj(x)

        x = window_reverse(x.view(-1, self.window_size, self.window_size, C), B, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x.view(B, H * W, C), size


class ChannelAttention(nn.Module):
    def __init__(self, dim, groups=8, qkv_bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.groups = groups
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, x, size):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.groups, C // self.groups).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * (float(N) ** -0.5)
        attn = (q.transpose(-1, -2) @ k).softmax(dim=-1)
        x = (attn @ v.transpose(-1, -2)).transpose(-1, -2).transpose(1, 2).reshape(B, N, C)
        return self.proj(x), size


class SpatialBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., qkv_bias=True, conv_at_attn=True, conv_at_ffn=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1, dtype=dtype, device=device, operations=operations)) if conv_at_attn else None
        self.window_attn = PreNorm(operations.LayerNorm(dim, dtype=dtype, device=device), WindowAttention(dim, num_heads, window_size, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations))
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1, dtype=dtype, device=device, operations=operations)) if conv_at_ffn else None
        self.ffn = PreNorm(operations.LayerNorm(dim, dtype=dtype, device=device), Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), dtype=dtype, device=device, operations=operations))

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)
        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class ChannelBlock(nn.Module):
    def __init__(self, dim, groups, mlp_ratio=4., qkv_bias=True, conv_at_attn=True, conv_at_ffn=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1, dtype=dtype, device=device, operations=operations)) if conv_at_attn else None
        self.channel_attn = PreNorm(operations.LayerNorm(dim, dtype=dtype, device=device), ChannelAttention(dim, groups=groups, qkv_bias=qkv_bias, dtype=dtype, device=device, operations=operations))
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1, dtype=dtype, device=device, operations=operations)) if conv_at_ffn else None
        self.ffn = PreNorm(operations.LayerNorm(dim, dtype=dtype, device=device), Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), dtype=dtype, device=device, operations=operations))

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)
        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class DaViT(nn.Module):
    """DaViT: Dual-Attention Vision Transformer. [B, 3, H, W] -> [B, N, C]"""
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        embed_dims = config.dim_embed
        num_heads = config.num_heads
        num_groups = config.num_groups
        depths = config.depths
        self.embed_dims = embed_dims

        convs, blocks = [], []
        for i in range(len(embed_dims)):
            convs.append(ConvEmbed(
                patch_size=config.patch_size[i], in_chans=3 if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i], stride=config.patch_stride[i], padding=config.patch_padding[i],
                pre_norm=config.patch_prenorm[i], dtype=dtype, device=device, operations=operations,
            ))
            blocks.append(MySequential(*[
                MySequential(OrderedDict([
                    ('spatial_block', SpatialBlock(embed_dims[i], num_heads[i], config.window_size, dtype=dtype, device=device, operations=operations)),
                    ('channel_block', ChannelBlock(embed_dims[i], num_groups[i], dtype=dtype, device=device, operations=operations)),
                ])) for j in range(depths[i])
            ]))
        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

    @property
    def dim_out(self):
        return self.embed_dims[-1]

    def forward_features_unpool(self, x):
        """[B, 3, H, W] -> [B, N, C]"""
        input_size = (x.size(2), x.size(3))
        for conv, block in zip(self.convs, self.blocks):
            x, input_size = conv(x, input_size)
            x, input_size = block(x, input_size)
        return x
