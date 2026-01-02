import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
import math
from diffusers import UNet2DConditionModel
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.unets.unet_2d_blocks import (
    get_down_block,
    get_up_block,
    get_mid_block,
)
from diffusers.models.activations import get_activation
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers


def custom_prepare_attention_mask(
    self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
) -> torch.Tensor:
    r"""
    Prepare the attention mask for the attention computation.

    Args:
        attention_mask (`torch.Tensor`):
            The attention mask to prepare.
        target_length (`int`):
            The target length of the attention mask. This is the length of the attention mask after padding.
        batch_size (`int`):
            The batch size, which is used to repeat the attention mask.
        out_dim (`int`, *optional*, defaults to `3`):
            The output dimension of the attention mask. Can be either `3` or `4`.

    Returns:
        `torch.Tensor`: The prepared attention mask.
    """
    head_size = self.heads
    if attention_mask is None:
        return attention_mask

    current_length: int = attention_mask.shape[-1]
    if current_length != target_length:
        if attention_mask.device.type == "mps":
            # HACK: MPS: Does not support padding by greater than dimension of input tensor.
            # Instead, we can manually construct the padding tensor.
            padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
            padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, padding], dim=2)
        else:
            # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
            #       we want to instead pad by (0, remaining_length), where remaining_length is:
            #       remaining_length: int = target_length - current_length
            # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
            B = attention_mask.shape[0]
            current_size = int(math.sqrt(current_length))
            target_size = int(math.sqrt(target_length))
            assert current_size**2 == current_length, f"current_length ({current_length}) cannot be squared to an integer size"
            assert target_size**2 == target_length, f"target_length ({target_length}) cannot be squared to an integer size"
            attention_mask = attention_mask.view(B, -1, current_size, current_size)
            attention_mask = F.interpolate(attention_mask, size=(target_size, target_size), mode="nearest")
            attention_mask = attention_mask.view(B, 1, target_length)

    if out_dim == 3:
        if attention_mask.shape[0] < batch_size * head_size:
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
    elif out_dim == 4:
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

    return attention_mask


def custom_get_attention_scores(self, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
    r"""
    Compute the attention scores.

    Args:
        query (`torch.Tensor`): The query tensor.
        key (`torch.Tensor`): The key tensor.
        attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

    Returns:
        `torch.Tensor`: The attention probabilities/scores.
    """
    dtype = query.dtype
    if self.upcast_attention:
        query = query.float()
        key = key.float()

    # if attention_mask is not None and len(torch.unique(attention_mask)) <= 2:
    if attention_mask is not None:
        baddbmm_input = attention_mask
        beta = 1
    else:
        baddbmm_input = torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device)
        beta = 0

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=self.scale,
    )

    # if attention_mask is not None and len(torch.unique(attention_mask)) > 2:
    #     m = 1 - (attention_mask / -10000.0)
    #     attention_scores = m * attention_scores

    del baddbmm_input

    if self.upcast_softmax:
        attention_scores = attention_scores.float()

    attention_probs = attention_scores.softmax(dim=-1)
    del attention_scores

    attention_probs = attention_probs.to(dtype)

    return attention_probs


class CustomUNet(UNet2DConditionModel):
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_dim: Optional[int] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        bbox_time_embed_dim: Optional[int] = None,
        point_embeddings_input_dim: Optional[int] = None,
        bbox_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        use_attention_mask_list=[True, True, True],
        use_encoder_hidden_states_list=[True, True, True],
    ):
        super().__init__()
        self.use_attention_mask_list = use_attention_mask_list
        self.use_encoder_hidden_states_list = use_encoder_hidden_states_list
        self.sample_size = sample_size
        num_attention_heads = num_attention_heads or attention_head_dim

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding)

        # time
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        self.point_embedding = TimestepEmbedding(point_embeddings_input_dim, time_embed_dim)
        self.bbox_time_proj = Timesteps(bbox_time_embed_dim, flip_sin_to_cos, freq_shift)
        self.bbox_embedding = TimestepEmbedding(bbox_embeddings_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if class_embeddings_concat:
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            mid_block_type,
            temb_channels=blocks_time_embed_dim,
            in_channels=block_out_channels[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            output_scale_factor=mid_block_scale_factor,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            resnet_skip_time_act=resnet_skip_time_act,
            cross_attention_norm=cross_attention_norm,
            attention_head_dim=attention_head_dim[-1],
            dropout=dropout,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)

            self.conv_act = get_activation(act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding)

        # distillation
        self.feature_map = []

    def _get_value(self, use_list, true_value, false_value):
        down_value = mid_value = up_value = false_value

        if use_list[0]:
            down_value = true_value
        if use_list[1]:
            mid_value = true_value
        if use_list[2]:
            up_value = true_value

        return down_value, mid_value, up_value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        trans: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        down_attn_mask, mid_attn_mask, up_attn_mask = self._get_value(self.use_attention_mask_list, attention_mask, None)
        down_encoder_hidden_states, mid_encoder_hidden_states, up_encoder_hidden_states = self._get_value(
            self.use_encoder_hidden_states_list, encoder_hidden_states, encoder_hidden_states_2
        )

        # 1. time
        t_emb, op_emb, aug_emb = None, None, None

        if timestep is not None:
            timesteps = timestep
            timesteps = timesteps.expand(sample.shape[0])
            t_emb = self.time_proj(timesteps)
            t_emb = t_emb.to(dtype=sample.dtype)

            t_emb = self.time_embedding(t_emb, timestep_cond)

        # opacity
        if trans is not None:
            trans = trans.expand(sample.shape[0])
            op_emb = self.time_proj(trans)
            op_emb = op_emb.to(dtype=sample.dtype)

            op_emb = self.time_embedding(op_emb, timestep_cond)

        if t_emb is not None and op_emb is not None:
            emb = t_emb + op_emb
        elif op_emb is not None:
            emb = op_emb
        elif t_emb is not None:
            emb = t_emb
        else:
            raise ValueError("Missing required field: 'timestep' and 'trans'. Please ensure it is included in your input.")

        if "point_coords" in added_cond_kwargs:
            coords_embeds = added_cond_kwargs.get("point_coords")
            coords_embeds = coords_embeds.reshape((sample.shape[0], -1))
            coords_embeds = coords_embeds.to(emb.dtype)
            aug_emb = self.point_embedding(coords_embeds)
        elif "bbox_mask_coords" in added_cond_kwargs:
            coords = added_cond_kwargs.get("bbox_mask_coords")
            coords_embeds = self.bbox_time_proj(coords.flatten())
            coords_embeds = coords_embeds.reshape((sample.shape[0], -1))
            coords_embeds = coords_embeds.to(emb.dtype)
            aug_emb = self.bbox_embedding(coords_embeds)
        else:
            raise ValueError(f"{self.__class__} cannot find point_coords or bbox_coords in added_cond_kwargs.")

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # distillation
        self.feature_map = []

        # 3. down
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=down_encoder_hidden_states,
                    attention_mask=down_attn_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)

            down_block_res_samples += res_samples

        self.feature_map.append(sample)

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=mid_encoder_hidden_states,
                    attention_mask=mid_attn_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

        self.feature_map.append(sample)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=up_encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=up_attn_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    scale=lora_scale,
                )

        self.feature_map.append(sample)

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        return UNet2DConditionOutput(sample=sample)
