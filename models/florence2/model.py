# Based on code from ComfyUI-Florence2 by Jukka Seppänen (kijai), MIT License
"""BART encoder-decoder language model, Florence2 top-level model, and autoregressive generation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .davit import DaViT
import comfy.ops
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.utils import ProgressBar
from tqdm.auto import tqdm

_POSITION_OFFSET = 2  # BART positional id offset


def make_attention_mask(mask_2d, dtype):
    """[B, S] -> [B, 1, 1, S] with -inf for padding positions."""
    inverted = 1.0 - mask_2d[:, None, None, :].to(dtype)
    return inverted.masked_fill(inverted.bool(), torch.finfo(dtype).min)


def make_causal_mask(shape, dtype, device, past_kv_len=0):
    """Create [B, 1, T, T+past] causal mask with -inf above diagonal."""
    bsz, tgt_len = shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    cond = torch.arange(tgt_len, device=device)
    mask.masked_fill_(cond < (cond + 1).view(-1, 1), 0)
    if past_kv_len > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_kv_len, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None].expand(bsz, 1, tgt_len, tgt_len + past_kv_len)


def _make_position_ids(input_ids, past_key_values_length, device):
    bsz, seq_len = input_ids.shape[:2]
    positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=device).expand(bsz, -1)
    return positions + _POSITION_OFFSET


class Attention(nn.Module):
    """Multi-headed attention with optional KV cache for decoder layers."""

    def __init__(self, embed_dim, num_heads, is_decoder, is_causal, dtype=None, device=None, operations=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.q_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.out_proj = operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, key_value_states=None, past_key_value=None, attention_mask=None):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)

        if is_cross_attention and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        # Ensure matching dtypes (cross-attention cache may differ when using manual_cast ops)
        if key_states.dtype != query_states.dtype:
            key_states = key_states.to(query_states.dtype)
            value_states = value_states.to(query_states.dtype)

        optimized_attention = optimized_attention_for_device(hidden_states.device, mask=attention_mask is not None, small_input=True)
        attn_output = optimized_attention(query_states, key_states, value_states, self.num_heads, mask=attention_mask, skip_reshape=True, skip_output_reshape=True)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, past_key_value


class EncoderLayer(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        d = config.d_model
        self.self_attn = Attention(d, config.encoder_attention_heads, is_decoder=False, is_causal=False, dtype=dtype, device=device, operations=operations)
        self.self_attn_layer_norm = operations.LayerNorm(d, dtype=dtype, device=device)
        self.fc1 = operations.Linear(d, config.encoder_ffn_dim, dtype=dtype, device=device)
        self.fc2 = operations.Linear(config.encoder_ffn_dim, d, dtype=dtype, device=device)
        self.final_layer_norm = operations.LayerNorm(d, dtype=dtype, device=device)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class DecoderLayer(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        d = config.d_model
        self.self_attn = Attention(d, config.decoder_attention_heads, is_decoder=True, is_causal=True, dtype=dtype, device=device, operations=operations)
        self.self_attn_layer_norm = operations.LayerNorm(d, dtype=dtype, device=device)
        self.encoder_attn = Attention(d, config.decoder_attention_heads, is_decoder=True, is_causal=False, dtype=dtype, device=device, operations=operations)
        self.encoder_attn_layer_norm = operations.LayerNorm(d, dtype=dtype, device=device)
        self.fc1 = operations.Linear(d, config.decoder_ffn_dim, dtype=dtype, device=device)
        self.fc2 = operations.Linear(config.decoder_ffn_dim, d, dtype=dtype, device=device)
        self.final_layer_norm = operations.LayerNorm(d, dtype=dtype, device=device)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None):
        # Self-attention
        self_attn_past = past_key_value[:2] if past_key_value is not None else None
        residual = hidden_states
        hidden_states, self_attn_cache = self.self_attn(hidden_states=hidden_states, past_key_value=self_attn_past, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-attention
        cross_attn_cache = None
        if encoder_hidden_states is not None:
            cross_attn_past = past_key_value[2:] if past_key_value is not None else None
            residual = hidden_states
            hidden_states, cross_attn_cache = self.encoder_attn(
                hidden_states=hidden_states, key_value_states=encoder_hidden_states,
                past_key_value=cross_attn_past, attention_mask=encoder_attention_mask,
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # FFN
        residual = hidden_states
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        present_key_value = None
        if self_attn_cache is not None:
            present_key_value = self_attn_cache
            if cross_attn_cache is not None:
                present_key_value = present_key_value + cross_attn_cache
        return hidden_states, present_key_value


class Encoder(nn.Module):
    def __init__(self, config, embed_tokens=None, dtype=None, device=None, operations=None):
        super().__init__()
        d = config.d_model
        self.embed_scale = math.sqrt(d) if config.scale_embedding else 1.0
        self.embed_tokens = operations.Embedding(config.vocab_size, d, padding_idx=config.pad_token_id, dtype=dtype, device=device)
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight
        self.embed_positions = operations.Embedding(config.max_position_embeddings + _POSITION_OFFSET, d, dtype=dtype, device=device)
        self.layers = nn.ModuleList([EncoderLayer(config, dtype=dtype, device=device, operations=operations) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = operations.LayerNorm(d, dtype=dtype, device=device)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        pos_input = input_ids if input_ids is not None else inputs_embeds[:, :, -1]
        position_ids = _make_position_ids(pos_input, 0, self.embed_positions.weight.device)
        embed_pos = self.embed_positions(position_ids).to(inputs_embeds.device)

        hidden_states = self.layernorm_embedding(inputs_embeds + embed_pos)

        if attention_mask is not None:
            attention_mask = make_attention_mask(attention_mask, hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        return (hidden_states,)


class Decoder(nn.Module):
    def __init__(self, config, embed_tokens=None, dtype=None, device=None, operations=None):
        super().__init__()
        d = config.d_model
        self.embed_scale = math.sqrt(d) if config.scale_embedding else 1.0
        self.embed_tokens = operations.Embedding(config.vocab_size, d, padding_idx=config.pad_token_id, dtype=dtype, device=device)
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight
        self.embed_positions = operations.Embedding(config.max_position_embeddings + _POSITION_OFFSET, d, dtype=dtype, device=device)
        self.layers = nn.ModuleList([DecoderLayer(config, dtype=dtype, device=device, operations=operations) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = operations.LayerNorm(d, dtype=dtype, device=device)

    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_values=None, use_cache=True, inputs_embeds=None):
        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if input_ids is not None:
            input_shape = input_ids.shape
            pos_input = input_ids
        else:
            input_shape = inputs_embeds.size()[:-1]
            pos_input = inputs_embeds[:, :, -1]

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        position_ids = _make_position_ids(pos_input, past_key_values_length, self.embed_positions.weight.device)
        positions = self.embed_positions(position_ids).to(inputs_embeds.device)

        hidden_states = self.layernorm_embedding(inputs_embeds + positions)

        # When decoding with KV cache and single token, the causal mask is trivially all-attend
        # (a 1xN mask with no masking). Skip mask computation entirely for this fast path.
        tgt_len = input_shape[-1]
        if past_key_values is not None and tgt_len == 1:
            causal_mask = None
        else:
            causal_mask = make_causal_mask(input_shape, hidden_states.dtype, hidden_states.device, past_kv_len=past_key_values_length)
            if attention_mask is not None:
                causal_mask = causal_mask + make_attention_mask(attention_mask, hidden_states.dtype)

        enc_attn_mask = None
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if encoder_attention_mask.ndim == 4:
                enc_attn_mask = encoder_attention_mask  # already expanded
            else:
                enc_attn_mask = make_attention_mask(encoder_attention_mask, hidden_states.dtype)

        next_cache = () if use_cache else None
        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present_kv = layer(
                hidden_states, attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=enc_attn_mask,
                past_key_value=past_kv,
            )
            if use_cache:
                next_cache = next_cache + (present_kv,)
        return (hidden_states, next_cache)


class LanguageModel(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.shared = operations.Embedding(config.vocab_size, config.d_model, dtype=dtype, device=device)
        self.encoder = Encoder(config, self.shared, dtype=dtype, device=device, operations=operations)
        self.decoder = Decoder(config, self.shared, dtype=dtype, device=device, operations=operations)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, encoder_outputs=None, past_key_values=None, inputs_embeds=None, use_cache=True):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids, encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache,
        )
        return decoder_outputs + encoder_outputs


class LanguageModelWithLMHead(nn.Module):
    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.config = config
        self.model = LanguageModel(config, dtype=dtype, device=device, operations=operations)
        self.lm_head = operations.Linear(config.d_model, config.vocab_size, bias=False, dtype=dtype, device=device)
        # Always zero — exists so the key loads from safetensors without warnings
        self.register_buffer("final_logits_bias", torch.zeros((1, config.vocab_size)))

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, encoder_outputs=None, past_key_values=None, inputs_embeds=None, use_cache=True):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache)
        return self.lm_head(outputs[0]), outputs

    def tie_weights(self):
        self.model.encoder.embed_tokens.weight = self.model.shared.weight
        self.model.decoder.embed_tokens.weight = self.model.shared.weight
        self.lm_head.weight = self.model.shared.weight


class PositionEmbedding2D(nn.Module):
    """Learned absolute 2D positional embedding (row + column)."""
    def __init__(self, embedding_dim, num_pos, dtype=None, device=None, operations=None):
        super().__init__()
        self.row_embeddings = operations.Embedding(num_pos, embedding_dim // 2, dtype=dtype, device=device)
        self.column_embeddings = operations.Embedding(num_pos, embedding_dim - (embedding_dim // 2), dtype=dtype, device=device)

    def forward(self, pixel_values):
        """pixel_values: [B, H, W, C] -> [B, H, W, embedding_dim]"""
        height, width = pixel_values.shape[1:3]
        x_emb = self.column_embeddings(torch.arange(width, device=pixel_values.device))
        y_emb = self.row_embeddings(torch.arange(height, device=pixel_values.device))
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(pixel_values.shape[0], 1, 1, 1).permute(0, 2, 3, 1)
        return pos


class CosinePositionEmbedding1D(nn.Module):
    """Fixed sinusoidal 1D positional encoding. Buffer key: pos_idx_to_embed"""
    def __init__(self, embed_dim, max_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        factor = math.log(10000)
        denominator = torch.exp(-factor * torch.arange(0, embed_dim, 2) / embed_dim)
        frequencies = torch.arange(0, max_seq_len).reshape(max_seq_len, 1) * denominator
        pos_idx_to_embed = torch.zeros((max_seq_len, embed_dim))
        pos_idx_to_embed[:, 0::2] = torch.sin(frequencies)
        pos_idx_to_embed[:, 1::2] = torch.cos(frequencies)
        self.register_buffer("pos_idx_to_embed", pos_idx_to_embed)

    def forward(self, seq_embeds):
        """seq_embeds: [T, D] or [B, T, D] -> positional embeddings of matching shape."""
        shape_len = len(seq_embeds.shape)
        assert 2 <= shape_len <= 3
        pos_embeds = comfy.ops.cast_to_input(self.pos_idx_to_embed[:seq_embeds.size(-2), :], seq_embeds)
        if shape_len == 3:
            pos_embeds = pos_embeds.view(1, pos_embeds.size(0), pos_embeds.size(1))
        return pos_embeds


class Florence2(nn.Module):
    """Florence2 = DaViT vision encoder + image projection + BART LM."""

    def __init__(self, config, dtype=None, device=None, operations=None):
        super().__init__()
        self.config = config
        vision_cfg = config.vision_config
        text_cfg = config.text_config

        self.vision_tower = DaViT(vision_cfg, dtype=dtype, device=device, operations=operations)

        image_dim_out = vision_cfg.dim_embed[-1]
        projection_dim = vision_cfg.projection_dim
        self.image_projection = nn.Parameter(torch.empty(image_dim_out, projection_dim, dtype=dtype, device=device))
        self.image_proj_norm = operations.LayerNorm(projection_dim, dtype=dtype, device=device)

        image_pos_embed_config = vision_cfg.image_pos_embed
        if image_pos_embed_config is not None and image_pos_embed_config.get("type") == "learned_abs_2d":
            self.image_pos_embed = PositionEmbedding2D(embedding_dim=image_dim_out, num_pos=image_pos_embed_config["max_pos_embeddings"], dtype=dtype, device=device, operations=operations)
        else:
            self.image_pos_embed = None

        visual_temporal_config = vision_cfg.visual_temporal_embedding
        if visual_temporal_config is not None and visual_temporal_config.get("type") == "COSINE":
            self.visual_temporal_embed = CosinePositionEmbedding1D(embed_dim=image_dim_out, max_seq_len=visual_temporal_config["max_temporal_embeddings"])
        else:
            self.visual_temporal_embed = None

        self.image_feature_source = vision_cfg.image_feature_source
        self.language_model = LanguageModelWithLMHead(text_cfg, dtype=dtype, device=device, operations=operations)

    def encode_image(self, pixel_values):
        """Encode image through DaViT + projection. [B, 3, H, W] -> [B, num_tokens, projection_dim]"""
        batch_size = pixel_values.shape[0]
        T = 1
        x = self.vision_tower.forward_features_unpool(pixel_values)

        if self.image_pos_embed is not None:
            x = x.view(batch_size * T, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            h = w = int(num_tokens ** 0.5)
            assert h * w == num_tokens, "only square feature maps supported"
            x = x.view(batch_size * T, h, w, x.shape[-1])
            x = x + self.image_pos_embed(x)
            x = x.view(batch_size, T * h * w, x.shape[-1])

        if self.visual_temporal_embed is not None:
            temporal = self.visual_temporal_embed(x.view(batch_size, T, -1, x.shape[-1])[:, :, 0])
            x = x.view(batch_size, T, -1, x.shape[-1]) + temporal.view(1, T, 1, x.shape[-1])

        x_feat_dict = {
            "spatial_avg_pool": x.view(batch_size, T, -1, x.shape[-1]).mean(dim=2),
            "temporal_avg_pool": x.view(batch_size, T, -1, x.shape[-1]).mean(dim=1),
            "last_frame": x.view(batch_size, T, -1, x.shape[-1])[:, -1],
        }
        x = torch.cat([x_feat_dict[src] for src in self.image_feature_source], dim=1)

        # Project — cast nn.Parameter to match x dtype (manual_cast doesn't cover raw params)
        x = x @ self.image_projection.to(dtype=x.dtype, device=x.device)
        x = self.image_proj_norm(x)
        return x

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds):
        batch_size, image_token_length = image_features.size()[:-1]
        device = image_features.device
        image_attention_mask = torch.ones(batch_size, image_token_length, device=device)
        if inputs_embeds is None:
            return image_features, image_attention_mask
        text_attention_mask = torch.ones(batch_size, inputs_embeds.size(1), device=device)
        return torch.cat([image_features, inputs_embeds], dim=1), torch.cat([image_attention_mask, text_attention_mask], dim=1)

    @torch.inference_mode()
    def generate(self, input_ids, pixel_values, max_new_tokens=1024, num_beams=3, do_sample=False, temperature=1.0):
        eos_token_id = self.config.text_config.eos_token_id
        decoder_start_token_id = self.config.text_config.decoder_start_token_id
        forced_bos_token_id = self.config.text_config.forced_bos_token_id

        image_features = self.encode_image(pixel_values)

        if input_ids is not None:
            embed_scale = self.language_model.model.encoder.embed_scale
            inputs_embeds = self.language_model.model.shared(input_ids) * embed_scale
        else:
            inputs_embeds = None

        merged_embeds, attention_mask = self._merge_input_ids_with_image_features(image_features, inputs_embeds)
        attention_mask = attention_mask.to(merged_embeds.dtype)
        encoder_outputs = self.language_model.model.encoder(inputs_embeds=merged_embeds, attention_mask=attention_mask)

        # Florence2 has no padding — the encoder attention mask is all-attend.
        # Passing None skips mask computation in cross-attention (faster).
        batch_size = merged_embeds.shape[0]
        args = dict(encoder_outputs=encoder_outputs, encoder_attention_mask=None, batch_size=batch_size,
                    decoder_start_token_id=decoder_start_token_id, eos_token_id=eos_token_id,
                    forced_bos_token_id=forced_bos_token_id, max_new_tokens=max_new_tokens)

        if num_beams > 1:
            return self._beam_search(**args, num_beams=num_beams, do_sample=do_sample)
        else:
            return self._greedy_or_sample(**args, do_sample=do_sample, temperature=temperature)

    def _greedy_or_sample(self, encoder_outputs, encoder_attention_mask, batch_size,
                          decoder_start_token_id, eos_token_id, forced_bos_token_id,
                          max_new_tokens, do_sample, temperature):
        device = encoder_outputs[0].device
        generated = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=device)
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        pbar = ProgressBar(max_new_tokens)
        tbar = tqdm(total=max_new_tokens, desc="Florence2", unit="tok")
        for step in range(max_new_tokens):
            cur_input_ids = generated[:, -1:] if past_key_values is not None else generated
            decoder_outputs = self.language_model.model.decoder(
                input_ids=cur_input_ids, encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, use_cache=True,
            )
            past_key_values = decoder_outputs[1]
            logits = self.language_model.lm_head(decoder_outputs[0][:, -1:, :])[:, -1, :]

            if step == 0 and forced_bos_token_id is not None:
                next_token = torch.full((batch_size, 1), forced_bos_token_id, dtype=torch.long, device=device)
            else:
                if forced_bos_token_id is not None and step > 0:
                    logits[:, forced_bos_token_id] = torch.finfo(logits.dtype).min
                if do_sample:
                    next_token = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            pbar.update(1)
            tbar.update(1)
            if finished.all():
                break
        tbar.close()
        return generated

    def _beam_search(self, encoder_outputs, encoder_attention_mask, batch_size,
                     decoder_start_token_id, eos_token_id, forced_bos_token_id,
                     max_new_tokens, num_beams, do_sample=False):
        device = encoder_outputs[0].device
        vocab_size = self.language_model.config.vocab_size

        # Expand encoder outputs for beams
        encoder_hidden = encoder_outputs[0].unsqueeze(1).expand(-1, num_beams, -1, -1).reshape(batch_size * num_beams, -1, encoder_outputs[0].shape[-1])
        encoder_outputs_expanded = (encoder_hidden,)

        generated = torch.full((batch_size * num_beams, 1), decoder_start_token_id, dtype=torch.long, device=device)
        past_key_values = None
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = torch.finfo(beam_scores.dtype).min
        beam_scores = beam_scores.view(-1)

        pbar = ProgressBar(max_new_tokens)
        tbar = tqdm(total=max_new_tokens, desc="Florence2 beam", unit="tok")
        for step in range(max_new_tokens):
            cur_input_ids = generated[:, -1:] if past_key_values is not None else generated
            decoder_outputs = self.language_model.model.decoder(
                input_ids=cur_input_ids, encoder_hidden_states=encoder_outputs_expanded[0],
                encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, use_cache=True,
            )
            past_key_values = decoder_outputs[1]
            next_token_logits = self.language_model.lm_head(decoder_outputs[0][:, -1:, :])[:, -1, :]

            # Force bos at step 0, suppress it after
            if step == 0 and forced_bos_token_id is not None:
                forced_scores = torch.full_like(next_token_logits, torch.finfo(next_token_logits.dtype).min)
                forced_scores[:, forced_bos_token_id] = 0
                next_token_scores = forced_scores
            else:
                if forced_bos_token_id is not None and step > 0:
                    next_token_logits[:, forced_bos_token_id] = torch.finfo(next_token_logits.dtype).min
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            next_scores = (next_token_scores + beam_scores[:, None]).view(batch_size, num_beams * vocab_size)

            # Step 0 with forced_bos is deterministic (only one valid token across all beams),
            # so skip sampling there — multinomial would fail with fewer nonzero entries than samples.
            sampling = do_sample and not (step == 0 and forced_bos_token_id is not None)
            if sampling:
                # HF beam-sample: multinomial without replacement over softmax(next_scores), then sort desc
                probs = F.softmax(next_scores, dim=-1)
                topk_indices = torch.multinomial(probs, num_samples=2 * num_beams)
                topk_scores = torch.gather(next_scores, -1, topk_indices)
                topk_scores, _sort_idx = torch.sort(topk_scores, descending=True, dim=-1)
                topk_indices = torch.gather(topk_indices, -1, _sort_idx)
            else:
                topk_scores, topk_indices = torch.topk(next_scores, 2 * num_beams, dim=-1, largest=True, sorted=True)

            topk_beam_indices = topk_indices.div(vocab_size, rounding_mode='floor')
            topk_token_ids = topk_indices % vocab_size

            next_beam_scores = topk_scores[:, :num_beams]
            next_beam_tokens = topk_token_ids[:, :num_beams]
            next_beam_flat_indices = topk_beam_indices[:, :num_beams]

            beam_offset = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            beam_idx_flat = (next_beam_flat_indices + beam_offset).view(-1)

            beam_scores = next_beam_scores.view(-1)
            beam_tokens = next_beam_tokens.view(-1)

            generated = torch.cat([generated[beam_idx_flat], beam_tokens.unsqueeze(-1)], dim=-1)
            past_key_values = self._reorder_cache(past_key_values, beam_idx_flat)

            pbar.update(1)
            tbar.update(1)
            if (beam_tokens.view(batch_size, num_beams) == eos_token_id).all():
                break
        tbar.close()

        # Select best beam, truncate at first eos
        beam_scores = beam_scores.view(batch_size, num_beams)
        best_indices = beam_scores.argmax(dim=-1) + torch.arange(batch_size, device=device) * num_beams
        generated = generated[best_indices]

        results = []
        for b in range(batch_size):
            seq = generated[b]
            eos_positions = (seq == eos_token_id).nonzero(as_tuple=False)
            for pos in eos_positions:
                if pos.item() > 0:
                    seq = seq[:pos.item() + 1]
                    break
            results.append(seq)

        max_len = max(r.shape[0] for r in results)
        pad_token_id = self.language_model.config.pad_token_id
        padded = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
        for b, r in enumerate(results):
            padded[b, :r.shape[0]] = r
        return padded

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
