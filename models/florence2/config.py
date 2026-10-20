# Based on code from ComfyUI-Florence2 by Jukka Seppänen (kijai), MIT License
import json
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional


@dataclass
class VisionConfig:
    drop_path_rate: float = 0.1
    patch_size: List[int] = field(default_factory=lambda: [7, 3, 3, 3])
    patch_stride: List[int] = field(default_factory=lambda: [4, 2, 2, 2])
    patch_padding: List[int] = field(default_factory=lambda: [3, 1, 1, 1])
    patch_prenorm: List[bool] = field(default_factory=lambda: [False, True, True, True])
    dim_embed: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    num_heads: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    num_groups: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    depths: List[int] = field(default_factory=lambda: [1, 1, 9, 1])
    window_size: int = 12
    projection_dim: int = 1024
    visual_temporal_embedding: Optional[Dict] = None
    image_pos_embed: Optional[Dict] = None
    image_feature_source: List[str] = field(default_factory=lambda: ["spatial_avg_pool", "temporal_avg_pool"])


@dataclass
class LanguageConfig:
    vocab_size: int = 51289
    max_position_embeddings: int = 1024
    encoder_layers: int = 12
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_layers: int = 12
    decoder_ffn_dim: int = 4096
    decoder_attention_heads: int = 16
    activation_function: str = "gelu"
    d_model: int = 1024
    scale_embedding: bool = False
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    decoder_start_token_id: int = 2
    forced_bos_token_id: int = 0
    forced_eos_token_id: int = 2


@dataclass
class Florence2Config:
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    text_config: LanguageConfig = field(default_factory=LanguageConfig)
    vocab_size: int = 51289
    projection_dim: int = 1024

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)

        vision_data = data.get("vision_config", {})
        text_data = data.get("text_config", {})

        vision_fields = {f.name for f in fields(VisionConfig)}
        text_fields = {f.name for f in fields(LanguageConfig)}

        vision_config = VisionConfig(**{k: v for k, v in vision_data.items() if k in vision_fields})
        text_config = LanguageConfig(**{k: v for k, v in text_data.items() if k in text_fields})

        return cls(
            vision_config=vision_config,
            text_config=text_config,
            vocab_size=data.get("vocab_size", text_config.vocab_size),
            projection_dim=data.get("projection_dim", vision_config.projection_dim),
        )

    @classmethod
    def from_state_dict(cls, sd):
        """Detect base vs large from weight shapes."""
        q_key = "language_model.model.encoder.layers.0.self_attn.q_proj.weight"
        if q_key in sd:
            d_model = sd[q_key].shape[0]
        else:
            d_model = 1024

        encoder_layer_keys = [k for k in sd if k.startswith("language_model.model.encoder.layers.")]
        if encoder_layer_keys:
            max_layer = max(int(k.split(".")[4]) for k in encoder_layer_keys)
            encoder_layers = max_layer + 1
        else:
            encoder_layers = 12

        decoder_layer_keys = [k for k in sd if k.startswith("language_model.model.decoder.layers.")]
        if decoder_layer_keys:
            max_layer = max(int(k.split(".")[4]) for k in decoder_layer_keys)
            decoder_layers = max_layer + 1
        else:
            decoder_layers = 12

        shared_key = "language_model.model.shared.weight"
        vocab_size = sd[shared_key].shape[0] if shared_key in sd else 51289

        fc1_key = "language_model.model.encoder.layers.0.fc1.weight"
        encoder_ffn_dim = sd[fc1_key].shape[0] if fc1_key in sd else d_model * 4

        dec_fc1_key = "language_model.model.decoder.layers.0.fc1.weight"
        decoder_ffn_dim = sd[dec_fc1_key].shape[0] if dec_fc1_key in sd else d_model * 4

        # Detect vision config from state dict
        dim_embed = []
        for i in range(4):
            conv_key = f"vision_tower.convs.{i}.proj.weight"
            if conv_key in sd:
                dim_embed.append(sd[conv_key].shape[0])

        if not dim_embed:
            dim_embed = [256, 512, 1024, 2048]

        num_heads = [d // 32 for d in dim_embed]  # head_dim=32 for DaViT spatial attention

        # Count depths
        depths = []
        for stage in range(4):
            stage_keys = [k for k in sd if k.startswith(f"vision_tower.blocks.{stage}.")]
            if stage_keys:
                max_depth = max(int(k.split(".")[3]) for k in stage_keys) + 1
                depths.append(max_depth)
            else:
                depths.append(1)

        if not any(k.startswith("vision_tower.") for k in sd):
            depths = [1, 1, 9, 1]

        projection_dim = dim_embed[-1] if dim_embed else 1024
        proj_key = "image_projection"
        if proj_key in sd:
            projection_dim = sd[proj_key].shape[1]

        vision_config = VisionConfig(
            dim_embed=dim_embed,
            num_heads=num_heads,
            num_groups=num_heads,
            depths=depths,
            projection_dim=projection_dim,
            image_pos_embed={"type": "learned_abs_2d", "max_pos_embeddings": 50},
            visual_temporal_embedding={"type": "COSINE", "max_temporal_embeddings": 100},
        )

        text_config = LanguageConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_attention_heads=d_model // 64,
            decoder_attention_heads=d_model // 64,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_ffn_dim=decoder_ffn_dim,
            max_position_embeddings=1024,
            scale_embedding=False,
        )

        return cls(
            vision_config=vision_config,
            text_config=text_config,
            vocab_size=vocab_size,
            projection_dim=projection_dim,
        )
