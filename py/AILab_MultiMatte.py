import os
import sys
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import folder_paths
from torch.hub import download_url_to_file
from safetensors.torch import load_file

from AILab_utils import (
    get_device,
    clean_vram,
    unload_model as release_model,
    process_mask,
    apply_background_color,
    tensor2pil,
    pil2tensor,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM3_MODELS_DIR = os.path.join(REPO_ROOT, "models")
SAM3_BPE_PATH = os.path.join(SAM3_MODELS_DIR, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

device = get_device()
folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

MODEL_REPO = "1038lab/MultiMatte"
MODEL_CACHE_NAME = "MultiMatte"
MODEL_CHOICES = {
    "MultiMatte-bf16 (1.57GB)": "MultiMatte-bf16.safetensors",
    "MultiMatte-fp8 (0.84GB)": "MultiMatte-fp8.safetensors",
    "MultiMatte-fp32 (3.13GB)": "MultiMatte.safetensors",
}
DEFAULT_MODEL = "MultiMatte-bf16 (1.57GB)"
DEFAULT_PROMPT = "the main foreground subject"


def _dequantize_fp8_state_dict(sd):
    out = {}
    for k, v in sd.items():
        if k.endswith("_scale"):
            continue
        scale_key = f"{k}_scale"
        if v.dtype == torch.float8_e4m3fn and scale_key in sd:
            out[k] = v.to(torch.bfloat16) * sd[scale_key].to(torch.bfloat16)
        else:
            out[k] = v
    return out


def _convert_hf_sam3_state_dict(hf_sd):
    out = {}

    def cat_qkv(prefix, dst, sep="_"):
        for kind in ("weight", "bias"):
            q = hf_sd[f"{prefix}.q_proj.{kind}"]
            k = hf_sd[f"{prefix}.k_proj.{kind}"]
            v = hf_sd[f"{prefix}.v_proj.{kind}"]
            out[f"{dst}{sep}{kind}"] = torch.cat([q, k, v], dim=0)

    out["backbone.vision_backbone.trunk.patch_embed.proj.weight"] = hf_sd[
        "model.vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"
    ]
    pe = hf_sd["model.vision_encoder.backbone.embeddings.position_embeddings"]
    out["backbone.vision_backbone.trunk.pos_embed"] = torch.cat([torch.zeros_like(pe[:, :1]), pe], dim=1)
    out["backbone.vision_backbone.trunk.ln_pre.weight"] = hf_sd["model.vision_encoder.backbone.layer_norm.weight"]
    out["backbone.vision_backbone.trunk.ln_pre.bias"] = hf_sd["model.vision_encoder.backbone.layer_norm.bias"]

    for i in range(32):
        hp = f"model.vision_encoder.backbone.layers.{i}"
        lp = f"backbone.vision_backbone.trunk.blocks.{i}"
        out[f"{lp}.norm1.weight"] = hf_sd[f"{hp}.layer_norm1.weight"]
        out[f"{lp}.norm1.bias"] = hf_sd[f"{hp}.layer_norm1.bias"]
        out[f"{lp}.norm2.weight"] = hf_sd[f"{hp}.layer_norm2.weight"]
        out[f"{lp}.norm2.bias"] = hf_sd[f"{hp}.layer_norm2.bias"]
        cat_qkv(f"{hp}.attention", f"{lp}.attn.qkv", sep=".")
        out[f"{lp}.attn.proj.weight"] = hf_sd[f"{hp}.attention.o_proj.weight"]
        out[f"{lp}.attn.proj.bias"] = hf_sd[f"{hp}.attention.o_proj.bias"]
        for fc in ("fc1", "fc2"):
            for wb in ("weight", "bias"):
                out[f"{lp}.mlp.{fc}.{wb}"] = hf_sd[f"{hp}.mlp.{fc}.{wb}"]

    for i in range(4):
        hp = f"model.vision_encoder.neck.fpn_layers.{i}"
        lp = f"backbone.vision_backbone.convs.{i}"
        for wb in ("weight", "bias"):
            out[f"{lp}.conv_1x1.{wb}"] = hf_sd[f"{hp}.proj1.{wb}"]
            out[f"{lp}.conv_3x3.{wb}"] = hf_sd[f"{hp}.proj2.{wb}"]
    for wb in ("weight", "bias"):
        out[f"backbone.vision_backbone.convs.0.dconv_2x2_0.{wb}"] = hf_sd[
            f"model.vision_encoder.neck.fpn_layers.0.scale_layers.0.{wb}"
        ]
        out[f"backbone.vision_backbone.convs.0.dconv_2x2_1.{wb}"] = hf_sd[
            f"model.vision_encoder.neck.fpn_layers.0.scale_layers.2.{wb}"
        ]
        out[f"backbone.vision_backbone.convs.1.dconv_2x2.{wb}"] = hf_sd[
            f"model.vision_encoder.neck.fpn_layers.1.scale_layers.0.{wb}"
        ]

    out["backbone.language_backbone.encoder.positional_embedding"] = hf_sd[
        "model.text_encoder.text_model.embeddings.position_embedding.weight"
    ]
    out["backbone.language_backbone.encoder.token_embedding.weight"] = hf_sd[
        "model.text_encoder.text_model.embeddings.token_embedding.weight"
    ]
    out["backbone.language_backbone.encoder.text_projection"] = (
        hf_sd["model.text_encoder.text_projection.weight"].t().contiguous()
    )
    out["backbone.language_backbone.encoder.ln_final.weight"] = hf_sd[
        "model.text_encoder.text_model.final_layer_norm.weight"
    ]
    out["backbone.language_backbone.encoder.ln_final.bias"] = hf_sd[
        "model.text_encoder.text_model.final_layer_norm.bias"
    ]
    out["backbone.language_backbone.resizer.weight"] = hf_sd["model.text_projection.weight"]
    out["backbone.language_backbone.resizer.bias"] = hf_sd["model.text_projection.bias"]

    for i in range(24):
        hp = f"model.text_encoder.text_model.encoder.layers.{i}"
        lp = f"backbone.language_backbone.encoder.transformer.resblocks.{i}"
        for wb in ("weight", "bias"):
            out[f"{lp}.ln_1.{wb}"] = hf_sd[f"{hp}.layer_norm1.{wb}"]
            out[f"{lp}.ln_2.{wb}"] = hf_sd[f"{hp}.layer_norm2.{wb}"]
            out[f"{lp}.attn.out_proj.{wb}"] = hf_sd[f"{hp}.self_attn.out_proj.{wb}"]
            out[f"{lp}.mlp.c_fc.{wb}"] = hf_sd[f"{hp}.mlp.fc1.{wb}"]
            out[f"{lp}.mlp.c_proj.{wb}"] = hf_sd[f"{hp}.mlp.fc2.{wb}"]
        cat_qkv(f"{hp}.self_attn", f"{lp}.attn.in_proj")

    for i in range(6):
        hp = f"model.detr_encoder.layers.{i}"
        lp = f"transformer.encoder.layers.{i}"
        for wb in ("weight", "bias"):
            out[f"{lp}.norm1.{wb}"] = hf_sd[f"{hp}.layer_norm1.{wb}"]
            out[f"{lp}.norm2.{wb}"] = hf_sd[f"{hp}.layer_norm2.{wb}"]
            out[f"{lp}.norm3.{wb}"] = hf_sd[f"{hp}.layer_norm3.{wb}"]
            out[f"{lp}.self_attn.out_proj.{wb}"] = hf_sd[f"{hp}.self_attn.o_proj.{wb}"]
            out[f"{lp}.cross_attn_image.out_proj.{wb}"] = hf_sd[f"{hp}.cross_attn.o_proj.{wb}"]
            out[f"{lp}.linear1.{wb}"] = hf_sd[f"{hp}.mlp.fc1.{wb}"]
            out[f"{lp}.linear2.{wb}"] = hf_sd[f"{hp}.mlp.fc2.{wb}"]
        cat_qkv(f"{hp}.self_attn", f"{lp}.self_attn.in_proj")
        cat_qkv(f"{hp}.cross_attn", f"{lp}.cross_attn_image.in_proj")

    for i in range(6):
        hp = f"model.detr_decoder.layers.{i}"
        lp = f"transformer.decoder.layers.{i}"
        for wb in ("weight", "bias"):
            out[f"{lp}.norm1.{wb}"] = hf_sd[f"{hp}.vision_cross_attn_layer_norm.{wb}"]
            out[f"{lp}.norm2.{wb}"] = hf_sd[f"{hp}.self_attn_layer_norm.{wb}"]
            out[f"{lp}.norm3.{wb}"] = hf_sd[f"{hp}.mlp_layer_norm.{wb}"]
            out[f"{lp}.catext_norm.{wb}"] = hf_sd[f"{hp}.text_cross_attn_layer_norm.{wb}"]
            out[f"{lp}.cross_attn.out_proj.{wb}"] = hf_sd[f"{hp}.vision_cross_attn.o_proj.{wb}"]
            out[f"{lp}.ca_text.out_proj.{wb}"] = hf_sd[f"{hp}.text_cross_attn.o_proj.{wb}"]
            out[f"{lp}.self_attn.out_proj.{wb}"] = hf_sd[f"{hp}.self_attn.o_proj.{wb}"]
            out[f"{lp}.linear1.{wb}"] = hf_sd[f"{hp}.mlp.fc1.{wb}"]
            out[f"{lp}.linear2.{wb}"] = hf_sd[f"{hp}.mlp.fc2.{wb}"]
        cat_qkv(f"{hp}.vision_cross_attn", f"{lp}.cross_attn.in_proj")
        cat_qkv(f"{hp}.text_cross_attn", f"{lp}.ca_text.in_proj")
        cat_qkv(f"{hp}.self_attn", f"{lp}.self_attn.in_proj")

    for wb in ("weight", "bias"):
        out[f"transformer.decoder.norm.{wb}"] = hf_sd[f"model.detr_decoder.output_layer_norm.{wb}"]
        out[f"transformer.decoder.presence_token_out_norm.{wb}"] = hf_sd[
            f"model.detr_decoder.presence_layer_norm.{wb}"
        ]
        for idx, l_name in enumerate(("layer1", "layer2", "layer3")):
            out[f"transformer.decoder.bbox_embed.layers.{idx}.{wb}"] = hf_sd[
                f"model.detr_decoder.box_head.{l_name}.{wb}"
            ]
            out[f"transformer.decoder.presence_token_head.layers.{idx}.{wb}"] = hf_sd[
                f"model.detr_decoder.presence_head.{l_name}.{wb}"
            ]
        for idx, l_name in enumerate(("layer1", "layer2")):
            out[f"transformer.decoder.boxRPB_embed_x.layers.{idx}.{wb}"] = hf_sd[
                f"model.detr_decoder.box_rpb_embed_x.{l_name}.{wb}"
            ]
            out[f"transformer.decoder.boxRPB_embed_y.layers.{idx}.{wb}"] = hf_sd[
                f"model.detr_decoder.box_rpb_embed_y.{l_name}.{wb}"
            ]
            out[f"transformer.decoder.ref_point_head.layers.{idx}.{wb}"] = hf_sd[
                f"model.detr_decoder.ref_point_head.{l_name}.{wb}"
            ]
    out["transformer.decoder.presence_token.weight"] = hf_sd["model.detr_decoder.presence_token.weight"]
    out["transformer.decoder.query_embed.weight"] = hf_sd["model.detr_decoder.query_embed.weight"]
    out["transformer.decoder.reference_points.weight"] = hf_sd["model.detr_decoder.reference_points.weight"]

    for wb in ("weight", "bias"):
        out[f"dot_prod_scoring.hs_proj.{wb}"] = hf_sd[f"model.dot_product_scoring.query_proj.{wb}"]
        out[f"dot_prod_scoring.prompt_proj.{wb}"] = hf_sd[f"model.dot_product_scoring.text_proj.{wb}"]
        out[f"dot_prod_scoring.prompt_mlp.out_norm.{wb}"] = hf_sd[
            f"model.dot_product_scoring.text_mlp_out_norm.{wb}"
        ]
        out[f"dot_prod_scoring.prompt_mlp.layers.0.{wb}"] = hf_sd[
            f"model.dot_product_scoring.text_mlp.layer1.{wb}"
        ]
        out[f"dot_prod_scoring.prompt_mlp.layers.1.{wb}"] = hf_sd[
            f"model.dot_product_scoring.text_mlp.layer2.{wb}"
        ]

    for wb in ("weight", "bias"):
        for name in ("boxes_direct_project", "boxes_pool_project", "boxes_pos_enc_project", "final_proj"):
            out[f"geometry_encoder.{name}.{wb}"] = hf_sd[f"model.geometry_encoder.{name}.{wb}"]
        out[f"geometry_encoder.encode_norm.{wb}"] = hf_sd[f"model.geometry_encoder.output_layer_norm.{wb}"]
        out[f"geometry_encoder.norm.{wb}"] = hf_sd[f"model.geometry_encoder.prompt_layer_norm.{wb}"]
        out[f"geometry_encoder.img_pre_norm.{wb}"] = hf_sd[f"model.geometry_encoder.vision_layer_norm.{wb}"]
    out["geometry_encoder.cls_embed.weight"] = hf_sd["model.geometry_encoder.cls_embed.weight"]
    out["geometry_encoder.label_embed.weight"] = hf_sd["model.geometry_encoder.label_embed.weight"]
    for i in range(3):
        hp = f"model.geometry_encoder.layers.{i}"
        lp = f"geometry_encoder.encode.{i}"
        for wb in ("weight", "bias"):
            out[f"{lp}.norm1.{wb}"] = hf_sd[f"{hp}.layer_norm1.{wb}"]
            out[f"{lp}.norm2.{wb}"] = hf_sd[f"{hp}.layer_norm2.{wb}"]
            out[f"{lp}.norm3.{wb}"] = hf_sd[f"{hp}.layer_norm3.{wb}"]
            out[f"{lp}.self_attn.out_proj.{wb}"] = hf_sd[f"{hp}.self_attn.o_proj.{wb}"]
            out[f"{lp}.cross_attn_image.out_proj.{wb}"] = hf_sd[f"{hp}.cross_attn.o_proj.{wb}"]
            out[f"{lp}.linear1.{wb}"] = hf_sd[f"{hp}.mlp.fc1.{wb}"]
            out[f"{lp}.linear2.{wb}"] = hf_sd[f"{hp}.mlp.fc2.{wb}"]
        cat_qkv(f"{hp}.self_attn", f"{lp}.self_attn.in_proj")
        cat_qkv(f"{hp}.cross_attn", f"{lp}.cross_attn_image.in_proj")

    for wb in ("weight", "bias"):
        out[f"segmentation_head.instance_seg_head.{wb}"] = hf_sd[f"model.mask_decoder.instance_projection.{wb}"]
        out[f"segmentation_head.semantic_seg_head.{wb}"] = hf_sd[f"model.mask_decoder.semantic_projection.{wb}"]
        out[f"segmentation_head.cross_attn_norm.{wb}"] = hf_sd[f"model.mask_decoder.prompt_cross_attn_norm.{wb}"]
        out[f"segmentation_head.cross_attend_prompt.out_proj.{wb}"] = hf_sd[
            f"model.mask_decoder.prompt_cross_attn.o_proj.{wb}"
        ]
        for idx in range(3):
            out[f"segmentation_head.mask_predictor.mask_embed.layers.{idx}.{wb}"] = hf_sd[
                f"model.mask_decoder.mask_embedder.layers.{idx}.{wb}"
            ]
            out[f"segmentation_head.pixel_decoder.conv_layers.{idx}.{wb}"] = hf_sd[
                f"model.mask_decoder.pixel_decoder.conv_layers.{idx}.{wb}"
            ]
            out[f"segmentation_head.pixel_decoder.norms.{idx}.{wb}"] = hf_sd[
                f"model.mask_decoder.pixel_decoder.norms.{idx}.{wb}"
            ]
    cat_qkv("model.mask_decoder.prompt_cross_attn", "segmentation_head.cross_attend_prompt.in_proj")

    return out


def _refine_foreground_pil(pil_img, mask_img, r=90):
    img_t = torch.from_numpy(np.array(pil_img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    alpha_t = torch.from_numpy(np.array(mask_img, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    def _blur_fusion(image, fg, bg, alpha, radius):
        k = radius * 2 + 1
        pad = radius
        img_a = F.avg_pool2d(image * alpha, k, stride=1, padding=pad)
        a_mean = F.avg_pool2d(alpha, k, stride=1, padding=pad).clamp(min=1e-6)
        mean_fg = img_a / a_mean
        img_b = F.avg_pool2d(image * (1.0 - alpha), k, stride=1, padding=pad)
        b_mean = F.avg_pool2d(1.0 - alpha, k, stride=1, padding=pad).clamp(min=1e-6)
        mean_bg = img_b / b_mean
        fg_est = mean_fg + alpha * (image - alpha * mean_fg - (1.0 - alpha) * mean_bg)
        return fg_est.clamp(0.0, 1.0)

    fg = _blur_fusion(img_t, img_t, img_t, alpha_t, r)
    fg = _blur_fusion(img_t, fg, img_t, alpha_t, 6)
    fg_np = (fg[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(fg_np, mode="RGB")


class AILab_MultiMatte:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image batch for promptable alpha matting."}),
            },
            "optional": {
                "model": (list(MODEL_CHOICES.keys()), {"default": DEFAULT_MODEL, "tooltip": "Select model precision: BF16 (1.57GB, recommended), FP8 (0.84GB, low VRAM), or FP32 (3.13GB, full precision)."}),
                "prompt": ("STRING", {"default": "", "multiline": False, "placeholder": "Empty = main subject | e.g. person, woman in red, flowing hair, glass bottle, veil", "tooltip": "Describe the target to matte in English (e.g. 'person', 'woman in red dress', 'flowing hair', 'glass cup', 'wedding veil', 'cat fur'). Leave empty to automatically matte the main foreground subject."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "Gaussian blur radius for softening mask edges."}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": "Expand (positive) or shrink (negative) the mask boundary in pixels."}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": "Invert the output mask."}),
                "refine_foreground": ("BOOLEAN", {"default": True, "tooltip": "Sub-pixel foreground boundary refinement for fine hairs and semi-transparent edges."}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Unload model weights from VRAM immediately after execution."}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": "Output format: Alpha for transparent PNG, or Color to composite against background."}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": "Hex background color used when background mode is set to Color."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "process"
    CATEGORY = "🧪AILab/🧽RMBG"

    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded_model_name = None
        self.target_device = device

    def _ensure_model(self, model_name=DEFAULT_MODEL):
        if self.model is not None and self.processor is not None and self.loaded_model_name == model_name:
            return self.model, self.processor

        if self.model is not None:
            release_model(self, "model")
            self.processor = None
            self.loaded_model_name = None

        if SAM3_MODELS_DIR not in sys.path:
            sys.path.insert(0, SAM3_MODELS_DIR)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        filename = MODEL_CHOICES.get(model_name, MODEL_CHOICES[DEFAULT_MODEL])
        local_cache_dir = os.path.join(folder_paths.models_dir, "RMBG", MODEL_CACHE_NAME)
        os.makedirs(local_cache_dir, exist_ok=True)
        ckpt_path = os.path.join(local_cache_dir, filename)
        if not os.path.exists(ckpt_path):
            url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{filename}"
            print(f"[MultiMatte] Downloading {filename} from {MODEL_REPO} to {local_cache_dir}...")
            download_url_to_file(url, ckpt_path)

        model = build_sam3_image_model(
            bpe_path=SAM3_BPE_PATH,
            device="cpu",
            eval_mode=True,
            checkpoint_path=None,
            load_from_HF=False,
            enable_segmentation=True,
        )
        hf_sd = load_file(ckpt_path, device="cpu")
        converted_sd = _convert_hf_sam3_state_dict(_dequantize_fp8_state_dict(hf_sd))
        del hf_sd
        model.load_state_dict(converted_sd, strict=False)
        del converted_sd

        model = model.to(self.target_device).eval()
        self.model = model
        self.processor = Sam3Processor(model, resolution=1008, device=self.target_device)
        self.loaded_model_name = model_name
        return self.model, self.processor

    def process(
        self,
        image,
        model=DEFAULT_MODEL,
        prompt="",
        mask_blur=0,
        mask_offset=0,
        invert_output=False,
        refine_foreground=True,
        unload_model=False,
        background="Alpha",
        background_color="#222222",
    ):
        sam3_model, processor = self._ensure_model(model)

        text_prompt = prompt.strip() if prompt and prompt.strip() else DEFAULT_PROMPT
        processed_images = []
        processed_masks = []
        use_cuda_amp = hasattr(self.target_device, "type") and self.target_device.type == "cuda"

        try:
            for single_img in image:
                pil_img = tensor2pil(single_img).convert("RGB")
                orig_w, orig_h = pil_img.size

                amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_cuda_amp else nullcontext()
                with torch.inference_mode(), amp_ctx:
                    state = processor.set_image(pil_img)
                    text_outputs = sam3_model.backbone.forward_text([text_prompt], device=self.target_device)
                    state["backbone_out"].update(text_outputs)
                    state["geometric_prompt"] = sam3_model._get_dummy_prompt()
                    outputs = sam3_model.forward_grounding(
                        backbone_out=state["backbone_out"],
                        find_input=processor.find_stage,
                        geometric_prompt=state["geometric_prompt"],
                        find_target=None,
                    )

                    alpha = F.interpolate(
                        outputs["semantic_seg"].float(),
                        size=(orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    ).sigmoid()[0, 0]
                    alpha_np = alpha.detach().cpu().numpy()

                mask_img = Image.fromarray((np.clip(alpha_np, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
                mask_img = process_mask(mask_img, invert_output, mask_blur, mask_offset)

                mask_tensor = torch.from_numpy(np.array(mask_img, dtype=np.float32) / 255.0).unsqueeze(0)
                processed_masks.append(mask_tensor)

                if background == "Alpha" and refine_foreground:
                    refined = _refine_foreground_pil(pil_img, mask_img)
                    cutout = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
                    cutout.paste(refined, (0, 0), mask_img)
                    processed_images.append(pil2tensor(cutout))
                else:
                    composed = apply_background_color(pil_img, mask_img, background, background_color)
                    composed = composed.convert("RGBA") if background == "Alpha" else composed.convert("RGB")
                    processed_images.append(pil2tensor(composed))

            out_image = torch.cat(processed_images, dim=0)
            out_mask = torch.cat(processed_masks, dim=0)
            out_mask_image = out_mask.unsqueeze(-1).expand(-1, -1, -1, 3)

            return (out_image, out_mask, out_mask_image)

        finally:
            if unload_model:
                release_model(self, "model")
                self.processor = None
                self.loaded_model_name = None
            else:
                clean_vram()


NODE_CLASS_MAPPINGS = {
    "AILab_MultiMatte": AILab_MultiMatte,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_MultiMatte": "MultiMatte (RMBG)",
}
