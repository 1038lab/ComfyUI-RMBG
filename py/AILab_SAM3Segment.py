import os
import sys
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.hub import download_url_to_file

import folder_paths
import comfy.model_management

from AILab_ImageMaskTools import pil2tensor, tensor2pil
from AILab_utils import clean_vram, process_mask, apply_background_color, get_or_download_model_file

# ── paths ──────────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SAM3_LOCAL_DIR = REPO_ROOT / "models" / "sam3"
MODELS_ROOT = REPO_ROOT / "models"

if "sam3" not in sys.modules:
    sys.path.insert(0, str(SAM3_LOCAL_DIR))
    sys.path.insert(0, str(MODELS_ROOT))

SAM3_BPE_PATH = SAM3_LOCAL_DIR / "assets" / "bpe_simple_vocab_16e6.txt.gz"
if not os.path.isfile(SAM3_BPE_PATH):
    raise RuntimeError("SAM3 assets missing; ensure sam3/assets/bpe_simple_vocab_16e6.txt.gz exists.")

# ── safetensors import ─────────────────────────────────────────────────
try:
    from safetensors.torch import load_file as safetensors_load_file
except ImportError:
    safetensors_load_file = None

# ── model registry ─────────────────────────────────────────────────────
SAM3_SAFETENSOR_MODELS = {
    "sam3.1_multiplex_fp16": {
        "model_url": "https://huggingface.co/1038lab/sam3/resolve/main/sam3.1_multiplex_fp16.safetensors",
        "filename": "sam3.1_multiplex_fp16.safetensors",
        "dtype": "float16",
    },
    "sam3": {
        "model_url": "https://huggingface.co/1038lab/sam3/resolve/main/sam3.safetensors",
        "filename": "sam3.safetensors",
        "dtype": "float32",
    },
}


def _resolve_device(user_choice):
    auto_device = comfy.model_management.get_torch_device()
    if user_choice == "CPU":
        return torch.device("cpu")
    if user_choice == "GPU":
        if auto_device.type != "cuda":
            raise RuntimeError("GPU unavailable")
        return torch.device("cuda")
    return auto_device


# ── safetensors-aware checkpoint loader ────────────────────────────────
def _load_safetensors_checkpoint(model, checkpoint_path):
    """Load safetensors checkpoint with smart key-mapping."""
    if safetensors_load_file is None:
        raise ImportError(
            "safetensors is required for SAM3 Multiplex. "
            "Install with: pip install safetensors"
        )
    state_dict = safetensors_load_file(checkpoint_path, device="cpu")

    # Sniff key structure: if keys have 'detector.' prefix, strip it
    has_detector_prefix = any(k.startswith("detector.") for k in state_dict)
    if has_detector_prefix:
        sam3_image_ckpt = {
            k.replace("detector.", ""): v
            for k, v in state_dict.items()
            if "detector" in k
        }
        if model.inst_interactive_predictor is not None:
            sam3_image_ckpt.update({
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in state_dict.items()
                if "tracker" in k
            })
    else:
        # Flat keys — load as-is
        sam3_image_ckpt = state_dict

    missing_keys, unexpected_keys = model.load_state_dict(sam3_image_ckpt, strict=False)
    if missing_keys:
        print(f"[SAM3 Multiplex] loaded {checkpoint_path}")
        print(f"  missing keys ({len(missing_keys)}): {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
    if unexpected_keys:
        print(f"  unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")


# ── model builder import ───────────────────────────────────────────────
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402


class SAM3Segment:
    @classmethod
    def INPUT_TYPES(cls):
        model_names = list(SAM3_SAFETENSOR_MODELS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (model_names, {"default": model_names[0]}),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Describe the concept"}),
                "output_mode": (["Merged", "Separate"], {"default": "Merged"}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.01}),
            },
            "optional": {
                "max_segments": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "segment_pick": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "device": (["Auto", "CPU", "GPU"], {"default": "Auto"}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "unload_model": ("BOOLEAN", {"default": False}),
                "background": (["Alpha", "Color"], {"default": "Alpha"}),
                "background_color": ("COLORCODE", {"default": "#222222"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment"
    CATEGORY = "🧪AILab/🧽RMBG"

    def __init__(self):
        self.processor_cache = {}

    def _load_processor(self, model_name, device_choice):
        torch_device = _resolve_device(device_choice)
        device_str = "cuda" if torch_device.type == "cuda" else "cpu"
        cache_key = (model_name, device_str)
        if cache_key not in self.processor_cache:
            model_info = SAM3_SAFETENSOR_MODELS[model_name]
            ckpt_path = get_or_download_model_file(model_info["filename"], model_info["model_url"], "sam3")

            # Build the model architecture (without loading weights)
            model = build_sam3_image_model(
                bpe_path=SAM3_BPE_PATH,
                device="cpu",  # load on CPU first, move later
                eval_mode=True,
                checkpoint_path=None,  # we load manually below
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )

            # Load safetensors checkpoint
            _load_safetensors_checkpoint(model, ckpt_path)

            # Move to target device (keep FP32 — autocast handles precision at inference)
            if device_str == "cuda":
                model = model.cuda()
            model.eval()

            processor = Sam3Processor(model, device=device_str)
            self.processor_cache[cache_key] = processor
            print(f"[SAM3 Multiplex] Loaded {model_name} on {device_str} ({model_info['dtype']})")
        return self.processor_cache[cache_key], torch_device

    def _empty_result(self, img_pil, background, background_color):
        w, h = img_pil.size
        mask_image = Image.new("L", (w, h), 0)
        result_image = apply_background_color(img_pil, mask_image, background, background_color)
        result_image = result_image.convert("RGBA") if background == "Alpha" else result_image.convert("RGB")
        empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
        mask_rgb = empty_mask.reshape((-1, 1, h, w)).movedim(1, -1).expand(-1, -1, -1, 3)
        return result_image, empty_mask, mask_rgb

    def _empty_batch(self, img_pil):
        w, h = img_pil.size
        empty_imgs = torch.zeros((0, h, w, 3), dtype=torch.float32)
        empty_masks = torch.zeros((0, h, w), dtype=torch.float32)
        empty_mask_images = torch.zeros((0, h, w, 3), dtype=torch.float32)
        return empty_imgs, empty_masks, empty_mask_images

    def _run_single_per_instance(self, processor, img_tensor, prompt, confidence, max_segments, segment_pick, mask_blur, mask_offset, invert, unload_model, background, background_color):
        img_pil = tensor2pil(img_tensor)
        text = prompt.strip() or "object"
        state = processor.set_image(img_pil)
        processor.reset_all_prompts(state)
        processor.set_confidence_threshold(confidence, state)
        state = processor.set_text_prompt(text, state)
        masks = state.get("masks")
        logits = state.get("masks_logits")
        if masks is None or masks.numel() == 0:
            return self._empty_batch(img_pil)
        masks = masks.float()
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        scores = None
        if logits is not None:
            logits = logits.float()
            if logits.ndim == 4:
                logits = logits.squeeze(1)
            scores = logits.mean(dim=(-2, -1))
        if scores is None:
            scores = torch.ones((masks.shape[0],), device=masks.device)
        if max_segments > 0 and masks.shape[0] > max_segments:
            topk = torch.topk(scores, k=max_segments)
            masks = masks[topk.indices]
            scores = scores[topk.indices]
        sorted_idx = torch.argsort(scores, descending=True)
        masks = masks[sorted_idx]
        if segment_pick > 0:
            idx = segment_pick - 1
            if idx >= masks.shape[0]:
                return self._empty_batch(img_pil)
            masks = masks[idx : idx + 1]

        mask_imgs, mask_tensors, mask_rgb_list = [], [], []
        for single_mask in masks:
            mask_np = (single_mask.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            mask_image = Image.fromarray(mask_np, mode="L")
            mask_image = process_mask(mask_image, invert, mask_blur, mask_offset)
            composed = apply_background_color(img_pil, mask_image, background, background_color)
            composed = composed.convert("RGBA") if background == "Alpha" else composed.convert("RGB")
            mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
            mask_rgb = mask_tensor.reshape((1, mask_image.height, mask_image.width, 1)).expand(-1, -1, -1, 3)
            mask_imgs.append(pil2tensor(composed))
            mask_tensors.append(mask_tensor)
            mask_rgb_list.append(mask_rgb)
        return (
            torch.cat(mask_imgs, dim=0),
            torch.cat(mask_tensors, dim=0),
            torch.cat(mask_rgb_list, dim=0),
        )

    def _run_single_merged(self, processor, img_tensor, prompt, confidence, max_segments, segment_pick, mask_blur, mask_offset, invert, unload_model, background, background_color):
        img_pil = tensor2pil(img_tensor)
        imgs, masks, _ = self._run_single_per_instance(
            processor, img_tensor, prompt, confidence, max_segments, segment_pick,
            mask_blur, mask_offset, invert, unload_model, background, background_color,
        )
        if masks.shape[0] == 0:
            return self._empty_result(img_pil, background, background_color)
        merged = masks.amax(dim=0)
        mask_np = (merged.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_np, mode="L")
        mask_image = process_mask(mask_image, invert, mask_blur, mask_offset)
        result_image = apply_background_color(img_pil, mask_image, background, background_color)
        result_image = result_image.convert("RGBA") if background == "Alpha" else result_image.convert("RGB")
        mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
        mask_rgb = mask_tensor.reshape((1, mask_image.height, mask_image.width, 1)).expand(-1, -1, -1, 3)
        return result_image, mask_tensor, mask_rgb

    def segment(self, image, model_name, prompt, device, confidence_threshold=0.5, max_segments=0, segment_pick=0, mask_blur=0, mask_offset=0, invert_output=False, unload_model=False, background="Alpha", background_color="#222222", output_mode="Merged"):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        processor, torch_device = self._load_processor(model_name, device)
        autocast_device = comfy.model_management.get_autocast_device(torch_device)
        autocast_enabled = torch_device.type == "cuda" and not comfy.model_management.is_device_mps(torch_device)

        # Use bfloat16 for original sam3, float16 for multiplex fp16
        model_info = SAM3_SAFETENSOR_MODELS[model_name]
        autocast_dtype = torch.float16 if model_info["dtype"] == "float16" else torch.bfloat16
        ctx = torch.autocast(autocast_device, dtype=autocast_dtype) if autocast_enabled else nullcontext()

        result_images, result_masks, result_mask_images = [], [], []
        with ctx:
            for tensor_img in image:
                if output_mode == "Separate":
                    imgs_batch, masks_batch, mask_imgs_batch = self._run_single_per_instance(
                        processor, tensor_img, prompt, confidence_threshold,
                        max_segments, segment_pick, mask_blur, mask_offset,
                        invert_output, unload_model, background, background_color,
                    )
                    result_images.append(imgs_batch)
                    result_masks.append(masks_batch)
                    result_mask_images.append(mask_imgs_batch)
                else:
                    img_pil, mask_tensor, mask_rgb = self._run_single_merged(
                        processor, tensor_img, prompt, confidence_threshold,
                        max_segments, segment_pick, mask_blur, mask_offset,
                        invert_output, unload_model, background, background_color,
                    )
                    result_images.append(pil2tensor(img_pil))
                    result_masks.append(mask_tensor)
                    result_mask_images.append(mask_rgb)

        if unload_model:
            device_str = "cuda" if torch_device.type == "cuda" else "cpu"
            cache_key = (model_name, device_str)
            if cache_key in self.processor_cache:
                del self.processor_cache[cache_key]
            clean_vram()

        final_images = torch.cat(result_images, dim=0)
        final_masks = torch.cat(result_masks, dim=0)
        final_mask_images = torch.cat(result_mask_images, dim=0)

        # If no segments found in Separate mode, return at least one empty result
        if final_images.shape[0] == 0:
            img_pil = tensor2pil(image[0])
            empty_img, empty_mask, empty_mask_img = self._empty_result(img_pil, background, background_color)
            final_images = pil2tensor(empty_img)
            final_masks = empty_mask
            final_mask_images = empty_mask_img

        return final_images, final_masks, final_mask_images


NODE_CLASS_MAPPINGS = {"SAM3Segment": SAM3Segment}
NODE_DISPLAY_NAME_MAPPINGS = {"SAM3Segment": "SAM3 Segmentation (RMBG)"}
