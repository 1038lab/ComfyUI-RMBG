# ComfyUI-RMBG
# VRMBG Video Background Removal Node
#
# Model License Notice:
# - VRMBG-3.0: Provided by BRIA AI (https://huggingface.co/briaai/VRMBG-3.0)
#   Please respect the original BRIA AI model license and usage terms.
#
# This integration script follows GPL-3.0 License.
# Source: https://github.com/1038lab/ComfyUI-RMBG

import os
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import folder_paths
import comfy.model_management as mm

from AILab_utils import get_device, clean_vram, unload_model as release_model, tensor2pil, pil2tensor

device = get_device()
folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

MODEL_SUBDIR = os.path.join(folder_paths.models_dir, "RMBG", "VRMBG-3.0")
REPO_URL = "https://huggingface.co/briaai/VRMBG-3.0"

INFER_SIZE = 1024
REQUIRED_FILES = ["config.json", "model.safetensors", "model.py", "vrmbg3_config.py"]


class AILab_VRMBG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of consecutive video frames [B, H, W, C] for temporal background removal."}),
            },
            "optional": {
                "reset_state": ("BOOLEAN", {"default": True, "tooltip": "Reset temporal prior state at the beginning of this batch. Keep True for standalone clips, or False when feeding consecutive chunks."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "Gaussian blur radius for softening mask edges."}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": "Expand (positive) or shrink (negative) the mask boundary in pixels."}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": "Invert the output mask."}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Unload model weights from VRAM immediately after execution."}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": "Output format: Alpha for transparent cutout, or Color to composite against solid background."}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": "Hex background color used when background mode is set to Color."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "process"
    CATEGORY = "🧪AILab/🧽RMBG"

    def __init__(self):
        self.model = None
        self.prev_rgb_t = None
        self.prev_alpha = None
        self.target_device = device
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _find_model_dir(self) -> str:
        for base in folder_paths.get_folder_paths("rmbg"):
            cand = os.path.join(base, "VRMBG-3.0")
            if os.path.isdir(cand) and all(os.path.exists(os.path.join(cand, f)) for f in REQUIRED_FILES):
                return cand
        return MODEL_SUBDIR

    def _ensure_model(self):
        if self.model is not None:
            return self.model

        model_dir = self._find_model_dir()
        missing_files = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(model_dir, f))]

        if missing_files:
            raise FileNotFoundError(
                f"[VRMBG-3.0] Model files missing: {missing_files}\n"
                f"Automatic download has been disabled. Please download the model manually and place all required files in:\n"
                f"  {model_dir}\n\n"
                f"Required files:\n"
                f"  - " + "\n  - ".join(REQUIRED_FILES) + "\n\n"
                f"Download link:\n"
                f"  - {REPO_URL}"
            )

        from transformers import AutoModelForImageSegmentation, PreTrainedModel
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}

        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_dir,
                trust_remote_code=True,
                local_files_only=True
            ).eval()
        except Exception as e:
            raise RuntimeError(f"[VRMBG-3.0] Failed to load local model from '{model_dir}': {e}")

        if not hasattr(self.model, "all_tied_weights_keys"):
            self.model.all_tied_weights_keys = {}

        if torch.cuda.is_available() and str(self.target_device).startswith("cuda"):
            self.model = self.model.half().to(self.target_device)
        else:
            self.model = self.model.to(self.target_device)

        return self.model

    def process(self, images, reset_state=True, mask_blur=0, mask_offset=0,
                invert_output=False, unload_model=False,
                background="Alpha", background_color="#222222", **kwargs):
        do_unload = kwargs.get("unload_model", unload_model)
        model = self._ensure_model()

        model_dtype = next(model.parameters()).dtype

        if reset_state or self.prev_rgb_t is None or self.prev_alpha is None:
            self.prev_rgb_t = torch.zeros(3, INFER_SIZE, INFER_SIZE, device=self.target_device, dtype=model_dtype)
            self.prev_alpha = torch.zeros(1, INFER_SIZE, INFER_SIZE, device=self.target_device, dtype=model_dtype)

        processed_images = []
        processed_masks = []

        try:
            for single_img in images:
                orig_h, orig_w = single_img.shape[0], single_img.shape[1]
                pil_img = tensor2pil(single_img).convert("RGB")

                # Permute [H, W, C] -> [C, H, W]
                img_chw = single_img.permute(2, 0, 1).to(self.target_device, dtype=torch.float32)
                # Resize to 1024x1024
                img_resized = F.interpolate(img_chw.unsqueeze(0), size=(INFER_SIZE, INFER_SIZE), mode="bilinear", align_corners=False).squeeze(0)
                current_t = self.normalize(img_resized).to(dtype=model_dtype)

                # Pair with temporal prior: [current_RGB, prev_RGB * prev_alpha]
                temporal_prior = (self.prev_rgb_t * self.prev_alpha).to(dtype=model_dtype)
                paired = torch.cat([current_t, temporal_prior], dim=0).unsqueeze(0)
                paired = paired.contiguous(memory_format=torch.channels_last)

                with torch.no_grad():
                    raw_out = model(paired)
                    if isinstance(raw_out, (list, tuple)):
                        out_tensor = raw_out[-1]
                    else:
                        out_tensor = raw_out
                    pred_alpha = out_tensor.sigmoid().squeeze(0)  # [1, 1024, 1024]

                # Update autoregressive temporal states for next frame
                self.prev_rgb_t = current_t
                self.prev_alpha = pred_alpha

                # Resize alpha back to native resolution
                alpha_resized = F.interpolate(pred_alpha.unsqueeze(0).float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
                alpha_np = alpha_resized.detach().cpu().numpy()

                mask_img = Image.fromarray((np.clip(alpha_np, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")

                if mask_offset != 0:
                    offset_k = abs(mask_offset) * 2 + 1
                    filter_type = ImageFilter.MaxFilter if mask_offset > 0 else ImageFilter.MinFilter
                    mask_img = mask_img.filter(filter_type(offset_k))

                if mask_blur > 0:
                    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=mask_blur))

                if invert_output:
                    mask_np_inv = 255.0 - np.array(mask_img, dtype=np.float32)
                    mask_img = Image.fromarray(np.clip(mask_np_inv, 0, 255).astype(np.uint8), mode="L")

                mask_tensor = pil2tensor(mask_img)
                processed_masks.append(mask_tensor)

                if background == "Alpha":
                    cutout = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
                    cutout.paste(pil_img, (0, 0), mask_img)
                    processed_images.append(pil2tensor(cutout))
                else:
                    bg_hex = background_color.lstrip("#")
                    if len(bg_hex) == 6:
                        bg_rgb = tuple(int(bg_hex[i:i+2], 16) for i in (0, 2, 4))
                    else:
                        bg_rgb = (34, 34, 34)

                    bg_canvas = Image.new("RGB", (orig_w, orig_h), bg_rgb)
                    bg_canvas.paste(pil_img, (0, 0), mask_img)
                    processed_images.append(pil2tensor(bg_canvas))

            out_image = torch.cat(processed_images, dim=0)
            out_mask = torch.cat(processed_masks, dim=0)
            out_mask_image = out_mask.unsqueeze(-1).expand(-1, -1, -1, 3)

            return (out_image, out_mask, out_mask_image)

        finally:
            if do_unload:
                release_model(self, "model")
                self.prev_rgb_t = None
                self.prev_alpha = None
            else:
                clean_vram()


NODE_CLASS_MAPPINGS = {
    "AILab_VRMBG": AILab_VRMBG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_VRMBG": "VRMBG (Video Background Removal)"
}
