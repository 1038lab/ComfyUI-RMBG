import os
import json
from pathlib import Path
from typing import Any, List, Tuple, Optional
import torch
import numpy as np
from PIL import Image, ImageFilter
from huggingface_hub import hf_hub_download
import folder_paths

from AILab_utils import get_device, clean_vram, unload_model as release_model, tensor2pil, pil2tensor

device = get_device()
folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

MODEL_SUBDIR = os.path.join(folder_paths.models_dir, "RMBG", "finegrain-box-segmenter")
REPO_ID = "1038lab/finegrain-box-segmenter"
WEIGHTS_FILENAME = "finegrain-box-segmenter.safetensors"


def parse_bounding_boxes(bboxes_input: Any, box_str: str, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    boxes = []

    if bboxes_input is not None:
        raw_list = []
        if isinstance(bboxes_input, str):
            cleaned = bboxes_input.strip()
            if cleaned:
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, list):
                        raw_list = parsed
                    elif isinstance(parsed, dict):
                        raw_list = parsed.get("bboxes", parsed.get("boxes", []))
                except Exception:
                    pass
        elif isinstance(bboxes_input, dict):
            raw_list = bboxes_input.get("bboxes", bboxes_input.get("boxes", []))
        elif isinstance(bboxes_input, (list, tuple)):
            if len(bboxes_input) == 4 and all(isinstance(x, (int, float)) for x in bboxes_input):
                raw_list = [bboxes_input]
            else:
                raw_list = bboxes_input

        for item in raw_list:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                x1, y1, x2, y2 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
                if max(x1, y1, x2, y2) <= 1.0:
                    x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
                x1_c = max(0, min(img_w, int(round(min(x1, x2)))))
                y1_c = max(0, min(img_h, int(round(min(y1, y2)))))
                x2_c = max(0, min(img_w, int(round(max(x1, x2)))))
                y2_c = max(0, min(img_h, int(round(max(y1, y2)))))
                if x2_c > x1_c and y2_c > y1_c:
                    boxes.append((x1_c, y1_c, x2_c, y2_c))

    if not boxes and box_str:
        cleaned = box_str.strip().strip("[]()").replace(" ", "")
        parts = [p for p in cleaned.split(",") if p]
        if len(parts) >= 4:
            try:
                x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                if max(x1, y1, x2, y2) <= 1.0:
                    x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
                x1_c = max(0, min(img_w, int(round(min(x1, x2)))))
                y1_c = max(0, min(img_h, int(round(min(y1, y2)))))
                x2_c = max(0, min(img_w, int(round(max(x1, x2)))))
                y2_c = max(0, min(img_h, int(round(max(y1, y2)))))
                if x2_c > x1_c and y2_c > y1_c:
                    boxes.append((x1_c, y1_c, x2_c, y2_c))
            except ValueError:
                pass

    return boxes


class AILab_FinegrainBoxSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image batch to segment with high definition 1024px quality."}),
            },
            "optional": {
                "bboxes": ("BBOX", {"tooltip": "Bounding box(es) from Florence-2 or YOLO detection nodes."}),
                "box_coordinates": ("STRING", {"default": "", "tooltip": "Manual bounding box as 'x1, y1, x2, y2' in pixel or normalized [0..1] coordinates."}),
                "margin": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "Safety padding margin around bounding boxes."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "Blur mask edges (0 to disable)."}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": "Expand (positive) or shrink (negative) mask edges in pixels."}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": "Invert the output mask."}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Unload model from VRAM immediately after execution."}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": "Background type: Alpha (transparent) or Color (custom solid background)."}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": "Hex color code for background when Color mode is selected."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment"
    CATEGORY = "🧪AILab/🧽RMBG"

    def __init__(self):
        self.segmenter = None
        self.current_weights_path = None
        self.target_device = device

    def _ensure_model(self, margin=0.05):
        if self.segmenter is not None:
            self.segmenter.margin = margin
            return self.segmenter

        try:
            from refiners.solutions import BoxSegmenter
        except ImportError as e:
            raise RuntimeError("The 'refiners' library is required for Finegrain Box Segmenter. Please run: pip install refiners") from e

        os.makedirs(MODEL_SUBDIR, exist_ok=True)
        local_weights = os.path.join(MODEL_SUBDIR, WEIGHTS_FILENAME)

        if not os.path.exists(local_weights):
            print(f"[Finegrain] Downloading {WEIGHTS_FILENAME} from {REPO_ID} to {MODEL_SUBDIR}...")
            local_weights = hf_hub_download(
                repo_id=REPO_ID,
                filename=WEIGHTS_FILENAME,
                local_dir=MODEL_SUBDIR,
                local_dir_use_symlinks=False
            )

        self.segmenter = BoxSegmenter(
            margin=margin,
            weights=local_weights,
            device=str(self.target_device)
        )
        self.current_weights_path = local_weights
        return self.segmenter

    def segment(self, image, bboxes=None, box_coordinates="", margin=0.05,
                mask_blur=0, mask_offset=0, invert_output=False, unload_model=False,
                background="Alpha", background_color="#222222", **kwargs):
        do_unload = kwargs.get("unload_model", unload_model)
        segmenter = self._ensure_model(margin=margin)

        processed_images = []
        processed_masks = []

        try:
            for single_img in image:
                pil_img = tensor2pil(single_img).convert("RGB")
                orig_w, orig_h = pil_img.size

                parsed_boxes = parse_bounding_boxes(bboxes, box_coordinates, orig_w, orig_h)

                if not parsed_boxes:
                    # Segment whole image / default subject
                    mask_pil = segmenter(pil_img, box_prompt=None)
                    combined_mask_np = np.array(mask_pil, dtype=np.float32)
                else:
                    combined_mask_np = np.zeros((orig_h, orig_w), dtype=np.float32)
                    for box in parsed_boxes:
                        mask_box = segmenter(pil_img, box_prompt=box)
                        mask_np = np.array(mask_box, dtype=np.float32)
                        combined_mask_np = np.maximum(combined_mask_np, mask_np)

                if combined_mask_np.max() > 1.0:
                    combined_mask_np = combined_mask_np / 255.0

                mask_img = Image.fromarray((np.clip(combined_mask_np, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")

                if mask_offset != 0:
                    offset_k = abs(mask_offset) * 2 + 1
                    filter_type = ImageFilter.MaxFilter if mask_offset > 0 else ImageFilter.MinFilter
                    mask_img = mask_img.filter(filter_type(offset_k))

                if mask_blur > 0:
                    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=mask_blur))

                if invert_output:
                    mask_inv = 255.0 - np.array(mask_img, dtype=np.float32)
                    mask_img = Image.fromarray(np.clip(mask_inv, 0, 255).astype(np.uint8), mode="L")

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
                if self.segmenter is not None:
                    if hasattr(self.segmenter, "model") and hasattr(self.segmenter.model, "to"):
                        try:
                            self.segmenter.model.to("cpu")
                        except Exception:
                            pass
                    self.segmenter = None
                clean_vram()
            else:
                clean_vram()


NODE_CLASS_MAPPINGS = {
    "AILab_FinegrainBoxSegment": AILab_FinegrainBoxSegment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_FinegrainBoxSegment": "Finegrain Box Segmenter (RMBG)"
}
