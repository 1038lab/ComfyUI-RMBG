from AILab_model_paths import model_file
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

from AILab_ImageMaskTools import pil2tensor, tensor2pil
from AILab_utils import apply_background_color, clean_vram, process_mask


DEVICE_CHOICES = ("auto", "cuda", "cpu", "mps")
MASK_COUNT_CHOICES = ("all", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10")
MASK_INDEX_CHOICES = ("none", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10")
ANNOTATION_MODES = ("Mask & Box", "Box Only", "Mask Only", "None")
BACKGROUND_CHOICES = ("Alpha", "Color")

class AILab_Yolo:
    CATEGORY = "🧪AILab/🧽RMBG"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("IMAGE", "ANNOTATED_IMAGE", "MASK", "MASK_LIST")
    FUNCTION = "yolo_detect"

    _MODEL_CACHE: Dict[str, "YOLO"] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_path": ("STRING", {"default": "", "tooltip": "Absolute path to a local YOLO .pt weight file on the ComfyUI server."}),
                "mask_count": (MASK_COUNT_CHOICES, {"default": "all", "tooltip": "Merge this many detections. 'all' merges everything (or just the selected index when specified)."}),
            },
            "optional": {
                "select_mask_index": (MASK_INDEX_CHOICES, {"default": "none", "tooltip": "1-based index of the first mask to keep. Use 'none' to start from the first detection."}),
                "conf": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Confidence threshold forwarded to Ultralytics."}),
                "iou": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "IOU used during NMS."}),
                "classes": ("STRING", {"default": "", "placeholder": "e.g. 0,2,5-7", "tooltip": "Comma list or ranges of class IDs; empty keeps every class."}),
                "annotation_mode": (ANNOTATION_MODES, {"default": "Mask & Box", "tooltip": "Display mode for ANNOTATED_IMAGE output."}),
                "background": (BACKGROUND_CHOICES, {"default": "Alpha", "tooltip": "Background type for cutout IMAGE output."}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": "Background color when background is set to Color."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "Apply Gaussian blur to mask edges."}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": "Expand/Shrink mask boundary (positive = expand, negative = shrink)."}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": "Invert the mask output."}),
                "device": (DEVICE_CHOICES, {"default": "auto", "tooltip": "Force a device or auto-detect CUDA → MPS → CPU."}),
                "max_det": ("INT", {"default": 300, "min": 1, "max": 1000, "step": 1, "tooltip": "Maximum detections per image."}),
                "retina_masks": ("BOOLEAN", {"default": True, "tooltip": "Use high-resolution masks (Ultralytics retina_masks flag)."}),
                "agnostic_nms": ("BOOLEAN", {"default": False, "tooltip": "Enable class-agnostic NMS."}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Unload model from VRAM after execution."}),
            },
        }

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _parse_classes(self, value: str) -> Optional[List[int]]:
        if not value or not value.strip():
            return None
        classes: List[int] = []
        try:
            for chunk in value.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if "-" in chunk:
                    start, end = [int(x) for x in chunk.split("-", 1)]
                    if start > end:
                        start, end = end, start
                    classes.extend(range(start, end + 1))
                else:
                    classes.append(int(chunk))
            return sorted(set(classes))
        except ValueError:
            print(f"[AILab_Yolo] Invalid classes string: {value}. Ignoring filter.")
            return None

    def _resolve_model_path(self, path: str) -> str:
        return model_file(path)

    def _get_model(self, model_path: str):
        model = self._MODEL_CACHE.get(model_path)
        if model is None:
            from ultralytics import YOLO
            model = YOLO(model_path)
            self._MODEL_CACHE[model_path] = model
        return model

    def _result_to_annotated(self, result, image_pil: Image.Image, annotation_mode: str) -> torch.Tensor:
        if annotation_mode == "None":
            return pil2tensor(image_pil)
        show_boxes = annotation_mode in ("Mask & Box", "Box Only")
        show_masks = annotation_mode in ("Mask & Box", "Mask Only")
        plotted = result.plot(boxes=show_boxes, masks=show_masks)
        rgb = plotted[..., ::-1]
        return pil2tensor(Image.fromarray(rgb))

    def _mask_from_tensor(self, mask_tensor: torch.Tensor, size: Image.Image.size):
        mask_np = mask_tensor.detach().cpu().numpy()
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        if mask_img.size != size:
            mask_img = mask_img.resize(size, Image.Resampling.NEAREST)
        return torch.from_numpy(np.array(mask_img).astype(np.float32) / 255.0)

    def _collect_masks(self, result, size) -> List[torch.Tensor]:
        width, height = size
        masks: List[torch.Tensor] = []

        if getattr(result, "masks", None) is not None and result.masks.data is not None:
            for mask_tensor in result.masks.data:
                masks.append(self._mask_from_tensor(mask_tensor, size))

        elif getattr(result, "boxes", None) is not None and len(result.boxes.xyxy) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                mask_img = Image.new("L", size, 0)
                draw = ImageDraw.Draw(mask_img)
                draw.rectangle([x1, y1, x2, y2], fill=255)
                masks.append(torch.from_numpy(np.array(mask_img).astype(np.float32) / 255.0))

        if not masks:
            masks.append(torch.zeros((height, width), dtype=torch.float32))

        return masks

    def _merge_masks(self, masks: List[torch.Tensor]) -> torch.Tensor:
        if not masks:
            raise ValueError("Cannot merge empty mask list.")
        merged = torch.zeros_like(masks[0])
        for mask in masks:
            merged = torch.maximum(merged, mask)
        return merged

    def yolo_detect(
        self,
        images,
        model_path="",
        mask_count="all",
        select_mask_index: str = "none",
        conf=0.25,
        iou=0.45,
        classes="",
        annotation_mode="Mask & Box",
        background="Alpha",
        background_color="#222222",
        mask_blur=0,
        mask_offset=0,
        invert_output=False,
        device="auto",
        max_det=300,
        retina_masks=True,
        agnostic_nms=False,
        unload_model: bool = False,
    ):
        try:
            model_path = self._resolve_model_path(model_path)
            model_inst = self._get_model(model_path)
            device_target = self._resolve_device(device)
            class_filter = self._parse_classes(classes)

            cutout_images: List[torch.Tensor] = []
            annotated_images: List[torch.Tensor] = []
            merged_masks: List[torch.Tensor] = []
            mask_list: List[torch.Tensor] = []

            count_limit = 0 if mask_count == "all" else max(0, int(mask_count))
            chosen_index: Optional[int] = None
            if select_mask_index != "none":
                chosen_index = int(select_mask_index) - 1

            for idx in range(images.shape[0]):
                image_pil = tensor2pil(images[idx])
                width, height = image_pil.size

                results = model_inst(
                    image_pil,
                    conf=conf,
                    iou=iou,
                    classes=class_filter,
                    device=device_target,
                    max_det=max_det,
                    retina_masks=retina_masks,
                    agnostic_nms=agnostic_nms,
                )

                result = results[0] if results else None
                if result is not None:
                    annotated_images.append(self._result_to_annotated(result, image_pil, annotation_mode))
                    frame_masks = self._collect_masks(result, (width, height))
                else:
                    annotated_images.append(pil2tensor(image_pil))
                    frame_masks = [torch.zeros((height, width), dtype=torch.float32)]

                selected_masks: List[torch.Tensor]
                if chosen_index is None:
                    if count_limit <= 0 or count_limit >= len(frame_masks):
                        selected_masks = frame_masks
                    else:
                        selected_masks = frame_masks[:count_limit]
                else:
                    if chosen_index >= len(frame_masks):
                        selected_masks = []
                    else:
                        span = count_limit if count_limit > 0 else 1
                        selected_masks = frame_masks[chosen_index : chosen_index + span]

                if selected_masks:
                    raw_merged = self._merge_masks(selected_masks)
                    mask_list.extend(selected_masks)
                else:
                    raw_merged = torch.zeros((height, width), dtype=torch.float32)
                    mask_list.append(raw_merged)

                mask_np = (raw_merged.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np, mode="L")
                processed_mask_pil = process_mask(
                    mask_pil,
                    invert_output=invert_output,
                    mask_blur=mask_blur,
                    mask_offset=mask_offset,
                )

                cutout_pil = apply_background_color(
                    image_pil,
                    processed_mask_pil,
                    background=background,
                    background_color=background_color,
                )
                if background == "Color":
                    cutout_pil = cutout_pil.convert("RGB")
                else:
                    cutout_pil = cutout_pil.convert("RGBA")
                cutout_images.append(pil2tensor(cutout_pil))

                processed_mask_tensor = torch.from_numpy(np.array(processed_mask_pil).astype(np.float32) / 255.0)
                merged_masks.append(processed_mask_tensor)

            if not cutout_images:
                cutout_images = [images]
            if not annotated_images:
                annotated_images = [images]
            if not merged_masks:
                width, height = tensor2pil(images[0]).size
                merged_masks = [torch.zeros((height, width), dtype=torch.float32)]
            if not mask_list:
                width, height = merged_masks[0].shape[1], merged_masks[0].shape[0]
                mask_list = [torch.zeros((height, width), dtype=torch.float32)]

            cutout_tensor = torch.cat(cutout_images, dim=0)
            annotated_tensor = torch.cat(annotated_images, dim=0)
            merged_tensor = torch.stack(merged_masks, dim=0)
            mask_tensor = torch.stack(mask_list, dim=0)

            return cutout_tensor, annotated_tensor, merged_tensor, mask_tensor
        finally:
            if unload_model:
                self._MODEL_CACHE.clear()
            clean_vram()


NODE_CLASS_MAPPINGS = {
    "AILab_Yolo": AILab_Yolo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_Yolo": "YOLO Segmentation (RMBG)",
}
