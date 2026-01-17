import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw

import folder_paths
from AILab_ImageMaskTools import pil2tensor, tensor2pil

ULTRALYTICS_DIR = os.path.join(folder_paths.models_dir, "ultralytics")
YOLO_LEGACY_DIR = os.path.join(folder_paths.models_dir, "yolo")
os.makedirs(ULTRALYTICS_DIR, exist_ok=True)
os.makedirs(YOLO_LEGACY_DIR, exist_ok=True)
folder_paths.add_model_folder_path("ultralytics", ULTRALYTICS_DIR, is_default=True)
folder_paths.add_model_folder_path("ultralytics", YOLO_LEGACY_DIR)

DEVICE_CHOICES = ("auto", "cuda", "cpu", "mps")
MASK_COUNT_CHOICES = ("all", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10")
MASK_INDEX_CHOICES = ("none", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10")


class AILab_YoloV8Adv:
    CATEGORY = "🧪AILab/🧽RMBG"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("ANNOTATED_IMAGE", "MASK", "MASK_LIST")
    FUNCTION = "yolo_detect"

    _MODEL_CACHE: Dict[str, "YOLO"] = {}

    @classmethod
    def _list_models(cls) -> List[str]:
        files = folder_paths.get_filename_list("ultralytics")
        return sorted(f for f in files if f.lower().endswith(".pt"))

    @classmethod
    def INPUT_TYPES(cls):
        models = cls._list_models()
        if not models:
            models = [f"Put .pt models into {ULTRALYTICS_DIR}"]
        default_model = models[0]

        return {
            "required": {
                "images": ("IMAGE",),
                "yolo_model": (tuple(models), {"default": default_model, "tooltip": f"YOLOv8 weights stored under {ULTRALYTICS_DIR} (subfolders allowed)."}),
                "mask_count": (MASK_COUNT_CHOICES, {"default": "all", "tooltip": "Merge this many detections. 'all' merges everything (or just the selected index when specified)."}),
            },
            "optional": {
                "select_mask_index": (MASK_INDEX_CHOICES, {"default": "none", "tooltip": "1-based index of the first mask to keep. Use 'none' to start from the first detection."}),
                "conf": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Confidence threshold forwarded to Ultralytics."}),
                "iou": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "IOU used during NMS."}),
                "classes": ("STRING", {"default": "", "placeholder": "e.g. 0,2,5-7", "tooltip": "Comma list or ranges of class IDs; empty keeps every class."}),
                "device": (DEVICE_CHOICES, {"default": "auto", "tooltip": "Force a device or auto-detect CUDA → MPS → CPU."}),
                "max_det": ("INT", {"default": 300, "min": 1, "max": 1000, "step": 1, "tooltip": "Maximum detections per image."}),
                "retina_masks": ("BOOLEAN", {"default": True, "tooltip": "Use high-resolution masks (Ultralytics retina_masks flag)."}),
                "agnostic_nms": ("BOOLEAN", {"default": False, "tooltip": "Enable class-agnostic NMS."}),
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
            print(f"[AILab_YoloV8] Invalid classes string: {value}. Ignoring filter.")
            return None

    def _resolve_model_path(self, name: str) -> str:
        return folder_paths.get_full_path_or_raise("ultralytics", name)

    def _get_model(self, model_path: str):
        model = self._MODEL_CACHE.get(model_path)
        if model is None:
            from ultralytics import YOLO
            model = YOLO(model_path)
            self._MODEL_CACHE[model_path] = model
        return model

    def _result_to_tensor(self, result) -> torch.Tensor:
        plotted = result.plot()
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
        yolo_model,
        mask_count="all",
        conf=0.25,
        iou=0.45,
        classes="",
        device="auto",
        max_det=300,
        retina_masks=True,
        agnostic_nms=False,
        select_mask_index: str = "none",
    ):
        model_path = self._resolve_model_path(yolo_model)
        model = self._get_model(model_path)
        device_target = self._resolve_device(device)
        class_filter = self._parse_classes(classes)

        merged_masks: List[torch.Tensor] = []
        annotated_images: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []

        count_limit = 0 if mask_count == "all" else max(0, int(mask_count))
        chosen_index: Optional[int] = None
        if select_mask_index != "none":
            chosen_index = int(select_mask_index) - 1

        for idx in range(images.shape[0]):
            image_pil = tensor2pil(images[idx])

            results = model(
                image_pil,
                conf=conf,
                iou=iou,
                classes=class_filter,
                device=device_target,
                max_det=max_det,
                retina_masks=retina_masks,
                agnostic_nms=agnostic_nms,
            )

            if not results:
                continue

            result = results[0]
            annotated_images.append(self._result_to_tensor(result))

            frame_masks = self._collect_masks(result, image_pil.size)

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
                merged_masks.append(self._merge_masks(selected_masks))
                mask_list.extend(selected_masks)
            else:
                fallback = torch.zeros_like(frame_masks[0])
                merged_masks.append(fallback)
                mask_list.append(fallback)

        if not merged_masks:
            width, height = tensor2pil(images[0]).size
            merged_masks = [torch.zeros((height, width), dtype=torch.float32)]

        if not mask_list:
            width, height = merged_masks[0].shape[1], merged_masks[0].shape[0]
            mask_list = [torch.zeros((height, width), dtype=torch.float32)]

        if not annotated_images:
            annotated_images = [images]

        merged_tensor = torch.stack(merged_masks, dim=0)
        annotated_tensor = torch.cat(annotated_images, dim=0)
        mask_tensor = torch.stack(mask_list, dim=0)

        return annotated_tensor, merged_tensor, mask_tensor


class AILab_YoloV8(AILab_YoloV8Adv):
    FUNCTION = "yolo_detect_simple"

    @classmethod
    def INPUT_TYPES(cls):
        models = cls._list_models()
        if not models:
            models = [f"Put .pt models into {ULTRALYTICS_DIR}"]
        default_model = models[0]

        return {
            "required": {
                "images": ("IMAGE",),
                "yolo_model": (tuple(models), {"default": default_model, "tooltip": f"YOLOv8 weights stored under {ULTRALYTICS_DIR}. Advanced controls available on YOLOv8 Adv."}),
                "mask_count": (MASK_COUNT_CHOICES, {"default": "all", "tooltip": "Merge this many detections. 'all' merges everything (or just the selected index when specified)."}),
            },
            "optional": {
                "select_mask_index": (MASK_INDEX_CHOICES, {"default": "none", "tooltip": "1-based index of the first mask to keep. Use 'none' to start from the first detection."}),
            },
        }

    def yolo_detect_simple(self, images, yolo_model, mask_count="all", select_mask_index="none"):
        return super().yolo_detect(
            images=images,
            yolo_model=yolo_model,
            mask_count=mask_count,
            conf=0.25,
            iou=0.45,
            classes="",
            device="auto",
            max_det=300,
            retina_masks=True,
            agnostic_nms=False,
            select_mask_index=select_mask_index,
        )


NODE_CLASS_MAPPINGS = {
    "AILab_YoloV8": AILab_YoloV8,
    "AILab_YoloV8Adv": AILab_YoloV8Adv,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_YoloV8": "YOLOv8 (RMBG)",
    "AILab_YoloV8Adv": "YOLOv8 Adv (RMBG)",
}
