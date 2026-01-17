import json
from typing import Any, List, Tuple

import torch


class AILab_Florence2ToCoordinates:
    CATEGORY = "🧪AILab/🧽RMBG"
    RETURN_TYPES = ("STRING", "BBOX", "MASK")
    RETURN_NAMES = ("CENTER_COORDINATES", "BBOXES", "MASK")
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("JSON", {"tooltip": "Florence2 JSON output (list per image)."}),
                "index": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated indexes; blank = use all boxes from first item.",
                    },
                ),
                "batch": ("BOOLEAN", {"default": False, "tooltip": "If true, gather boxes across the batch."}),
            },
            "optional": {
                "image": ("IMAGE",)
            },
        }

    @staticmethod
    def _parse_payload(payload: Any) -> List[Any]:
        if payload is None:
            return []
        if isinstance(payload, str):
            cleaned = payload.strip()
            if not cleaned:
                return []
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # Fallback: some upstream nodes stringify Python lists with single quotes
                try:
                    normalized = cleaned.replace("'", '"')
                    return json.loads(normalized)
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid JSON payload for Florence2 data") from exc
        if isinstance(payload, list):
            return payload
        return []

    @staticmethod
    def _get_bboxes(entry: Any) -> List[List[float]]:
        if isinstance(entry, dict):
            if "bboxes" in entry:
                return entry["bboxes"]
            raise ValueError("Entry does not contain 'bboxes'.")
        if isinstance(entry, list):
            return entry
        raise ValueError("Unsupported entry type; expected dict with 'bboxes' or list of boxes.")

    @staticmethod
    def _parse_indexes(index_str: str, default_count: int) -> List[int]:
        text = index_str.strip()
        if not text:
            return list(range(default_count))
        try:
            return [int(part.strip()) for part in text.split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError("Index must be comma-separated integers.") from exc

    def convert(self, data, index: str, batch: bool = False, image=None):
        records = self._parse_payload(data)
        if not records:
            empty = json.dumps([{"x": 0, "y": 0}])
            mask = self._build_empty_mask(image)
            return (empty, [], mask)

        first_bboxes = self._get_bboxes(records[0])
        if not first_bboxes:
            empty = json.dumps([{"x": 0, "y": 0}])
            mask = self._build_empty_mask(image)
            return (empty, [], mask)

        indexes = self._parse_indexes(index, len(first_bboxes))
        centers = []
        selected_boxes = []
        selections: List[Tuple[int, List[float]]] = []
        max_dims = {}

        def append_box(batch_idx: int, box: List[float]):
            min_x, min_y, max_x, max_y = box
            center_x = int((min_x + max_x) / 2)
            center_y = int((min_y + max_y) / 2)
            centers.append({"x": center_x, "y": center_y})
            selected_boxes.append(box)
            selections.append((batch_idx, box))
            dims = max_dims.setdefault(batch_idx, [1, 1])
            dims[0] = max(dims[0], int(max_x) + 1)
            dims[1] = max(dims[1], int(max_y) + 1)

        if batch:
            for batch_idx, record in enumerate(records):
                boxes = self._get_bboxes(record)
                for idx in indexes:
                    if 0 <= idx < len(boxes):
                        append_box(batch_idx, boxes[idx])
        else:
            boxes = first_bboxes
            for idx in indexes:
                if not 0 <= idx < len(boxes):
                    raise ValueError(f"Index {idx} is out of range for available boxes")
                append_box(0, boxes[idx])

        mask_tensor = self._build_mask_tensor(image, selections, max_dims, batch, len(records))

        return (json.dumps(centers), selected_boxes, mask_tensor)

    @staticmethod
    def _build_empty_mask(image):
        if image is not None:
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            return torch.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2]), dtype=torch.float32, device=tensor.device)
        return torch.zeros((1, 1, 1), dtype=torch.float32)

    def _build_mask_tensor(self, image, selections, max_dims, batch_mode, record_count):
        if not selections:
            return self._build_empty_mask(image)

        if image is not None:
            tensor = image
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            base = torch.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2]), dtype=torch.float32, device=tensor.device)
        else:
            batch_size = record_count if batch_mode else 1
            max_width = max((dims[0] for dims in max_dims.values()), default=1)
            max_height = max((dims[1] for dims in max_dims.values()), default=1)
            base = torch.zeros((batch_size, max_height, max_width), dtype=torch.float32)

        for batch_idx, box in selections:
            min_x, min_y, max_x, max_y = box
            width = base.shape[2]
            height = base.shape[1]
            x0 = max(0, min(int(min_x), width - 1))
            y0 = max(0, min(int(min_y), height - 1))
            x1 = max(x0 + 1, min(int(max_x), width))
            y1 = max(y0 + 1, min(int(max_y), height))
            base[batch_idx, y0:y1, x0:x1] = 1.0

        return base


NODE_CLASS_MAPPINGS = {
    "AILab_Florence2ToCoordinates": AILab_Florence2ToCoordinates,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_Florence2ToCoordinates": "Florence2 Box Coordinates (RMBG)",
}
