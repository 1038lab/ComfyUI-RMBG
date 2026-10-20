import json
import os
import torch
import torchvision.transforms.functional as TF
from typing import Dict, Any, Tuple, List
from PIL import Image
import numpy as np
import folder_paths
from AILab_utils import clean_vram, get_device

def hex_to_rgba(hex_str: str) -> Tuple[float, float, float, float]:
    if not isinstance(hex_str, str):
        return (0.0, 0.0, 0.0, 1.0)
    cleaned = hex_str.strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join([c * 2 for c in cleaned]) + "FF"
    elif len(cleaned) == 6:
        cleaned = cleaned + "FF"
    elif len(cleaned) != 8:
        return (0.0, 0.0, 0.0, 1.0)
    try:
        r = int(cleaned[0:2], 16) / 255.0
        g = int(cleaned[2:4], 16) / 255.0
        b = int(cleaned[4:6], 16) / 255.0
        a = int(cleaned[6:8], 16) / 255.0
        return (r, g, b, a)
    except Exception:
        return (0.0, 0.0, 0.0, 1.0)

class AILab_Canvas:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "canvas_width": (
                    "INT",
                    {
                        "default": 640,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Canvas width in pixels when not fitting background image size.",
                    },
                ),
                "canvas_height": (
                    "INT",
                    {
                        "default": 360,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Canvas height in pixels when not fitting background image size.",
                    },
                ),
                "background": (
                    ["Alpha", "Color"],
                    {
                        "default": "Alpha",
                        "tooltip": "Choose background type: Alpha (transparent) or Color (custom background color).",
                    },
                ),
                "background_color": (
                    "COLORCODE",
                    {
                        "default": "#222222",
                        "tooltip": "Pick background color (use color picker).",
                    },
                ),
                "fit_background_size": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When True, canvas dimensions match the background image. When False, canvas dimensions use canvas_width and canvas_height.",
                    },
                ),
                "canvas_state": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": "Serialized JSON layout containing layer coordinates, scales, rotations, and order.",
                    },
                ),
            },
            "optional": {
                "Background": (
                    "IMAGE",
                    {"tooltip": "Optional background image input."},
                ),
                "layer_1": (
                    "IMAGE",
                    {"tooltip": "Layer 1 image input."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "composite"
    OUTPUT_NODE = True
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def _save_images(self, slot_tensors: List[Tuple[str, torch.Tensor]]) -> List[Dict[str, Any]]:
        results = []
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        for slot_name, tensor in slot_tensors:
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            t = tensor[0].detach().cpu()
            h, w = t.shape[0], t.shape[1]
            full_out_dir, base_filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                f"ailab_canvas_{slot_name.lower()}", output_dir, w, h
            )
            arr = np.clip(255.0 * t.numpy(), 0, 255).astype(np.uint8)
            if arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
                img = Image.fromarray(arr, mode="L")
            elif arr.shape[-1] == 3:
                img = Image.fromarray(arr, mode="RGB")
            elif arr.shape[-1] == 4:
                img = Image.fromarray(arr, mode="RGBA")
            else:
                img = Image.fromarray(arr[..., :3], mode="RGB")

            file = f"{base_filename}_{counter:05}_{slot_name}.png"
            img.save(os.path.join(full_out_dir, file), compress_level=1)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "temp",
                "slot": slot_name,
                "width": w,
                "height": h,
            })
        return results

    def composite(
        self,
        canvas_width: int,
        canvas_height: int,
        background: str = "Alpha",
        background_color: str = "#222222",
        fit_background_size: bool = True,
        canvas_state: str = "{}",
        **kwargs: Any,
    ):
        try:
            bg_tensor = kwargs.get("Background", kwargs.get("background_image", None))
            if not canvas_state or canvas_state == "{}":
                canvas_state = kwargs.get("canvas_state", "{}")

            layer_dict = {}
            for i in range(1, 6):
                key = f"layer_{i}"
                if key in kwargs and isinstance(kwargs[key], torch.Tensor):
                    layer_dict[key] = kwargs[key]

            for k, v in kwargs.items():
                k_clean = k.lower().replace(" ", "_")
                if k_clean.startswith("layer_") and isinstance(v, torch.Tensor):
                    layer_dict[k_clean] = v

            device = bg_tensor.device if bg_tensor is not None else (
                next(iter(layer_dict.values())).device if layer_dict else get_device()
            )

            if bg_tensor is not None and fit_background_size:
                target_h = int(bg_tensor.shape[1])
                target_w = int(bg_tensor.shape[2])
            else:
                target_h = max(64, int(canvas_height))
                target_w = max(64, int(canvas_width))
                target_h = (target_h // 8) * 8
                target_w = (target_w // 8) * 8

            batch_size = 1
            if bg_tensor is not None:
                batch_size = max(batch_size, bg_tensor.shape[0])
            for l_tensor in layer_dict.values():
                batch_size = max(batch_size, l_tensor.shape[0])

            canvas = torch.zeros((batch_size, 4, target_h, target_w), dtype=torch.float32, device=device)
            fg_mask = torch.zeros((batch_size, 1, target_h, target_w), dtype=torch.float32, device=device)

            if bg_tensor is not None:
                bg = bg_tensor.to(device=device, dtype=torch.float32)
                if bg.shape[0] < batch_size:
                    bg = bg.repeat(batch_size, 1, 1, 1)

                if bg.shape[1] != target_h or bg.shape[2] != target_w:
                    bg_perm = bg.permute(0, 3, 1, 2)
                    bg_resized = TF.resize(bg_perm, [target_h, target_w], interpolation=TF.InterpolationMode.BILINEAR)
                    bg = bg_resized.permute(0, 2, 3, 1)

                if bg.shape[-1] >= 4:
                    canvas[:, :3, :, :] = bg[..., :3].permute(0, 3, 1, 2)
                    canvas[:, 3:4, :, :] = bg[..., 3:4].permute(0, 3, 1, 2)
                else:
                    canvas[:, :3, :, :] = bg[..., :3].permute(0, 3, 1, 2)
                    canvas[:, 3:4, :, :] = 1.0
            else:
                if background == "Alpha":
                    canvas[:, :, :, :] = 0.0
                else:
                    r, g, b, a = hex_to_rgba(background_color)
                    canvas[:, 0, :, :] = r
                    canvas[:, 1, :, :] = g
                    canvas[:, 2, :, :] = b
                    canvas[:, 3, :, :] = a

            raw_layers = []
            for k in sorted(layer_dict.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 99):
                v = layer_dict[k]
                if v is not None and isinstance(v, torch.Tensor):
                    raw_layers.append((k, v))

            parsed_state = {}
            if canvas_state and isinstance(canvas_state, str) and canvas_state.strip():
                try:
                    parsed_state = json.loads(canvas_state)
                except Exception:
                    parsed_state = {}

            state_layer_map = {}
            for item in parsed_state.get("layers", []):
                if isinstance(item, dict) and "id" in item:
                    state_layer_map[str(item["id"]).lower()] = item

            state_width = float(parsed_state.get("canvas_width", target_w) or target_w)
            state_height = float(parsed_state.get("canvas_height", target_h) or target_h)
            scale_ratio_x = target_w / max(1.0, state_width)
            scale_ratio_y = target_h / max(1.0, state_height)

            ordered_layers = []
            for layer_name, tensor in raw_layers:
                layer_cfg = state_layer_map.get(layer_name.lower(), {})
                z_index = int(layer_cfg.get("zIndex", layer_cfg.get("z_index", 0)))
                ordered_layers.append((z_index, layer_name, tensor, layer_cfg))

            ordered_layers.sort(key=lambda x: x[0])

            for _, _, fg_tensor, cfg in ordered_layers:
                fg = fg_tensor.to(device=device, dtype=torch.float32)
                if fg.shape[0] < batch_size:
                    fg = fg.repeat(batch_size, 1, 1, 1)

                orig_h, orig_w = int(fg.shape[1]), int(fg.shape[2])

                if fg.shape[-1] >= 4:
                    fg_rgb = fg[..., :3].permute(0, 3, 1, 2)
                    fg_a = fg[..., 3:4].permute(0, 3, 1, 2)
                else:
                    fg_rgb = fg[..., :3].permute(0, 3, 1, 2)
                    fg_a = torch.ones((batch_size, 1, orig_h, orig_w), dtype=torch.float32, device=device)

                opacity = float(cfg.get("opacity", 1.0))
                opacity = max(0.0, min(1.0, opacity))
                fg_a = fg_a * opacity

                flip_x = bool(cfg.get("flipX", cfg.get("flip_x", False)))
                flip_y = bool(cfg.get("flipY", cfg.get("flip_y", False)))
                if flip_x:
                    fg_rgb = torch.flip(fg_rgb, dims=[3])
                    fg_a = torch.flip(fg_a, dims=[3])
                if flip_y:
                    fg_rgb = torch.flip(fg_rgb, dims=[2])
                    fg_a = torch.flip(fg_a, dims=[2])

                max_w = target_w * 0.85
                max_h = target_h * 0.85
                if orig_w > max_w or orig_h > max_h:
                    default_scale = min(max_w / max(1, orig_w), max_h / max(1, orig_h))
                else:
                    default_scale = 1.0

                scale = float(cfg.get("scaleX", cfg.get("scale_x", cfg.get("scaleY", default_scale))))
                scale_x = scale
                scale_y = scale

                new_w = max(1, int(round(orig_w * scale_x * scale_ratio_x)))
                new_h = max(1, int(round(orig_h * scale_y * scale_ratio_y)))

                fg_rgba = torch.cat([fg_rgb, fg_a], dim=1)
                if new_w != orig_w or new_h != orig_h:
                    fg_rgba = TF.resize(fg_rgba, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)

                angle = float(cfg.get("angle", 0.0))
                if abs(angle) > 1e-3:
                    fg_rgba = TF.rotate(fg_rgba, angle=-angle, expand=True, interpolation=TF.InterpolationMode.BILINEAR)

                rot_h, rot_w = fg_rgba.shape[2], fg_rgba.shape[3]

                if "x" in cfg and "y" in cfg:
                    raw_cx = float(cfg["x"]) * scale_ratio_x
                    raw_cy = float(cfg["y"]) * scale_ratio_y
                else:
                    raw_cx = target_w / 2.0
                    raw_cy = target_h / 2.0

                x0 = int(round(raw_cx - rot_w / 2.0))
                y0 = int(round(raw_cy - rot_h / 2.0))
                x1 = x0 + rot_w
                y1 = y0 + rot_h

                dst_x0, dst_y0 = max(0, x0), max(0, y0)
                dst_x1, dst_y1 = min(target_w, x1), min(target_h, y1)
                src_x0 = max(0, -x0)
                src_y0 = max(0, -y0)
                src_x1 = src_x0 + (dst_x1 - dst_x0)
                src_y1 = src_y0 + (dst_y1 - dst_y0)

                if dst_x1 > dst_x0 and dst_y1 > dst_y0:
                    fg_patch = fg_rgba[:, :, src_y0:src_y1, src_x0:src_x1]
                    bg_patch = canvas[:, :, dst_y0:dst_y1, dst_x0:dst_x1]

                    p_fg_rgb = fg_patch[:, :3]
                    p_fg_a = fg_patch[:, 3:4]
                    p_bg_rgb = bg_patch[:, :3]
                    p_bg_a = bg_patch[:, 3:4]

                    out_a = p_fg_a + p_bg_a * (1.0 - p_fg_a)
                    out_rgb = torch.where(
                        out_a > 1e-6,
                        (p_fg_rgb * p_fg_a + p_bg_rgb * p_bg_a * (1.0 - p_fg_a)) / out_a,
                        torch.zeros_like(p_fg_rgb),
                    )

                    canvas[:, :3, dst_y0:dst_y1, dst_x0:dst_x1] = out_rgb
                    canvas[:, 3:4, dst_y0:dst_y1, dst_x0:dst_x1] = out_a

                    curr_fg_a = fg_mask[:, :, dst_y0:dst_y1, dst_x0:dst_x1]
                    fg_mask[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = p_fg_a + curr_fg_a * (1.0 - p_fg_a)

            if background == "Alpha" and (bg_tensor is None or bg_tensor.shape[-1] >= 4):
                out_image = canvas.permute(0, 2, 3, 1).clamp(0.0, 1.0)
            else:
                out_image = canvas[:, :3, :, :].permute(0, 2, 3, 1).clamp(0.0, 1.0)
            out_mask = fg_mask.squeeze(1).clamp(0.0, 1.0)

            slot_tensors = [("Background", bg_tensor)]
            for i in range(1, 6):
                slot_tensors.append((f"layer_{i}", layer_dict.get(f"layer_{i}")))

            ui_images = self._save_images(slot_tensors)

            return {
                "ui": {"images": ui_images},
                "result": (out_image.cpu(), out_mask.cpu()),
            }

        finally:
            clean_vram()

NODE_CLASS_MAPPINGS = {
    "AILab_Canvas": AILab_Canvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_Canvas": "Canvas (RMBG)",
}