# ComfyUI-RMBG
# Interactive Image Compare Node (ComfyUI Standard)
# License: GPL-3.0

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import folder_paths


class AILab_ImageCompareView:
    """Real-time Interactive Image Compare node.

    Renders two images side-by-side or with various comparison modes directly
    on the node canvas.  All UI controls (mode dropdown, match_size checkbox)
    are native ComfyUI widgets — the frontend JS handles canvas rendering and
    mouse interaction only.

    Outputs ``IMAGE`` (the composed result for the current mode) and
    ``DIFF_MASK`` (the absolute-difference mask between the two inputs).
    """

    DESCRIPTION = (
        "Real-time Interactive Image Compare (RMBG) - Compare two images on-node. "
        "Supports wipe (L/R & T/B), overlay opacity blend, pixel difference, "
        "side-by-side, and highlight-diff (red on grey).  Uses native ComfyUI "
        "widgets for mode selection; the frontend handles interactive canvas "
        "rendering."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE", {"tooltip": "First image to compare (e.g. before / original)."}),
                "image2": ("IMAGE", {"tooltip": "Second image to compare (e.g. after / processed)."}),
                "mode": (
                    [
                        "left_right",
                        "up_down",
                        "overlay",
                        "difference",
                        "side_by_side",
                        "highlight_diff",
                    ],
                    {"default": "left_right", "tooltip": "Comparison mode."},
                ),
                "match_size": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If True, resize image2 to match image1 dimensions when they differ.",
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "DIFF_MASK")
    FUNCTION = "compare_images"
    OUTPUT_NODE = True
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_cmp_" + "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5)
        )
        self.compress_level = 4

    # ── Core entry point ───────────────────────────────────────
    def compare_images(self, image1, image2, mode="left_right",
                       match_size=True, prompt=None, extra_pnginfo=None):
        # ── Both images provided (required inputs) ──────────────
        # Ensure RGB (strip alpha if present)
        t1 = image1[..., :3] if image1.shape[-1] == 4 else image1
        t2 = image2[..., :3] if image2.shape[-1] == 4 else image2

        # Align batch size
        b1, b2 = t1.shape[0], t2.shape[0]
        if b1 != b2:
            if b1 == 1:
                t1 = t1.repeat(b2, 1, 1, 1)
            elif b2 == 1:
                t2 = t2.repeat(b1, 1, 1, 1)
            else:
                min_b = min(b1, b2)
                t1 = t1[:min_b]
                t2 = t2[:min_b]

        # Match spatial size if requested (resize image2 → image1)
        if match_size and t1.shape[1:3] != t2.shape[1:3]:
            nchw2 = t2.permute(0, 3, 1, 2)
            resized2 = F.interpolate(
                nchw2, size=(t1.shape[1], t1.shape[2]),
                mode="bicubic", align_corners=False,
            )
            t2 = torch.clamp(resized2.permute(0, 2, 3, 1), 0.0, 1.0)

        # Safety: if sizes still differ (match_size=False), crop both to the
        # common minimum so _compose and diff_mask never crash on shape mismatch.
        if t1.shape[1:3] != t2.shape[1:3]:
            min_h = min(t1.shape[1], t2.shape[1])
            min_w = min(t1.shape[2], t2.shape[2])
            t1 = t1[:, :min_h, :min_w, :]
            t2 = t2[:, :min_h, :min_w, :]

        # Save both images to temp directory for frontend interaction
        results = self._save_images([(1, t1), (2, t2)])

        # Difference mask (used by all modes)
        diff_mask = torch.clamp(torch.abs(t1 - t2).mean(dim=-1), 0.0, 1.0)

        # Compose output based on mode
        out_img = self._compose(mode, t1, t2)

        return {"ui": {"images": results}, "result": (out_img, diff_mask)}

    # ── Mode composition ──────────────────────────────────────
    @staticmethod
    def _compose(mode, t1, t2):
        """Return the composed output tensor for *mode*."""
        if mode == "left_right":
            mid = t1.shape[2] // 2
            out = t2.clone()
            out[:, :, :mid, :] = t1[:, :, :mid, :]
        elif mode == "up_down":
            mid = t1.shape[1] // 2
            out = t2.clone()
            out[:, :mid, :, :] = t1[:, :mid, :, :]
        elif mode == "overlay":
            out = torch.clamp(0.5 * t1 + 0.5 * t2, 0.0, 1.0)
        elif mode == "difference":
            out = torch.clamp(torch.abs(t1 - t2), 0.0, 1.0)
        elif mode == "side_by_side":
            out = torch.cat([t1, t2], dim=2)
        elif mode == "highlight_diff":
            # grey base (luminance of image1), red where pixels differ
            grey = t1.mean(dim=-1, keepdim=True).repeat(1, 1, 1, 3)
            diff = torch.abs(t1 - t2).mean(dim=-1, keepdim=True)
            red = torch.zeros_like(t1)
            red[..., 0] = 1.0  # pure red channel
            threshold = 0.15
            mask = (diff > threshold).expand_as(t1).float()
            out = grey * (1 - mask) + red * mask
        else:
            out = t1
        return out

    # ── Image saving helper ────────────────────────────────────
    def _save_images(self, slot_tensors):
        """Save tensors to temp dir, tagged with *slot* (1 or 2).

        The frontend uses ``slot`` to distinguish image1 from image2.
        """
        results = []
        os.makedirs(self.output_dir, exist_ok=True)
        first_tensor = slot_tensors[0][1]
        prefix = "ailab_cmp" + self.prefix_append
        w = first_tensor.shape[2]
        h = first_tensor.shape[1]

        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(prefix, self.output_dir, w, h)
        )

        for slot, tensor in slot_tensors:
            arr = np.clip(255.0 * tensor[0].cpu().numpy(), 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            file = f"{filename}_{counter:05}_{slot}.png"
            img.save(
                os.path.join(full_output_folder, file),
                compress_level=self.compress_level,
            )
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type,
                "slot": slot,
            })
            counter += 1

        return results


NODE_CLASS_MAPPINGS = {
    "AILab_ImageCompareView": AILab_ImageCompareView,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_ImageCompareView": "Image Compare (RMBG) 🔀🖼️",
}
