import torch
from comfy.utils import ProgressBar

def color_format(color: str) -> str:
    if not color:
        return ""
    color = color.strip().upper()
    if not color.startswith('#'):
        color = f"#{color}"
    color = color[1:]
    if len(color) == 3:
        r, g, b = color[0], color[1], color[2]
        return f"#{r}{r}{g}{g}{b}{b}"
    elif len(color) < 6:
        raise ValueError(f"Invalid color format: {color}")
    elif len(color) > 6:
        color = color[:6]
    return f"#{color}"

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b

def validate_image_data(images):
    if not isinstance(images, torch.Tensor):
        raise TypeError("Images must be a torch tensor.")
    if images.min() < 0 or images.max() > 255:
        raise ValueError("Images must have pixel values in the range [0, 255].")

def compute_color_distances(images, color):
    return torch.norm(images - color, dim=-1)

class AILab_ColorToMask:
    RETURN_TYPES = ("MASK",)
    FUNCTION = "color_to_mask"
    CATEGORY = "🧪AILab/🧽RMBG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "invert": ("BOOLEAN", {"default": False}),
                 "threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                 "mask_color": ("COLORCODE", {"default": "#FFFFFF", "tooltip": "Mask color (hex)"}),
        },
    }

    def color_to_mask(self, images, mask_color, threshold, invert):
        validate_image_data(images)

        mask_color = color_format(mask_color)
        r, g, b = hex_to_rgb(mask_color)
        color = torch.tensor([r, g, b], dtype=torch.float32)

        black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        white = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        
        if invert:
            black, white = white, black

        per_batch = 16
        device = images.device
        color = color.to(device)
        black = black.to(device)
        white = white.to(device)

        if images.max() > 1.0:
            images = images / 255.0

        steps = images.shape[0]
        pbar = ProgressBar(steps)
        tensors_out = []

        for start_idx in range(0, steps, per_batch):
            end_idx = min(start_idx + per_batch, steps)
            batch = images[start_idx:end_idx]

            color_distances = compute_color_distances(batch, color)
            mask = color_distances <= threshold / 255.0
            mask_out = torch.where(mask.unsqueeze(-1), white, black).float()
            mask_out = mask_out.mean(dim=-1)

            tensors_out.append(mask_out.cpu())
            pbar.update(end_idx - start_idx)

        tensors_out = torch.cat(tensors_out, dim=0)
        tensors_out = torch.clamp(tensors_out, min=0.0, max=1.0)
        return tensors_out,

NODE_CLASS_MAPPINGS = {
    "AILab_ColorToMask": AILab_ColorToMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_ColorToMask": "Color to Mask (RMBG) 🎭"
}
