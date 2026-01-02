# Shared utility helpers
import numpy as np
import torch
from PIL import Image
from comfy.utils import common_upscale


def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def pil2mask(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image.convert("L")).astype(np.float32) / 255.0).unsqueeze(0)


def resize_image(img: Image.Image, width: int, height: int) -> Image.Image:
    return img.resize((width, height), resample=Image.LANCZOS)


def blend_overlay(img_1: Image.Image, img_2: Image.Image) -> Image.Image:
    arr1 = np.array(img_1).astype(float) / 255.0
    arr2 = np.array(img_2).astype(float) / 255.0
    mask = arr2 < 0.5
    result = np.zeros_like(arr1)
    result[mask] = 2 * arr1[mask] * arr2[mask]
    result[~mask] = 1 - 2 * (1 - arr1[~mask]) * (1 - arr2[~mask])
    return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8))


def fill_mask(width: int, height: int, mask: Image.Image, box=(0, 0), color=0) -> Image.Image:
    bg = Image.new("L", (width, height), color)
    bg.paste(mask, box, mask)
    return bg


def empty_image(width: int, height: int, batch_size: int = 1) -> torch.Tensor:
    return torch.zeros([batch_size, height, width, 3])


def upscale_mask(mask: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = common_upscale(mask, width, height, "bicubic", "disabled")
    mask = mask.squeeze(1)
    return mask


def extract_alpha_mask(image: torch.Tensor) -> torch.Tensor:
    alpha = image[..., 3]
    if alpha.max() > 1.0:
        alpha = alpha / 255.0
    if len(alpha.shape) == 4:
        alpha = alpha[:, :, :, 0]
    return alpha.unsqueeze(1) if alpha.ndim == 3 else alpha


def ensure_mask_shape(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[1] == 1:
        return mask.squeeze(1)
    return mask


COLOR_PRESETS = {
    "black": "#000000",
    "white": "#FFFFFF",
    "red": "#FF0000",
    "green": "#00FF00",
    "blue": "#0000FF",
    "yellow": "#FFFF00",
    "cyan": "#00FFFF",
    "magenta": "#FF00FF",
    "gray": "#808080",
    "silver": "#C0C0C0",
    "maroon": "#800000",
    "olive": "#808000",
    "purple": "#800080",
    "teal": "#008080",
    "navy": "#000080",
    "orange": "#FFA500",
    "pink": "#FFC0CB",
    "brown": "#A52A2A",
    "violet": "#EE82EE",
    "indigo": "#4B0082",
    "light_gray": "#D3D3D3",
    "dark_gray": "#A9A9A9",
    "light_blue": "#ADD8E6",
    "dark_blue": "#00008B",
    "light_green": "#90EE90",
    "dark_green": "#006400",
}


def color_format(color: str) -> str:
    if not color:
        return ""

    color = color.strip().upper()
    if not color.startswith("#"):
        color = f"#{color}"

    color = color[1:]
    if len(color) == 3:
        r, g, b = color[0], color[1], color[2]
        return f"#{r}{r}{g}{g}{b}{b}"
    if len(color) < 6:
        raise ValueError(f"Invalid color format: {color}")
    if len(color) > 6:
        color = color[:6]

    return f"#{color}"
