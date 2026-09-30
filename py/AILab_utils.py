# Shared utility helpers
import os
import gc
from typing import Union, Optional
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.hub import download_url_to_file
from comfy.utils import common_upscale
import comfy.model_management as mm
import folder_paths


def get_device():
    """Return the torch device ComfyUI selected (CUDA, MPS, or CPU)."""
    return mm.get_torch_device()


def clean_vram():
    """Safely clear cached VRAM across CUDA, MPS, and CPU backends."""
    gc.collect()
    mm.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def unload_model(loader_or_model, attr_name="model"):
    """Unload a model from memory and release VRAM.

    Accepts either:
    1. A loader object that owns a model attribute (e.g. loader, attr_name='model')
    2. A raw torch.nn.Module instance directly.
    """
    if hasattr(loader_or_model, attr_name):
        model = getattr(loader_or_model, attr_name)
        if model is not None:
            if hasattr(model, "to"):
                try:
                    model.to("cpu")
                except Exception:
                    pass
            setattr(loader_or_model, attr_name, None)
            if hasattr(loader_or_model, "current_model_version"):
                loader_or_model.current_model_version = None
    elif isinstance(loader_or_model, torch.nn.Module):
        try:
            loader_or_model.to("cpu")
        except Exception:
            pass
        del loader_or_model

    clean_vram()


def patch_transformers_bert():
    """Ensure BertModel and BertModelWarper compatibility for GroundingDINO on transformers >= 5.0."""
    try:
        def _get_head_mask(self, head_mask=None, num_hidden_layers=12, is_attention_chunked=False):
            if head_mask is not None:
                if hasattr(head_mask, "dim"):
                    if head_mask.dim() == 1:
                        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                    elif head_mask.dim() == 2:
                        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                if is_attention_chunked and hasattr(head_mask, "unsqueeze"):
                    head_mask = head_mask.unsqueeze(-1)
                return head_mask
            return [None] * num_hidden_layers

        def _adapted_get_extended_attention_mask(self, attention_mask, input_shape=None, *args, **kwargs):
            device = kwargs.pop("device", None)
            dtype = kwargs.pop("dtype", None)
            if len(args) >= 1:
                arg = args[0]
                if isinstance(arg, torch.device) or (isinstance(arg, str) and str(arg).startswith(("cuda", "cpu", "mps"))):
                    device = arg
                elif isinstance(arg, torch.dtype):
                    dtype = arg
            if len(args) >= 2:
                arg2 = args[1]
                if isinstance(arg2, torch.dtype):
                    dtype = arg2
                elif isinstance(arg2, (torch.device, str)):
                    device = arg2

            if dtype is None:
                dtype = getattr(self, "dtype", torch.float32)

            if attention_mask.dim() == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            else:
                raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

            if device is not None:
                extended_attention_mask = extended_attention_mask.to(device=device)
            if dtype is not None:
                extended_attention_mask = extended_attention_mask.to(dtype=dtype)

            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
            return extended_attention_mask

        # 1. Patch BertModelWarper directly in GroundingDINO
        try:
            import groundingdino.models.GroundingDINO.bertwarper as bw
            if not getattr(bw, "_rmbg_patched", False):
                orig_init = bw.BertModelWarper.__init__
                def new_init(self, bert_model):
                    if not hasattr(bert_model, "get_head_mask"):
                        bert_model.get_head_mask = lambda *a, **k: _get_head_mask(bert_model, *a, **k)
                    bert_model.get_extended_attention_mask = lambda *a, **k: _adapted_get_extended_attention_mask(bert_model, *a, **k)
                    orig_init(self, bert_model)
                    if not hasattr(self, "get_head_mask") or self.get_head_mask is None:
                        self.get_head_mask = lambda *a, **k: _get_head_mask(self, *a, **k)
                    self.get_extended_attention_mask = lambda *a, **k: _adapted_get_extended_attention_mask(self, *a, **k)
                bw.BertModelWarper.__init__ = new_init
                bw._rmbg_patched = True
        except Exception:
            pass

        # 2. Patch BertModel in transformers
        try:
            import transformers
            if hasattr(transformers, "BertModel"):
                if not hasattr(transformers.BertModel, "get_head_mask"):
                    transformers.BertModel.get_head_mask = _get_head_mask
                transformers.BertModel.get_extended_attention_mask = _adapted_get_extended_attention_mask
            if hasattr(transformers, "PreTrainedModel"):
                if not hasattr(transformers.PreTrainedModel, "get_head_mask"):
                    transformers.PreTrainedModel.get_head_mask = _get_head_mask
                transformers.PreTrainedModel.get_extended_attention_mask = _adapted_get_extended_attention_mask
            from transformers.models.bert.modeling_bert import BertModel
            if not hasattr(BertModel, "get_head_mask"):
                BertModel.get_head_mask = _get_head_mask
            BertModel.get_extended_attention_mask = _adapted_get_extended_attention_mask
        except Exception:
            pass

    except Exception:
        pass


def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def pil2mask(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image.convert("L")).astype(np.float32) / 255.0).unsqueeze(0)


def mask2image(mask: torch.Tensor) -> Image.Image:
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    return tensor2pil(mask)


def image2mask(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image.convert("L")).astype(np.float32) / 255.0)


def RGB2RGBA(image: Image.Image, mask: Union[Image.Image, torch.Tensor]) -> Image.Image:
    if isinstance(mask, torch.Tensor):
        mask = mask2image(mask)
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return Image.merge('RGBA', (*image.convert('RGB').split(), mask.convert('L')))


def process_mask(mask_image: Image.Image, invert_output: bool = False, mask_blur: int = 0, mask_offset: int = 0) -> Image.Image:
    if invert_output:
        mask_np = np.array(mask_image)
        mask_image = Image.fromarray(255 - mask_np)
    if mask_blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))
    if mask_offset != 0:
        filter_type = ImageFilter.MaxFilter if mask_offset > 0 else ImageFilter.MinFilter
        size = abs(mask_offset) * 2 + 1
        for _ in range(abs(mask_offset)):
            mask_image = mask_image.filter(filter_type(size))
    return mask_image


def apply_background_color(image: Image.Image, mask_image: Image.Image, background: str = "Alpha", background_color: str = "#222222") -> Image.Image:
    rgba_image = image.copy().convert('RGBA')
    rgba_image.putalpha(mask_image.convert('L'))
    if background == "Color":
        def hex_to_rgba(hex_color):
            hex_color = hex_color.lstrip('#')
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            return (r, g, b, 255)
        rgba = hex_to_rgba(background_color)
        bg_image = Image.new('RGBA', image.size, rgba)
        composite_image = Image.alpha_composite(bg_image, rgba_image)
        return composite_image.convert('RGB')
    return rgba_image


def get_or_download_model_file(filename: str, url: str, dirname: str = "rmbg") -> str:
    local_path = folder_paths.get_full_path(dirname, filename)
    if local_path:
        return local_path
    folder = os.path.join(folder_paths.models_dir, dirname)
    os.makedirs(folder, exist_ok=True)
    local_path = os.path.join(folder, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from {url} ...")
        download_url_to_file(url, local_path)
    return local_path


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

ASPECT_RATIOS = {
    # Square
    "1:1 (Square)":               (1, 1),

    # Landscape
    "5:4 (Landscape)":            (5, 4),
    "4:3 (Standard)":             (4, 3),
    "3:2 (Photo)":                (3, 2),
    "16:10 (Landscape)":          (16, 10),
    "16:9 (Widescreen)":          (16, 9),
    "1.85:1 (Film)":              (185, 100),
    "2:1 (Univisium)":            (2, 1),
    "21:9 (Ultrawide)":           (21, 9),
    "2.35:1 (Anamorphic)":        (235, 100),
    "2.39:1 (Panavision)":        (239, 100),
    "32:9 (Super Ultrawide)":     (32, 9),
    "3:1 (Panorama)":             (3, 1),

    # Portrait (mirrors of the landscape ratios)
    "4:5 (Portrait)":             (4, 5),
    "3:4 (Portrait Standard)":    (3, 4),
    "2:3 (Portrait Photo)":       (2, 3),
    "10:16 (Portrait)":           (10, 16),
    "9:16 (Portrait Widescreen)": (9, 16),
    "9:21 (Portrait Ultrawide)":  (9, 21),
    "1:3 (Portrait Panorama)":    (1, 3),
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
