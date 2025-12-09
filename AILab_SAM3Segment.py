import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.hub import download_url_to_file

import folder_paths
import comfy.model_management

from AILab_ImageMaskTools import pil2tensor, tensor2pil

CURRENT_DIR = os.path.dirname(__file__)
SAM3_LOCAL_DIR = os.path.join(CURRENT_DIR, "sam3")
if SAM3_LOCAL_DIR not in sys.path:
    sys.path.insert(0, SAM3_LOCAL_DIR)

SAM3_BPE_PATH = os.path.join(SAM3_LOCAL_DIR, "assets", "bpe_simple_vocab_16e6.txt.gz")
if not os.path.isfile(SAM3_BPE_PATH):
    raise RuntimeError("SAM3 assets missing; ensure sam3/assets/bpe_simple_vocab_16e6.txt.gz exists.")

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

_DEFAULT_PT_ENTRY = {
    "model_url": "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt",
    "filename": "sam3.pt",
}

SAM3_MODELS = {
    "sam3": _DEFAULT_PT_ENTRY.copy(),
}


def get_sam3_pt_models():
    """Return a dictionary containing the PT model definition."""
    entry = SAM3_MODELS.get("sam3")
    if entry and entry.get("filename", "").endswith(".pt"):
        return {"sam3": entry}
    # Fallback: upgrade any legacy entry to PT naming
    for key, value in SAM3_MODELS.items():
        if value.get("filename", "").endswith(".pt"):
            return {"sam3": value}
        if "sam3" in key and value:
            candidate = value.copy()
            candidate["model_url"] = _DEFAULT_PT_ENTRY["model_url"]
            candidate["filename"] = _DEFAULT_PT_ENTRY["filename"]
            return {"sam3": candidate}
    return {"sam3": _DEFAULT_PT_ENTRY.copy()}


def process_mask(mask_image, invert_output=False, mask_blur=0, mask_offset=0):
    if invert_output:
        mask_np = np.array(mask_image)
        mask_image = Image.fromarray(255 - mask_np)
    if mask_blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))
    if mask_offset != 0:
        filt = ImageFilter.MaxFilter if mask_offset > 0 else ImageFilter.MinFilter
        size = abs(mask_offset) * 2 + 1
        for _ in range(abs(mask_offset)):
            mask_image = mask_image.filter(filt(size))
    return mask_image


def apply_background_color(image, mask_image, background="Alpha", background_color="#222222"):
    rgba_image = image.copy().convert("RGBA")
    rgba_image.putalpha(mask_image.convert("L"))
    if background == "Color":
        hex_color = background_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        bg_image = Image.new("RGBA", image.size, (r, g, b, 255))
        composite = Image.alpha_composite(bg_image, rgba_image)
        return composite.convert("RGB")
    return rgba_image


def get_or_download_model_file(filename, url):
    local_path = None
    if hasattr(folder_paths, "get_full_path"):
        local_path = folder_paths.get_full_path("sam3", filename)
    if local_path and os.path.isfile(local_path):
        return local_path
    base_models_dir = getattr(folder_paths, "models_dir", os.path.join(CURRENT_DIR, "models"))
    models_dir = os.path.join(base_models_dir, "sam3")
    os.makedirs(models_dir, exist_ok=True)
    local_path = os.path.join(models_dir, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from {url} ...")
        download_url_to_file(url, local_path)
    return local_path


def _resolve_device(user_choice):
    auto_device = comfy.model_management.get_torch_device()
    if user_choice == "CPU":
        return torch.device("cpu")
    if user_choice == "GPU":
        if auto_device.type != "cuda":
            raise RuntimeError("GPU unavailable")
        return torch.device("cuda")
    return auto_device


class SAM3Segment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Describe the concept"}),
                "sam3_model": (list(SAM3_MODELS.keys()), {"default": "sam3"}),
                "device": (["Auto", "CPU", "GPU"], {"default": "Auto"}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.01}),
            },
            "optional": {
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "unload_model": ("BOOLEAN", {"default": False}),
                "background": (["Alpha", "Color"], {"default": "Alpha"}),
                "background_color": ("COLORCODE", {"default": "#222222"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment"
    CATEGORY = "🧪AILab/🧽RMBG"

    def __init__(self):
        self.processor_cache = {}

    def _load_processor(self, model_choice, device_choice):
        torch_device = _resolve_device(device_choice)
        device_str = "cuda" if torch_device.type == "cuda" else "cpu"
        cache_key = (model_choice, device_str)
        if cache_key not in self.processor_cache:
            model_info = SAM3_MODELS[model_choice]
            ckpt_path = get_or_download_model_file(model_info["filename"], model_info["model_url"])
            model = build_sam3_image_model(
                bpe_path=SAM3_BPE_PATH,
                device=device_str,
                eval_mode=True,
                checkpoint_path=ckpt_path,
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )
            processor = Sam3Processor(model, device=device_str)
            self.processor_cache[cache_key] = processor
        return self.processor_cache[cache_key], torch_device

    def _empty_result(self, img_pil, background, background_color):
        w, h = img_pil.size
        mask_image = Image.new("L", (w, h), 0)
        result_image = apply_background_color(img_pil, mask_image, background, background_color)
        if background == "Alpha":
            result_image = result_image.convert("RGBA")
        else:
            result_image = result_image.convert("RGB")
        empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
        mask_rgb = empty_mask.reshape((-1, 1, h, w)).movedim(1, -1).expand(-1, -1, -1, 3)
        return result_image, empty_mask, mask_rgb

    def _run_single(self, processor, img_tensor, prompt, confidence, mask_blur, mask_offset, invert, background, background_color):
        img_pil = tensor2pil(img_tensor)
        text = prompt.strip() or "object"
        state = processor.set_image(img_pil)
        processor.reset_all_prompts(state)
        processor.set_confidence_threshold(confidence, state)
        state = processor.set_text_prompt(text, state)
        masks = state.get("masks")
        if masks is None or masks.numel() == 0:
            return self._empty_result(img_pil, background, background_color)
        masks = masks.float().to("cpu")
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        combined = masks.amax(dim=0)
        mask_np = (combined.clamp(0, 1).numpy() * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_np, mode="L")
        mask_image = process_mask(mask_image, invert, mask_blur, mask_offset)
        result_image = apply_background_color(img_pil, mask_image, background, background_color)
        if background == "Alpha":
            result_image = result_image.convert("RGBA")
        else:
            result_image = result_image.convert("RGB")
        mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
        mask_rgb = mask_tensor.reshape((-1, 1, mask_image.height, mask_image.width)).movedim(1, -1).expand(-1, -1, -1, 3)
        return result_image, mask_tensor, mask_rgb

    def segment(self, image, prompt, sam3_model, device, confidence_threshold=0.5, mask_blur=0, mask_offset=0, invert_output=False, unload_model=False, background="Alpha", background_color="#222222"):

        if image.ndim == 3:
            image = image.unsqueeze(0)

        processor, torch_device = self._load_processor(sam3_model, device)
        autocast_device = comfy.model_management.get_autocast_device(torch_device)
        autocast_enabled = torch_device.type == "cuda" and not comfy.model_management.is_device_mps(torch_device)
        ctx = torch.autocast(autocast_device, dtype=torch.bfloat16) if autocast_enabled else nullcontext()

        result_images, result_masks, result_mask_images = [], [], []

        with ctx:
            for tensor_img in image:
                img_pil, mask_tensor, mask_rgb = self._run_single(
                    processor,
                    tensor_img,
                    prompt,
                    confidence_threshold,
                    mask_blur,
                    mask_offset,
                    invert_output,
                    background,
                    background_color,
                )
                result_images.append(pil2tensor(img_pil))
                result_masks.append(mask_tensor)
                result_mask_images.append(mask_rgb)

        if unload_model:
            device_str = "cuda" if torch_device.type == "cuda" else "cpu"
            cache_key = (sam3_model, device_str)
            if cache_key in self.processor_cache:
                del self.processor_cache[cache_key]
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

        return torch.cat(result_images, dim=0), torch.cat(result_masks, dim=0), torch.cat(result_mask_images, dim=0)


NODE_CLASS_MAPPINGS = {
    "SAM3Segment": SAM3Segment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3Segment": "SAM3 Segmentation (RMBG)",
}