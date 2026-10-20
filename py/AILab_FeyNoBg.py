import os
import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2
import json
import safetensors.torch
import folder_paths
import comfy.model_management as mm
from huggingface_hub import hf_hub_download

from AILab_utils import get_device, clean_vram, unload_model as release_model, tensor2pil, pil2tensor

device = get_device()
folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

MODEL_REPO = "1038lab/FeyNobg"
MODEL_CACHE_NAME = "FeyNoBg"
REQUIRED_FILES = ["config.json", "FeyNobg.safetensors", "preprocessor_config.json"]


def _refine_foreground_edges(image_bchw, masks_b1hw):
    b, c, h, w = image_bchw.shape
    if b != masks_b1hw.shape[0]:
        raise ValueError("images and masks must have the same batch size")
    
    image_np = image_bchw.cpu().numpy()
    mask_np = masks_b1hw.cpu().numpy()
    
    refined_fg = []
    for i in range(b):
        mask = mask_np[i, 0]      
        thresh = 0.45
        mask_binary = (mask > thresh).astype(np.float32)
        
        edge_blur = cv2.GaussianBlur(mask_binary, (3, 3), 0)
        transition_mask = np.logical_and(mask > 0.05, mask < 0.95)
        
        alpha = 0.85
        mask_refined = np.where(transition_mask,
                              alpha * mask + (1-alpha) * edge_blur,
                              mask_binary)
        
        edge_region = np.logical_and(mask > 0.2, mask < 0.8)
        mask_refined = np.where(edge_region,
                              mask_refined * 0.98,
                              mask_refined)
        
        result = []
        for ch in range(image_np.shape[1]):
            channel = image_np[i, ch]
            refined = channel * mask_refined
            result.append(refined)
            
        refined_fg.append(np.stack(result))
    
    return torch.from_numpy(np.stack(refined_fg))


class AILab_FeyNoBg:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image batch to remove background from."}),
            },
            "optional": {
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "Gaussian blur radius for softening mask edges."}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": "Expand (positive) or shrink (negative) the mask boundary in pixels."}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": "Invert the output mask."}),
                "refine_foreground": ("BOOLEAN", {"default": True, "tooltip": "Sub-pixel foreground boundary refinement for fine hairs and semi-transparent edges."}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Unload model weights from VRAM immediately after execution."}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": "Output format: Alpha for transparent PNG, or Color to composite against background."}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": "Hex background color used when background mode is set to Color."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "process"
    CATEGORY = "🧪AILab/🧽RMBG"

    def __init__(self):
        self.model = None
        self.processor = None
        self.target_device = device

    def _ensure_model(self):
        if self.model is not None and self.processor is not None:
            return self.model, self.processor

        try:
            from nobg.birefnet.modeling_birefnet import BiRefNet, BiRefNetConfig
            from nobg import BiRefNetImageProcessor
        except ImportError as e:
            raise RuntimeError("The 'nobg' library is required for FeyNoBg. Please run: pip install nobg") from e

        local_cache_dir = os.path.join(folder_paths.models_dir, "RMBG", MODEL_CACHE_NAME)
        os.makedirs(local_cache_dir, exist_ok=True)

        for filename in REQUIRED_FILES:
            filepath = os.path.join(local_cache_dir, filename)
            if not os.path.exists(filepath):
                hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=filename,
                    local_dir=local_cache_dir,
                    local_dir_use_symlinks=False
                )

        config_path = os.path.join(local_cache_dir, "config.json")
        weights_path = os.path.join(local_cache_dir, "FeyNobg.safetensors")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        config = BiRefNetConfig(**config_data)
        self.model = BiRefNet(config)
        state_dict = safetensors.torch.load_file(weights_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.processor = BiRefNetImageProcessor.from_pretrained(local_cache_dir)

        if torch.cuda.is_available() and str(self.target_device).startswith("cuda"):
            self.model = self.model.half().to(self.target_device)
        else:
            self.model = self.model.to(self.target_device)

        return self.model, self.processor

    def process(self, image, mask_blur=0, mask_offset=0, invert_output=False,
                refine_foreground=True, unload_model=False,
                background="Alpha", background_color="#222222", **kwargs):
        do_unload = kwargs.get("unload_model", unload_model)
        refine = kwargs.get("refine_foreground", refine_foreground)

        model, processor = self._ensure_model()

        processed_images = []
        processed_masks = []

        try:
            for single_img in image:
                pil_img = tensor2pil(single_img).convert("RGB")
                orig_w, orig_h = pil_img.size

                inputs = processor(pil_img, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.target_device)
                if next(model.parameters()).dtype == torch.float16:
                    pixel_values = pixel_values.half()

                with torch.no_grad():
                    outputs = model(pixel_values=pixel_values)
                    alpha = processor.post_process_alpha_matting(outputs, target_sizes=[(orig_h, orig_w)])[0]

                if isinstance(alpha, torch.Tensor):
                    alpha_np = alpha.detach().cpu().float().numpy()
                elif isinstance(alpha, Image.Image):
                    alpha_np = np.array(alpha).astype(np.float32) / 255.0
                else:
                    alpha_np = np.array(alpha).astype(np.float32)
                    if alpha_np.max() > 1.0:
                        alpha_np /= 255.0

                mask_img = Image.fromarray((np.clip(alpha_np, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")

                if mask_offset != 0:
                    offset_k = abs(mask_offset) * 2 + 1
                    filter_type = ImageFilter.MaxFilter if mask_offset > 0 else ImageFilter.MinFilter
                    mask_img = mask_img.filter(filter_type(offset_k))

                if mask_blur > 0:
                    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=mask_blur))

                if invert_output:
                    mask_np_proc = 255.0 - np.array(mask_img, dtype=np.float32)
                    mask_img = Image.fromarray(np.clip(mask_np_proc, 0, 255).astype(np.uint8), mode="L")

                mask_tensor = pil2tensor(mask_img)
                processed_masks.append(mask_tensor)

                if background == "Alpha":
                    if refine:
                        refined_single = _refine_foreground_edges(single_img.unsqueeze(0).permute(0, 3, 1, 2), mask_tensor.unsqueeze(1))
                        refined_tensor = refined_single.squeeze(0).permute(1, 2, 0)
                        refined_pil = tensor2pil(refined_tensor)
                    else:
                        refined_pil = pil_img

                    cutout = Image.new("RGBA", (orig_w, orig_h), (0, 0, 0, 0))
                    cutout.paste(refined_pil, (0, 0), mask_img)
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
                release_model(self, "model")
                self.processor = None
            else:
                clean_vram()


NODE_CLASS_MAPPINGS = {
    "AILab_FeyNoBg": AILab_FeyNoBg
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_FeyNoBg": "FeyNoBg (RMBG)"
}
