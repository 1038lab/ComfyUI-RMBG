import json
import math
import torch
import numpy as np
from PIL import Image

import comfy.model_management
import comfy.utils
import node_helpers
import nodes

from AILab_utils import (
    clean_vram,
    process_mask,
    apply_background_color,
    tensor2pil,
    pil2tensor,
)

RMBG3_INSTRUCTION = (
    "Generate a detailed grayscale alpha matte. Map the opaque foreground to white and the "
    "background to black. Produce soft, anti-aliased grayscale gradients at the edges of the "
    "subject to represent fine details and transparency."
)
RMBG3_PROMPT = json.dumps({"edit_instruction": RMBG3_INSTRUCTION})


def _trained_size(aspect_ratio: float):
    area = 1024 * 1024
    sizes = [
        (w, h)
        for w in range(64, 2 * area // 64, 16)
        for h in range(
            max(64, 16 * math.ceil(0.9 * area / w / 16)),
            16 * math.floor(1.1 * area / w / 16) + 1,
            16,
        )
    ]
    return min(
        sizes,
        key=lambda s: (
            round(1 - min(s[0] / s[1] / aspect_ratio, aspect_ratio * s[1] / s[0]), 9),
            abs(s[0] * s[1] - area),
            -s[0],
        ),
    )


def _refine_foreground(pil_img: Image.Image, mask_img: Image.Image, radius: int = 45) -> Image.Image:
    import cv2

    img_f = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    alpha = np.asarray(mask_img.convert("L"), dtype=np.float32) / 255.0
    a = alpha[..., None]
    k = radius * 2 + 1
    mean_a = cv2.blur(a, (k, k))[..., None]
    mean_ia = cv2.blur(img_f * a, (k, k))
    fg = np.clip(mean_ia / np.maximum(mean_a, 1e-4), 0.0, 1.0)
    blended = fg * (1.0 - a) + img_f * a
    return Image.fromarray((np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def _unload_diffusion_model(model):
    for loaded in list(comfy.model_management.current_loaded_models):
        if loaded.model is model:
            loaded.model_unload()
            try:
                comfy.model_management.current_loaded_models.remove(loaded)
            except ValueError:
                pass


class AILab_RMBG3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "RMBG-3.0 diffusion model loaded via Load Diffusion Model."}),
                "clip": ("CLIP", {"tooltip": "Fibo 2.0 text encoder (qwen3vl_4b, type=fibo2) loaded via Load CLIP."}),
                "vae": ("VAE", {"tooltip": "FLUX.2 VAE (flux2-vae.safetensors) loaded via Load VAE."}),
                "image": ("IMAGE", {"tooltip": "Input image batch for RMBG-3.0 background removal."}),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Adjust matte detection strength (lower values boost semi-transparent foreground)."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "Gaussian blur radius for softening mask edges."}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": "Expand (positive) or shrink (negative) the mask boundary in pixels."}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": "Invert the output mask."}),
                "refine_foreground": ("BOOLEAN", {"default": False, "tooltip": "Fast foreground color estimation to remove edge color fringes."}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Unload the RMBG-3.0 diffusion model from VRAM after execution (keeps CLIP and VAE cached)."}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1, "tooltip": "Sampling steps (RMBG-3.0 is distilled for 4 steps)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Random seed for sampling."}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": "Output format: Alpha for transparent PNG, or Color to composite against background."}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": "Hex background color used when background mode is set to Color."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "process"
    CATEGORY = "🧪AILab/🧽RMBG"

    def process(self, model, clip, vae, image, sensitivity=1.0, mask_blur=0, mask_offset=0,
                invert_output=False, refine_foreground=False, unload_model=False,
                steps=4, seed=0, background="Alpha", background_color="#222222"):
        processed_images = []
        processed_masks = []

        try:
            for idx in range(image.shape[0]):
                single_img = image[idx:idx + 1]
                pil_img = tensor2pil(single_img).convert("RGB")
                orig_w, orig_h = pil_img.size

                target_w, target_h = _trained_size(orig_w / orig_h)
                if (single_img.shape[2], single_img.shape[1]) == (target_w, target_h):
                    resized = single_img[..., :3]
                else:
                    resized = comfy.utils.common_upscale(
                        single_img[..., :3].movedim(-1, 1), target_w, target_h, "lanczos", "center"
                    ).movedim(1, -1)

                tokens = clip.tokenize(RMBG3_PROMPT, images=[resized])
                positive = clip.encode_from_tokens_scheduled(tokens)
                if positive[0][0].shape[-1] != 46080:
                    raise RuntimeError(
                        f"RMBG-3.0 requires Load CLIP with type='fibo2' (expected 46080 text dims, got {positive[0][0].shape[-1]})."
                    )
                negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

                ref_latents = [vae.encode(resized)]
                if ref_latents[0].shape[1] != 128:
                    raise RuntimeError(
                        f"RMBG-3.0 requires FLUX.2 VAE (128 latent channels, e.g. flux2-vae.safetensors), "
                        f"but got {ref_latents[0].shape[1]} channels."
                    )
                positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
                negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

                latent = {
                    "samples": torch.zeros(
                        [1, 128, target_h // 16, target_w // 16],
                        device=comfy.model_management.intermediate_device(),
                    )
                }

                sampled = nodes.common_ksampler(
                    model, seed, steps, 1.0, "euler", "simple", positive, negative, latent, denoise=1.0
                )[0]
                decoded = vae.decode(sampled["samples"])

                matte = decoded[0].detach().cpu().float().mean(dim=-1)
                if sensitivity < 1.0:
                    matte = (matte * (1.0 + (1.0 - sensitivity))).clamp(0.0, 1.0)
                matte_np = (matte.numpy() * 255.0).clip(0, 255).astype(np.uint8)
                mask_img = Image.fromarray(matte_np, mode="L").resize((orig_w, orig_h), Image.LANCZOS)
                mask_img = process_mask(mask_img, invert_output, mask_blur, mask_offset)

                mask_tensor = torch.from_numpy(np.array(mask_img, dtype=np.float32) / 255.0).unsqueeze(0)
                processed_masks.append(mask_tensor)

                fg_img = _refine_foreground(pil_img, mask_img) if refine_foreground else pil_img
                composed = apply_background_color(fg_img, mask_img, background, background_color)
                composed = composed.convert("RGBA") if background == "Alpha" else composed.convert("RGB")
                processed_images.append(pil2tensor(composed))

            out_image = torch.cat(processed_images, dim=0)
            out_mask = torch.cat(processed_masks, dim=0)
            out_mask_image = out_mask.unsqueeze(-1).expand(-1, -1, -1, 3)

            return (out_image, out_mask, out_mask_image)
        finally:
            if unload_model:
                _unload_diffusion_model(model)
            clean_vram()


NODE_CLASS_MAPPINGS = {
    "AILab_RMBG3": AILab_RMBG3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_RMBG3": "RMBG-3.0 (RMBG)",
}
