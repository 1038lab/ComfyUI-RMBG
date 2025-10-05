# ComfyUI-RMBG
#
# AILab Inpaint Tools
# A collection of specialized nodes for inpainting tasks in ComfyUI.
# Features a set of utilities for mask processing, latent conditioning, and inpainting workflows.
#
# 1. Inpaint Nodes:
#    - AILab_ReferenceLatentMask: A node for inpainting tasks with the Flux Kontext model, using a reference latent and mask for precise region conditioning
#
# License: GPL-3.0
# These nodes are crafted to streamline common image and mask operations within ComfyUI workflows.

import torch
import node_helpers

def expand_mask(mask, expand_amount):
    if expand_amount == 0:
        return mask
        
    import torch.nn.functional as F
    
    binary_mask = (mask > 0.5).float()
    kernel_size = abs(expand_amount) * 2 + 1
    kernel_size = max(3, kernel_size)
    
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    
    if expand_amount > 0:
        expanded = F.conv2d(
            binary_mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1]),
            kernel,
            padding=kernel_size // 2
        )
        result = (expanded > 0).float()
    else:
        eroded = F.conv2d(
            binary_mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1]),
            kernel,
            padding=kernel_size // 2
        )
        result = (eroded >= kernel_size * kernel_size).float()
    
    if len(mask.shape) == 3:
        result = result.squeeze(1)
        
    return result


def blur_mask(mask, blur_amount):
    if blur_amount == 0:
        return mask
        
    import torch.nn.functional as F
    import math
    
    x = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1])
    kernel_size = max(3, math.ceil(blur_amount * 3) * 2 + 1)
    
    sigma = blur_amount
    half_kernel = kernel_size // 2
    grid = torch.arange(-half_kernel, half_kernel + 1, device=mask.device).float()
    
    gaussian = torch.exp(-0.5 * (grid / sigma) ** 2)
    gaussian = gaussian / gaussian.sum()
    
    gaussian_x = gaussian.view(1, 1, 1, kernel_size)
    gaussian_y = gaussian.view(1, 1, kernel_size, 1)
    
    blurred = F.conv2d(x, gaussian_x, padding=(0, half_kernel))
    blurred = F.conv2d(blurred, gaussian_y, padding=(half_kernel, 0))
    
    if len(mask.shape) == 3:
        blurred = blurred.squeeze(1)
        
    return blurred

class AILab_ReferenceLatentMask:
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "conditioning": "Base conditioning input for inpainting task",
            "latent": "Encoded latent from VAE",
            "mask": "Area to inpaint (white regions)",
            "expand": "Grow mask (+) or shrink mask (-)",
            "blur": "Soften mask edges",
            "mask_only": "Only generate content in masked area"
        }
        
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": tooltips["conditioning"]}),
                "latent": ("LATENT", {"tooltip": tooltips["latent"]}),
                "mask": ("MASK", {"tooltip": tooltips["mask"]}),
                "expand": ("INT", {"default": 5, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["expand"]}),
                "blur": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 64.0, "step": 0.1, "tooltip": tooltips["blur"]}),
                "mask_only": ("BOOLEAN", {"default": True, "tooltip": tooltips["mask_only"]}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "MASK")
    RETURN_NAMES = ("CONDITIONING", "LATENT", "MASK")
    FUNCTION = "prepare_inpaint_conditioning"
    CATEGORY = "🧪AILab/🎭Inpaint"

    def add_latent_to_conditioning(self, conditioning, latent=None):
        if latent is not None:
            return node_helpers.conditioning_set_values(
                conditioning, 
                {"reference_latents": [latent["samples"]]}, 
                append=True
            )
        return conditioning

    def prepare_inpaint_conditioning(self, conditioning, latent, mask, expand=5, blur=3.0, mask_only=True):
        processed_mask = mask
        
        if expand != 0:
            processed_mask = expand_mask(processed_mask, expand)
            
        if blur > 0:
            processed_mask = blur_mask(processed_mask, blur)
        
        modified_cond = node_helpers.conditioning_set_values(
            conditioning, 
            {
                "concat_latent_image": latent["samples"],
                "concat_mask": processed_mask
            }
        )
        
        final_cond = self.add_latent_to_conditioning(modified_cond, latent)
        
        output_latent = {"samples": latent["samples"]}
        if mask_only:
            output_latent["noise_mask"] = processed_mask
            
        return (final_cond, output_latent, processed_mask)


NODE_CLASS_MAPPINGS = {
    "AILab_ReferenceLatentMask": AILab_ReferenceLatentMask,
}

NODE_DISPLAY_NAME_MAPPINGS = { 
    "AILab_ReferenceLatentMask": "Reference Latent Mask (RMBG) 🖼️🎭",
} 