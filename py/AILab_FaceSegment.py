from AILab_model_paths import local_model_inputs, model_bundle
# ComfyUI-RMBG
# This custom node for ComfyUI provides functionality for face parsing using Segformer model.
# 
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/AILab-AI/ComfyUI-RMBG

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from PIL import Image, ImageFilter
from torchvision import transforms

from torchvision import transforms
from AILab_utils import get_device, clean_vram, unload_model, pil2tensor, tensor2pil, RGB2RGBA, mask2image, image2mask
device = get_device()


AVAILABLE_MODELS = {
    "face_parsing": "1038lab/segformer_face"
}

@local_model_inputs('model_path')
class FaceSegment:
    def __init__(self):
        self.processor = None
        self.model = None
        self.cache_dir = None
    
    @classmethod
    def INPUT_TYPES(cls):
        available_classes = [
            "Skin", "Nose", "Eyeglasses", "Left-eye", "Right-eye",
            "Left-eyebrow", "Right-eyebrow", "Left-ear", "Right-ear", "Mouth",
            "Upper-lip", "Lower-lip", "Hair", "Earring", "Neck",
        ]
        tooltips = {
            "process_res": "Processing resolution (higher = more VRAM)",
            "mask_blur": "Blur amount for mask edges",
            "mask_offset": "Expand/Shrink mask boundary",
            "invert_output": "Invert both image and mask output",
            "background": "Choose background type: Alpha (transparent) or Color (custom background color).",
            "background_color": "Choose background color (Alpha = transparent)"
        }
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                **{cls_name: ("BOOLEAN", {"default": False}) 
                   for cls_name in available_classes},
                "process_res": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 32, "tooltip": tooltips["process_res"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "unload_model": ("BOOLEAN", {"default": False, "tooltip": "Unload model from VRAM after execution"}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": tooltips["background_color"]}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment_face"
    CATEGORY = "🧪AILab/🧽RMBG"

    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            torch.cuda.empty_cache()
        self.processor = None

    def segment_face(self, images, process_res=512, mask_blur=0, mask_offset=0, invert_output=False, unload_model=False, background="Alpha", background_color="#222222", **class_selections):
        self.cache_dir = model_bundle('1038lab/segformer_face', {"model.safetensors": self.model_path})
        try:
            if self.processor is None or self.model is None:
                from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
                self.processor = SegformerImageProcessor.from_pretrained(self.cache_dir, local_files_only=True)
                self.model = AutoModelForSemanticSegmentation.from_pretrained(self.cache_dir, local_files_only=True)
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.to(device)

            # Class mapping for segmentation
            class_map = {
                "Background": 0, "Skin": 1, "Nose": 2, "Eyeglasses": 3,
                "Left-eye": 4, "Right-eye": 5, "Left-eyebrow": 6, "Right-eyebrow": 7,
                "Left-ear": 8, "Right-ear": 9, "Mouth": 10, "Upper-lip": 11,
                "Lower-lip": 12, "Hair": 13, "Hat": 14, "Earring": 15,
                "Necklace": 16, "Neck": 17, "Clothing": 18
            }

            # Get selected classes
            selected_classes = [name for name, selected in class_selections.items() if selected]
            if not selected_classes:
                selected_classes = ["Skin", "Nose", "Left-eye", "Right-eye", "Mouth"]
            
            # Validate selected classes
            invalid_classes = [cls for cls in selected_classes if cls not in class_map]
            if invalid_classes:
                raise ValueError(f"Invalid class selections: {', '.join(invalid_classes)}. Valid classes are: {', '.join(class_map.keys())}")

            # Image preprocessing
            transform_image = transforms.Compose([
                transforms.Resize((process_res, process_res)),
                transforms.ToTensor(),
            ])

            batch_tensor = []
            batch_masks = []
            
            for image in images:
                orig_image = tensor2pil(image)
                w, h = orig_image.size
                
                input_tensor = transform_image(orig_image)

                if input_tensor.shape[0] == 4:
                    input_tensor = input_tensor[:3]
                
                input_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_tensor)
                
                input_tensor = input_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    logits = outputs.logits.cpu()
                    upsampled_logits = nn.functional.interpolate(
                        logits,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred_seg = upsampled_logits.argmax(dim=1)[0]

                    # Combine selected class masks
                    combined_mask = None
                    for class_name in selected_classes:
                        mask = (pred_seg == class_map[class_name]).float()
                        if combined_mask is None:
                            combined_mask = mask
                        else:
                            combined_mask = torch.clamp(combined_mask + mask, 0, 1)

                    # Convert mask to PIL for processing
                    mask_image = Image.fromarray((combined_mask.numpy() * 255).astype(np.uint8))

                    if mask_blur > 0:
                        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))

                    if mask_offset != 0:
                        if mask_offset > 0:
                            mask_image = mask_image.filter(ImageFilter.MaxFilter(size=mask_offset * 2 + 1))
                        else:
                            mask_image = mask_image.filter(ImageFilter.MinFilter(size=-mask_offset * 2 + 1))

                    if invert_output:
                        mask_image = Image.fromarray(255 - np.array(mask_image))

                    # Handle background color
                    if background == "Alpha":
                        rgba_image = RGB2RGBA(orig_image, mask_image)
                        result_image = pil2tensor(rgba_image)
                    else:
                        def hex_to_rgba(hex_color):
                            hex_color = hex_color.lstrip('#')
                            if len(hex_color) == 6:
                                r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                                a = 255
                            elif len(hex_color) == 8:
                                r, g, b, a = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), int(hex_color[6:8], 16)
                            else:
                                raise ValueError("Invalid color format")
                            return (r, g, b, a)
                        rgba_image = RGB2RGBA(orig_image, mask_image)
                        rgba = hex_to_rgba(background_color)
                        bg_image = Image.new('RGBA', orig_image.size, rgba)
                        composite_image = Image.alpha_composite(bg_image, rgba_image)
                        result_image = pil2tensor(composite_image.convert('RGB'))

                    batch_tensor.append(result_image)
                    batch_masks.append(pil2tensor(mask_image))

            # Create mask image for visualization
            mask_images = []
            for mask_tensor in batch_masks:
                # Convert mask to RGB image format for visualization
                mask_image = mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
                mask_images.append(mask_image)
            
            mask_image_output = torch.cat(mask_images, dim=0)
            
            # Prepare final output
            batch_tensor = torch.cat(batch_tensor, dim=0)
            batch_masks = torch.cat(batch_masks, dim=0)
            
            return (batch_tensor, batch_masks, mask_image_output)

        except Exception as e:
            self.clear_model()
            raise RuntimeError(f"Error in Face Parsing processing: {str(e)}")
        finally:
            if unload_model:
                self.clear_model()
            clean_vram()

NODE_CLASS_MAPPINGS = {
    "FaceSegment": FaceSegment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceSegment": "Face Segment (RMBG)"
} 
