# ComfyUI-RMBG
# This custom node for ComfyUI provides functionality for fashion segmentation using segformer-b3-fashion model.
# It leverages deep learning techniques to process images and generate masks for fashion items segmentation.
# 
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/AILab-AI/ComfyUI-RMBG

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from PIL import Image, ImageFilter
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import folder_paths
from huggingface_hub import hf_hub_download
import shutil
from torchvision import transforms

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

def RGB2RGBA(image: Image.Image, mask: Union[Image.Image, torch.Tensor]) -> Image.Image:
    if isinstance(mask, torch.Tensor):
        mask = mask2image(mask)
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return Image.merge('RGBA', (*image.convert('RGB').split(), mask.convert('L')))

def mask2image(mask: torch.Tensor) -> Image.Image:
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    return tensor2pil(mask)

device = "cuda" if torch.cuda.is_available() else "cpu"

folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

AVAILABLE_MODELS = {
    "segformer_fashion": "1038lab/segformer_fashion"
}

class FashionSegmentAccessories:
    @classmethod
    def INPUT_TYPES(cls):
        accessories_classes = [
            # Head accessories
            "hat",
            "glasses", 
            "headband, head covering, hair accessory",
            # Neck and upper body accessories
            "scarf",
            "tie",
            # Hand accessories
            "glove",
            "watch",
            # Waist accessories
            "belt",
            # Leg accessories
            "leg warmer",
            # Other accessories
            "bag, wallet", 
            "umbrella"
        ]
        
        details_classes = [
            # Upper body details
            "collar", 
            "lapel", 
            "neckline",
            "epaulette", 
            "pocket",
            # Decorative details
            "buckle", 
            "zipper", 
            "applique", 
            "bow",
            "flower", 
            "bead", 
            "fringe",
            "ribbon", 
            "rivet", 
            "ruffle", 
            "sequin", 
            "tassel"
        ]
        
        return {
            "required": {},
            "optional": {
                **{cls_name: ("BOOLEAN", {"default": False}) 
                   for cls_name in accessories_classes + details_classes},
            },
        }

    RETURN_TYPES = ("ACCESSORIES_OPTIONS",)
    RETURN_NAMES = ("accessories_options",)
    FUNCTION = "get_options"
    CATEGORY = "🧪AILab/🧽RMBG"

    def get_options(self, **class_selections):
        selected = [name for name, selected in class_selections.items() if selected]
        return (selected,)

class FashionSegmentClothing:
    def __init__(self):
        self.processor = None
        self.model = None
        self.cache_dir = os.path.join(folder_paths.models_dir, "RMBG", "segformer_fashion")
        self.class_map = {
            "Unlabelled": 0, "shirt, blouse": 1, "top, t-shirt, sweatshirt": 2, "sweater": 3,
            "cardigan": 4, "jacket": 5, "vest": 6, "pants": 7, "shorts": 8, "skirt": 9, "coat": 10,
            "dress": 11, "jumpsuit": 12, "cape": 13, 
            "glasses": 14, "hat": 15, "headband, head covering, hair accessory": 16, "tie": 17, "glove": 18,
            "watch": 19, "belt": 20, "leg warmer": 21, "tights, stockings": 22,
            "sock": 23, "shoe": 24, "bag, wallet": 25, "scarf": 26, "umbrella": 27,
            "hood": 28, "collar": 29, "lapel": 30, "epaulette": 31, "sleeve": 32,
            "pocket": 33, "neckline": 34, "buckle": 35, "zipper": 36, "applique": 37,
            "bead": 38, "bow": 39, "flower": 40, "fringe": 41, "ribbon": 42,
            "rivet": 43, "ruffle": 44, "sequin": 45, "tassel": 46
        }
    
    @classmethod
    def INPUT_TYPES(cls):
        clothing_classes = [
            # Upper body
            "coat",
            "jacket",
            "cardigan",
            "vest",
            "sweater",
            "hood",
            "shirt, blouse",
            "top, t-shirt, sweatshirt",
            "sleeve",
            # Full body
            "dress",
            "jumpsuit",
            "cape",
            # Lower body
            "pants",
            "shorts",
            "skirt",
            # Socks and shoes
            "tights, stockings",
            "sock",
            "shoe"
        ]
        
        tooltips = {
            "accessories_options": "Select the accessories to be segmented",
            "process_res": "Processing resolution (higher = more VRAM)",
            "mask_blur": "Blur amount for mask edges",
            "mask_offset": "Expand/Shrink mask boundary",
            "invert_output": "Invert both image and mask output",
            "background": "Choose background type: Alpha (transparent) or Color (custom background color).",
            "background_color": "Choose background color (Alpha = transparent)",
        }
        
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "accessories_options": ("ACCESSORIES_OPTIONS",),
                **{cls_name: ("BOOLEAN", {"default": False,}) 
                   for cls_name in clothing_classes},
                "process_res": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 32, "tooltip": tooltips["process_res"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": tooltips["background_color"]}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment_fashion"
    CATEGORY = "🧪AILab/🧽RMBG"

    def check_model_cache(self):
        if not os.path.exists(self.cache_dir):
            return False, "Model directory not found"
        
        required_files = [
            'config.json',
            'model.safetensors',
            'preprocessor_config.json'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.cache_dir, f))]
        if missing_files:
            return False, f"Missing required model files: {', '.join(missing_files)}"
        return True, "Model cache verified"

    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()

    def download_model_files(self):
        model_id = AVAILABLE_MODELS["segformer_fashion"]
        model_files = {
            'config.json': 'config.json',
            'model.safetensors': 'model.safetensors',
            'preprocessor_config.json': 'preprocessor_config.json'
        }
        
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Downloading fashion segmentation model files...")
        
        try:
            for save_name, repo_path in model_files.items():
                print(f"Downloading {save_name}...")
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=repo_path,
                    local_dir=self.cache_dir,
                    local_dir_use_symlinks=False
                )
                
                if os.path.dirname(downloaded_path) != self.cache_dir:
                    target_path = os.path.join(self.cache_dir, save_name)
                    shutil.move(downloaded_path, target_path)
            return True, "Model files downloaded successfully"
        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"

    def segment_fashion(self, images, accessories_options=None, process_res=512, mask_blur=0, mask_offset=0, 
                       background="Alpha", background_color="#222222", invert_output=False, **class_selections):
        if accessories_options is None:
            accessories_options = []
        try:
            # Check and download model
            cache_status, message = self.check_model_cache()
            if not cache_status:
                print(f"Cache check: {message}")
                download_status, download_message = self.download_model_files()
                if not download_status:
                    raise RuntimeError(download_message)
            
            # Load model if needed
            if self.processor is None:
                self.processor = SegformerImageProcessor.from_pretrained(self.cache_dir)
                self.model = AutoModelForSemanticSegmentation.from_pretrained(self.cache_dir)
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.to(device)

            # Get all selected classes
            selected_classes = []
            # Add clothing selections
            selected_classes.extend([name for name, selected in class_selections.items() if selected])
            # Add accessories selections
            selected_classes.extend(accessories_options)
            
            if not selected_classes:
                selected_classes = ["shirt, blouse"]

            transform_image = transforms.Compose([
                transforms.Resize((process_res, process_res)),
                transforms.ToTensor(),
            ])

            batch_tensor = []
            batch_masks = []
            
            for image in images:
                orig_image = tensor2pil(image)
                w, h = orig_image.size
                
                inputs = self.processor(images=orig_image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits.cpu()
                    
                    upsampled_logits = nn.functional.interpolate(
                        logits,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred_seg = upsampled_logits.argmax(dim=1)[0]

                    # Merge masks for selected classes
                    combined_mask = None
                    for class_name in selected_classes:
                        mask = (pred_seg == self.class_map[class_name]).float()
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

                    # Process background color
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
            
            batch_tensor = torch.cat(batch_tensor, dim=0)
            batch_masks = torch.cat(batch_masks, dim=0)
            
            return (batch_tensor, batch_masks, mask_image_output)

        except Exception as e:
            self.clear_model()
            raise RuntimeError(f"Error in fashion segmentation: {str(e)}")
        finally:
            if self.model is not None and not self.model.training:
                self.clear_model()

    def __del__(self):
        self.clear_model()

NODE_CLASS_MAPPINGS = {
    "FashionSegmentAccessories": FashionSegmentAccessories,
    "FashionSegmentClothing": FashionSegmentClothing
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FashionSegmentAccessories": "Accessories Segment (RMBG)",
    "FashionSegmentClothing": "Fashion Segment (RMBG)"
} 