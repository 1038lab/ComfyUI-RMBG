# ComfyUI-RMBG
#
# This node facilitates background removal using various models, including RMBG-2.0, INSPYRENET, BEN, BEN2, and BIREFNET-HR.
# It utilizes advanced deep learning techniques to process images and generate accurate masks for background removal.
#
# AILab Image and Mask Tools
# This module is specifically designed for ComfyUI-RMBG, enhancing workflows within ComfyUI.
# It offers a collection of utility nodes for efficient handling of images and masks:
#
# 1. Preview Nodes:
#    - Preview: A universal preview tool for both images and masks.
#    - ImagePreview: A specialized preview tool for images.
#    - MaskPreview: A specialized preview tool for masks.
#
# 2. Load Image Nodes:
#    - LoadImage: A node for loading images with some frequently used options.
#    - LoadImageSimple: A node for loading images with some frequently used options.
#    - LoadImageAdvanced: A node for loading images with advanced options.
#    - LoadImageBatch: A node for loading batch images from local path or URL.
#
# 3. Image and Mask Processing Nodes:
#    - MaskOverlay: A node for overlaying a mask on an image.
#    - ImageMaskConvert: Converts between image and mask formats and extracts masks from image channels.
#    - ImageToList: Converts a batch of images into an image list.
#    - MaskToList: Converts a batch of masks into a mask list.
#    - ImageMaskToList: Converts a batch of images and masks into an image and mask list.
#
# 4. Mask Processing Nodes:
#    - MaskEnhancer: Refines masks through techniques such as blur, smoothing, expansion/contraction, and hole filling.
#    - MaskCombiner: Combines multiple masks using union, intersection, or difference operations.
#
# 5. Image Processing Nodes:
#    - ImageCombiner: Combines foreground and background images with various blending modes and positioning options.
#    - ImageStitch: Stitches multiple images together in various directions.
#    - ImageCrop: Crops an image to a specified size and position.44
#    - ICLoRAConcat: Concatenates images with a mask using IC LoRA.
#    - CropObject: Crops an image to the object in the image.
#    - ImageCompare: Compares two images and returns a mask of the differences.
#    - ImageResize: Full feature Image and mask Resize.
#    - UnbatchImages: Unbatch images into individual image. 
#
# 5. Input Nodes:
#    - ColorInput: A node for inputting colors in various formats.
#
# License: GPL-3.0
# These nodes are crafted to streamline common image and mask operations within ComfyUI workflows.

import os
import random
import folder_paths
import numpy as np
import hashlib
import torch
import cv2
import re
from nodes import MAX_RESOLUTION
from comfy.utils import common_upscale
from PIL import Image, ImageFilter, ImageOps, ImageSequence, ImageChops, ImageDraw, ImageFont
import torchvision.transforms.functional as T
import torch.nn.functional as F
from comfy import model_management
from comfy_extras.nodes_mask import ImageCompositeMasked
from scipy import ndimage
from AILab_utils import (
    tensor2pil,
    pil2tensor,
    pil2mask,
    resize_image,
    blend_overlay,
    fill_mask,
    empty_image,
    upscale_mask,
    extract_alpha_mask,
    ensure_mask_shape,
    color_format,
    COLOR_PRESETS,
)

# Base class for preview
class AILab_PreviewBase:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = ""

    def get_unique_filename(self, filename_prefix):
        os.makedirs(self.output_dir, exist_ok=True)
        filename = filename_prefix + self.prefix_append
        counter = 1
        while True:
            file = f"{filename}_{counter:04d}.png"
            full_path = os.path.join(self.output_dir, file)
            if not os.path.exists(full_path):
                return full_path, file
            counter += 1

    def save_image(self, image, filename_prefix, prompt=None, extra_pnginfo=None):
        results = []
      
        try:
            if isinstance(image, torch.Tensor):
                if len(image.shape) == 4:  # Batch of images
                    for i in range(image.shape[0]):
                        full_output_path, file = self.get_unique_filename(filename_prefix)
                        img = Image.fromarray(np.clip(image[i].cpu().numpy() * 255, 0, 255).astype(np.uint8))
                        img.save(full_output_path)        
                        results.append({"filename": file, "subfolder": "", "type": self.type})
                else:
                    full_output_path, file = self.get_unique_filename(filename_prefix)
                    img = Image.fromarray(np.clip(image.cpu().numpy() * 255, 0, 255).astype(np.uint8))
                    img.save(full_output_path)
                    results.append({"filename": file, "subfolder": "", "type": self.type})
            else:
                full_output_path, file = self.get_unique_filename(filename_prefix)
                image.save(full_output_path)
                results.append({"filename": file, "subfolder": "", "type": self.type})

            return {
                "ui": {"images": results},
            }
        except Exception as e:
            print(f"Error saving image: {e}")
            return {"ui": {}}

# Preview node
class AILab_Preview(AILab_PreviewBase):
    def __init__(self):
        super().__init__()
        self.prefix_append = "_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def preview(self, image=None, mask=None, prompt=None, extra_pnginfo=None):
        results = []
        
        if image is not None:
            image_result = self.save_image(image, "image_preview", prompt, extra_pnginfo)
            if "ui" in image_result and "images" in image_result["ui"]:
                results.extend(image_result["ui"]["images"])
        
        if mask is not None:
            preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            mask_result = self.save_image(preview, "mask_preview", prompt, extra_pnginfo)
            if "ui" in mask_result and "images" in mask_result["ui"]:
                results.extend(mask_result["ui"]["images"])
        
        return {
            "ui": {"images": results},
            "result": (image if image is not None else None, mask if mask is not None else None)
        }

# Mask overlay node
class AILab_MaskOverlay(AILab_PreviewBase):
    def __init__(self):
        super().__init__()
        self.prefix_append = "_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "mask_opacity": "Control mask opacity (0.0-1.0)",
            "mask_color": "Color for the mask overlay",
            "image": "Input image (RGBA will be converted to RGB)",
            "mask": "Input mask"
        }
        
        return {
            "required": {
                "mask_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tooltips["mask_opacity"]}),
                "mask_color": ("COLORCODE", {"default": "#0000FF", "tooltip": tooltips["mask_color"]}),
             },
            "optional": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "mask": ("MASK", {"tooltip": tooltips["mask"]}),                
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "🧪AILab/🖼️IMAGE"
    OUTPUT_NODE = True

    def hex_to_rgb(self, hex_color):
        """Convert hex color code to RGB values (0-1 range)"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return r, g, b

    def ensure_rgb(self, image):
        """Ensure image is RGB format, convert from RGBA if needed"""
        if image.shape[-1] == 4:
            rgb_image = image[..., :3]
            return rgb_image
        return image

    def execute(self, mask_opacity, mask_color, filename_prefix="ComfyUI", image=None, mask=None, prompt=None, extra_pnginfo=None):
        """Execute image and mask composition"""
        if image is not None:
            image = self.ensure_rgb(image)
        
        preview = None
        
        if mask is not None and image is None:
            preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        elif mask is None and image is not None:
            preview = image
        elif mask is not None and image is not None:
            mask_adjusted = mask * mask_opacity
            mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3).clone()

            r, g, b = self.hex_to_rgb(mask_color)
            mask_image[:, :, :, 0] = r
            mask_image[:, :, :, 1] = g
            mask_image[:, :, :, 2] = b

            if hasattr(ImageCompositeMasked, "execute"):
                preview, = ImageCompositeMasked.execute(image, mask_image, 0, 0, True, mask_adjusted)
            else:
                preview, = ImageCompositeMasked.composite(image, mask_image, 0, 0, True, mask_adjusted)
        
        if preview is None:
            preview = empty_image(64, 64)
            
        if mask is None:
            mask = torch.zeros((1, 64, 64))

        # Save preview for display
        result = self.save_image(preview, filename_prefix, prompt, extra_pnginfo)
        
        # Return both the image and mask for further processing
        return {
            "ui": result["ui"] if "ui" in result else {},
            "result": (preview, mask)
        }

# Mask preview node
class AILab_MaskPreview(AILab_PreviewBase):
    def __init__(self):
        super().__init__()
        self.prefix_append = "_mask_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": ("MASK",),},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "preview_mask"
    OUTPUT_NODE = True
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def preview_mask(self, mask, prompt=None, extra_pnginfo=None):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        result = self.save_image(preview, "mask_preview", prompt, extra_pnginfo)
        return {
            "ui": result["ui"],
            "result": (mask,)
        }

# Image preview node
class AILab_ImagePreview(AILab_PreviewBase):
    def __init__(self):
        super().__init__()
        self.prefix_append = "_image_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",),},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "preview_image"
    OUTPUT_NODE = True
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def preview_image(self, image, prompt=None, extra_pnginfo=None):
        result = self.save_image(image, "image_preview", prompt, extra_pnginfo)
        return {
            "ui": result["ui"],
            "result": (image,)
        }

# Image mask conversion node
class AILab_ImageMaskConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_channel": (["alpha", "red", "green", "blue"], {"default": "alpha"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "convert"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def convert(self, image=None, mask=None, mask_channel="alpha"):
        # Case 1: No inputs
        if image is None and mask is None:
            empty_image = torch.zeros(1, 3, 64, 64)
            empty_mask = torch.zeros(1, 64, 64)
            return (empty_image, empty_mask)
            
        # Case 2: Only mask input
        if image is None and mask is not None:
            if mask.ndim == 4:
                tensor = mask.permute(0, 2, 3, 1)
                tensor_rgb = torch.cat([tensor] * 3, dim=-1)
                return (tensor_rgb, mask)
            elif mask.ndim == 3:
                tensor = mask.unsqueeze(-1)
                tensor_rgb = torch.cat([tensor] * 3, dim=-1)
                return (tensor_rgb, mask)
            elif mask.ndim == 2:
                tensor = mask.unsqueeze(0).unsqueeze(-1)
                tensor_rgb = torch.cat([tensor] * 3, dim=-1)
                return (tensor_rgb, mask.unsqueeze(0))
            else:
                print(f"Invalid mask shape: {mask.shape}")
                empty_image = torch.zeros(1, 3, 64, 64)
                return (empty_image, mask)
            
        # Case 3: Only image input
        if image is not None and mask is None:
            mask_list = []
            for img in image:
                pil_img = tensor2pil(img)
                pil_img = pil_img.convert("RGBA")
                r, g, b, a = pil_img.split()
                if mask_channel == "red":
                    channel_img = r
                elif mask_channel == "green":
                    channel_img = g
                elif mask_channel == "blue":
                    channel_img = b
                elif mask_channel == "alpha":
                    channel_img = a
                mask = np.array(channel_img.convert("L")).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask)
                mask_list.append(mask_tensor)
            result_mask = torch.stack(mask_list)
            return (image, result_mask)

        if image is not None and mask is not None:
            if mask.ndim == 4:  # [B,C,H,W]
                mask = mask.squeeze(1)  # Convert to [B,H,W]
            return (image, mask)

# Mask enhancer node
class AILab_MaskEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "mask": "Input mask to be processed.",
            "sensitivity": "Adjust the strength of mask detection (higher values result in more aggressive detection).",
            "mask_blur": "Specify the amount of blur to apply to the mask edges (0 for no blur, higher values for more blur).",
            "mask_offset": "Adjust the mask boundary (positive values expand the mask, negative values shrink it).",
            "smooth": "Smooth the mask edges (0 for no smoothing, higher values create smoother edges).",
            "fill_holes": "Enable to fill holes in the mask.",
            "invert_output": "Enable to invert the mask output (useful for certain effects)."
        }
        
        return {
            "required": {
                "mask": ("MASK", {"tooltip": tooltips["mask"]}),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tooltips["sensitivity"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 128.0, "step": 0.5, "tooltip": tooltips["smooth"]}),
                "fill_holes": ("BOOLEAN", {"default": False, "tooltip": tooltips["fill_holes"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "process_mask"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def fill_mask_region(self, mask_pil):
        """Fill holes in the mask"""
        mask_np = np.array(mask_pil)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask_np)
        for contour in contours:
            cv2.drawContours(filled_mask, [contour], 0, 255, -1)  # -1 means fill
        return Image.fromarray(filled_mask)

    def process_mask(self, mask, sensitivity=1.0, mask_blur=0, mask_offset=0, smooth=0.0, 
                    fill_holes=False, invert_output=False):
        processed_masks = []
        
        for mask_item in mask:
            m = mask_item * (1 + (1 - sensitivity))
            m = torch.clamp(m, 0, 1)
            
            if smooth > 0:
                mask_np = m.cpu().numpy()
                binary_mask = (mask_np > 0.5).astype(np.float32)
                blurred_mask = ndimage.gaussian_filter(binary_mask, sigma=smooth)
                final_mask = (blurred_mask > 0.5).astype(np.float32)
                m = torch.from_numpy(final_mask)
            
            if fill_holes:
                mask_pil = tensor2pil(m)
                mask_pil = self.fill_mask_region(mask_pil)
                m = pil2tensor(mask_pil).squeeze(0)
            
            if mask_blur > 0:
                mask_pil = tensor2pil(m)
                mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=mask_blur))
                m = pil2tensor(mask_pil).squeeze(0)
            
            if mask_offset != 0:
                mask_pil = tensor2pil(m)
                if mask_offset > 0:
                    for _ in range(mask_offset):
                        mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
                else:
                    for _ in range(-mask_offset):
                        mask_pil = mask_pil.filter(ImageFilter.MinFilter(3))
                m = pil2tensor(mask_pil).squeeze(0)
            
            if invert_output:
                m = 1.0 - m
            
            processed_masks.append(m.unsqueeze(0))
        
        return (torch.cat(processed_masks, dim=0),)

# Mask combiner node
class AILab_MaskCombiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_1": ("MASK",),
                "mode": (["combine", "intersection", "difference"], {"default": "combine"})
            },
            "optional": {
                "mask_2": ("MASK", {"default": None}),
                "mask_3": ("MASK", {"default": None}),
                "mask_4": ("MASK", {"default": None})
            }
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "combine_masks"

    def combine_masks(self, mask_1, mode="combine", mask_2=None, mask_3=None, mask_4=None):
        try:
            masks = [m for m in [mask_1, mask_2, mask_3, mask_4] if m is not None]
            
            if len(masks) <= 1:
                return (masks[0] if masks else torch.zeros((1, 64, 64), dtype=torch.float32),)
                
            ref_shape = masks[0].shape
            masks = [self._resize_if_needed(m, ref_shape) for m in masks]
            
            if mode == "combine":
                result = torch.maximum(masks[0], masks[1])
                for mask in masks[2:]:
                    result = torch.maximum(result, mask)
            elif mode == "intersection":
                result = torch.minimum(masks[0], masks[1])
            else:
                result = torch.abs(masks[0] - masks[1])
                
            return (torch.clamp(result, 0, 1),)
        except Exception as e:
            print(f"Error in combine_masks: {str(e)}")
            print(f"Mask shapes: {[m.shape for m in masks]}")
            raise e
    
    def _resize_if_needed(self, mask, target_shape):
        try:
            if mask.shape == target_shape:
                return mask
                
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            elif len(mask.shape) == 4:
                mask = mask.squeeze(1)
            
            target_height = target_shape[-2] if len(target_shape) >= 2 else target_shape[0]
            target_width = target_shape[-1] if len(target_shape) >= 2 else target_shape[1]
            
            resized_masks = []
            for i in range(mask.shape[0]):
                mask_np = mask[i].cpu().numpy()
                img = Image.fromarray((mask_np * 255).astype(np.uint8))
                img_resized = img.resize((target_width, target_height), Image.LANCZOS)
                mask_resized = np.array(img_resized).astype(np.float32) / 255.0
                resized_masks.append(torch.from_numpy(mask_resized))
            
            return torch.stack(resized_masks)
            
        except Exception as e:
            print(f"Error in _resize_if_needed: {str(e)}")
            print(f"Input mask shape: {mask.shape}, Target shape: {target_shape}")
            raise e

# Base class for image loaders
class AILab_BaseImageLoader:
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif')

    @classmethod
    def get_image_files(cls):
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        return [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and 
                f.lower().endswith(cls.IMAGE_EXTENSIONS)]
    
    @classmethod
    def get_folder_list(cls):
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        return [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

    def download_image(self, url):
        try:
            import requests
            from io import BytesIO

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, stream=True, timeout=10, headers=headers)
            if response.status_code != 200:
                raise ValueError(f"Failed to download image from URL: {url}, status code: {response.status_code}")
            return Image.open(BytesIO(response.content))

        except Exception as e:
            print(f"Error downloading image from URL: {str(e)}")
            raise e

    def get_image(self, image_path_or_URL="", image=""):
        if not image_path_or_URL and (not image or image == ""):
            return None

        if image_path_or_URL:
            image_path_or_URL = image_path_or_URL.strip()
            
            if image_path_or_URL.startswith(('http://', 'https://')):
                return self.download_image(image_path_or_URL)
            else:
                if os.path.isfile(image_path_or_URL):
                    return Image.open(image_path_or_URL)
                else:
                    input_dir = folder_paths.get_input_directory()
                    full_path = os.path.join(input_dir, image_path_or_URL)
                    if os.path.isfile(full_path):
                        return Image.open(full_path)
                    else:
                        raise ValueError(f"Image file not found: {image_path_or_URL}")
        else:
            if not image or image == "":
                return None
            image_path = folder_paths.get_annotated_filepath(image)
            return Image.open(image_path)

    def get_metadata(self, img):
        metadata_text = "No metadata found"
        try:
            if hasattr(img, 'text') and 'parameters' in img.text:
                metadata_text = img.text['parameters']
            elif hasattr(img, 'info') and 'parameters' in img.info:
                metadata_text = img.info['parameters']
            elif hasattr(img, 'text') and img.text:
                metadata_text = "\n".join([f"{k}: {v}" for k, v in img.text.items()])
        except Exception as e:
            print(f"Could not read metadata: {e}")
        return metadata_text

    @classmethod
    def calculate_hash(cls, image_path_or_URL="", image=""):
        if not image_path_or_URL and (not image or image == ""):
            return "no_input"

        if image_path_or_URL:
            try:
                if image_path_or_URL.startswith(('http://', 'https://')):
                    m = hashlib.sha256()
                    m.update(image_path_or_URL.encode('utf-8'))
                    return m.digest().hex()
                else:
                    if os.path.isfile(image_path_or_URL):
                        file_path = image_path_or_URL
                    else:
                        input_dir = folder_paths.get_input_directory()
                        file_path = os.path.join(input_dir, image_path_or_URL)
                        if not os.path.isfile(file_path):
                            return None
                    m = hashlib.sha256()
                    with open(file_path, 'rb') as f:
                        m.update(f.read())
                    return m.digest().hex()
            except:
                return None
        else:
            image_path = folder_paths.get_annotated_filepath(image)
            m = hashlib.sha256()
            with open(image_path, 'rb') as f:
                m.update(f.read())
            return m.hexdigest()

    @classmethod
    def validate_inputs(cls, image_path_or_URL="", image=""):
        if not image_path_or_URL and (not image or image == ""):
            return True
        if image_path_or_URL:
            return True
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True

    def process_image_to_tensor(self, img):
        if img is None:
            return None

        img_rgb = img.convert('RGB')
        output_images = []

        for i in ImageSequence.Iterator(img_rgb):
            i = ImageOps.exif_transpose(i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            if i.mode != 'RGB':
                i = i.convert('RGB')

            image = np.array(i).astype(np.float32) / 255.0
            if len(image.shape) == 3:
                image = torch.from_numpy(image)[None,]
            else:
                image = torch.from_numpy(image).unsqueeze(0)
            output_images.append(image)

        return torch.cat(output_images, dim=0) if len(output_images) > 1 else output_images[0]

    def resize_image_to_target(self, image, megapixels=0.0, scale_by=1.0, size=0, resize_mode="longest_side", resampling=Image.LANCZOS):
        orig_width, orig_height = image.size

        if megapixels > 0:
            aspect_ratio = orig_width / orig_height
            target_pixels = int(megapixels * 1024 * 1024)
            final_height = int((target_pixels / aspect_ratio) ** 0.5)
            final_width = int(aspect_ratio * final_height)
            
            if final_width != orig_width or final_height != orig_height:
                image = image.resize((final_width, final_height), resampling)
            return image, final_width, final_height

        target_w, target_h = orig_width, orig_height

        if size > 0:
            if resize_mode == "longest_side":
                if orig_width >= orig_height:
                    target_w = size
                    target_h = int(orig_height * (size / orig_width))
                else:
                    target_h = size
                    target_w = int(orig_width * (size / orig_height))
            elif resize_mode == "shortest_side":
                if orig_width <= orig_height:
                    target_w = size
                    target_h = int(orig_height * (size / orig_width))
                else:
                    target_h = size
                    target_w = int(orig_width * (size / orig_height))
            elif resize_mode == "width":
                target_w = size
                target_h = int(orig_height * (size / orig_width))
            elif resize_mode == "height":
                target_h = size
                target_w = int(orig_width * (size / orig_height))

        if scale_by != 1.0:
            target_w = int(target_w * scale_by)
            target_h = int(target_h * scale_by)

        if target_w != orig_width or target_h != orig_height:
            image = image.resize((target_w, target_h), resampling)

        return image, target_w, target_h
    
    def calculate_target_dimensions(self, orig_width, orig_height, megapixels=0.0, scale_by=1.0, size=0, resize_mode="longest_side"):
        if megapixels > 0:
            aspect_ratio = orig_width / orig_height
            target_pixels = int(megapixels * 1024 * 1024)
            final_height = int((target_pixels / aspect_ratio) ** 0.5)
            final_width = int(aspect_ratio * final_height)
            return final_width, final_height

        target_w, target_h = orig_width, orig_height

        if size > 0:
            if resize_mode == "longest_side":
                if orig_width >= orig_height:
                    target_w = size
                    target_h = int(orig_height * (size / orig_width))
                else:
                    target_h = size
                    target_w = int(orig_width * (size / orig_height))
            elif resize_mode == "shortest_side":
                if orig_width <= orig_height:
                    target_w = size
                    target_h = int(orig_height * (size / orig_width))
                else:
                    target_h = size
                    target_w = int(orig_width * (size / orig_height))
            elif resize_mode == "width":
                target_w = size
                target_h = int(orig_height * (size / orig_width))
            elif resize_mode == "height":
                target_h = size
                target_w = int(orig_width * (size / orig_height))

        if scale_by != 1.0:
            target_w = int(target_w * scale_by)
            target_h = int(target_h * scale_by)

        return target_w, target_h

    def process_and_resize_image(self, img, mask_channel="alpha", resampling=Image.LANCZOS,
                                 megapixels=0.0, scale_by=1.0, size=0, resize_mode="longest_side",
                                 advanced_mask=False):

        resized_img, width, height = self.resize_image_to_target(img, megapixels=megapixels, scale_by=scale_by, size=size, resize_mode=resize_mode, resampling=resampling)

        img_rgb = resized_img.convert("RGB")

        mask = None
        if advanced_mask:
            if mask_channel == "alpha" and "A" in resized_img.getbands():
                mask = np.array(resized_img.getchannel("A")).astype(np.float32) / 255.0
            elif mask_channel == "red":
                mask = np.array(img_rgb.getchannel("R")).astype(np.float32) / 255.0
            elif mask_channel == "green":
                mask = np.array(img_rgb.getchannel("G")).astype(np.float32) / 255.0
            elif mask_channel == "blue":
                mask = np.array(img_rgb.getchannel("B")).astype(np.float32) / 255.0
        else:
            if "A" in resized_img.getbands():
                mask = np.array(resized_img.getchannel("A")).astype(np.float32) / 255.0

        if mask is None:
            mask = np.ones((height, width), dtype=np.float32)

        image_tensor = self.process_image_to_tensor(img_rgb)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        if advanced_mask:
            mask_image = mask_tensor.reshape((-1, 1, height, width)).movedim(1, -1).expand(-1, -1, -1, 3)
        else:
            mask_image = None

        return image_tensor, mask_tensor, mask_image, width, height

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return cls.calculate_hash(kwargs.get("image_path_or_URL", ""), kwargs.get("image", ""))

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        for i in range(1, 4):
             if not cls.validate_inputs(kwargs.get(f"image_path_or_URL_{i}", ""), kwargs.get(f"image_{i}", "")):
                 return f"Invalid image file for image_{i}"
        return True

# Simple image loader node (basic functionality)
class AILab_LoadImageSimple(AILab_BaseImageLoader):
    @classmethod
    def INPUT_TYPES(cls):
        files = cls.get_image_files()
        return {
            "required": {
                "image_path_or_URL": ("STRING", {"default": "", "placeholder": "Local path, network path or URL"}),
                "image": ([""] + sorted(files) if files else [""], {"image_upload": True}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "WIDTH", "HEIGHT")
    FUNCTION = "load_image"
    OUTPUT_NODE = False

    def load_image(self, image_path_or_URL="", image="", extra_pnginfo=None):
        try:
            img = self.get_image(image_path_or_URL, image)

            if img is None:
                print("No image input provided, returning empty image and mask")
                empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
                return (empty_image, empty_mask, 64, 64)
            
            width, height = img.size
            output_image = self.process_image_to_tensor(img)
            
            mask = None
            if 'A' in img.getbands():
                mask_np = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask_np).unsqueeze(0)
            else:
                mask = torch.ones((1, height, width), dtype=torch.float32)
            
            return (output_image, mask, width, height)
            
        except Exception as e:
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (empty_image, empty_mask, 64, 64)

# Standard image loader node (with resize and basic mask)
class AILab_LoadImage(AILab_BaseImageLoader):
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        files = cls.get_image_files()
        return {
            "required": {
                "image_path_or_URL": ("STRING", {"default": "", "placeholder": "Local path or URL"}),
                "image": ([""] + sorted(files) if files else [""], {"image_upload": True}),
                "upscale_method": (cls.upscale_methods, {"default": "lanczos"}),
                "megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.01}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "resize_mode": (["longest_side", "shortest_side", "width", "height"], {"default": "longest_side"}),
                "size": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "WIDTH", "HEIGHT")
    FUNCTION = "load_image"
    OUTPUT_NODE = False

    def load_image(self, image_path_or_URL="", image="", upscale_method="lanczos", megapixels=0.0,
                   scale_by=1.0, resize_mode="longest_side", size=0, extra_pnginfo=None):
        try:
            img = self.get_image(image_path_or_URL, image)
            if img is None:
                raise ValueError("Image is None")

            resampling = {
                "nearest-exact": Image.NEAREST,
                "bilinear": Image.BILINEAR,
                "area": Image.BOX,
                "bicubic": Image.BICUBIC,
                "lanczos": Image.LANCZOS
            }.get(upscale_method, Image.LANCZOS)

            image_tensor, mask_tensor, _, width, height = self.process_and_resize_image(
                img, resampling=resampling, megapixels=megapixels, scale_by=scale_by,
                size=size, resize_mode=resize_mode, advanced_mask=False
            )
            return image_tensor, mask_tensor, width, height

        except Exception as e:
            traceback.print_exc()
            empty_image = torch.zeros((1, 64, 64, 3))
            empty_mask = torch.zeros((1, 64, 64))
            return empty_image, empty_mask, 64, 64

# Advanced image loader node (with full mask processing AND metadata output)
class AILab_LoadImageAdvanced(AILab_BaseImageLoader):
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        files = cls.get_image_files()
        return {
            "required": {
                "image_path_or_URL": ("STRING", {"default": "", "placeholder": "Local path or URL"}),
                "image": ([""] + sorted(files) if files else [""], {"image_upload": True}),
                "mask_channel": (["alpha", "red", "green", "blue"], {"default": "alpha"}),
                "upscale_method": (cls.upscale_methods, {"default": "lanczos"}),
                "megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.01}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "resize_mode": (["longest_side", "shortest_side", "width", "height"], {"default": "longest_side"}),
                "size": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE", "WIDTH", "HEIGHT", "METADATA")
    FUNCTION = "load_image"
    OUTPUT_NODE = False

    def load_image(self, image_path_or_URL="", image="", mask_channel="alpha", upscale_method="lanczos", megapixels=0.0,
                   scale_by=1.0, resize_mode="longest_side", size=0, extra_pnginfo=None):
        try:
            img = self.get_image(image_path_or_URL, image)
            if img is None:
                raise ValueError("Image is None")
    
            metadata_text = self.get_metadata(img)

            resampling = {
                "nearest-exact": Image.NEAREST,
                "bilinear": Image.BILINEAR,
                "area": Image.BOX,
                "bicubic": Image.BICUBIC,
                "lanczos": Image.LANCZOS
            }.get(upscale_method, Image.LANCZOS)

            image_tensor, mask_tensor, mask_image, width, height = self.process_and_resize_image(
                img, mask_channel=mask_channel, resampling=resampling, megapixels=megapixels,
                scale_by=scale_by, size=size, resize_mode=resize_mode, advanced_mask=True
            )

            return (image_tensor, mask_tensor, mask_image, width, height, metadata_text)

        except Exception as e:
            traceback.print_exc()
            empty_image = torch.zeros((1, 64, 64, 3))
            empty_mask = torch.zeros((1, 64, 64))
            empty_mask_image = empty_mask.reshape((-1, 1, 64, 64)).movedim(1, -1).expand(-1, -1, -1, 3)
            return (empty_image, empty_mask, empty_mask_image, 64, 64, "Error loading image")

#Batch Image loader node
class AILab_LoadImageBatch(AILab_BaseImageLoader):
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path_or_urls": ("STRING", {"default": "", "multiline": True, "placeholder": "Path to a directory, comma/new-line separated file paths, OR comma/new-line separated URLs"}),
                "upscale_method": (cls.upscale_methods, {"default": "lanczos"}),
                "megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.01}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "resize_mode": (["longest_side", "shortest_side", "width", "height"], {"default": "longest_side"}),
                "size": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            },
            "optional": {
                "batch_size": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "Number of images to load (0 = all images)"}),
                "start_from": ("INT", {"default": 1, "min": 1, "step": 1, "tooltip": "Start from Nth image (1 = first image)"}),
                "sort_method": (["sequential", "reverse", "random"], {"default": "sequential", "tooltip": "Image loading order: sequential/reverse/random"}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "WIDTH", "HEIGHT")
    FUNCTION = "load_image_batch"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'sort_method' in kwargs and kwargs['sort_method'] == "random":
            return float("NaN")
        return hashlib.sha256(str(kwargs).encode('utf-8')).hexdigest()

    def load_image_batch(self, path_or_urls="", upscale_method="lanczos", megapixels=0.0,
                         scale_by=1.0, resize_mode="longest_side", size=0, 
                         batch_size=0, start_from=1, sort_method="sequential", extra_pnginfo=None):
        
        image_list = []
        
        input_path = path_or_urls.strip()
        
        if not input_path:
                raise ValueError("No input provided. Please specify a path/URL list.")
        
        potential_paths = [path.strip() for path in re.split(r'[,\n]+', input_path) if path.strip()]

        if not potential_paths:
            raise ValueError("Input is empty or contains only whitespace.")

        first_path = potential_paths[0]
        
        if first_path.startswith(('http://', 'https://')):
            image_list = [path.strip() for path in re.split(r'[,\n]\s*(?=http)', input_path) if path.strip()]
        
        elif os.path.isdir(first_path):
            image_list = [
                os.path.join(first_path, f)
                for f in os.listdir(first_path)
                if os.path.isfile(os.path.join(first_path, f)) and f.lower().endswith(self.IMAGE_EXTENSIONS)
            ]
            image_list.sort()
        
        elif os.path.isfile(first_path):
            image_list = [p for p in potential_paths if os.path.isfile(p)]
        
        else:
            relative_path_check = os.path.join(folder_paths.get_input_directory(), first_path)
            
            if os.path.isdir(relative_path_check):
                image_list = [
                    os.path.join(relative_path_check, f)
                    for f in os.listdir(relative_path_check)
                    if os.path.isfile(os.path.join(relative_path_check, f)) and f.lower().endswith(self.IMAGE_EXTENSIONS)
                ]
                image_list.sort()
            
            elif os.path.isfile(relative_path_check):
                image_list = [os.path.join(folder_paths.get_input_directory(), p) for p in potential_paths 
                              if os.path.isfile(os.path.join(folder_paths.get_input_directory(), p))]
            
            else:
                raise ValueError(f"Input is not a valid URL, directory, or file path: {first_path}")

        if not image_list:
            raise ValueError("No valid images found from the provided input.")
        
        if sort_method == "reverse":
            image_list.reverse()
        elif sort_method == "random":
            import random
            random.shuffle(image_list)

        start_index = max(0, start_from - 1)
        if start_index > 0 and start_index < len(image_list):
            image_list = image_list[start_index:]
        elif start_index >= len(image_list):
             raise ValueError(f"start_from ({start_from}) is out of bounds. Only {len(image_list)} images found.")

        if batch_size > 0:
            image_list = image_list[:batch_size]
        
        if not image_list:
            raise ValueError("No images left after applying start_from/batch_size filters.")

        resampling = {
            "nearest-exact": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "area": Image.BOX,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS
        }.get(upscale_method, Image.LANCZOS)

        output_images = []
        output_masks = []
        master_width = 0
        master_height = 0

        first_img = self.get_image(image_list[0])
        resized_first_img, master_width, master_height = self.resize_image_to_target(
            first_img, megapixels=megapixels, scale_by=scale_by, size=size,
            resize_mode=resize_mode, resampling=resampling
        )
        
        img_tensor = self.process_image_to_tensor(resized_first_img)
        
        mask = None
        if 'A' in resized_first_img.getbands():
            mask_np = np.array(resized_first_img.getchannel('A')).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask_np).unsqueeze(0)
        else:
            mask = torch.ones((1, master_height, master_width), dtype=torch.float32)
        
        output_images.append(img_tensor)
        output_masks.append(mask)

        for img_path_or_url in image_list[1:]:
            img = self.get_image(img_path_or_url)
            resized_img = img.resize((master_width, master_height), resampling)
            img_tensor = self.process_image_to_tensor(resized_img)
            
            mask = None
            if 'A' in resized_img.getbands():
                mask_np = np.array(resized_img.getchannel('A')).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask_np).unsqueeze(0)
            else:
                mask = torch.ones((1, master_height, master_width), dtype=torch.float32)
            
            output_images.append(img_tensor)
            output_masks.append(mask)
        
        if not output_images:
            raise ValueError("All images in the batch failed to load or process.")

        return (torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0), master_width, master_height)

# Image combiner node
class AILab_ImageCombiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE",),
                "background": ("IMAGE",),
                "mode": (["normal", "multiply", "screen", "overlay", "add", "subtract"], 
                              {"default": "normal"}),
                "foreground_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "foreground_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "position_x": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
                "position_y": ("INT", {"default": 50, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "foreground_mask": ("MASK", {"default": None}),
            }
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT")
    FUNCTION = "combine_images"
    
    def combine_images(self, foreground, background, mode="normal", foreground_opacity=1.0, 
                      foreground_scale=1.0, position_x=50, position_y=50, foreground_mask=None):
        if len(foreground.shape) == 3:
            foreground = foreground.unsqueeze(0)
        if len(background.shape) == 3:
            background = background.unsqueeze(0)
        
        batch_size = foreground.shape[0]
        output_images = []
        
        for b in range(batch_size):
            fg_pil = tensor2pil(foreground[b])
            bg_pil = tensor2pil(background[b])
            
            if fg_pil.mode != 'RGBA':
                fg_pil = fg_pil.convert('RGBA')
            
            if foreground_scale != 1.0:
                new_width = int(fg_pil.width * foreground_scale)
                new_height = int(fg_pil.height * foreground_scale)
                fg_pil = fg_pil.resize((new_width, new_height), Image.LANCZOS)
            
            if foreground_mask is not None:
                mask_tensor = foreground_mask[b] if len(foreground_mask.shape) > 2 else foreground_mask
                mask_pil = Image.fromarray(np.uint8(mask_tensor.cpu().numpy() * 255))
                if mask_pil.size != fg_pil.size:
                    mask_pil = mask_pil.resize(fg_pil.size, Image.LANCZOS)
                r, g, b, a = fg_pil.split()
                a = ImageChops.multiply(a, mask_pil)
                fg_pil = Image.merge('RGBA', (r, g, b, a))
            
            fg_w, fg_h = fg_pil.size
            bg_w, bg_h = bg_pil.size
            
            x = int(bg_w * position_x / 100 - fg_w / 2)
            y = int(bg_h * position_y / 100 - fg_h / 2)
            
            new_fg = Image.new('RGBA', (bg_w, bg_h), (0, 0, 0, 0))
            new_fg.paste(fg_pil, (x, y), fg_pil)
            fg_pil = new_fg
            
            if bg_pil.mode != 'RGBA':
                bg_pil = bg_pil.convert('RGBA')
            
            if foreground_opacity < 1.0:
                r, g, b, a = fg_pil.split()
                a = Image.eval(a, lambda x: int(x * foreground_opacity))
                fg_pil = Image.merge('RGBA', (r, g, b, a))
            
            if mode == "normal":
                result = bg_pil.copy()
                result = Image.alpha_composite(result, fg_pil)
            else:
                alpha = fg_pil.split()[3]
                fg_rgb = fg_pil.convert('RGB')
                bg_rgb = bg_pil.convert('RGB')
                
                if mode == "multiply":
                    blended = ImageChops.multiply(fg_rgb, bg_rgb)
                elif mode == "screen":
                    blended = ImageChops.screen(fg_rgb, bg_rgb)
                elif mode == "add":
                    blended = ImageChops.add(fg_rgb, bg_rgb, 1.0)
                elif mode == "subtract":
                    blended = ImageChops.subtract(fg_rgb, bg_rgb, 1.0)
                elif mode == "overlay":
                    blended = blend_overlay(fg_rgb, bg_rgb)
                else:
                    blended = fg_rgb
                
                blended = blended.convert('RGBA')
                r, g, b, _ = blended.split()
                blended = Image.merge('RGBA', (r, g, b, alpha))
                result = bg_pil.copy()
                result = Image.alpha_composite(result, blended)
            
            if result.mode != 'RGB':
                white_bg = Image.new('RGB', result.size, 'white')
                result = Image.alpha_composite(white_bg.convert('RGBA'), result)
                result = result.convert('RGB')
            
            output_images.append(pil2tensor(result))
        
        final_image = torch.cat(output_images, dim=0)
        width = final_image.shape[2]
        height = final_image.shape[1]
        
        return (final_image, width, height)

# Mask extractor node
class AILab_MaskExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["extract_masked_area", "apply_mask", "invert_mask"], {"default": "extract_masked_area"}),
                "background": (["Alpha", "original", "Color"], {"default": "Alpha", "tooltip": "Choose background type"}),
                "background_color": ("COLORCODE", {"default": "#FFFFFF", "tooltip": "Choose background color (Alpha = transparent)"})
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract_masked_area"

    def _prepare_mask(self, mask_np, image_shape):
        try:
            if isinstance(mask_np, torch.Tensor):
                mask_np = mask_np.cpu().numpy()
            mask_np = np.array(mask_np)
            while len(mask_np.shape) > 2 and mask_np.shape[-1] == 1:
                mask_np = mask_np.squeeze(-1)
            while len(mask_np.shape) > 2 and mask_np.shape[0] == 1:
                mask_np = mask_np.squeeze(0)
            if len(mask_np.shape) > 2:
                mask_np = mask_np.squeeze()
            if mask_np.shape != image_shape[:2]:
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
                mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            mask_np = mask_np[..., np.newaxis]
            mask_np = np.repeat(mask_np, image_shape[2], axis=2)
            return mask_np
        except Exception as e:
            print(f"Error in _prepare_mask: {str(e)}")
            raise e

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)

    def extract_masked_area(self, image, mode="extract_masked_area", background="Alpha", background_color="#FFFFFF", mask=None):
        try:
            if mask is None and image.shape[-1] == 4:
                alpha = image[..., 3]
                mask = 1.0 - alpha
                image = image[..., :3]
            elif mask is None:
                mask = torch.ones((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32)

            pil_image = tensor2pil(image)
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            mask_np = self._prepare_mask(mask, image_np.shape)
            result_np = np.zeros_like(image_np)
            
            if mode == "extract_masked_area":
                result_np = image_np * mask_np
                if background == "Alpha":
                    if pil_image.mode != "RGBA":
                        pil_image = pil_image.convert("RGBA")
                    result_rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
                    result_rgba[:, :, :3] = image_np * mask_np
                    result_rgba[:, :, 3] = mask_np[..., 0]
                    result_pil = Image.fromarray((result_rgba * 255).astype(np.uint8), mode="RGBA")
                    return (torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0).unsqueeze(0),)
                elif background == "original":
                    result_np = image_np * mask_np
                elif background == "Color":
                    r, g, b = self.hex_to_rgb(background_color)
                    result_np = result_np + (1 - mask_np) * np.array([r, g, b])
            
            elif mode == "apply_mask":
                result_np = image_np * mask_np
                if background == "Alpha":
                    if pil_image.mode != "RGBA":
                        pil_image = pil_image.convert("RGBA")
                    result_rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
                    result_rgba[:, :, :3] = image_np * mask_np
                    result_rgba[:, :, 3] = mask_np[..., 0]
                    result_pil = Image.fromarray((result_rgba * 255).astype(np.uint8), mode="RGBA")
                    return (torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0).unsqueeze(0),)
                elif background == "original":
                    result_np = image_np * mask_np + image_np * (1 - mask_np)
                elif background == "Color":
                    r, g, b = self.hex_to_rgb(background_color)
                    result_np = result_np + (1 - mask_np) * np.array([r, g, b])
            
            elif mode == "invert_mask":
                result_np = image_np * (1 - mask_np)
                if background == "Alpha":
                    if pil_image.mode != "RGBA":
                        pil_image = pil_image.convert("RGBA")
                    result_rgba = np.zeros((*image_np.shape[:2], 4), dtype=np.float32)
                    result_rgba[:, :, :3] = image_np * (1 - mask_np)
                    result_rgba[:, :, 3] = (1 - mask_np)[..., 0]
                    result_pil = Image.fromarray((result_rgba * 255).astype(np.uint8), mode="RGBA")
                    return (torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0).unsqueeze(0),)
                elif background == "original":
                    result_np = image_np * (1 - mask_np) + image_np * mask_np
                elif background == "Color":
                    r, g, b = self.hex_to_rgb(background_color)
                    result_np = result_np + (1 - mask_np) * np.array([r, g, b])
            
            result_pil = Image.fromarray(np.clip(result_np * 255, 0, 255).astype(np.uint8))
            return (pil2tensor(result_pil),)
        except Exception as e:
            print(f"Error in extract_masked_area: {str(e)}")
            raise e

# Image Stitch node
class AILab_ImageStitch:
    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "image1": "First image to stitch",
            "stitch_mode": "Mode for stitching images together",
            "match_image_size": "If True, resize image2 to match image1's aspect ratio",
            "megapixels": "Target megapixels for final output (0 = no limit, overrides max_width/max_height)",
            "max_width": "Maximum width of output image (0 = no limit, ignored if megapixels > 0)",
            "max_height": "Maximum height of output image (0 = no limit, ignored if megapixels > 0)",
            "upscale_method": "Upscaling method for all resize operations",
            "spacing_width": "Width of spacing between images",
            "background_color": "Color for spacing between images and padding background",
            "kontext_mode": "Special mode that arranges 3 images in a specific layout (image1 and image2 stacked vertically, image3 on the right)"
        }
        return {
            "required": {
                "image1": ("IMAGE",),
                "stitch_mode": (["right", "down", "left", "up", "2x2", "kontext_mode"], {"default": "right", "tooltip": tooltips["stitch_mode"]}),
                "match_image_size": ("BOOLEAN", {"default": True, "tooltip": tooltips["match_image_size"]}),
                "megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.01, "tooltip": tooltips["megapixels"]}),
                "max_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": tooltips["max_width"]}),
                "max_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": tooltips["max_height"]}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "lanczos", "tooltip": tooltips["upscale_method"]}),
                "spacing_width": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1, "tooltip": tooltips["spacing_width"]}),
                "background_color": ("COLORCODE", {"default": "#FFFFFF", "tooltip": tooltips["background_color"]}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT")
    FUNCTION = "stitch"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)

    def pad_with_color(self, image, padding, color_val):
        batch, height, width, channels = image.shape
        r, g, b = color_val
        pad_top, pad_bottom, pad_left, pad_right = padding
        new_height = height + pad_top + pad_bottom
        new_width = width + pad_left + pad_right
        result = torch.zeros((batch, new_height, new_width, channels), device=image.device)
        if channels >= 3:
            result[..., 0] = r
            result[..., 1] = g
            result[..., 2] = b
            if channels == 4:
                result[..., 3] = 1.0
        result[:, pad_top:pad_top+height, pad_left:pad_left+width, :] = image
        return result

    def match_dimensions(self, image1, image2, stitch_mode, color_val):
        h1, w1 = image1.shape[1:3]
        h2, w2 = image2.shape[1:3]
        if stitch_mode in ["left", "right"]:
            if h1 != h2:
                target_h = max(h1, h2)
                if h1 < target_h:
                    pad_h = target_h - h1
                    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                    image1 = self.pad_with_color(image1, (pad_top, pad_bottom, 0, 0), color_val)
                if h2 < target_h:
                    pad_h = target_h - h2
                    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                    image2 = self.pad_with_color(image2, (pad_top, pad_bottom, 0, 0), color_val)
        else:
            if w1 != w2:
                target_w = max(w1, w2)
                if w1 < target_w:
                    pad_w = target_w - w1
                    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                    image1 = self.pad_with_color(image1, (0, 0, pad_left, pad_right), color_val)
                if w2 < target_w:
                    pad_w = target_w - w2
                    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                    image2 = self.pad_with_color(image2, (0, 0, pad_left, pad_right), color_val)
        return image1, image2

    def ensure_same_channels(self, image1, image2):
        if image1.shape[-1] != image2.shape[-1]:
            max_channels = max(image1.shape[-1], image2.shape[-1])
            if image1.shape[-1] < max_channels:
                image1 = torch.cat([
                    image1,
                    torch.ones(*image1.shape[:-1], max_channels - image1.shape[-1], device=image1.device),
                ], dim=-1)
            if image2.shape[-1] < max_channels:
                image2 = torch.cat([
                    image2,
                    torch.ones(*image2.shape[:-1], max_channels - image2.shape[-1], device=image2.device),
                ], dim=-1)
        return image1, image2

    def create_spacing(self, image1, image2, spacing_width, stitch_mode, color_val):
        if spacing_width <= 0:
            return None
        spacing_width = spacing_width + (spacing_width % 2)
        if stitch_mode in ["left", "right"]:
            spacing_shape = (
                image1.shape[0],
                max(image1.shape[1], image2.shape[1]),
                spacing_width,
                image1.shape[-1],
            )
        else:
            spacing_shape = (
                image1.shape[0],
                spacing_width,
                max(image1.shape[2], image2.shape[2]),
                image1.shape[-1],
            )
        spacing = torch.zeros(spacing_shape, device=image1.device)
        r, g, b = color_val
        if spacing.shape[-1] >= 3:
            spacing[..., 0] = r
            spacing[..., 1] = g
            spacing[..., 2] = b
            if spacing.shape[-1] == 4:
                spacing[..., 3] = 1.0
        return spacing

    def stitch_kontext_mode(self, image1, image2, image3, match_image_size, spacing_width, color_val, upscale_method, image4=None):
        has_image4 = image4 is not None
        if image1 is None or image2 is None:
            if image2 is None and image3 is not None:
                return self.stitch_two_images(image1, image3, "right", match_image_size, spacing_width, color_val, upscale_method)
            else:
                return image1
        images_to_align = [image1, image2]
        if image3 is not None:
            images_to_align.append(image3)
        if has_image4:
            images_to_align.append(image4)
        max_batch = max(img.shape[0] for img in images_to_align)
        for i, img in enumerate(images_to_align):
            if img.shape[0] < max_batch:
                images_to_align[i] = torch.cat([img, img[-1:].repeat(max_batch - img.shape[0], 1, 1, 1)])
        image1, image2 = images_to_align[0], images_to_align[1]
        image3 = images_to_align[2] if len(images_to_align) > 2 else None
        image4 = images_to_align[3] if len(images_to_align) > 3 else None
        if has_image4:
            left_images = [image1, image2, image3]
            right_image = image4
        else:
            left_images = [image1, image2]
            right_image = image3
        if match_image_size:
            w1 = image1.shape[2]
            for i, img in enumerate(left_images[1:], 1):
                h, w = img.shape[1:3]
                aspect_ratio = h / w
                target_w = w1
                target_h = int(w1 * aspect_ratio)
                left_images[i] = common_upscale(
                    img.movedim(-1, 1), target_w, target_h, upscale_method, "disabled"
                ).movedim(1, -1)
        else:
            for i in range(1, len(left_images)):
                left_images[0], left_images[i] = self.match_dimensions(left_images[0], left_images[i], "down", color_val)
        for i in range(1, len(left_images)):
            left_images[0], left_images[i] = self.ensure_same_channels(left_images[0], left_images[i])
        left_column_parts = [left_images[0]]
        for i in range(1, len(left_images)):
            spacing = self.create_spacing(left_images[i-1], left_images[i], spacing_width, "down", color_val)
            if spacing is not None:
                left_column_parts.append(spacing)
            left_column_parts.append(left_images[i])
        left_column = torch.cat(left_column_parts, dim=1)
        if match_image_size:
            h_left = left_column.shape[1]
            hr, wr = right_image.shape[1:3]
            aspect_ratio = wr / hr
            target_h = h_left
            target_w = int(h_left * aspect_ratio)
            right_image = common_upscale(
                right_image.movedim(-1, 1), target_w, target_h, upscale_method, "disabled"
            ).movedim(1, -1)
        else:
            left_column, right_image = self.match_dimensions(left_column, right_image, "right", color_val)
        left_column, right_image = self.ensure_same_channels(left_column, right_image)
        h_spacing = self.create_spacing(left_column, right_image, spacing_width, "right", color_val)
        h_images = [left_column]
        if h_spacing is not None:
            h_images.append(h_spacing)
        h_images.append(right_image)
        result = torch.cat(h_images, dim=2)
        return result

    def stitch_two_images(self, image1, image2, stitch_mode, match_image_size, spacing_width, color_val, upscale_method):
        if image2 is None:
            return image1
        if image1.shape[0] != image2.shape[0]:
            max_batch = max(image1.shape[0], image2.shape[0])
            if image1.shape[0] < max_batch:
                image1 = torch.cat(
                    [image1, image1[-1:].repeat(max_batch - image1.shape[0], 1, 1, 1)]
                )
            if image2.shape[0] < max_batch:
                image2 = torch.cat(
                    [image2, image2[-1:].repeat(max_batch - image2.shape[0], 1, 1, 1)]
                )
        if match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            aspect_ratio = w2 / h2
            if stitch_mode in ["left", "right"]:
                target_h, target_w = h1, int(h1 * aspect_ratio)
            else:
                target_w, target_h = w1, int(w1 / aspect_ratio)
            image2 = common_upscale(
                image2.movedim(-1, 1), target_w, target_h, upscale_method, "disabled"
            ).movedim(1, -1)
        else:
            image1, image2 = self.match_dimensions(image1, image2, stitch_mode, color_val)
        image1, image2 = self.ensure_same_channels(image1, image2)
        spacing = self.create_spacing(image1, image2, spacing_width, stitch_mode, color_val)
        images = [image2, image1] if stitch_mode in ["left", "up"] else [image1, image2]
        if spacing is not None:
            images.insert(1, spacing)
        concat_dim = 2 if stitch_mode in ["left", "right"] else 1
        result = torch.cat(images, dim=concat_dim)
        return result

    def create_blank_like(self, reference_image, color_val):
        batch, height, width, channels = reference_image.shape
        result = torch.zeros((batch, height, width, channels), device=reference_image.device)
        r, g, b = color_val
        if channels >= 3:
            result[..., 0] = r
            result[..., 1] = g
            result[..., 2] = b
            if channels == 4:
                result[..., 3] = 1.0
        return result

    def stitch_multi_mode(self, image1, image2, image3, image4, stitch_mode, match_image_size, spacing_width, color_val, upscale_method):
        images = [image for image in [image1, image2, image3, image4] if image is not None]
        if len(images) == 0:
            return torch.zeros((1, 64, 64, 3))
        if len(images) == 1:
            return images[0]
        current = images[0]
        for next_img in images[1:]:
            current = self.stitch_two_images(current, next_img, stitch_mode, match_image_size, spacing_width, color_val, upscale_method)
        return current

    def stitch_grid_2x2(self, image1, image2, image3, image4, match_image_size, spacing_width, color_val, upscale_method):
        ref = image1
        img2 = image2 if image2 is not None else self.create_blank_like(ref, color_val)
        row1 = self.stitch_two_images(ref, img2, "right", match_image_size, spacing_width, color_val, upscale_method)
        img3 = image3 if image3 is not None else self.create_blank_like(ref, color_val)
        img4 = image4 if image4 is not None else self.create_blank_like(ref, color_val)
        row2 = self.stitch_two_images(img3, img4, "right", match_image_size, spacing_width, color_val, upscale_method)
        result = self.stitch_two_images(row1, row2, "down", match_image_size, spacing_width, color_val, upscale_method)
        return result

    def stitch(self, image1, stitch_mode, match_image_size, megapixels, max_width, max_height, upscale_method, spacing_width, background_color, image2=None, image3=None, image4=None,):
        if image1 is None:
            return (torch.zeros((1, 64, 64, 3)),)
        color_val = self.hex_to_rgb(background_color)
        if stitch_mode == "kontext_mode":
            result = self.stitch_kontext_mode(image1, image2, image3, match_image_size, spacing_width, color_val, upscale_method, image4=image4)
        elif stitch_mode == "2x2":
            result = self.stitch_grid_2x2(image1, image2, image3, image4, match_image_size, spacing_width, color_val, upscale_method)
        else:
            result = self.stitch_multi_mode(image1, image2, image3, image4, stitch_mode, match_image_size, spacing_width, color_val, upscale_method)
        h, w = result.shape[1:3]
        need_resize = False
        target_w, target_h = w, h
        if megapixels > 0:
            aspect_ratio = w / h
            target_pixels = int(megapixels * 1024 * 1024)
            target_h = int((target_pixels / aspect_ratio) ** 0.5)
            target_w = int(aspect_ratio * target_h)
            need_resize = True
        elif max_width > 0 or max_height > 0:
            if max_width > 0 and w > max_width:
                scale_factor = max_width / w
                target_w = max_width
                target_h = int(h * scale_factor)
                need_resize = True
            else:
                target_w, target_h = w, h
            if max_height > 0 and (target_h > max_height or (target_h == h and h > max_height)):
                scale_factor = max_height / target_h
                target_h = max_height
                target_w = int(target_w * scale_factor)
                need_resize = True
        if need_resize:
            result = common_upscale(
                result.movedim(-1, 1), target_w, target_h, upscale_method, "disabled"
            ).movedim(1, -1)
        final_height, final_width = result.shape[1:3]
        return (result, final_width, final_height)
   

# Image Crop node
class AILab_ImageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Width of the crop region in pixels. Will be clamped to image width."}),
                "height": ("INT", {"default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Height of the crop region in pixels. Will be clamped to image height."}),
                "x_offset": ("INT", {"default": 0, "min": -99999, "step": 1, "tooltip": "Horizontal offset (in pixels) added to the crop position. Positive values move right, negative left."}),
                "y_offset": ("INT", {"default": 0, "min": -99999, "step": 1, "tooltip": "Vertical offset (in pixels) added to the crop position. Positive values move down, negative up."}),
                "split": ("BOOLEAN", {"default": False, "tooltip": "If True, output the cropped region and the rest of the image with the crop area set to zero. If False, the rest is a zero image."}),
                "position": (["top-left", "top-center", "top-right", "right-center", "bottom-right", "bottom-center", "bottom-left", "left-center", "center"], {"tooltip": "Anchor position for the crop region. Determines where the crop is placed relative to the image."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("CROP", "REST")
    FUNCTION = "execute"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def execute(self, image, width, height, position, x_offset, y_offset, split=False):
        _, oh, ow, _ = image.shape

        width = min(ow, width)
        height = min(oh, height)

        if "center" in position:
            x = round((ow-width) / 2)
            y = round((oh-height) / 2)
        if "top" in position:
            y = 0
        if "bottom" in position:
            y = oh-height
        if "left" in position:
            x = 0
        if "right" in position:
            x = ow-width

        x += x_offset
        y += y_offset

        x2 = x+width
        y2 = y+height

        if x2 > ow:
            x2 = ow
        if x < 0:
            x = 0
        if y2 > oh:
            y2 = oh
        if y < 0:
            y = 0

        crop = image[:, y:y2, x:x2, :]
        rest = None
        if split:
            top = image[:, 0:y, :, :] if y > 0 else None
            bottom = image[:, y2:oh, :, :] if y2 < oh else None
            left = image[:, y:y2, 0:x, :] if x > 0 else None
            right = image[:, y:y2, x2:ow, :] if x2 < ow else None

            parts = []
            if top is not None:
                parts.append(top)
            if left is not None or right is not None:
                row_parts = []
                if left is not None:
                    row_parts.append(left)
                if right is not None:
                    row_parts.append(right)
                if row_parts:
                    row = torch.cat(row_parts, dim=2)
                    parts.append(row)
            if bottom is not None:
                parts.append(bottom)
            if parts:
                rest = torch.cat(parts, dim=1)
            else:
                rest = torch.zeros_like(image[:, :0, :0, :])
        else:
            rest = image.clone()
            rest[:] = 0
        return (crop, rest)

# ICLoRA Concat node
class AILab_ICLoRAConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "object_image": ("IMAGE",{"tooltip": ("The main image to be used as the foreground (object) in the concatenation.\nIf the image has 4 channels (RGBA), the alpha channel will be automatically extracted and used as the object mask if no mask is provided.")}),
                "layout": (["top-bottom", "left-right"], {"default": "left-right", "tooltip": "The direction in which to concatenate the images: top-bottom or left-right."}),
                "custom_size": ("INT", {"default": 0, "max": MAX_RESOLUTION, "min": 0, "step": 8, "tooltip": "If 0, the output image size is unchanged. Otherwise, sets the base image height (for left-right) or base image width (for top-bottom) in pixels for the concatenation. The object image will be scaled proportionally to match the base image in the concatenation direction."}),
            },
            "optional": {
                "object_mask": ("MASK", {"tooltip": "Mask for the object_image. Defines the region of the object_image to be blended into the base_image."}),
                "base_image": ("IMAGE", {"tooltip": "The background image to be concatenated with the object_image.\nIf the image has 4 channels (RGBA), the alpha channel will be automatically extracted and used as the base mask if no mask is provided."}),
                "base_mask": ("MASK", {"tooltip": "Mask for the base_image. Defines the region of the base_image to be blended with the object_image."}),
            },
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    FUNCTION = "create"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "OBJECT_MASK", "BASE_MASK", "WIDTH", "HEIGHT", "X", "Y")

    def create(self, object_image, layout, custom_size=0, base_image=None, object_mask=None, base_mask=None):
        if object_image.shape[-1] == 4 and object_mask is None:
            object_mask = extract_alpha_mask(object_image)
            object_image = object_image[..., :3]
        if base_image is not None and base_image.shape[-1] == 4 and base_mask is None:
            base_mask = extract_alpha_mask(base_image)
            base_image = base_image[..., :3]

        if base_image is None:
            base_image = empty_image(object_image.shape[2], object_image.shape[1])
            base_mask = torch.full((1, object_image.shape[1], object_image.shape[2]), 1, dtype=torch.float32, device="cpu")
        elif base_image is not None and base_mask is None:
            # raise ValueError("base_mask is required when base_image is provided")
            base_mask = torch.full((1, object_image.shape[1], object_image.shape[2]), 1, dtype=torch.float32, device="cpu")
            
        object_mask = ensure_mask_shape(object_mask)
        base_mask = ensure_mask_shape(base_mask)

        _, base_h, base_w, base_c = base_image.shape
        _, obj_h, obj_w, obj_c = object_image.shape

        if layout == 'left-right':
            if custom_size > 0:
                new_base_h = custom_size
                new_base_w = int(base_w * (custom_size / base_h))
                base_image = base_image.movedim(-1, 1)
                base_image = common_upscale(base_image, new_base_w, new_base_h, 'bicubic', 'disabled')
                base_image = base_image.movedim(1, -1)
                if base_mask is not None:
                    base_mask = upscale_mask(base_mask, new_base_w, new_base_h)
                base_h, base_w = new_base_h, new_base_w

            scale = base_h / obj_h
            new_obj_w = int(obj_w * scale)
            object_image = object_image.movedim(-1, 1)
            object_image = common_upscale(object_image, new_obj_w, base_h, 'bicubic', 'disabled')
            object_image = object_image.movedim(1, -1)
            if object_mask is not None:
                object_mask = upscale_mask(object_mask, new_obj_w, base_h)
            else:
                object_mask = torch.full((1, base_h, new_obj_w), 1, dtype=torch.float32, device="cpu")

            if object_image.shape[-1] != base_image.shape[-1]:
                min_c = min(object_image.shape[-1], base_image.shape[-1])
                object_image = object_image[..., :min_c]
                base_image = base_image[..., :min_c]
            
            image = torch.cat((object_image, base_image), dim=2)
            batch = object_mask.shape[0]
            out_h = base_h
            out_w = new_obj_w + base_w
            object_mask_resized = object_mask
            base_mask_resized = base_mask
            
            if object_mask_resized.shape[-2:] != (base_h, new_obj_w):
                object_mask_resized = upscale_mask(object_mask_resized, new_obj_w, base_h)
            if base_mask_resized.shape[-2:] != (base_h, base_w):
                base_mask_resized = upscale_mask(base_mask_resized, base_w, base_h)
            
            OBJECT_MASK = torch.zeros((batch, out_h, out_w), dtype=object_mask_resized.dtype, device=object_mask_resized.device)
            BASE_MASK = torch.zeros((batch, out_h, out_w), dtype=base_mask_resized.dtype, device=base_mask_resized.device)
            OBJECT_MASK[:, :, :new_obj_w] = object_mask_resized
            BASE_MASK[:, :, new_obj_w:] = base_mask_resized

        elif layout == 'top-bottom':
            if custom_size > 0:
                new_base_w = custom_size
                new_base_h = int(base_h * (custom_size / base_w))
                base_image = base_image.movedim(-1, 1)
                base_image = common_upscale(base_image, new_base_w, new_base_h, 'bicubic', 'disabled')
                base_image = base_image.movedim(1, -1)
                if base_mask is not None:
                    base_mask = upscale_mask(base_mask, new_base_w, new_base_h)
                base_h, base_w = new_base_h, new_base_w

            scale = base_w / obj_w
            new_obj_h = int(obj_h * scale)
            object_image = object_image.movedim(-1, 1)
            object_image = common_upscale(object_image, base_w, new_obj_h, 'bicubic', 'disabled')
            object_image = object_image.movedim(1, -1)
            if object_mask is not None:
                object_mask = upscale_mask(object_mask, base_w, new_obj_h)
            else:
                object_mask = torch.full((1, new_obj_h, base_w), 1, dtype=torch.float32, device="cpu")

            if object_image.shape[-1] != base_image.shape[-1]:
                min_c = min(object_image.shape[-1], base_image.shape[-1])
                object_image = object_image[..., :min_c]
                base_image = base_image[..., :min_c]
            
            image = torch.cat((object_image, base_image), dim=1)
            batch = object_mask.shape[0]
            out_h = new_obj_h + base_h
            out_w = base_w
            object_mask_resized = object_mask
            base_mask_resized = base_mask
            
            if object_mask_resized.shape[-2:] != (new_obj_h, base_w):
                object_mask_resized = upscale_mask(object_mask_resized, base_w, new_obj_h)
            if base_mask_resized.shape[-2:] != (base_h, base_w):
                base_mask_resized = upscale_mask(base_mask_resized, base_w, base_h)
            
            OBJECT_MASK = torch.zeros((batch, out_h, out_w), dtype=object_mask_resized.dtype, device=object_mask_resized.device)
            BASE_MASK = torch.zeros((batch, out_h, out_w), dtype=base_mask_resized.dtype, device=base_mask_resized.device)
            OBJECT_MASK[:, :new_obj_h, :] = object_mask_resized
            BASE_MASK[:, new_obj_h:, :] = base_mask_resized

        x = object_image.shape[2] if layout == 'left-right' else 0
        y = object_image.shape[1] if layout == 'top-bottom' else 0

        return (image, OBJECT_MASK, BASE_MASK, out_w, out_h, x, y)

class AILab_CropObject:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "crop_object"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def get_bbox_from_tensor(self, tensor, padding):
        rows = torch.any(tensor > 0, dim=1)
        cols = torch.any(tensor > 0, dim=0)
        if not torch.any(rows) or not torch.any(cols):
            return None
        rmin, rmax = torch.where(rows)[0][[0, -1]]
        cmin, cmax = torch.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - padding)
        rmax = min(tensor.shape[0] - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(tensor.shape[1] - 1, cmax + padding)
        return rmin, rmax, cmin, cmax

    def crop_object(self, image=None, mask=None, padding=0):
        if mask is None and image is None:
            raise ValueError("At least one of image or mask must be provided")
        bbox = None
        if mask is not None:
            mask_tensor = mask.squeeze()
            bbox = self.get_bbox_from_tensor(mask_tensor, padding)
        elif image is not None and image.shape[-1] == 4:
            alpha = image[0, :, :, 3]
            bbox = self.get_bbox_from_tensor(alpha, padding)
        if bbox is None:
            return (image, mask)
        rmin, rmax, cmin, cmax = bbox
        if mask is not None:
            cropped_mask = mask[:, rmin:rmax+1, cmin:cmax+1]
        else:
            if image is not None and image.shape[-1] == 4:
                alpha = image[0, rmin:rmax+1, cmin:cmax+1, 3]
                cropped_mask = alpha.unsqueeze(0)
            else:
                cropped_mask = None
        if image is not None:
            cropped_image = image[:, rmin:rmax+1, cmin:cmax+1, :]
        else:
            cropped_image = None
        return (
            cropped_image if image is not None else image,
            cropped_mask if mask is not None else mask
        )

# Image Compare node
class AILab_ImageCompare:
    def __init__(self):
        self.font_size = 20
        self.padding = 10
        #self.text_align = "center"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {"default": "Image 1"}),
                "text2": ("STRING", {"default": "Image 2"}),
                "text3": ("STRING", {"default": "Image 3"}),
                "size_base": (["largest", "smallest", "image1", "image2", "image3"], {"default": "largest"}),
                "text_color": ("COLORCODE", {"default": "#000000"}),
                "bg_color": ("COLORCODE", {"default": "#FFFFFF"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🖼️IMAGE"

    def get_font(self):
        try:
            if os.name == "nt":
                return ImageFont.truetype("arial.ttf", self.font_size)
            elif os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
                return ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    self.font_size,
                )
            else:
                return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    def create_text_panel(self, width, text):
        font = self.get_font()
        temp_img = Image.new("RGB", (width, self.font_size * 2), self.bg_color)
        temp_draw = ImageDraw.Draw(temp_img)
        text_bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        final_height = max(int(text_height * 1.5), self.font_size * 2)
        panel = Image.new("RGB", (width, final_height), self.bg_color)
        draw = ImageDraw.Draw(panel)
        x = (width - text_width) // 2
        y = (final_height - text_height) // 2
        draw.text((x, y), text, font=font, fill=self.text_color)
        return panel

    def _select_base_image(self, pil_map, size_base):
        if size_base in ("image1", "image2", "image3") and size_base in pil_map:
            return size_base
        if size_base == "smallest":
            best_label = None
            best_area = float('inf')
            for label, img in pil_map.items():
                area = img.width * img.height
                if area < best_area:
                    best_area = area
                    best_label = label
            return best_label
        if size_base != "largest":
            print(
                f"Warning: size_base '{size_base}' is not available, fallback to 'largest'."
            )
        best_label = None
        best_area = -1
        for label, img in pil_map.items():
            area = img.width * img.height
            if area > best_area:
                best_area = area
                best_label = label
        return best_label

    def generate(self, text1, text2, text3, size_base="largest", text_color="#000000", bg_color="#FFFFFF", image1=None, image2=None, image3=None,):
        self.bg_color = bg_color
        self.text_color = text_color

        tensors = []
        texts = []
        labels = []

        if image1 is not None and hasattr(image1, "shape") and image1.shape[0] > 0:
            tensors.append(image1)
            texts.append(text1)
            labels.append("image1")
        if image2 is not None and hasattr(image2, "shape") and image2.shape[0] > 0:
            tensors.append(image2)
            texts.append(text2)
            labels.append("image2")
        if image3 is not None and hasattr(image3, "shape") and image3.shape[0] > 0:
            tensors.append(image3)
            texts.append(text3)
            labels.append("image3")

        if len(tensors) < 2:
            print("Warning: At least two images are required.")
            return (torch.zeros((1, 64, 64, 3)),)

        batch_sizes = [t.shape[0] for t in tensors]
        batch_size = min(batch_sizes)
        if len(set(batch_sizes)) > 1:
            print(
                f"Warning: Input batches have different sizes {batch_sizes}. "
                f"Only processing the minimum size of {batch_size}."
            )

        output_images = []

        for i in range(batch_size):
            pil_map = {}
            for t, label in zip(tensors, labels):
                frame = t[i].unsqueeze(0)
                img_pil = tensor2pil(frame)
                pil_map[label] = img_pil

            base_label = self._select_base_image(pil_map, size_base)
            base_img = pil_map[base_label]
            base_h = base_img.height

            resized_pils = []
            for label in labels:
                img = pil_map[label]
                w, h = img.size
                
                if w == 0 or h == 0:
                    resized_pils.append(Image.new("RGB", (1, 1), self.bg_color))
                    continue
                
                if label == base_label:
                    resized_pils.append(img)
                    continue
                
                scale = base_h / h
                new_h = base_h
                new_w = max(1, int(round(w * scale)))
                
                if img.size != (new_w, new_h):
                    img = resize_image(img, new_w, new_h)
                resized_pils.append(img)

            panels = []
            for img, text in zip(resized_pils, texts):
                if text.strip():
                    panels.append(self.create_text_panel(img.width, text))
                else:
                    panels.append(None)

            total_width = (
                sum(img.width for img in resized_pils)
                + self.padding * (len(resized_pils) + 1)
            )
            img_height = max(img.height for img in resized_pils)

            panel_heights = [p.height for p in panels if p is not None]
            panel_height = max(panel_heights) if panel_heights else 0
            panel_area_height = (panel_height + self.padding) if panel_height > 0 else 0

            total_height = img_height + panel_area_height + self.padding * 2

            result_pil = Image.new("RGB", (total_width, total_height), self.bg_color)

            y_img = self.padding
            y_panel = y_img + img_height + self.padding

            x = self.padding
            for img, panel in zip(resized_pils, panels):
                result_pil.paste(img, (x, y_img))
                if panel is not None and panel_height > 0:
                    y_offset = (
                        y_panel + (panel_height - panel.height)
                        if panel.height < panel_height
                        else y_panel
                    )
                    result_pil.paste(panel, (x, y_offset))
                x += img.width + self.padding

            output_images.append(pil2tensor(result_pil))

        if not output_images:
            return (torch.zeros((1, 64, 64, 3)),)

        final_batch_tensor = torch.cat(output_images, dim=0)
        return (final_batch_tensor,)

# Color Input node
class AILab_ColorInput:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "preset": (list(COLOR_PRESETS.keys()),),
                "color": ("STRING", {"default": "", "placeholder": "Enter color code (e.g. #FF0000 or #F00)"}),
            },
        }

    RETURN_TYPES = ("COLORCODE",)
    RETURN_NAMES = ("COLOR",)
    FUNCTION = 'get_color'
    CATEGORY = '🧪AILab/🛠️UTIL/🔄IO'

    def get_color(self, preset, color):
        if not color:
            return (COLOR_PRESETS[preset],)
            
        try:
            fixed_color = color_format(color)
            if not all(c in '0123456789ABCDEFabcdef' for c in fixed_color[1:]):
                raise ValueError(f"Invalid hex characters in {color}")
            return (fixed_color,)
        except Exception as e:
            raise RuntimeError(f"Invalid color format: {color}. Please use format like #FF0000 or #F00")

# Image Resize
class AILab_ImageResize:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return r, g, b

    def _pad_image_and_mask(self, image, pad_left, pad_right, pad_top, pad_bottom, pad_color, pad_mode, mask=None):
        B, H, W, C = image.shape
        padded_width = W + pad_left + pad_right
        padded_height = H + pad_top + pad_bottom
        
        pad_color = color_format(pad_color)
        r, g, b = self.hex_to_rgb(pad_color)
        bg_color = torch.tensor([r, g, b], dtype=image.dtype, device=image.device)
        
        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        out_masks = None
        
        if pad_mode == "pillarbox_blur":
            for b in range(B):
                scale_fill = max(padded_width / float(W), padded_height / float(H)) if (W > 0 and H > 0) else 1.0
                bg_w = max(1, int(round(W * scale_fill)))
                bg_h = max(1, int(round(H * scale_fill)))
                
                src_b = image[b].movedim(-1, 0).unsqueeze(0)
                bg = common_upscale(src_b, bg_w, bg_h, "bilinear", crop="disabled").squeeze(0)
                y0 = max(0, (bg_h - padded_height) // 2)
                x0 = max(0, (bg_w - padded_width) // 2)
                bg = bg[:, y0:y0+padded_height, x0:x0+padded_width]

                if bg.shape[1] != padded_height or bg.shape[2] != padded_width:
                    pad_h = padded_height - bg.shape[1]
                    pad_w = padded_width - bg.shape[2]
                    pad_top_fix = max(0, pad_h // 2)
                    pad_bottom_fix = max(0, pad_h - pad_top_fix)
                    pad_left_fix = max(0, pad_w // 2)
                    pad_right_fix = max(0, pad_w - pad_left_fix)
                    bg = F.pad(bg.unsqueeze(0), (pad_left_fix, pad_right_fix, pad_top_fix, pad_bottom_fix), mode="replicate").squeeze(0)

                sigma = max(1.0, 0.006 * float(min(padded_height, padded_width)))
                kernel_size = int(round(sigma * 2)) * 2 + 1
                bg = gaussian_blur(bg, kernel_size=[kernel_size, kernel_size], sigma=sigma)
                
                dim = 0.35
                bg = torch.clamp(bg * dim, 0.0, 1.0)
                out_image[b] = bg.movedim(0, -1)
            
            out_image[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = image
        elif pad_mode == "edge":
            out_image[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = image
            for b in range(B):
                out_image[b, :pad_top, :, :] = out_image[b, pad_top:pad_top+1, :, :].repeat(pad_top, 1, 1)
                out_image[b, pad_top+H:, :, :] = out_image[b, pad_top+H-1:pad_top+H, :, :].repeat(pad_bottom, 1, 1)
                out_image[b, :, :pad_left, :] = out_image[b, :, pad_left:pad_left+1, :].repeat(1, pad_left, 1)
                out_image[b, :, pad_left+W:, :] = out_image[b, :, pad_left+W-1:pad_left+W, :].repeat(1, pad_right, 1)
        elif pad_mode == "edge_pixel":
            out_image[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = image
            for b in range(B):
                out_image[b, :pad_top, pad_left:pad_left+W, :] = image[b, 0, :, :].unsqueeze(0).repeat(pad_top, 1, 1)
                out_image[b, pad_top+H:, pad_left:pad_left+W, :] = image[b, H-1, :, :].unsqueeze(0).repeat(pad_bottom, 1, 1)
                out_image[b, pad_top:pad_top+H, :pad_left, :] = image[b, :, 0, :].unsqueeze(1).repeat(1, pad_left, 1)
                out_image[b, pad_top:pad_top+H, pad_left+W:, :] = image[b, :, W-1, :].unsqueeze(1).repeat(1, pad_right, 1)
                out_image[b, :pad_top, :pad_left, :] = image[b, 0, 0, :]
                out_image[b, :pad_top, pad_left+W:, :] = image[b, 0, W-1, :]
                out_image[b, pad_top+H:, :pad_left, :] = image[b, H-1, 0, :]
                out_image[b, pad_top+H:, pad_left+W:, :] = image[b, H-1, W-1, :]
        else:
            for b in range(B):
                out_image[b, :, :, :] = bg_color
                out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]

        if mask is not None:
            out_masks = torch.nn.functional.pad(
                mask, 
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='replicate'
            )
        else:
            out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
            for m in range(B):
                out_masks[m, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0

        return out_image, out_masks

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.01}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "resize_mode": (["longest_side", "shortest_side"], {"default": "longest_side"}),
                "resize_value": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_method": (cls.upscale_methods, {"default": "lanczos"}),
                "device": (["cpu", "gpu"], {"default": "cpu"}),
                "divisible_by": ("INT", {"default": 2, "min": 1, "max": 512, "step": 1}),
                "output_mode": (["stretch", "pad", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur"], {"default": "stretch"}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
                "pad_color": ("COLORCODE", {"default": "#FFFFFF", "tooltip": "Padding color (hex)"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "WIDTH", "HEIGHT")
    FUNCTION = "resize"
    OUTPUT_NODE = False

    def resize(self, image, custom_width, custom_height, megapixels, scale_by, resize_mode, resize_value, upscale_method, device, divisible_by, output_mode, crop_position, pad_color, mask=None):
        B, orig_height, orig_width, C = image.shape
        
        target_device = torch.device("cpu")
        if device == "gpu":
            if upscale_method == "lanczos":
                raise Exception("Lanczos is not supported on the GPU")
            target_device = model_management.get_torch_device()
        
        out_image = image.clone().to(target_device)
        out_mask = mask.clone().to(target_device) if mask is not None else None
        
        target_width, target_height = orig_width, orig_height
        
        is_megapixels_used = megapixels > 0
        if is_megapixels_used:
            aspect_ratio = orig_width / orig_height
            target_pixels = int(megapixels * 1024 * 1024)
            target_height = int((target_pixels / aspect_ratio) ** 0.5)
            target_width = int(aspect_ratio * target_height)
        elif resize_value > 0:
            if resize_mode == "longest_side":
                ratio = resize_value / max(orig_width, orig_height)
            else:
                ratio = resize_value / min(orig_width, orig_height)
            target_width = round(orig_width * ratio)
            target_height = round(orig_height * ratio)
        elif custom_width > 0 or custom_height > 0:
            if custom_width > 0 and custom_height == 0:
                target_width = custom_width
                target_height = int(orig_height * (target_width / orig_width))
            elif custom_height > 0 and custom_width == 0:
                target_height = custom_height
                target_width = int(orig_width * (target_height / orig_height))
            elif custom_width > 0 and custom_height > 0:
                target_width = custom_width
                target_height = custom_height
        elif scale_by != 1.0:
            target_width = int(orig_width * scale_by)
            target_height = int(orig_height * scale_by)
        
        final_width, final_height = target_width, target_height

        if not is_megapixels_used and scale_by != 1.0:
            final_width = int(final_width * scale_by)
            final_height = int(final_height * scale_by)

        if output_mode == "crop":
            old_width, old_height = orig_width, orig_height
            
            old_aspect = old_width / old_height
            new_aspect = final_width / final_height
            
            if old_aspect > new_aspect:
                crop_w = round(old_height * new_aspect)
                crop_h = old_height
            else:
                crop_w = old_width
                crop_h = round(old_width / new_aspect)
            
            x, y = 0, 0
            if crop_position == "center":
                x = (old_width - crop_w) // 2
                y = (old_height - crop_h) // 2
            elif crop_position == "top":
                x = (old_width - crop_w) // 2
                y = 0
            elif crop_position == "bottom":
                x = (old_width - crop_w) // 2
                y = old_height - crop_h
            elif crop_position == "left":
                x = 0
                y = (old_height - crop_h) // 2
            elif crop_position == "right":
                x = old_width - crop_w
                y = (old_height - crop_h) // 2
            
            out_image = out_image.narrow(-2, x, crop_w).narrow(-3, y, crop_h)
            if out_mask is not None:
                out_mask = out_mask.narrow(-1, x, crop_w).narrow(-2, y, crop_h)

        if output_mode in ["pad", "pad_edge", "pad_edge_pixel", "pillarbox_blur"]:
            orig_aspect = out_image.shape[2] / out_image.shape[1]
            new_aspect = final_width / final_height
            
            if orig_aspect > new_aspect:
                resize_width = final_width
                resize_height = int(final_width / orig_aspect)
            else:
                resize_height = final_height
                resize_width = int(final_height * orig_aspect)
            
            out_image = common_upscale(out_image.movedim(-1, 1), resize_width, resize_height, upscale_method, crop="disabled").movedim(1, -1)
            if out_mask is not None:
                out_mask = F.interpolate(out_mask.unsqueeze(1), size=(resize_height, resize_width), mode='bicubic', align_corners=False).squeeze(1)

            remaining_w = final_width - out_image.shape[2]
            remaining_h = final_height - out_image.shape[1]
            if crop_position == "left":
                pad_left = 0
                pad_right = remaining_w
                pad_top = remaining_h // 2
                pad_bottom = remaining_h - pad_top
            elif crop_position == "right":
                pad_left = remaining_w
                pad_right = 0
                pad_top = remaining_h // 2
                pad_bottom = remaining_h - pad_top
            elif crop_position == "top":
                pad_left = remaining_w // 2
                pad_right = remaining_w - pad_left
                pad_top = 0
                pad_bottom = remaining_h
            elif crop_position == "bottom":
                pad_left = remaining_w // 2
                pad_right = remaining_w - pad_left
                pad_top = remaining_h
                pad_bottom = 0
            else:
                pad_left = remaining_w // 2
                pad_right = remaining_w - pad_left
                pad_top = remaining_h // 2
                pad_bottom = remaining_h - pad_top
            
            pad_mode_mapped = {
                "pad_edge": "edge",
                "pad_edge_pixel": "edge_pixel",
            }.get(output_mode, output_mode)
            out_image, out_mask = self._pad_image_and_mask(
                out_image, pad_left, pad_right, pad_top, pad_bottom, pad_color, pad_mode_mapped, mask=out_mask
            )
            final_width, final_height = out_image.shape[2], out_image.shape[1]
            
        elif output_mode == "stretch" or output_mode == "crop":
            out_image = common_upscale(out_image.movedim(-1, 1), final_width, final_height, upscale_method, crop="disabled").movedim(1, -1)
            if out_mask is not None:
                out_mask = F.interpolate(out_mask.unsqueeze(1), size=(final_height, final_width), mode='bicubic', align_corners=False).squeeze(1)
                
        if divisible_by > 1:
            final_width = final_width - (final_width % divisible_by)
            final_height = final_height - (final_height % divisible_by)
            out_image = common_upscale(out_image.movedim(-1, 1), final_width, final_height, upscale_method, crop="disabled").movedim(1, -1)
            if out_mask is not None:
                 out_mask = F.interpolate(out_mask.unsqueeze(1), size=(final_height, final_width), mode='bicubic', align_corners=False).squeeze(1)

        return (out_image.cpu(), out_mask.cpu() if out_mask is not None else torch.zeros((1, 64, 64)), final_width, final_height)

# Image to Batch List
class AILab_ImageToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
            },
            "required": {
                "resize_mode": (["off", "crop", "fit"], {"default": "crop"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "WIDTH", "HEIGHT", "BATCH_SIZE")
    OUTPUT_IS_LIST = (True, False, False, False)
    FUNCTION = "collect_images"
    CATEGORY = "🧪AILab/🖼️IMAGE"
    
    def resize_and_center_crop(self, img, target_h, target_w):
        b, h, w, c = img.shape
        scale = max(target_h / h, target_w / w)
        new_h = max(int(h * scale), target_h)
        new_w = max(int(w * scale), target_w)
        img = img.permute(0, 3, 1, 2)
        resized = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
        top = max((new_h - target_h) // 2, 0)
        left = max((new_w - target_w) // 2, 0)
        bottom = top + target_h
        right = left + target_w
        if bottom > new_h or right > new_w:
            cropped = resized[:, :, :target_h, :target_w]
        else:
            cropped = resized[:, :, top:bottom, left:right]
        return cropped.permute(0, 2, 3, 1)
    
    def collect_images(self, resize_mode="off", **kwargs):
        images = []
        for i in range(1, 7):
            key = f"image_{i}"
            img = kwargs.get(key)
            if img is not None:
                images.append(img)
        
        if not images:
            return ([], 0, 0, 0)
        
        base_img = images[0]
        _, base_h, base_w, _ = base_img.shape
        
        batch_size = len(images)
        
        if resize_mode == "off":
            return (images, base_w, base_h, batch_size)
        
        resized_images = []
        for img in images:
            if resize_mode == "fit":
                img_nchw = img.permute(0, 3, 1, 2)
                resized = F.interpolate(img_nchw, size=(base_h, base_w), mode='bilinear', align_corners=False)
                resized = resized.permute(0, 2, 3, 1)
            elif resize_mode == "crop":
                resized = self.resize_and_center_crop(img, base_h, base_w)
            resized_images.append(resized)
        
        return (resized_images, base_w, base_h, batch_size)

# Mask to Batch List
class AILab_MaskToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "mask_1": ("MASK",),
                "mask_2": ("MASK",),
                "mask_3": ("MASK",),
                "mask_4": ("MASK",),
                "mask_5": ("MASK",),
                "mask_6": ("MASK",),
            },
            "required": {
                "resize_mode": (["off", "crop", "fit"], {"default": "off"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "collect_masks"
    CATEGORY = "🧪AILab/🦠MASK"

    def resize_and_center_crop(self, mask, target_h, target_w):
        b, h, w = mask.shape
        scale = max(target_h / h, target_w / w)
        new_h = max(int(h * scale), target_h)
        new_w = max(int(w * scale), target_w)
        mask = mask.unsqueeze(1)
        resized = F.interpolate(mask, size=(new_h, new_w), mode='nearest')
        top = max((new_h - target_h) // 2, 0)
        left = max((new_w - target_w) // 2, 0)
        bottom = top + target_h
        right = left + target_w
        cropped = resized[:, :, top:bottom, left:right]
        return cropped.squeeze(1)

    def collect_masks(self, resize_mode="off", **kwargs):
        masks = []
        for i in range(1, 7):
            key = f"mask_{i}"
            mask = kwargs.get(key)
            if mask is not None:
                masks.append(mask)
        if not masks:
            return ([],)

        base_mask = masks[0]
        _, base_h, base_w = base_mask.shape

        if resize_mode == "off":
            return (masks,)

        resized_masks = []
        for mask in masks:
            if resize_mode == "fit":
                mask_resized = F.interpolate(mask.unsqueeze(1), size=(base_h, base_w), mode='nearest').squeeze(1)
            elif resize_mode == "crop":
                mask_resized = self.resize_and_center_crop(mask, base_h, base_w)
            resized_masks.append(mask_resized)

        return (resized_masks,)

# Image and Mask to Batch List
class AILab_ImageMaskToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image_1": ("IMAGE",),
                "mask_1": ("MASK",),
                "image_2": ("IMAGE",),
                "mask_2": ("MASK",),
                "image_3": ("IMAGE",),
                "mask_3": ("MASK",),
                "image_4": ("IMAGE",),
                "mask_4": ("MASK",),
                "image_5": ("IMAGE",),
                "mask_5": ("MASK",),
                "image_6": ("IMAGE",),
                "mask_6": ("MASK",),
            },
            "required": {
                "resize_mode": (["off", "crop", "fit"], {"default": "crop"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "WIDTH", "HEIGHT", "BATCH_SIZE")
    OUTPUT_IS_LIST = (True, True, False, False, False)
    FUNCTION = "collect_images_and_masks"
    CATEGORY = "🧪AILab/🖼️IMAGE"
    
    def resize_and_center_crop_image(self, img, target_h, target_w):
        b, h, w, c = img.shape
        scale = max(target_h / h, target_w / w)
        new_h = max(int(h * scale), target_h)
        new_w = max(int(w * scale), target_w)
        img = img.permute(0, 3, 1, 2)
        resized = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
        top = max((new_h - target_h) // 2, 0)
        left = max((new_w - target_w) // 2, 0)
        bottom = top + target_h
        right = left + target_w
        if bottom > new_h or right > new_w:
            cropped = resized[:, :, :target_h, :target_w]
        else:
            cropped = resized[:, :, top:bottom, left:right]
        return cropped.permute(0, 2, 3, 1)
    
    def resize_and_center_crop_mask(self, mask, target_h, target_w):
        b, h, w = mask.shape
        scale = max(target_h / h, target_w / w)
        new_h = max(int(h * scale), target_h)
        new_w = max(int(w * scale), target_w)
        mask = mask.unsqueeze(1)
        resized = F.interpolate(mask, size=(new_h, new_w), mode='nearest')
        top = max((new_h - target_h) // 2, 0)
        left = max((new_w - target_w) // 2, 0)
        bottom = top + target_h
        right = left + target_w
        cropped = resized[:, :, top:bottom, left:right]
        return cropped.squeeze(1)
    
    def create_empty_mask(self, height, width, batch_size=1, device=None):

        if device is None:
            device = torch.device('cpu')
        return torch.zeros((batch_size, height, width), dtype=torch.float32, device=device)
    
    def collect_images_and_masks(self, resize_mode="off", **kwargs):
        image_mask_pairs = []
        
        for i in range(1, 7):
            image_key = f"image_{i}"
            mask_key = f"mask_{i}"
            
            image = kwargs.get(image_key)
            mask = kwargs.get(mask_key)
            
            if image is not None:
                if mask is None:
                    _, h, w, _ = image.shape
                    mask = self.create_empty_mask(h, w, batch_size=image.shape[0], device=image.device)
                
                image_mask_pairs.append((image, mask))
        
        if not image_mask_pairs:
            return ([], [], 0, 0, 0)

        base_image, base_mask = image_mask_pairs[0]
        _, base_h, base_w, _ = base_image.shape
        
        batch_size = len(image_mask_pairs)
        
        if resize_mode == "off":
            processed_images = [pair[0] for pair in image_mask_pairs]
            processed_masks = [pair[1] for pair in image_mask_pairs]
            return (processed_images, processed_masks, base_w, base_h, batch_size)
        
        processed_images = []
        processed_masks = []
        
        for image, mask in image_mask_pairs:
            if resize_mode == "fit":
                img_nchw = image.permute(0, 3, 1, 2)
                resized_image = F.interpolate(img_nchw, size=(base_h, base_w), mode='bilinear', align_corners=False)
                resized_image = resized_image.permute(0, 2, 3, 1)
            elif resize_mode == "crop":
                resized_image = self.resize_and_center_crop_image(image, base_h, base_w)
            
            if resize_mode == "fit":
                resized_mask = F.interpolate(mask.unsqueeze(1), size=(base_h, base_w), mode='nearest').squeeze(1)
            elif resize_mode == "crop":
                resized_mask = self.resize_and_center_crop_mask(mask, base_h, base_w)
            
            processed_images.append(resized_image)
            processed_masks.append(resized_mask)
        
        return (processed_images, processed_masks, base_w, base_h, batch_size)

#Load Image Batch
class AILab_LoadImageBatch(AILab_BaseImageLoader):
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path_or_urls": ("STRING", {"default": "", "multiline": True, "placeholder": "Path to a directory, comma/new-line separated file paths, OR comma/new-line separated URLs"}),
                "upscale_method": (cls.upscale_methods, {"default": "lanczos"}),
                "megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.01}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "resize_mode": (["longest_side", "shortest_side", "width", "height"], {"default": "longest_side"}),
                "size": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            },
            "optional": {
                "batch_size": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "Number of images to load (0 = all images)"}),
                "start_from": ("INT", {"default": 1, "min": 1, "step": 1, "tooltip": "Start from Nth image (1 = first image)"}),
                "sort_method": (["sequential", "reverse", "random"], {"default": "sequential", "tooltip": "Image loading order: sequential/reverse/random"}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "WIDTH", "HEIGHT")
    FUNCTION = "load_image_batch"
    OUTPUT_NODE = False
    OUTPUT_IS_LIST = (True, True, True, True) 

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'sort_method' in kwargs and kwargs['sort_method'] == "random":
            return float("NaN")
        return hashlib.sha256(str(kwargs).encode('utf-8')).hexdigest()

    def load_image_batch(self, path_or_urls="", upscale_method="lanczos", megapixels=0.0,
                         scale_by=1.0, resize_mode="longest_side", size=0, 
                         batch_size=0, start_from=1, sort_method="sequential", extra_pnginfo=None):
        
        image_list = []
        
        input_path = path_or_urls.strip()
        
        if not input_path:
                raise ValueError("No input provided. Please specify a path/URL list.")
        
        potential_paths = [path.strip() for path in re.split(r'[,\n]+', input_path) if path.strip()]

        if not potential_paths:
            raise ValueError("Input is empty or contains only whitespace.")

        first_path = potential_paths[0]
        
        if first_path.startswith(('http://', 'https://')):
            image_list = [path.strip() for path in re.split(r'[,\n]\s*(?=http)', input_path) if path.strip()]
        
        elif os.path.isdir(first_path):
            image_list = [
                os.path.join(first_path, f)
                for f in os.listdir(first_path)
                if os.path.isfile(os.path.join(first_path, f)) and f.lower().endswith(self.IMAGE_EXTENSIONS)
            ]
            image_list.sort()
        
        elif os.path.isfile(first_path):
            image_list = [p for p in potential_paths if os.path.isfile(p)]
        
        else:
            relative_path_check = os.path.join(folder_paths.get_input_directory(), first_path)
            
            if os.path.isdir(relative_path_check):
                image_list = [
                    os.path.join(relative_path_check, f)
                    for f in os.listdir(relative_path_check)
                    if os.path.isfile(os.path.join(relative_path_check, f)) and f.lower().endswith(self.IMAGE_EXTENSIONS)
                ]
                image_list.sort()
            
            elif os.path.isfile(relative_path_check):
                image_list = [os.path.join(folder_paths.get_input_directory(), p) for p in potential_paths 
                              if os.path.isfile(os.path.join(folder_paths.get_input_directory(), p))]
            
            else:
                raise ValueError(f"Input is not a valid URL, directory, or file path: {first_path}")

        if not image_list:
            raise ValueError("No valid images found from the provided input.")
        
        if sort_method == "reverse":
            image_list.reverse()
        elif sort_method == "random":
            import random
            random.shuffle(image_list)

        start_index = max(0, start_from - 1)
        if start_index > 0 and start_index < len(image_list):
            image_list = image_list[start_index:]
        elif start_index >= len(image_list):
             raise ValueError(f"start_from ({start_from}) is out of bounds. Only {len(image_list)} images found.")

        if batch_size > 0:
            image_list = image_list[:batch_size]
        
        if not image_list:
            raise ValueError("No images left after applying start_from/batch_size filters.")

        resampling = {
            "nearest-exact": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "area": Image.BOX,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS
        }.get(upscale_method, Image.LANCZOS)

        output_images = []
        output_masks = []
        output_widths = []
        output_heights = []

        for img_path_or_url in image_list:
            img = self.get_image(img_path_or_url)
            resized_img, width, height = self.resize_image_to_target(
                img, megapixels=megapixels, scale_by=scale_by, size=size,
                resize_mode=resize_mode, resampling=resampling
            )
            
            img_tensor = self.process_image_to_tensor(resized_img)
            
            mask = None
            if 'A' in resized_img.getbands():
                mask_np = np.array(resized_img.getchannel('A')).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask_np).unsqueeze(0)
            else:
                mask = torch.ones((1, height, width), dtype=torch.float32)
            
            output_images.append(img_tensor)
            output_masks.append(mask)
            output_widths.append(width)
            output_heights.append(height)
        
        if not output_images:
            raise ValueError("All images in the batch failed to load or process.")

        return (output_images, output_masks, output_widths, output_heights)

#Unbatch Images
class AILab_UnbatchImages:
    CATEGORY = "🧪AILab/🖼️IMAGE"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8")
    FUNCTION = "unbatch_images"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    def unbatch_images(self, images):
        if images is None or images.shape[0] == 0:
            raise ValueError("Input batch is empty. Upstream node (e.g., LoadImageBatch) likely failed to load any images.")
        
        batch_size = images.shape[0]
        outputs = []

        for i in range(8):
            if i < batch_size:
                image = images[i:i+1, :, :, :]
                outputs.append(image)
            else:
                outputs.append(outputs[-1])
        
        return tuple(outputs)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "AILab_LoadImage": AILab_LoadImage,
    "AILab_LoadImageSimple": AILab_LoadImageSimple,
    "AILab_LoadImageAdvanced": AILab_LoadImageAdvanced,
    "AILab_LoadImageBatch": AILab_LoadImageBatch,
    "AILab_UnbatchImages": AILab_UnbatchImages,
    "AILab_Preview": AILab_Preview,
    "AILab_MaskOverlay": AILab_MaskOverlay,
    "AILab_ImagePreview": AILab_ImagePreview,
    "AILab_MaskPreview": AILab_MaskPreview,
    "AILab_ImageMaskConvert": AILab_ImageMaskConvert,
    "AILab_MaskEnhancer": AILab_MaskEnhancer,
    "AILab_MaskCombiner": AILab_MaskCombiner,
    "AILab_ImageCombiner": AILab_ImageCombiner,
    "AILab_MaskExtractor": AILab_MaskExtractor,
    "AILab_ImageStitch": AILab_ImageStitch,
    "AILab_ImageCrop": AILab_ImageCrop,
    "AILab_ICLoRAConcat": AILab_ICLoRAConcat,
    "AILab_CropObject": AILab_CropObject,
    "AILab_ImageCompare": AILab_ImageCompare,
    "AILab_ColorInput": AILab_ColorInput,
    "AILab_ImageResize": AILab_ImageResize,
    "AILab_ImageToList": AILab_ImageToList,
    "AILab_MaskToList": AILab_MaskToList,
    "AILab_ImageMaskToList": AILab_ImageMaskToList,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_LoadImage": "Load Image (RMBG) 🖼️",
    "AILab_LoadImageSimple": "Load Image Basic (RMBG) 🖼️",
    "AILab_LoadImageAdvanced": "Load Image Advanced (RMBG) 🖼️",
    "AILab_LoadImageBatch": "Load Image Batch (RMBG) 🖼️",
    "AILab_UnbatchImages": "Unbatch Images (RMBG) 🖼️",
    "AILab_Preview": "Image/Mask Preview (RMBG) 🖼️🎭",
    "AILab_MaskOverlay": "Mask Overlay (RMBG) 🖼️🎭",
    "AILab_ImagePreview": "Image Preview (RMBG) 🖼️",
    "AILab_MaskPreview": "Mask Preview (RMBG) 🎭",
    "AILab_ImageMaskConvert": "Image/Mask Converter (RMBG) 🖼️🎭",
    "AILab_MaskEnhancer": "Mask Enhancer (RMBG) 🎭",
    "AILab_MaskCombiner": "Mask Combiner (RMBG) 🎭",
    "AILab_ImageCombiner": "Image Combiner (RMBG) 🖼️",
    "AILab_MaskExtractor": "Mask Extractor (RMBG) 🎭",
    "AILab_ImageStitch": "Image Stitch (RMBG) 🖼️",
    "AILab_ImageCrop": "Image Crop (RMBG) 🖼️",
    "AILab_ICLoRAConcat": "IC LoRA Concat (RMBG) 🖼️🎭",
    "AILab_CropObject": "Crop To Object (RMBG) 🖼️🎭",
    "AILab_ImageCompare": "Side By Side Compare (RMBG) 🖼️🖼️",
    "AILab_ColorInput": "Color Input (RMBG) 🎨",
    "AILab_ImageResize": "Image Resize (RMBG) 🖼️🎭",
    "AILab_ImageToList": "Image to List (RMBG)",
    "AILab_MaskToList": "Mask to List (RMBG)",
    "AILab_ImageMaskToList": "Image and Mask to List (RMBG)",
}
