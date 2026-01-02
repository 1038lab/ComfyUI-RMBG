# ComfyUI-RMBG
# This custom node for ComfyUI provides functionality for background removal using BiRefNet models.
#
# Model License Notice:
# - BiRefNet Models: Apache-2.0 License (https://huggingface.co/ZhengPeng7)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/AILab-AI/ComfyUI-RMBG

import os
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
import numpy as np
import folder_paths
from huggingface_hub import hf_hub_download
import sys
import importlib.util
from safetensors.torch import load_file
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

# Add model path
folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

# Model configuration
MODEL_CONFIG = {
    "BiRefNet-general": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-general.safetensors": "BiRefNet-general.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "General purpose model with balanced performance",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet_512x512": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_512x512.safetensors": "BiRefNet_512x512.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "Optimized for 512x512 resolution, faster processing",
        "default_res": 512,
        "max_res": 1024,
        "min_res": 256,
        "force_res": True
    },
    "BiRefNet-HR": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-HR.safetensors": "BiRefNet-HR.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "High resolution general purpose model",
        "default_res": 2048,
        "max_res": 2560,
        "min_res": 1024
    },
    "BiRefNet-portrait": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-portrait.safetensors": "BiRefNet-portrait.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "Optimized for portrait/human matting",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet-matting": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-matting.safetensors": "BiRefNet-matting.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "General purpose matting model",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet-HR-matting": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-HR-matting.safetensors": "BiRefNet-HR-matting.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "High resolution matting model",
        "default_res": 2048,
        "max_res": 2560,
        "min_res": 1024
    },
    "BiRefNet_lite": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet_lite.py": "birefnet_lite.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_lite.safetensors": "BiRefNet_lite.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "Lightweight version for faster processing",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet_lite-2K": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet_lite.py": "birefnet_lite.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_lite-2K.safetensors": "BiRefNet_lite-2K.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "Lightweight version optimized for 2K resolution",
        "default_res": 2048,
        "max_res": 2560,
        "min_res": 1024
    },
    "BiRefNet_dynamic": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_dynamic.safetensors": "BiRefNet_dynamic.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "Dynamic model for high-resolution dichotomous image segmentation",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet_lite-matting": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet_lite.py": "birefnet_lite.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_lite-matting.safetensors": "BiRefNet_lite-matting.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "Lightweight matting model for general purpose",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet_toonout": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_toonout.safetensors": "BiRefNet_toonout.safetensors",
            "config.json": "config.json"
        },
        "cache_dir": "BiRefNet",
        "description": "A model to get a toon style outline from an image.",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    }
}

# Utility functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def handle_model_error(message):
    print(f"[BiRefNet ERROR] {message}")
    raise RuntimeError(message)

def refine_foreground(image_bchw, masks_b1hw):
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
        for c in range(image_np.shape[1]):
            channel = image_np[i, c]
            refined = channel * mask_refined
            result.append(refined)
            
        refined_fg.append(np.stack(result))
    
    return torch.from_numpy(np.stack(refined_fg))

class BiRefNetModel:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.base_cache_dir = os.path.join(folder_paths.models_dir, "RMBG")
    
    def get_cache_dir(self, model_name):
        return os.path.join(self.base_cache_dir, MODEL_CONFIG[model_name]["cache_dir"])
    
    def check_model_cache(self, model_name):
        cache_dir = self.get_cache_dir(model_name)
        
        if not os.path.exists(cache_dir):
            return False, "Model directory not found"
        
        missing_files = []
        for filename in MODEL_CONFIG[model_name]["files"].keys():
            if not os.path.exists(os.path.join(cache_dir, filename)):
                missing_files.append(filename)
        
        if missing_files:
            return False, f"Missing model files: {', '.join(missing_files)}"
            
        return True, "Model cache verified"
    
    def download_model(self, model_name):
        cache_dir = self.get_cache_dir(model_name)
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Downloading {model_name} model files...")
            
            for filename in MODEL_CONFIG[model_name]["files"].keys():
                print(f"Downloading {filename}...")
                hf_hub_download(
                    repo_id=MODEL_CONFIG[model_name]["repo_id"],
                    filename=filename,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                    
            return True, "Model files downloaded successfully"
            
        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"
    
    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            self.current_model_version = None
            torch.cuda.empty_cache()
            print("Model cleared from memory")

    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            cache_dir = self.get_cache_dir(model_name)
            model_filename = [k for k in MODEL_CONFIG[model_name]["files"].keys() if k.endswith('.py') and k != "BiRefNet_config.py"][0]
            model_path = os.path.join(cache_dir, model_filename)
            config_path = os.path.join(cache_dir, "BiRefNet_config.py")
            weights_filename = [k for k in MODEL_CONFIG[model_name]["files"].keys() if k.endswith('.safetensors')][0]
            weights_path = os.path.join(cache_dir, weights_filename)
            
            try:
                # Fix relative imports in model file
                with open(model_path, 'r', encoding='utf-8') as f:
                    model_content = f.read()
                model_content = model_content.replace("from .BiRefNet_config", "from BiRefNet_config")
                with open(model_path, 'w', encoding='utf-8') as f:
                    f.write(model_content)
                
                # Load config and model dynamically
                spec = importlib.util.spec_from_file_location("BiRefNet_config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                sys.modules["BiRefNet_config"] = config_module
                spec.loader.exec_module(config_module)
                
                spec = importlib.util.spec_from_file_location("birefnet", model_path)
                model_module = importlib.util.module_from_spec(spec)
                sys.modules["birefnet"] = model_module
                spec.loader.exec_module(model_module)
                
                # Initialize model
                self.model = model_module.BiRefNet(config_module.BiRefNetConfig())
                
                # Load weights
                state_dict = load_file(weights_path)
                self.model.load_state_dict(state_dict)
                
                self.model.eval()
                self.model.half()
                torch.set_float32_matmul_precision('high')
                self.model.to(device)
                self.current_model_version = model_name
                
            except Exception as e:
                handle_model_error(f"Error loading BiRefNet model: {str(e)}")

    def process_image(self, image, params):
        try:
            transform_image = transforms.Compose([
                transforms.Resize((params["process_res"], params["process_res"]), 
                                interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            
            input_tensor = transform_image(orig_image).unsqueeze(0).to(device).half()
            
            with torch.no_grad():
                preds = self.model(input_tensor)
                pred = preds[-1].sigmoid().cpu()
            
            pred = pred[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize((w, h), Image.BICUBIC)
            
            return mask
            
        except Exception as e:
            handle_model_error(f"Error in BiRefNet processing: {str(e)}")

class BiRefNetRMBG:
    def __init__(self):
        self.model = BiRefNetModel()
    
    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "image": "Input image to be processed for background removal.",
            "model": "Select the BiRefNet model variant to use.",
            "mask_blur": "Specify the amount of blur to apply to the mask edges (0 for no blur, higher values for more blur).",
            "mask_offset": "Adjust the mask boundary (positive values expand the mask, negative values shrink it).",
            "invert_output": "Enable to invert both the image and mask output (useful for certain effects).",
            "refine_foreground": "Use Fast Foreground Colour Estimation to optimize transparent background",
            "background": "Choose background type: Alpha (transparent) or Color (custom background color).",
            "background_color": "Choose background color (Alpha = transparent)"
        }
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "model": (list(MODEL_CONFIG.keys()), {"tooltip": tooltips["model"]}),
            },
            "optional": {
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "refine_foreground": ("BOOLEAN", {"default": False, "tooltip": tooltips["refine_foreground"]}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": tooltips["background_color"]}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "process_image"
    CATEGORY = "🧪AILab/🧽RMBG"

    def process_image(self, image, model, **params):
        try:
            model_config = MODEL_CONFIG[model]
            process_res = model_config.get("default_res", 1024)
            if model_config.get("force_res", False):
                base_res = 512
                process_res = ((process_res + base_res - 1) // base_res) * base_res
            else:
                process_res = process_res // 32 * 32
            print(f"Using {model} model with {process_res} resolution")
            params["process_res"] = process_res
            processed_images = []
            processed_masks = []
            cache_status, message = self.model.check_model_cache(model)
            if not cache_status:
                print(f"Cache check: {message}")
                print("Downloading required model files...")
                download_status, download_message = self.model.download_model(model)
                if not download_status:
                    handle_model_error(download_message)
                print("Model files downloaded successfully")
            self.model.load_model(model)
            for img in image:
                mask = self.model.process_image(img, params)
                if params["mask_blur"] > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=params["mask_blur"]))
                if params["mask_offset"] != 0:
                    if params["mask_offset"] > 0:
                        for _ in range(params["mask_offset"]):
                            mask = mask.filter(ImageFilter.MaxFilter(3))
                    else:
                        for _ in range(-params["mask_offset"]):
                            mask = mask.filter(ImageFilter.MinFilter(3))
                if params["invert_output"]:
                    mask = Image.fromarray(255 - np.array(mask))
                img_tensor = torch.from_numpy(np.array(tensor2pil(img))).permute(2, 0, 1).unsqueeze(0) / 255.0
                mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).unsqueeze(0) / 255.0
                if params.get("refine_foreground", False):
                    refined_fg = refine_foreground(img_tensor, mask_tensor)
                    refined_fg = tensor2pil(refined_fg[0].permute(1, 2, 0))
                    orig_image = tensor2pil(img)
                    r, g, b = refined_fg.split()
                    foreground = Image.merge('RGBA', (r, g, b, mask))
                else:
                    orig_image = tensor2pil(img)
                    orig_rgba = orig_image.convert("RGBA")
                    r, g, b, _ = orig_rgba.split()
                    foreground = Image.merge('RGBA', (r, g, b, mask))
                if params["background"] == "Alpha":
                    processed_images.append(pil2tensor(foreground))
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
                    background_color = params.get("background_color", "#222222")
                    rgba = hex_to_rgba(background_color)
                    bg_image = Image.new('RGBA', orig_image.size, rgba)
                    composite_image = Image.alpha_composite(bg_image, foreground)
                    processed_images.append(pil2tensor(composite_image.convert("RGB")))
                processed_masks.append(pil2tensor(mask))
            mask_images = []
            for mask_tensor in processed_masks:
                mask_image = mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
                mask_images.append(mask_image)
            mask_image_output = torch.cat(mask_images, dim=0)
            return (torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0), mask_image_output)
        except Exception as e:
            handle_model_error(f"Error in image processing: {str(e)}")

# Node Mapping
NODE_CLASS_MAPPINGS = {
    "BiRefNetRMBG": BiRefNetRMBG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNetRMBG": "BiRefNet (RMBG)"
}