import os
import sys
import copy
import torch
import numpy as np
from PIL import Image, ImageFilter
from torch.hub import download_url_to_file
from safetensors.torch import load_file

import folder_paths
import comfy.model_management

from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

try:
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.models import build_model
    from groundingdino.util.utils import clean_state_dict
    from groundingdino.util import box_ops
    from groundingdino.datasets.transforms import Compose, RandomResize, ToTensor, Normalize
    GROUNDINGDINO_AVAILABLE = True
except ImportError:
    GROUNDINGDINO_AVAILABLE = False
    print("Warning: GroundingDINO not available. Text prompts will use fallback method.")

# Disable PyTorch JIT compilation
torch.jit._state._python_cu = None
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

original_jit_script = torch.jit.script
def patched_jit_script(obj):
    return obj
torch.jit.script = patched_jit_script

current_dir = os.path.dirname(__file__)
sam2_path = os.path.join(current_dir, "sam2")
sys.path.insert(0, sam2_path)

import torch.serialization
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

from sam2_image_predictor import SAM2ImagePredictor
from AILab_ImageMaskTools import pil2tensor, tensor2pil

# SAM2 model definitions with FP32 and FP16 versions
SAM2_MODELS = {
    "sam2.1_hiera_tiny": {
        "fp32": {
            "model_url": "https://huggingface.co/1038lab/sam2/resolve/main/sam2.1_hiera_tiny.safetensors",
            "filename": "sam2.1_hiera_tiny.safetensors"
        },
        "fp16": {
            "model_url": "https://huggingface.co/1038lab/sam2/resolve/main/sam2.1_hiera_tiny-fp16.safetensors",
            "filename": "sam2.1_hiera_tiny-fp16.safetensors"
        }
    },
    "sam2.1_hiera_small": {
        "fp32": {
            "model_url": "https://huggingface.co/1038lab/sam2/resolve/main/sam2.1_hiera_small.safetensors",
            "filename": "sam2.1_hiera_small.safetensors"
        },
        "fp16": {
            "model_url": "https://huggingface.co/1038lab/sam2/resolve/main/sam2.1_hiera_small-fp16.safetensors",
            "filename": "sam2.1_hiera_small-fp16.safetensors"
        }
    },
    "sam2.1_hiera_base_plus": {
        "fp32": {
            "model_url": "https://huggingface.co/1038lab/sam2/resolve/main/sam2.1_hiera_base_plus.safetensors",
            "filename": "sam2.1_hiera_base_plus.safetensors"
        },
        "fp16": {
            "model_url": "https://huggingface.co/1038lab/sam2/resolve/main/sam2.1_hiera_base_plus-fp16.safetensors",
            "filename": "sam2.1_hiera_base_plus-fp16.safetensors"
        }
    },
    "sam2.1_hiera_large": {
        "fp32": {
            "model_url": "https://huggingface.co/1038lab/sam2/resolve/main/sam2.1_hiera_large.safetensors",
            "filename": "sam2.1_hiera_large.safetensors"
        },
        "fp16": {
            "model_url": "https://huggingface.co/1038lab/sam2/resolve/main/sam2.1_hiera_large-fp16.safetensors",
            "filename": "sam2.1_hiera_large-fp16.safetensors"
        }
    }
}

# GroundingDINO model definitions
DINO_MODELS = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/1038lab/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/1038lab/GroundingDINO/resolve/main/groundingdino_swint_ogc.safetensors",
        "config_filename": "GroundingDINO_SwinT_OGC.cfg.py",
        "model_filename": "groundingdino_swint_ogc.safetensors"
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/1038lab/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/1038lab/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.safetensors",
        "config_filename": "GroundingDINO_SwinB.cfg.py",
        "model_filename": "groundingdino_swinb_cogcoor.safetensors"
    }
}

def get_or_download_model_file(filename, url, dirname):
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

def process_mask(mask_image: Image.Image, invert_output: bool = False, 
                mask_blur: int = 0, mask_offset: int = 0) -> Image.Image:
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

def apply_background_color(image: Image.Image, mask_image: Image.Image, 
                         background: str = "Alpha",
                         background_color: str = "#222222") -> Image.Image:
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


class SAM2Segment:
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "prompt": "Enter text description of object to segment",
            "sam2_model": "SAM2 model size: Tiny (fastest) to Large (best quality)",
            "device": "Auto: smart detection, CPU: force CPU, GPU: force GPU",
            "dino_model": "GroundingDINO model for text-to-box detection",
            "threshold": "Detection threshold (higher = more strict)",
            "mask_blur": "Blur mask edges (0 = disabled)",
            "mask_offset": "Expand/shrink mask (positive = expand)",
            "invert_output": "Invert the mask output",
            "background": "Background type",
            "background_color": "Background color (when not Alpha)",
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "Object to segment", "tooltip": tooltips["prompt"]}),
                "sam2_model": (list(SAM2_MODELS.keys()), {"default": "sam2.1_hiera_tiny", "tooltip": tooltips["sam2_model"]}),
                "dino_model": (list(DINO_MODELS.keys()), {"default": "GroundingDINO_SwinT_OGC (694MB)", "tooltip": tooltips["dino_model"]}),
                "device": (["Auto", "CPU", "GPU"], {"default": "Auto", "tooltip": tooltips["device"]}),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.01, "tooltip": tooltips["threshold"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": tooltips["background_color"]}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "segment_v2"
    CATEGORY = "🧪AILab/🧽RMBG"

    def __init__(self):
        self.dino_model_cache = {}
        self.sam2_model_cache = {}

    def load_sam2(self, model_name, device="Auto"):
        cache_key = f"{model_name}_{device}"
        if cache_key not in self.sam2_model_cache:
            model_info = SAM2_MODELS[model_name]
            device_obj = comfy.model_management.get_torch_device()
            
            # Determine precision based on device preference
            if device == "Auto":
                precision = "fp16" if hasattr(device_obj, 'type') and device_obj.type == 'cuda' else "fp32"
            elif device == "GPU":
                precision = "fp16" if hasattr(device_obj, 'type') and device_obj.type == 'cuda' else "fp32"
            else:  # CPU
                precision = "fp32"
            
            print(f"Loading {model_name} in {precision.upper()} precision")
            
            model_path = get_or_download_model_file(model_info[precision]["filename"], model_info[precision]["model_url"], "sam2")
            
            # Clear any existing Hydra instance
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()
            
            initialize(config_path="sam2/configs")
            
            config_map = {
                "sam2.1_hiera_tiny": "sam2.1/sam2.1_hiera_t.yaml",
                "sam2.1_hiera_small": "sam2.1/sam2.1_hiera_s.yaml", 
                "sam2.1_hiera_base_plus": "sam2.1/sam2.1_hiera_b+.yaml",
                "sam2.1_hiera_large": "sam2.1/sam2.1_hiera_l.yaml"
            }
            
            config_file = config_map[model_name]
            sam_device = comfy.model_management.get_torch_device()
            
            from sam2.build_sam import build_sam2
            from hydra import compose
            from omegaconf import OmegaConf
            from hydra.utils import instantiate
            
            cfg = compose(config_name=config_file)
            OmegaConf.resolve(cfg)
            sam_model = instantiate(cfg.model, _recursive_=True)
            
            state_dict = load_file(model_path)
            
            # Apply precision to model
            dtype = {"fp16": torch.float16, "fp32": torch.float32}[precision]
            sam_model.load_state_dict(state_dict, strict=False)
            sam_model = sam_model.to(dtype).to(sam_device).eval()
            
            predictor = SAM2ImagePredictor(sam_model)
            self.sam2_model_cache[cache_key] = predictor
        return self.sam2_model_cache[cache_key]

    def segment_v2(self, image, prompt, sam2_model, dino_model, device, threshold=0.35,
                   mask_blur=0, mask_offset=0, background="Alpha", 
                   background_color="#222222", invert_output=False):
        device_obj = comfy.model_management.get_torch_device()

        # Process batch images
        batch_size = image.shape[0] if len(image.shape) == 4 else 1
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        result_images = []
        result_masks = []
        result_mask_images = []
        
        for b in range(batch_size):
            img_pil = tensor2pil(image[b])
            img_np = np.array(img_pil.convert("RGB"))

            # Load GroundingDINO config and weights
            dino_info = DINO_MODELS[dino_model]
            config_path = get_or_download_model_file(dino_info["config_filename"], dino_info["config_url"], "grounding-dino")
            weights_path = get_or_download_model_file(dino_info["model_filename"], dino_info["model_url"], "grounding-dino")

            # Load and cache GroundingDINO model
            dino_key = (config_path, weights_path, device_obj)
            if dino_key not in self.dino_model_cache:
                args = SLConfig.fromfile(config_path)
                model = build_model(args)
                checkpoint = load_file(weights_path)
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    checkpoint = clean_state_dict(checkpoint['model'])
                model.load_state_dict(checkpoint, strict=False)
                model.eval()
                model.to(device_obj)
                self.dino_model_cache[dino_key] = model
            dino = self.dino_model_cache[dino_key]

            # Load SAM2 model
            predictor = self.load_sam2(sam2_model, device)

            # Preprocess image for DINO
            transform = Compose([
                RandomResize([800], max_size=1333),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_tensor, _ = transform(img_pil.convert("RGB"), None)
            image_tensor = image_tensor.unsqueeze(0).to(device_obj)

            # Prepare text prompt
            text_prompt = prompt if prompt.endswith(".") else prompt + "."

            # Forward pass
            with torch.no_grad():
                outputs = dino(image_tensor, captions=[text_prompt])
            logits = outputs["pred_logits"].sigmoid()[0]
            boxes = outputs["pred_boxes"][0]

            # Filter boxes by threshold
            filt_mask = logits.max(dim=1)[0] > threshold
            boxes_filt = boxes[filt_mask]
            
            # Handle case with no detected boxes
            if boxes_filt.shape[0] == 0:
                width, height = img_pil.size
                empty_mask = torch.zeros((1, height, width), dtype=torch.float32, device="cpu")
                empty_mask_rgb = empty_mask.reshape((-1, 1, height, width)).movedim(1, -1).expand(-1, -1, -1, 3)
                result_image = apply_background_color(img_pil, Image.fromarray((empty_mask[0].numpy() * 255).astype(np.uint8)), background, background_color)
                result_images.append(pil2tensor(result_image))
                result_masks.append(empty_mask)
                result_mask_images.append(empty_mask_rgb)
                continue

            # Convert boxes to xyxy
            H, W = img_pil.size[1], img_pil.size[0]
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes_filt)
            boxes_xyxy = boxes_xyxy * torch.tensor([W, H, W, H], dtype=torch.float32, device=boxes_xyxy.device)
            boxes_xyxy = boxes_xyxy.cpu().numpy()

            # Set image and predict with autocast for precision handling
            from contextlib import nullcontext
            # Determine precision based on device preference
            if device == "Auto":
                precision = "fp16" if hasattr(device_obj, 'type') and device_obj.type == 'cuda' else "fp32"
            elif device == "GPU":
                precision = "fp16" if hasattr(device_obj, 'type') and device_obj.type == 'cuda' else "fp32"
            else:  # CPU
                precision = "fp32"
                
            autocast_condition = not comfy.model_management.is_device_mps(device_obj)
            with torch.autocast(comfy.model_management.get_autocast_device(device_obj), dtype=torch.float16 if precision == "fp16" else torch.float32) if autocast_condition else nullcontext():
                predictor.set_image(img_pil)
                
                # Process each box individually and combine masks
                all_masks = []
                for box in boxes_xyxy:
                    with torch.no_grad():
                        masks, iou_predictions, low_res_masks = predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=box,
                            multimask_output=False
                        )
                    all_masks.append(masks)
            
            # Combine all masks into one
            if len(all_masks) == 1:
                mask = all_masks[0]
            else:
                combined_mask = np.zeros_like(all_masks[0])
                for single_mask in all_masks:
                    combined_mask = np.maximum(combined_mask, single_mask)
                mask = combined_mask
            
            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            mask = (mask * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask, mode="L")

            # Process mask and apply background
            mask_image = process_mask(mask_pil, invert_output, mask_blur, mask_offset)
            result_image = apply_background_color(img_pil, mask_image, background, background_color)
            if background == "Color":
                result_image = result_image.convert("RGB")
            else:
                result_image = result_image.convert("RGBA")
            
            # Convert to tensors
            mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
            mask_image_vis = mask_tensor.reshape((-1, 1, mask_image.height, mask_image.width)).movedim(1, -1).expand(-1, -1, -1, 3)
            
            result_images.append(pil2tensor(result_image))
            result_masks.append(mask_tensor)
            result_mask_images.append(mask_image_vis)

        # If no images were successfully processed, return empty results
        if len(result_images) == 0:
            width, height = tensor2pil(image[0]).size
            empty_mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32, device="cpu")
            empty_mask_rgb = empty_mask.reshape((-1, 1, height, width)).movedim(1, -1).expand(-1, -1, -1, 3)
            return (image, empty_mask, empty_mask_rgb)
            
        # Combine all batch results
        return (torch.cat(result_images, dim=0), 
                torch.cat(result_masks, dim=0), 
                torch.cat(result_mask_images, dim=0))

NODE_CLASS_MAPPINGS = {
    "SAM2Segment": SAM2Segment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM2Segment": "SAM2 Segmentation (RMBG)",
}