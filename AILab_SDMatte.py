# ComfyUI-RMBG
# This custom node for ComfyUI provides functionality for Object removal using SDMatte model.
#
# reference from https://github.com/vivoCameraResearch/SDMatte
# model: https://huggingface.co/1038lab/SDMatte
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/AILab-AI/ComfyUI-RMBG


import os
import sys
import copy

if os.environ.get('SDMATTE_CPU_ONLY', '').lower() in ('1', 'true', 'yes'):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from torch.hub import download_url_to_file
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import folder_paths

try:
    import comfy.model_management
    COMFY_AVAILABLE = True
except Exception as e:
    print(f"Warning: ComfyUI model management not available: {e}")
    COMFY_AVAILABLE = False
    class MockModelManagement:
        @staticmethod
        def get_torch_device():
            return torch.device('cpu')

    class MockComfy:
        model_management = MockModelManagement()

    comfy = MockComfy()

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Will use torch.load for model loading.")

try:
    import diffusers
    import transformers
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers/transformers not available. SDMatte functionality will be limited.")

current_dir = os.path.dirname(__file__)
if str(current_dir) not in sys.path:
    sys.path.insert(0, current_dir)

SDMATTE_MODELS = {
    "SDMatte": {
        "model_url": "https://huggingface.co/1038lab/SDMatte/resolve/main/SDMatte.safetensors",
        "filename": "SDMatte.safetensors",
        "repo_id": "1038lab/SDMatte"
    },
    "SDMatte_plus": {
        "model_url": "https://huggingface.co/1038lab/SDMatte/resolve/main/SDMatte_plus.safetensors",
        "filename": "SDMatte_plus.safetensors",
        "repo_id": "1038lab/SDMatte"
    }
}

REQUIRED_COMPONENTS = ["scheduler", "text_encoder", "tokenizer", "unet", "vae"]

def get_or_download_model_file(filename, url, dirname):
    local_path = folder_paths.get_full_path(dirname, filename)
    if local_path:
        return local_path
    folder = os.path.join(folder_paths.models_dir, dirname)
    os.makedirs(folder, exist_ok=True)
    local_path = os.path.join(folder, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from {url} ...")
        try:
            download_url_to_file(url, local_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download {filename} from {url}: {e}")
    return local_path

def ensure_model_components(model_name):
    model_info = SDMATTE_MODELS[model_name]
    repo_id = model_info["repo_id"]

    components_dir = os.path.join(folder_paths.models_dir, "RMBG", "SDMatte")

    missing_components = []
    for component in REQUIRED_COMPONENTS:
        component_path = os.path.join(components_dir, component)
        if not os.path.exists(component_path) or not os.listdir(component_path):
            missing_components.append(component)

    if missing_components:
        print(f"Downloading missing SDMatte components: {missing_components}")
        base_url = f"https://huggingface.co/{repo_id}/resolve/main"

        for component in missing_components:
            component_dir = os.path.join(components_dir, component)
            os.makedirs(component_dir, exist_ok=True)

            if component == "scheduler":
                files = ["scheduler_config.json"]
            elif component == "text_encoder":
                files = ["config.json"]
            elif component == "tokenizer":
                files = ["merges.txt", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"]
            elif component == "unet":
                files = ["config.json"]
            elif component == "vae":
                files = ["config.json"]

            for file in files:
                file_url = f"{base_url}/{component}/{file}"
                file_path = os.path.join(component_dir, file)
                if not os.path.exists(file_path):
                    try:
                        print(f"  Downloading {component}/{file}...")
                        download_url_to_file(file_url, file_path)
                    except Exception as e:
                        print(f"  Warning: Failed to download {file}: {e}")

    return components_dir

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

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

SDMatteCore = None

def _resize_norm_image_bchw(image_bchw: torch.Tensor, size_hw=(1024, 1024)) -> torch.Tensor:
    if image_bchw.shape[1] == 4:
        image_bchw = image_bchw[:, :3, :, :]

    resize = transforms.Resize(size_hw, interpolation=InterpolationMode.BILINEAR, antialias=True)
    norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    x = resize(image_bchw)
    x = norm(x)
    return x

def _resize_mask_b1hw(mask_b1hw: torch.Tensor, size_hw=(1024, 1024)) -> torch.Tensor:
    resize = transforms.Resize(size_hw, interpolation=InterpolationMode.BILINEAR, antialias=True)
    return resize(mask_b1hw)

class AILab_SDMatte:
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "model": "SDMatte model variant: Standard or Plus version",
            "image": "Input image for matting extraction",
            "mask": "Mask: White=foreground, Black=background. If omitted and image has alpha, alpha will be used.",
            "process_res": "Processing resolution: higher = better quality but slower",
            "device": "Auto: smart detection, CPU: force CPU, GPU: force GPU",
            "transparent_object": "Whether input image contains transparent objects",
            "mask_refine": "Enable mask refinement using mask constraints",
            "sensitivity": "Sensitivity for mask constraint (0.1-1.0): higher = more strict",
            "mask_blur": "Blur mask edges (0 = disabled)",
            "mask_offset": "Expand/shrink mask (positive = expand)",
            "invert_output": "Invert the mask output",
            "background": "Background type for output",
            "background_color": "Background color (when not Alpha)",
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (list(SDMATTE_MODELS.keys()), {"default": "SDMatte", "tooltip": tooltips["model"]}),
                "device": (["Auto", "CPU", "GPU"], {"default": "Auto", "tooltip": tooltips["device"]}),
                "process_res": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8, "tooltip": tooltips["process_res"]}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": tooltips["mask"]}),
                "transparent_object": ("BOOLEAN", {"default": True, "tooltip": tooltips["transparent_object"]}),
                "mask_refine": ("BOOLEAN", {"default": True, "tooltip": tooltips["mask_refine"]}),
                "sensitivity": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.1, "tooltip": tooltips["sensitivity"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "background_color": ("COLORCODE", {"default": "#222222", "tooltip": tooltips["background_color"]}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "matting_inference"
    CATEGORY = "🧪AILab/🧽RMBG"

    def __init__(self):
        self.model_cache = {}

    def load_sdmatte_model(self, model_name, device="Auto"):
        cache_key = f"{model_name}_{device}"
        
        current_model_keys = [k for k in self.model_cache.keys() if k.startswith(model_name)]
        if cache_key not in self.model_cache and len(self.model_cache) > 0:
            for key in list(self.model_cache.keys()):
                if key not in current_model_keys:
                    del self.model_cache[key]
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if cache_key not in self.model_cache:
            if not DIFFUSERS_AVAILABLE:
                raise ImportError("diffusers and transformers are required for SDMatte functionality")
        
            global SDMatteCore
            if SDMatteCore is None:
                import sys
                import os
                current_dir = os.path.dirname(__file__)
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                from SDMatte.modeling.SDMatte.meta_arch import SDMatte as SDMatteCore
        
            model_info = SDMATTE_MODELS[model_name]
        
            model_path = get_or_download_model_file(
                model_info["filename"],
                model_info["model_url"],
                "RMBG/SDMatte"
            )
        
            pretrained_repo = ensure_model_components(model_name)
        
            sdmatte_model = SDMatteCore(
                pretrained_model_name_or_path=pretrained_repo,
                load_weight=False,
                use_aux_input=True,
                aux_input="trimap",
                use_encoder_hidden_states=True,
                use_attention_mask=True,
                add_noise=False,
            )
        
            self._load_model_weights(sdmatte_model, model_path)
        
            device_obj = comfy.model_management.get_torch_device()
            if device == "CPU":
                device_obj = torch.device('cpu')
            elif device == "GPU":
                if not torch.cuda.is_available():
                    print("SDMatte: GPU requested but CUDA not available, falling back to CPU")
                    device_obj = torch.device('cpu')
                else:
                    device_obj = comfy.model_management.get_torch_device()
        
            sdmatte_model.eval()
            sdmatte_model.to(device_obj)
        
            if device_obj.type == 'cuda':
                self._apply_memory_optimizations(sdmatte_model)
        
            self.model_cache[cache_key] = sdmatte_model
        
        return self.model_cache[cache_key]

    def _load_model_weights(self, model, model_path):
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required for SDMatte functionality")

        try:
            state_dict = load_file(model_path)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            if os.path.exists(model_path):
                print(f"[SDMatte] Model file appears corrupted, deleting: {model_path}")
                os.remove(model_path)
            raise RuntimeError(f"Failed to load model weights. File may be corrupted. Please try again to re-download. Error: {e}")

    def _apply_memory_optimizations(self, model):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            unet = getattr(model, 'unet', None)
            if unet is not None and hasattr(unet, 'set_attn_processor'):
                from diffusers.models.attention_processor import SlicedAttnProcessor
                unet.set_attn_processor(SlicedAttnProcessor(slice_size=1))
        except Exception:
            pass

    def matting_inference(self, image, model, process_res, device="Auto",
                   mask=None, transparent_object=True, mask_refine=True,
                   sensitivity=0.8, mask_blur=0, mask_offset=0,
                   invert_output=False, background="Alpha", background_color="#222222"):
        sdmatte_model = self.load_sdmatte_model(model, device)
        device_obj = comfy.model_management.get_torch_device()
        if device == "CPU":
            device_obj = torch.device('cpu')
        
        batch_size = image.shape[0]
        
        result_masks = []
        result_images = []
        result_mask_images = []
        
        for b in range(batch_size):
            img_pil = tensor2pil(image[b])
            B, H, W = 1, img_pil.height, img_pil.width
            orig_h, orig_w = H, W
        
            img_bchw = image[b:b+1].permute(0, 3, 1, 2).contiguous().to(device_obj)
            img_in = _resize_norm_image_bchw(img_bchw, (int(process_res), int(process_res)))
        
            if mask is not None:
                mask_b1hw = mask[b:b+1].unsqueeze(1).contiguous().to(device_obj)
                mask_for_refine = mask[b:b+1]
            else:
                if image.shape[-1] == 4:
                    alpha = image[b, :, :, 3]
                    mask_b1hw = alpha.unsqueeze(0).unsqueeze(0).contiguous().to(device_obj)
                    mask_for_refine = alpha.unsqueeze(0)
                else:
                    raise ValueError("Mask required: provide a mask or use an image with alpha.")
            
            tri = _resize_mask_b1hw(mask_b1hw, (int(process_res), int(process_res))) * 2 - 1
            data = {"image": img_in,
                    "is_trans": torch.tensor([1 if transparent_object else 0], device=device_obj),
                    "caption": [""],
                    "trimap": tri,
                    "trimap_coords": torch.tensor([[0,0,1,1]], dtype=tri.dtype, device=device_obj)}
        
            with torch.inference_mode():
                if device_obj.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pred_alpha = sdmatte_model(data)
                else:
                    pred_alpha = sdmatte_model(data)
        
            out = transforms.Resize((orig_h, orig_w), interpolation=InterpolationMode.BILINEAR, antialias=True)(pred_alpha)
            out = out.squeeze(1).clamp(0, 1).detach().cpu()
        
            if mask_refine:
                out = self._refine_mask(out, mask_for_refine, sensitivity)
        
            mask_pil = Image.fromarray((out[0].numpy() * 255).astype(np.uint8), mode="L")
        
            mask_image = process_mask(mask_pil, invert_output, mask_blur, mask_offset)
        
            result_image = apply_background_color(img_pil, mask_image, background, background_color)
            if background == "Color":
                result_image = result_image.convert("RGB")
            else:
                result_image = result_image.convert("RGBA")
        
            mask_tensor = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
            mask_image_vis = mask_tensor.reshape((-1, 1, mask_image.height, mask_image.width)).movedim(1, -1).expand(-1, -1, -1, 3)
        
            result_masks.append(mask_tensor)
            result_images.append(pil2tensor(result_image))
            result_mask_images.append(mask_image_vis)
        
        if device_obj.type == 'cuda':
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        return (torch.cat(result_images, dim=0), torch.cat(result_masks, dim=0), torch.cat(result_mask_images, dim=0))

    def _refine_mask(self, mask, trimap, constraint):
        trimap_cpu = trimap.cpu()
        foreground_regions = trimap_cpu > constraint
        background_regions = trimap_cpu < (1.0 - constraint)
        unknown_regions = ~(foreground_regions | background_regions)

        refined_mask = mask.clone()
        refined_mask[background_regions] = 0.0
        refined_mask[foreground_regions] = torch.clamp(refined_mask[foreground_regions] * 1.2, 0, 1)

        alpha_threshold = 0.3
        low_confidence = (refined_mask < alpha_threshold) & unknown_regions
        refined_mask[low_confidence] = 0.0

        return refined_mask


NODE_CLASS_MAPPINGS = {
    "AILab_SDMatte": AILab_SDMatte,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_SDMatte": "SDMatte Matting (RMBG)",
}