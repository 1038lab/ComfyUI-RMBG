# ComfyUI-RMBG
# This custom node for ComfyUI provides functionality for background removal using various models,
# including RMBG-2.0, INSPYRENET, BEN, BEN2 and BIREFNET-HR. It leverages deep learning techniques
# to process images and generate masks for background removal.
#
# Models License Notice:
# - RMBG-2.0: Apache-2.0 License (https://huggingface.co/briaai/RMBG-2.0)
# - INSPYRENET: MIT License (https://github.com/plemeri/InSPyReNet)
# - BEN: Apache-2.0 License (https://huggingface.co/PramaLLC/BEN)
# - BEN2: Apache-2.0 License (https://huggingface.co/PramaLLC/BEN2)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-RMBG

import os
import platform
import subprocess
import time
import uuid
import shutil
import torch
from PIL import Image
import numpy as np
import folder_paths
from PIL import ImageFilter
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import sys
import importlib.util
from transformers import AutoModelForImageSegmentation
import cv2
import types

device = "cuda" if torch.cuda.is_available() else "cpu"

folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))


def _rmbg_progress_path():
    try:
        user_dir = folder_paths.get_user_directory()
        os.makedirs(user_dir, exist_ok=True)
        return os.path.join(user_dir, "rmbg_progress.log")
    except Exception:
        return os.path.join(os.getcwd(), "rmbg_progress.log")


def _rmbg_progress(msg: str):
    """
    Crash-resilient marker logging to help locate native crashes.
    Enabled only when COMFYUI_RMBG_DEBUG_PROGRESS=1.
    """
    if not os.environ.get("COMFYUI_RMBG_DEBUG_PROGRESS"):
        return
    try:
        path = _rmbg_progress_path()
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] pid={os.getpid()} {msg}\r\n".encode("utf-8", errors="replace")
        fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o666)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception:
        pass


def _run_rmbg2_subprocess(images, model_name: str, params: dict):
    """
    Run RMBG-2.0 inference in a separate process to avoid native crashes taking down ComfyUI.
    Returns a list of PIL 'L' masks.
    """
    tmp_root = folder_paths.get_temp_directory()
    run_id = f"rmbg2_{os.getpid()}_{uuid.uuid4().hex}"
    run_dir = os.path.join(tmp_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    input_paths = []
    output_paths = []
    for i, img in enumerate(images):
        in_path = os.path.join(run_dir, f"in_{i}.png")
        out_path = os.path.join(run_dir, f"out_{i}.png")
        tensor2pil(img).convert("RGB").save(in_path)
        input_paths.append(in_path)
        output_paths.append(out_path)

    node_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    worker_path = os.path.join(os.path.dirname(__file__), "rmbg_worker.py")

    cache_dir = os.path.join(folder_paths.models_dir, "RMBG", AVAILABLE_MODELS[model_name]["cache_dir"])

    cmd = [
        sys.executable,
        "-s",
        worker_path,
        "--node-root",
        node_root,
        "--model",
        model_name,
        "--cache-dir",
        cache_dir,
        "--process-res",
        str(int(params.get("process_res", 1024))),
        "--sensitivity",
        str(float(params.get("sensitivity", 1.0))),
        "--inputs",
        *input_paths,
        "--outputs",
        *output_paths,
    ]

    debug_keep = bool(os.environ.get("COMFYUI_RMBG_DEBUG_KEEP_TEMP"))
    try:
        _rmbg_progress(f"RMBG-2.0 subprocess spawn: {worker_path} inputs={len(input_paths)}")
        comfy_root = os.path.dirname(folder_paths.__file__)
        result = subprocess.run(cmd, cwd=comfy_root, capture_output=True, text=True)
        _rmbg_progress(f"RMBG-2.0 subprocess exit={result.returncode}")

        if result.returncode != 0:
            stderr_tail = (result.stderr or "").strip().splitlines()[-40:]
            stdout_tail = (result.stdout or "").strip().splitlines()[-40:]
            _rmbg_progress(f"RMBG-2.0 subprocess stdout_tail={stdout_tail}")
            _rmbg_progress(f"RMBG-2.0 subprocess stderr_tail={stderr_tail}")

            print("[RMBG ERROR] RMBG-2.0 subprocess failed. Last output:")
            if stdout_tail:
                print("\n".join(stdout_tail))
            if stderr_tail:
                print("\n".join(stderr_tail))
            raise RuntimeError("RMBG-2.0 subprocess failed")

        masks = []
        for out_path in output_paths:
            mask = Image.open(out_path).convert("L")
            masks.append(mask)
        return masks
    finally:
        if not debug_keep:
            shutil.rmtree(run_dir, ignore_errors=True)

# Model configuration
AVAILABLE_MODELS = {
    "RMBG-2.0": {
        "type": "rmbg",
        "repo_id": "1038lab/RMBG-2.0",
        "files": {
            "config.json": "config.json",
            "model.safetensors": "model.safetensors",
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py"
        },
        "cache_dir": "RMBG-2.0"
    },
    "INSPYRENET": {
        "type": "inspyrenet",
        "repo_id": "1038lab/inspyrenet",
        "files": {
            "inspyrenet.safetensors": "inspyrenet.safetensors"
        },
        "cache_dir": "INSPYRENET"
    },
    "BEN": {
        "type": "ben",
        "repo_id": "1038lab/BEN",
        "files": {
            "model.py": "model.py",
            "BEN_Base.pth": "BEN_Base.pth"
        },
        "cache_dir": "BEN"
    },
    "BEN2": {
        "type": "ben2",
        "repo_id": "1038lab/BEN2",
        "files": {
            "BEN2_Base.pth": "BEN2_Base.pth",
            "BEN2.py": "BEN2.py"
        },
        "cache_dir": "BEN2"
    }
}

# Utility functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def handle_model_error(message):
    print(f"[RMBG ERROR] {message}")
    raise RuntimeError(message)

class BaseModelLoader:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.base_cache_dir = os.path.join(folder_paths.models_dir, "RMBG")
    
    def get_cache_dir(self, model_name):
        cache_path = os.path.join(self.base_cache_dir, AVAILABLE_MODELS[model_name]["cache_dir"])
        os.makedirs(cache_path, exist_ok=True)
        return cache_path
    
    def check_model_cache(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        
        if not os.path.exists(cache_dir):
            return False, "Model directory not found"
        
        missing_files = []
        for filename in model_info["files"].keys():
            if not os.path.exists(os.path.join(cache_dir, model_info["files"][filename])):
                missing_files.append(filename)
        
        if missing_files:
            return False, f"Missing model files: {', '.join(missing_files)}"
            
        return True, "Model cache verified"
    
    def download_model(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Downloading {model_name} model files...")
            
            for filename in model_info["files"].keys():
                print(f"Downloading {filename}...")
                hf_hub_download(
                    repo_id=model_info["repo_id"],
                    filename=filename,
                    local_dir=cache_dir
                )
                    
            return True, "Model files downloaded successfully"
            
        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"
    
    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model = None
        self.current_model_version = None

class RMBGModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()

            cache_dir = self.get_cache_dir(model_name)
            try:
                # Primary path: Modern transformers compatibility mode (optimized for newer versions)
                try:
                    from transformers import PreTrainedModel
                    import json

                    config_path = os.path.join(cache_dir, "config.json")
                    with open(config_path, 'r') as f:
                        config = json.load(f)

                    birefnet_path = os.path.join(cache_dir, "birefnet.py")
                    BiRefNetConfig_path = os.path.join(cache_dir, "BiRefNet_config.py")

                    # Load the BiRefNetConfig
                    config_spec = importlib.util.spec_from_file_location("BiRefNetConfig", BiRefNetConfig_path)
                    config_module = importlib.util.module_from_spec(config_spec)
                    sys.modules["BiRefNetConfig"] = config_module
                    config_spec.loader.exec_module(config_module)

                    # Fix and load birefnet module
                    with open(birefnet_path, 'r') as f:
                        birefnet_content = f.read()

                    birefnet_content = birefnet_content.replace(
                        "from .BiRefNet_config import BiRefNetConfig",
                        "from BiRefNetConfig import BiRefNetConfig"
                    )

                    module_name = f"custom_birefnet_model_{hash(birefnet_path)}"
                    module = types.ModuleType(module_name)
                    sys.modules[module_name] = module
                    exec(birefnet_content, module.__dict__)

                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, PreTrainedModel) and attr != PreTrainedModel:
                            BiRefNetConfig = getattr(config_module, "BiRefNetConfig")
                            model_config = BiRefNetConfig()
                            self.model = attr(model_config)

                            weights_path = os.path.join(cache_dir, "model.safetensors")
                            try:
                                try:
                                    import safetensors.torch
                                    self.model.load_state_dict(safetensors.torch.load_file(weights_path))
                                except ImportError:
                                    from transformers.modeling_utils import load_state_dict
                                    state_dict = load_state_dict(weights_path)
                                    self.model.load_state_dict(state_dict)
                            except Exception as load_error:
                                pytorch_weights = os.path.join(cache_dir, "pytorch_model.bin")
                                if os.path.exists(pytorch_weights):
                                    self.model.load_state_dict(torch.load(pytorch_weights, map_location="cpu"))
                                else:
                                    raise RuntimeError(f"Failed to load weights: {str(load_error)}")
                            break

                    if self.model is None:
                        raise RuntimeError("Could not find suitable model class")

                except Exception as modern_e:
                    print(f"[RMBG INFO] Using standard transformers loading (fallback mode)...")
                    try:
                        self.model = AutoModelForImageSegmentation.from_pretrained(
                            cache_dir,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                    except Exception as standard_e:
                        handle_model_error(f"Failed to load model with both modern and standard methods. Modern error: {str(modern_e)}. Standard error: {str(standard_e)}")

            except Exception as e:
                handle_model_error(f"Error loading model: {str(e)}")

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            torch.set_float32_matmul_precision('high')
            self.model.to(device)
            self.current_model_version = model_name
            
    def process_image(self, images, model_name, params):
        try:
            self.load_model(model_name)

            if isinstance(images, torch.Tensor):
                if len(images.shape) == 3:
                    images = [images]
                else:
                    images = [img for img in images]

            # Avoid torchvision CPU transforms on Windows (can crash in torch_cpu.dll on some setups).
            # ComfyUI IMAGE tensors are HWC in [0, 1] on CPU; move to GPU and do resize/normalize there.
            original_sizes = [(int(img.shape[1]), int(img.shape[0])) for img in images]

            input_tensors = []
            for img in images:
                if img.ndim != 3:
                    handle_model_error(f"Unexpected image tensor shape: {tuple(img.shape)}")
                chw = img.permute(2, 0, 1)
                if chw.shape[0] > 3:
                    chw = chw[:3, :, :]
                input_tensors.append(chw)

            input_batch = torch.stack(input_tensors, dim=0).to(device=device, dtype=torch.float32, non_blocking=True)
            input_batch = F.interpolate(
                input_batch,
                size=(params["process_res"], params["process_res"]),
                mode="bilinear",
                align_corners=False,
            )
            mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=input_batch.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=input_batch.dtype).view(1, 3, 1, 1)
            input_batch = (input_batch - mean) / std

            with torch.no_grad():
                outputs = self.model(input_batch)
                
                results = None
                if isinstance(outputs, list) and len(outputs) > 0:
                    results = outputs[-1]
                elif isinstance(outputs, dict) and "logits" in outputs:
                    results = outputs["logits"]
                elif isinstance(outputs, torch.Tensor):
                    results = outputs
                else:
                    try:
                        if hasattr(outputs, "last_hidden_state"):
                            results = outputs.last_hidden_state
                        else:
                            for _, v in outputs.items():
                                if isinstance(v, torch.Tensor):
                                    results = v
                                    break
                    except Exception:
                        results = None

                if results is None:
                    handle_model_error("Unable to recognize model output format")

                results = results.sigmoid()
                
                masks = []
                
                for result, (orig_w, orig_h) in zip(results, original_sizes):
                    mask = result.squeeze()
                    mask = mask * (1 + (1 - params["sensitivity"]))
                    mask = torch.clamp(mask, 0, 1)

                    mask = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()

                    masks.append(tensor2pil(mask.detach().float().cpu()))

                return masks

        except Exception as e:
            handle_model_error(f"Error in batch processing: {str(e)}")

class InspyrenetModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            try:
                import transparent_background
                self.model = transparent_background.Remover()
                self.current_model_version = model_name
            except Exception as e:
                handle_model_error(f"Failed to initialize transparent_background: {str(e)}")
    
    def process_image(self, image, model_name, params):
        try:
            self.load_model(model_name)
            
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            
            aspect_ratio = h / w
            new_w = params["process_res"]
            new_h = int(params["process_res"] * aspect_ratio)
            resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
            
            foreground = self.model.process(resized_image, type='rgba')
            foreground = foreground.resize((w, h), Image.LANCZOS)
            mask = foreground.split()[-1]
            
            return mask
            
        except Exception as e:
            handle_model_error(f"Error in Inspyrenet processing: {str(e)}")

class BENModel(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            cache_dir = self.get_cache_dir(model_name)
            model_path = os.path.join(cache_dir, "model.py")
            module_name = f"custom_ben_model_{hash(model_path)}"
            
            spec = importlib.util.spec_from_file_location(module_name, model_path)
            ben_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = ben_module
            spec.loader.exec_module(ben_module)
            
            model_weights_path = os.path.join(cache_dir, "BEN_Base.pth")
            self.model = ben_module.BEN_Base()
            self.model.loadcheckpoints(model_weights_path)
            
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            
            torch.set_float32_matmul_precision('high')
            self.model.to(device)
            self.current_model_version = model_name
    
    def process_image(self, image, model_name, params):
        try:
            self.load_model(model_name)
            
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            
            aspect_ratio = h / w
            new_w = params["process_res"]
            new_h = int(params["process_res"] * aspect_ratio)
            resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
            
            processed_input = resized_image.convert("RGBA")
            
            with torch.no_grad():
                _, foreground = self.model.inference(processed_input)
            
            foreground = foreground.resize((w, h), Image.LANCZOS)
            mask = foreground.split()[-1]
            
            return mask
            
        except Exception as e:
            handle_model_error(f"Error in BEN processing: {str(e)}")

class BEN2Model(BaseModelLoader):
    def __init__(self):
        super().__init__()
        
    def load_model(self, model_name):
        if self.current_model_version != model_name:
            self.clear_model()
            
            try:
                cache_dir = self.get_cache_dir(model_name)
                model_path = os.path.join(cache_dir, "BEN2.py")
                module_name = f"custom_ben2_model_{hash(model_path)}"
                
                spec = importlib.util.spec_from_file_location(module_name, model_path)
                ben2_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = ben2_module
                spec.loader.exec_module(ben2_module)
                
                model_weights_path = os.path.join(cache_dir, "BEN2_Base.pth")
                self.model = ben2_module.BEN_Base()
                self.model.loadcheckpoints(model_weights_path)
                
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                
                torch.set_float32_matmul_precision('high')
                self.model.to(device)
                self.current_model_version = model_name
                
            except Exception as e:
                handle_model_error(f"Error loading BEN2 model: {str(e)}")
    
    def process_image(self, images, model_name, params):
        try:
            self.load_model(model_name)
            
            if isinstance(images, torch.Tensor):
                if len(images.shape) == 3:
                    images = [images]
                else:
                    images = [img for img in images]
            
            batch_size = 3
            all_masks = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_pil_images = []
                original_sizes = []
                
                for img in batch_images:
                    orig_image = tensor2pil(img)
                    w, h = orig_image.size
                    original_sizes.append((w, h))
                    
                    aspect_ratio = h / w
                    new_w = params["process_res"]
                    new_h = int(params["process_res"] * aspect_ratio)
                    resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
                    processed_input = resized_image.convert("RGBA")
                    batch_pil_images.append(processed_input)
                
                with torch.no_grad():
                    try:
                        foregrounds = self.model.inference(batch_pil_images)
                        if not isinstance(foregrounds, list):
                            foregrounds = [foregrounds]
                    except Exception as e:
                        handle_model_error(f"Error in BEN2 inference: {str(e)}")
                
                for foreground, (orig_w, orig_h) in zip(foregrounds, original_sizes):
                    foreground = foreground.resize((orig_w, orig_h), Image.LANCZOS)
                    mask = foreground.split()[-1]
                    all_masks.append(mask)
            
            if len(all_masks) == 1:
                return all_masks[0]
            return all_masks

        except Exception as e:
            handle_model_error(f"Error in BEN2 processing: {str(e)}")

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

class RMBG:
    def __init__(self):
        self.models = {
            "RMBG-2.0": RMBGModel(),
            "INSPYRENET": InspyrenetModel(),
            "BEN": BENModel(),
            "BEN2": BEN2Model()
        }
    
    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "image": "Input image to be processed for background removal.",
            "model": "Select the background removal model to use (RMBG-2.0, INSPYRENET, BEN).",
            "sensitivity": "Adjust the strength of mask detection (higher values result in more aggressive detection).",
            "process_res": "Set the processing resolution (higher values require more VRAM and may increase processing time).",
            "mask_blur": "Specify the amount of blur to apply to the mask edges (0 for no blur, higher values for more blur).",
            "mask_offset": "Adjust the mask boundary (positive values expand the mask, negative values shrink it).",
            "background": "Choose output type: Alpha (transparent) or Color (custom background color).",
            "background_color": "Pick background color (supports alpha, use color picker).",
            "invert_output": "Enable to invert both the image and mask output (useful for certain effects).",
            "refine_foreground": "Use Fast Foreground Colour Estimation to optimize transparent background"
        }
        
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "model": (list(AVAILABLE_MODELS.keys()), {"tooltip": tooltips["model"]}),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tooltips["sensitivity"]}),
                "process_res": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8, "tooltip": tooltips["process_res"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
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
            _rmbg_progress(
                f"RMBG node start model={model} batch={getattr(image, 'shape', None)} "
                f"refine={params.get('refine_foreground', False)} res={params.get('process_res', None)}"
            )
            processed_images = []
            processed_masks = []
            
            model_instance = self.models[model]
            
            cache_status, message = model_instance.check_model_cache(model)
            if not cache_status:
                print(f"Cache check: {message}")
                print("Downloading required model files...")
                download_status, download_message = model_instance.download_model(model)
                if not download_status:
                    handle_model_error(download_message)
                print("Model files downloaded successfully")
            
            model_type = AVAILABLE_MODELS[model]["type"]
            
            def _process_pair(img, mask):
                if isinstance(mask, list):
                    masks = [m.convert("L") for m in mask if isinstance(m, Image.Image)]
                    mask_local = masks[0] if masks else None
                elif isinstance(mask, Image.Image):
                    mask_local = mask.convert("L")
                else:
                    mask_local = mask
                
                # Avoid CPU torch ops here (some Windows setups crash in torch_cpu.dll during clamp/interpolate).
                mask_arr = np.array(mask_local, dtype=np.float32) / 255.0
                mask_arr = mask_arr * (1 + (1 - params["sensitivity"]))
                mask_arr = np.clip(mask_arr, 0.0, 1.0)
                mask_img_local = Image.fromarray((mask_arr * 255.0).astype(np.uint8), mode="L")
                
                if params["mask_blur"] > 0:
                    mask_img_local = mask_img_local.filter(ImageFilter.GaussianBlur(radius=params["mask_blur"]))
                
                if params["mask_offset"] != 0:
                    if params["mask_offset"] > 0:
                        for _ in range(params["mask_offset"]):
                            mask_img_local = mask_img_local.filter(ImageFilter.MaxFilter(3))
                    else:
                        for _ in range(-params["mask_offset"]):
                            mask_img_local = mask_img_local.filter(ImageFilter.MinFilter(3))
                
                if params["invert_output"]:
                    mask_img_local = Image.fromarray(255 - np.array(mask_img_local))
                
                img_tensor_local = torch.from_numpy(np.array(tensor2pil(img))).permute(2, 0, 1).unsqueeze(0) / 255.0
                mask_tensor_b1hw = torch.from_numpy(np.array(mask_img_local)).unsqueeze(0).unsqueeze(0) / 255.0
                
                orig_image_local = tensor2pil(img)
                
                if params.get("refine_foreground", False):
                    refined_fg_local = refine_foreground(img_tensor_local, mask_tensor_b1hw)
                    refined_fg_local = tensor2pil(refined_fg_local[0].permute(1, 2, 0))
                    r, g, b = refined_fg_local.split()
                    foreground_local = Image.merge('RGBA', (r, g, b, mask_img_local))
                else:
                    orig_rgba_local = orig_image_local.convert("RGBA")
                    r, g, b, _ = orig_rgba_local.split()
                    foreground_local = Image.merge('RGBA', (r, g, b, mask_img_local))
                
                if params["background"] == "Color":
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
                    bg_image = Image.new('RGBA', orig_image_local.size, rgba)
                    composite_image = Image.alpha_composite(bg_image, foreground_local)
                    processed_images.append(pil2tensor(composite_image.convert("RGB")))
                else:
                    processed_images.append(pil2tensor(foreground_local))
                
                processed_masks.append(pil2tensor(mask_img_local))
            
            if model_type in ("rmbg", "ben2"):
                images_list = [img for img in image]
                chunk_size = 4
                for start in range(0, len(images_list), chunk_size):
                    batch_imgs = images_list[start:start + chunk_size]
                    _rmbg_progress(f"RMBG node calling model_instance.process_image type={model_type} chunk_start={start} chunk_len={len(batch_imgs)}")
                    use_subprocess = os.environ.get("COMFYUI_RMBG_RMBG2_SUBPROCESS", "1") != "0"
                    if use_subprocess and model == "RMBG-2.0" and platform.system() == "Windows":
                        masks = _run_rmbg2_subprocess(batch_imgs, model, params)
                    else:
                        masks = model_instance.process_image(batch_imgs, model, params)
                    _rmbg_progress("RMBG node returned from model_instance.process_image")
                    if isinstance(masks, Image.Image):
                        masks = [masks]
                    for img_item, mask_item in zip(batch_imgs, masks):
                        _process_pair(img_item, mask_item)
            else:
                for img in image:
                    mask = model_instance.process_image(img, model, params)
                    _process_pair(img, mask)
            
            mask_images = []
            for mask_tensor in processed_masks:
                mask_image = mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
                mask_images.append(mask_image)
            
            mask_image_output = torch.cat(mask_images, dim=0)
            _rmbg_progress(f"RMBG node finish processed_images={len(processed_images)} processed_masks={len(processed_masks)}")
            return (torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0), mask_image_output)
            
        except Exception as e:
            _rmbg_progress(f"RMBG node exception: {type(e).__name__}: {e}")
            handle_model_error(f"Error in image processing: {str(e)}")
            empty_mask = torch.zeros((image.shape[0], image.shape[2], image.shape[3]))
            empty_mask_image = empty_mask.reshape((-1, 1, empty_mask.shape[-2], empty_mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            return (image, empty_mask, empty_mask_image)

NODE_CLASS_MAPPINGS = {
    "RMBG": RMBG
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RMBG": "Remove Background (RMBG)"
} 
