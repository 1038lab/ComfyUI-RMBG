# ComfyUI-RMBG
# This custom node for ComfyUI provides functionality for Object removal using Big-Lama model.
#
# reference from https://github.com/advimman/lama
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/AILab-AI/ComfyUI-RMBG


import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import folder_paths
from comfy.model_management import get_torch_device
from torchvision import transforms
from huggingface_hub import hf_hub_download
import shutil
import gc

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def pil2comfy(image):
    img_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def pad_image(image, is_mask=False):
    w, h = image.size
    if w % 8 != 0:
        w = w + (8 - w % 8)
    if h % 8 != 0:
        h = h + (8 - h % 8)
    
    fill_color = 0 if is_mask else None
    padded = Image.new(image.mode, (w, h), color=fill_color)
    padded.paste(image, (0, 0))
    return padded

def cropimage(image, w, h):
    return image.crop((0, 0, w, h))

DEVICE = get_torch_device()
folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

class AILab_LamaRemover:
    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "images": "Input images to be processed",
            "masks": "Masks defining areas to be removed (white=remove)",
            "removal_strength": "Strength of the removal effect (higher values increase the effect area)",
            "edge_smoothness": "Controls edge smoothness (higher values create smoother transitions)"
        }
        
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": tooltips["images"]}),
                "masks": ("MASK", {"tooltip": tooltips["masks"]}),
                "removal_strength": ("INT", {"default": 230, "min": 0, "max": 255, "step": 1, "display": "slider", "tooltip": tooltips["removal_strength"]}),
                "edge_smoothness": ("INT", {"default": 8, "min": 0, "max": 20, "step": 1, "display": "slider", "tooltip": tooltips["edge_smoothness"]}),
            },
        }
    
    CATEGORY = "🧪AILab/🧽RMBG"
    RETURN_NAMES = ("images",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_object"
    
    def __init__(self):
        self.model = None
        self.device = DEVICE
        self.cache_dir = os.path.join(folder_paths.models_dir, "RMBG", "Lama")
        self.model_path = os.path.join(self.cache_dir, "big-lama.pt")
        self.to_pil = transforms.ToPILImage()
    
    def load_model(self):
        if self.model is not None:
            return
            
        if not os.path.exists(self.model_path):
            self.download_model()
        
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
        except Exception as e:
            print(f"Can't use comfy device: {str(e)}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = torch.jit.load(self.model_path, map_location=self.device)
        
        self.model.eval()
        self.model.to(self.device)
    
    def download_model(self):
        print("Downloading Big-Lama model...")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        try:
            downloaded_path = hf_hub_download(
                repo_id="1038lab/Lama",
                filename="big-lama.pt",
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False
            )
            
            if os.path.dirname(downloaded_path) != self.cache_dir:
                shutil.move(downloaded_path, self.model_path)
            
            print("Big-Lama model downloaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error downloading Big-Lama model: {str(e)}")

    def process_with_model(self, img_tensor, mask_tensor):
        with torch.inference_mode():
            img_tensor = img_tensor.to(self.device)
            mask_tensor = mask_tensor.to(self.device)
            
            result = self.model(img_tensor, mask_tensor)
            result_cpu = result[0].cpu()
            
            del img_tensor
            del mask_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return result_cpu

    def remove_object(self, images, masks, removal_strength, edge_smoothness):
        try:
            self.load_model()
            results = []
            
            for image, mask in zip(images, masks):
                ori_image = tensor2pil(image)
                w, h = ori_image.size
                p_image = pad_image(ori_image)
                
                mask_np = mask.cpu().numpy()
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                p_mask = pad_image(mask_pil, is_mask=True)
                
                if p_mask.size != p_image.size:
                    try:
                        p_mask = p_mask.resize(p_image.size, Image.LANCZOS)
                    except AttributeError:
                        p_mask = p_mask.resize(p_image.size, Image.ANTIALIAS)
                
                p_mask = ImageOps.invert(p_mask)
                p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=edge_smoothness))
                gray = p_mask.point(lambda x: 0 if x > removal_strength else 255)
                
                img_tensor = torch.FloatTensor(np.array(p_image)).permute(2, 0, 1).unsqueeze(0) / 255.0
                mask_tensor = torch.FloatTensor(np.array(gray)).unsqueeze(0).unsqueeze(0) / 255.0
                
                result = self.process_with_model(img_tensor, mask_tensor)
                result_img = self.to_pil(result.squeeze())
                
                if result_img.width > w or result_img.height > h:
                    result_img = cropimage(result_img, w, h)
                
                result_tensor = pil2comfy(result_img)
                results.append(result_tensor)
                
                del result
                gc.collect()
            
            return (torch.cat(results, dim=0),)
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise RuntimeError(f"Error in object removal: {str(e)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

NODE_CLASS_MAPPINGS = {
    "AILab_LamaRemover": AILab_LamaRemover,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_LamaRemover": "Lama Remover (RMBG)",
}