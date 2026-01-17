import io
import os
import random
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageColor, ImageDraw

import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

MODEL_DIR = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(MODEL_DIR, exist_ok=True)
folder_paths.add_model_folder_path("LLM", MODEL_DIR)

TASK_CONFIGS = {
    "Polygon Mask (text prompt)": {
        "token": "<REFERRING_EXPRESSION_SEGMENTATION>",
        "mode": "polygon",
        "allows_text": True,
    },
    "Phrase Grounding (text boxes)": {
        "token": "<CAPTION_TO_PHRASE_GROUNDING>",
        "mode": "bbox",
        "allows_text": True,
    },
    "Region Proposals (boxes only)": {
        "token": "<REGION_PROPOSAL>",
        "mode": "bbox",
        "allows_text": False,
    },
}

TASK_CHOICES = tuple(TASK_CONFIGS.keys())

GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "num_beams": 3,
    "do_sample": True,
}

MODEL_CHOICES = (
    "microsoft/Florence-2-base",
    "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-large",
    "microsoft/Florence-2-large-ft",
    "thwri/CogFlorence-2.1-Large",
    "thwri/CogFlorence-2.2-Large",
)

COLOR_BANK = ["blue", "orange", "green", "purple", "pink", "cyan"]


def _fixed_get_imports(filename):
    try:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        if "flash_attn" in imports:
            imports.remove("flash_attn")
        return imports
    except Exception:
        return get_imports(filename)


class AILab_Florence2:
    CATEGORY = "🧪AILab/🧽RMBG"
    RETURN_TYPES = ("IMAGE", "MASK", "JSON")
    RETURN_NAMES = ("IMAGE", "MASK", "DATA")
    FUNCTION = "analyze"
    MODEL_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (
                    MODEL_CHOICES,
                    {
                        "default": MODEL_CHOICES[0],
                        "tooltip": "Base = stable, +ft = fine-tuned captions, CogFlorence = sharper phrase alignment.",
                    },
                ),
                "task": (
                    TASK_CHOICES,
                    {
                        "default": TASK_CHOICES[0],
                        "tooltip": "Polygon masks use prompts; phrase grounding/region proposals return boxes.",
                    },
                ),
                "precision": (
                    ("fp16", "bf16", "fp32"),
                    {"default": "fp16", "tooltip": "Lower precision saves VRAM; fp32 is safest if you hit NaNs."},
                ),
                "attention": (
                    ("flash_attention_2", "sdpa", "eager"),
                    {"default": "sdpa", "tooltip": "flash_attn2 needs PyTorch 2.1+; use eager if kernels fail."},
                ),
                "fill_mask": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "When true, bbox tasks also output filled mask tensors."},
                ),
            },
            "optional": {
                "output_mask_select": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated indices or labels (e.g. 0,2,person) to limit masks.",
                    },
                ),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Keep weights on the current device after execution."},
                ),
                "text_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Prompt: e.g. a person wearing red coat",
                        "tooltip": "Used for polygon masks or phrase grounding; ignored for region proposals.",
                    },
                ),
            },
        }

    def _ensure_weights(self, model_name: str) -> str:
        target = os.path.join(MODEL_DIR, model_name.split("/")[-1])
        if os.path.exists(target):
            return target
        from huggingface_hub import snapshot_download

        snapshot_download(model_name, local_dir=target, local_dir_use_symlinks=False)
        return target

    def _get_model(self, model_name: str, precision: str, attention: str) -> Dict[str, Any]:
        key = (model_name, precision, attention)
        if key in self.MODEL_CACHE:
            return self.MODEL_CACHE[key]

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        weights = self._ensure_weights(model_name)
        offload = mm.unet_offload_device()

        if transformers.__version__ < "4.51.0":
            with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
                model = AutoModelForCausalLM.from_pretrained(
                    weights,
                    attn_implementation=attention,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to(offload)
        else:
            from models.modeling_florence2 import Florence2ForConditionalGeneration

            model = Florence2ForConditionalGeneration.from_pretrained(
                weights,
                attn_implementation=attention,
                torch_dtype=dtype,
            ).to(offload)

        processor = AutoProcessor.from_pretrained(weights, trust_remote_code=True)
        bundle = {"model": model, "processor": processor, "dtype": dtype}
        self.MODEL_CACHE[key] = bundle
        return bundle

    @staticmethod
    def _prepare_prompt(task: str, text_prompt: str) -> str:
        config = TASK_CONFIGS[task]
        text = text_prompt.strip()
        if text and not config["allows_text"]:
            text = ""
        base = config["token"]
        return f"{base} {text}" if text else base

    def _draw_regions(
        self,
        image_pil: Image.Image,
        predictions: Dict[str, Any],
        fill_mask: bool,
        select_filter: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        width, height = image_pil.size
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(image_pil)

        mask_layer = Image.new("RGB", image_pil.size, (0, 0, 0)) if fill_mask else None
        mask_draw = ImageDraw.Draw(mask_layer) if mask_layer else None

        for index, (bbox, label) in enumerate(zip(predictions["bboxes"], predictions["labels"])):
            x0, y0, x1, y1 = bbox
            if y1 < y0:
                y0, y1 = y1, y0
            if x1 < x0:
                x0, x1 = x1, x0
            filter_hit = not select_filter or str(index) in select_filter or label in select_filter
            if fill_mask and filter_hit:
                mask_draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(
                x0,
                max(0, y0 - 12),
                f"{index}.{label}",
                color="white",
                fontsize=12,
                bbox=dict(facecolor=random.choice(COLOR_BANK), alpha=0.5),
            )

        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", pad_inches=0)
        buf.seek(0)
        annotated = Image.open(buf)
        plt.close(fig)

        annotated_tensor = TF.to_tensor(annotated)[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
        if mask_layer is not None:
            mask_tensor = TF.to_tensor(mask_layer).unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
            mask_tensor = mask_tensor.mean(dim=0, keepdim=True)[:, :, :, 0]
        else:
            mask_tensor = torch.zeros((1, annotated_tensor.shape[1], annotated_tensor.shape[2]), dtype=torch.float32)

        return annotated_tensor, mask_tensor

    def _segment_polygons(
        self,
        image_pil: Image.Image,
        predictions: Dict[str, Any],
        fill_mask: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        width, height = image_pil.size
        mask_image = Image.new("RGB", (width, height), "black")
        mask_draw = ImageDraw.Draw(mask_image)

        for polygons, label in zip(predictions["polygons"], predictions["labels"]):
            color = random.choice(COLOR_BANK)
            for polygon in polygons:
                polygon = np.array(polygon).reshape(-1, 2)
                polygon = np.clip(polygon, [0, 0], [width - 1, height - 1])
                if len(polygon) < 3:
                    continue
                pts = polygon.reshape(-1).tolist()
                if fill_mask:
                    overlay = Image.new("RGBA", image_pil.size, (255, 255, 255, 0))
                    draw = ImageDraw.Draw(overlay)
                    rgba = ImageColor.getrgb(color) + (180,)
                    draw.polygon(pts, outline=color, fill=rgba, width=3)
                    image_pil = Image.alpha_composite(image_pil.convert("RGBA"), overlay).convert("RGB")
                else:
                    draw = ImageDraw.Draw(image_pil)
                    draw.polygon(pts, outline=color, width=3)
                mask_draw.polygon(pts, outline="white", fill="white")

        image_tensor = TF.to_tensor(image_pil)[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
        mask_tensor = TF.to_tensor(mask_image).unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
        mask_tensor = mask_tensor.mean(dim=0, keepdim=True)[:, :, :, 0]
        return image_tensor, mask_tensor

    def analyze(
        self,
        image: torch.Tensor,
        model_name: str,
        task: str,
        precision: str,
        attention: str,
        fill_mask: bool,
        output_mask_select: str = "",
        keep_model_loaded: bool = False,
        text_prompt: str = "",
    ):
        bundle = self._get_model(model_name, precision, attention)
        model = bundle["model"]
        processor = bundle["processor"]
        dtype = bundle["dtype"]

        device = mm.get_torch_device()
        offload = mm.unet_offload_device()
        model.to(device)
        images = image.permute(0, 3, 1, 2)

        task_config = TASK_CONFIGS[task]
        prompt = self._prepare_prompt(task, text_prompt)
        out_images, out_masks, out_data = [], [], []
        pbar = ProgressBar(len(images))
        select_filter = [s.strip() for s in output_mask_select.split(",") if s.strip()]

        for tensor in images:
            image_pil = TF.to_pil_image(tensor)
            inputs = processor(
                text=prompt,
                images=image_pil,
                return_tensors="pt",
                do_rescale=False,
            ).to(dtype).to(device)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                do_sample=GENERATION_CONFIG["do_sample"],
                num_beams=GENERATION_CONFIG["num_beams"],
                use_cache=False,
            )
            raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(raw, task=task_config["token"], image_size=image_pil.size)
            predictions = parsed[task_config["token"]]

            if task_config["mode"] == "bbox":
                annotated, mask = self._draw_regions(image_pil, predictions, fill_mask, select_filter)
                out_images.append(annotated)
                out_masks.append(mask)
            else:
                image_tensor, mask_tensor = self._segment_polygons(image_pil, predictions, fill_mask)
                out_images.append(image_tensor)
                out_masks.append(mask_tensor)

            out_data.append(predictions)
            pbar.update(1)

        image_out = torch.cat(out_images, dim=0)
        mask_out = torch.cat(out_masks, dim=0)

        if not keep_model_loaded:
            model.to(offload)
            mm.soft_empty_cache()

        return image_out, mask_out, out_data


NODE_CLASS_MAPPINGS = {
    "AILab_Florence2": AILab_Florence2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_Florence2": "Florence2 (RMBG)",
}
