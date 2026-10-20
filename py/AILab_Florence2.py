import hashlib
import io
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageColor, ImageDraw, ImageFont

import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file
import comfy.ops
import folder_paths

from models.florence2.config import Florence2Config
from models.florence2.model import Florence2
from models.florence2.processing import Processor

MODEL_DIR = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(MODEL_DIR, exist_ok=True)
folder_paths.add_model_folder_path("LLM", MODEL_DIR)

MODE_CONFIGS = {
    "Polygon Mask (prompt)": {
        "token": "<REFERRING_EXPRESSION_SEGMENTATION>",
        "type": "polygon",
        "allows_text": True,
    },
    "Phrase Grounding (boxes)": {
        "token": "<CAPTION_TO_PHRASE_GROUNDING>",
        "type": "bbox",
        "allows_text": True,
    },
    "Region Proposals (boxes only)": {
        "token": "<REGION_PROPOSAL>",
        "type": "bbox",
        "allows_text": False,
    },
}

MODE_CHOICES = tuple(MODE_CONFIGS.keys())
TASK_CONFIGS = MODE_CONFIGS
TASK_CHOICES = MODE_CHOICES


MODEL_CHOICES = (
    "microsoft/Florence-2-base",
    "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-large",
    "microsoft/Florence-2-large-ft",
    "thwri/CogFlorence-2.1-Large",
    "thwri/CogFlorence-2.2-Large",
)

COLOR_BANK = ["blue", "orange", "green", "purple", "pink", "cyan"]





class AILab_Florence2:
    CATEGORY = "🧪AILab/🧽RMBG"
    RETURN_TYPES = ("IMAGE", "MASK", "JSON", "BBOX")
    RETURN_NAMES = ("IMAGE", "MASK", "DATA", "BBOXES")
    FUNCTION = "analyze"
    MODEL_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Prompt: e.g. face, a person wearing red coat",
                        "tooltip": "Used for polygon masks or phrase grounding; ignored for region proposals.",
                    },
                ),
                "model": (
                    MODEL_CHOICES,
                    {
                        "default": MODEL_CHOICES[0],
                        "tooltip": "Base = stable, +ft = fine-tuned captions, CogFlorence = sharper phrase alignment.",
                    },
                ),
                "mode": (
                    MODE_CHOICES,
                    {
                        "default": MODE_CHOICES[0],
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
                "max_new_tokens": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 4096, "step": 32},
                ),
                "unload_model": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Unload model from VRAM after execution to free memory."},
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
        key = (model_name, precision)
        if key in self.MODEL_CACHE:
            return self.MODEL_CACHE[key]

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        weights = self._ensure_weights(model_name)
        offload = mm.unet_offload_device()

        config_path = os.path.join(weights, "config.json")
        checkpoint_path = os.path.join(weights, "model.safetensors")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(weights, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No model weights found at {weights}")

        config = Florence2Config.from_json(config_path)
        model = Florence2(config, dtype=dtype, device="cpu", operations=comfy.ops.manual_cast)

        sd = load_torch_file(checkpoint_path)
        for key_name in [
            "language_model.model.encoder.embed_tokens.weight",
            "language_model.model.decoder.embed_tokens.weight",
        ]:
            if key_name in sd and "language_model.model.shared.weight" in sd:
                sd.pop(key_name, None)

        model.load_state_dict(sd, strict=False)
        del sd
        model.language_model.tie_weights()
        model = model.eval().to(offload)

        processor = Processor(model_path=weights)
        bundle = {"model": model, "processor": processor, "dtype": dtype}
        self.MODEL_CACHE[key] = bundle
        return bundle

    @staticmethod
    def _prepare_prompt(mode: str, text_prompt: str) -> str:
        config = MODE_CONFIGS.get(mode, MODE_CONFIGS[MODE_CHOICES[0]])
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
        annotated = image_pil.copy()
        draw = ImageDraw.Draw(annotated)

        mask_layer = Image.new("RGB", image_pil.size, (0, 0, 0)) if fill_mask else None
        mask_draw = ImageDraw.Draw(mask_layer) if mask_layer else None

        try:
            font = ImageFont.load_default(size=14)
        except Exception:
            font = ImageFont.load_default()

        for index, (bbox, label) in enumerate(zip(predictions["bboxes"], predictions["labels"])):
            x0, y0, x1, y1 = bbox
            if y1 < y0:
                y0, y1 = y1, y0
            if x1 < x0:
                x0, x1 = x1, x0
            filter_hit = not select_filter or str(index) in select_filter or label in select_filter
            if fill_mask and filter_hit:
                mask_draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))

            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            label_text = f"{index}.{label}"
            color = random.choice(COLOR_BANK)
            text_pos = (x0, max(0, y0 - 16))
            text_bbox = draw.textbbox(text_pos, label_text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text(text_pos, label_text, fill="white", font=font)

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
        prompt: str = "",
        model: str = MODEL_CHOICES[0],
        mode: str = MODE_CHOICES[0],
        precision: str = "fp16",
        attention: str = "sdpa",
        fill_mask: bool = True,
        output_mask_select: str = "",
        max_new_tokens: int = 1024,
        unload_model: bool = False,
        model_name: str = None,
        task: str = None,
        text_prompt: str = None,
        keep_model_loaded: bool = None,
        **kwargs,
    ):
        if model_name is not None and (model is None or model == MODEL_CHOICES[0]):
            model = model_name
        if task is not None and not mode:
            mode = task
        if text_prompt is not None and not prompt:
            prompt = text_prompt
        if keep_model_loaded is not None:
            unload_model = not keep_model_loaded

        bundle = self._get_model(model, precision, attention)
        florence_model = bundle["model"]
        processor = bundle["processor"]
        dtype = bundle["dtype"]

        device = mm.get_torch_device()
        offload = mm.unet_offload_device()
        florence_model.to(device)
        images = image.permute(0, 3, 1, 2)

        mode_config = MODE_CONFIGS.get(mode, MODE_CONFIGS[MODE_CHOICES[0]])
        prepared_prompt = self._prepare_prompt(mode, prompt)
        out_images, out_masks, out_data = [], [], []
        pbar = ProgressBar(len(images))
        select_filter = [s.strip() for s in output_mask_select.split(",") if s.strip()]

        for tensor in images:
            image_pil = TF.to_pil_image(tensor)
            inputs = processor(
                text=prepared_prompt,
                images=tensor.unsqueeze(0),
            )

            generated_ids = florence_model.generate(
                input_ids=inputs["input_ids"].to(device),
                pixel_values=inputs["pixel_values"].to(dtype=dtype, device=device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=3,
            )
            raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(raw, task=mode_config["token"], image_size=image_pil.size)
            predictions = parsed[mode_config["token"]]

            if mode_config["type"] == "bbox":
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

        if unload_model:
            florence_model.to(offload)
            mm.soft_empty_cache()

        all_bboxes = []
        for entry in out_data:
            if isinstance(entry, dict) and "bboxes" in entry:
                all_bboxes.extend(entry["bboxes"])

        return image_out, mask_out, out_data, all_bboxes


NODE_CLASS_MAPPINGS = {
    "AILab_Florence2": AILab_Florence2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_Florence2": "Florence2 (RMBG)",
}
