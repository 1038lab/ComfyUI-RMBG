---
license: mit
base_model:
- microsoft/Florence-2-large
datasets:
- Ejafa/ye-pop
tags:
- art
pipeline_tag: image-to-text
language:
- en
library_name: transformers
---
# microsoft/Florence-2-large tuned on Ejafa/ye-pop captioned with CogVLM2

This repository contains a fine-tuned version of the `microsoft/Florence-2-large` model. The model has been tuned on a 40,000 image subset of the `Ejafa/ye-pop` dataset, with captions generated using `THUDM/cogvlm2-llama3-chat-19B`.

## Training Details

- **Vision Encoder**: The vision encoder was frozen during training.
- **Batch Size**: 64
- **Gradient Accumulation Steps**: 16
- **Learning Rate**: 5.12e-05
- **Optimizer**: AdamW
- **Scheduler**: polynomial
- **Epochs**: 8.36

## Dataset

The fine-tuning process utilized a 40,000 image subset from the `Ejafa/ye-pop` dataset. This dataset contains a wide array of images with varying subjects, providing a robust training ground for improving the model's captioning abilities.

## Captioning

The captions were generated using `THUDM/cogvlm2-llama3-chat-19B` and then post-processed with `google/gemma-2-9b` to remove vagueness.

## Usage

To use this model, you can load it directly from the Hugging Face Model Hub:

```python
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("thwri/CogFlorence-2.2-Large", trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained("thwri/CogFlorence-2.2-Large", trust_remote_code=True)
# Function to run the model on an example
def run_example(task_prompt, image):
    prompt = task_prompt
    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=True
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer
from PIL import Image
import requests
import copy
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)
result = run_example("<MORE_DETAILED_CAPTION>" , image)
print(result)
# {'<MORE_DETAILED_CAPTION>': 'A vivid portrayal of a classic Volkswagen Beetle parked on a cobblestone street. The car is painted a vibrant turquoise, contrasting with the muted yellow of the building behind it. The building has two wooden doors, one with a white frame and the other with a dark brown finish. The sky is clear, and the sun casts a warm glow on the scene, highlighting the car's details. The image evokes a nostalgic and nostalgic mood, capturing a moment in time without posed elements.'}
```