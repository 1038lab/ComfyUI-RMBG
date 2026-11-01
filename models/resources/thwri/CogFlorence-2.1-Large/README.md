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
- **Epochs**: 7.37

## Dataset

The fine-tuning process utilized a 40,000 image subset from the `Ejafa/ye-pop` dataset. This dataset contains a wide array of images with varying subjects, providing a robust training ground for improving the model's captioning abilities.

## Captioning

The captions were generated using `THUDM/cogvlm2-llama3-chat-19B`.

## Usage

To use this model, you can load it directly from the Hugging Face Model Hub:

```python
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("thwri/CogFlorence-2.1-Large", trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained("thwri/CogFlorence-2.1-Large", trust_remote_code=True)
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
# {'<MORE_DETAILED_CAPTION>': 'A vivid, close-up photograph of a classic car, specifically a Volkswagen Beetle, parked on a cobblestone street. The car is painted in a striking shade of turquoise, with a glossy finish that reflects the surrounding environment. The vehicle's rounded shape is accentuated by its rounded tires and chrome detailing. The background reveals a weathered yellow wall with a rustic wooden door, adding to the rustic charm of the scene. The sky above is clear, suggesting a sunny day. The overall style of the image is candid, capturing a moment in time without any posed or staged elements.'}
```