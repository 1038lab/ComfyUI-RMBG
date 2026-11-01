---
license: mit
license_link: https://huggingface.co/microsoft/Florence-2-large/resolve/main/LICENSE
pipeline_tag: image-text-to-text
tags:
- vision
---

# Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks

## Model Summary

**This is a continued pretrained version of Florence-2-large model with 4k context length, only 0.1B samples are used for continue pretraining, thus it might not be trained well. In addition, OCR task has been updated with line separator ('\n'). COCO OD AP 39.8**

This Hub repository contains a HuggingFace's `transformers` implementation of Florence-2 model from Microsoft.

Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks.  Florence-2 can interpret simple text prompts to perform tasks like captioning, object detection, and segmentation. It leverages our FLD-5B dataset, containing 5.4 billion annotations across 126 million images, to master multi-task learning. The model's sequence-to-sequence architecture enables it to excel in both zero-shot and fine-tuned settings, proving to be a competitive vision foundation model. 

Resources and Technical Documentation:
+ [Florence-2 technical report](https://arxiv.org/abs/2311.06242). 
+ [Jupyter Notebook for inference and visualization of Florence-2-large](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb)

| Model   | Model size | Model Description | 
| ------- | ------------- |   ------------- |  
| Florence-2-base[[HF]](https://huggingface.co/microsoft/Florence-2-base) | 0.23B | Pretrained model with FLD-5B  
| Florence-2-large[[HF]](https://huggingface.co/microsoft/Florence-2-large) | 0.77B  | Pretrained model with FLD-5B  
| Florence-2-base-ft[[HF]](https://huggingface.co/microsoft/Florence-2-base-ft) | 0.23B  | Finetuned model on a colletion of downstream tasks
| Florence-2-large-ft[[HF]](https://huggingface.co/microsoft/Florence-2-large-ft) | 0.77B | Finetuned model on a colletion of downstream tasks
 
## How to Get Started with the Model

Use the code below to get started with the model. All models are trained with float16. 

```python
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

prompt = "<OD>"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=4096,
    num_beams=3,
    do_sample=False
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

print(parsed_answer)

```


## Tasks

This model is capable of performing different tasks through changing the prompts.

First, let's define a function to run a prompt.

<details>
<summary> Click to expand </summary>

```python
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    print(parsed_answer)
```
</details>

Here are the tasks `Florence-2` could perform:

<details>
<summary> Click to expand </summary>



### Caption
```python
prompt = "<CAPTION>"
run_example(prompt)
```

### Detailed Caption
```python
prompt = "<DETAILED_CAPTION>"
run_example(prompt)
```

### More Detailed Caption
```python
prompt = "<MORE_DETAILED_CAPTION>"
run_example(prompt)
```

### Caption to Phrase Grounding 
caption to phrase grounding task requires additional text input, i.e. caption. 

Caption to phrase grounding results format: 
{'\<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['', '', ...]}}
```python
task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
results = run_example(task_prompt, text_input="A green car parked in front of a yellow building.")
```

### Object Detection

OD results format: 
{'\<OD>': {'bboxes': [[x1, y1, x2, y2], ...], 
'labels': ['label1', 'label2', ...]} }

```python
prompt = "<OD>"
run_example(prompt)
```

### Dense Region Caption
Dense region caption results format: 
{'\<DENSE_REGION_CAPTION>' : {'bboxes': [[x1, y1, x2, y2], ...], 
'labels': ['label1', 'label2', ...]} }
```python
prompt = "<DENSE_REGION_CAPTION>"
run_example(prompt)
```

### Region proposal
Dense region caption results format: 
{'\<REGION_PROPOSAL>': {'bboxes': [[x1, y1, x2, y2], ...], 
'labels': ['', '', ...]}}
```python
prompt = "<REGION_PROPOSAL>"
run_example(prompt)
```

### OCR 

```python
prompt = "<OCR>"
run_example(prompt)
```

### OCR with Region
OCR with region output format:
{'\<OCR_WITH_REGION>': {'quad_boxes': [[x1, y1, x2, y2, x3, y3, x4, y4], ...], 'labels': ['text1', ...]}}
```python
prompt = "<OCR_WITH_REGION>"
run_example(prompt)
```

### Output confidence score with Object Detection
```python

def run_example_with_score(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3,
      return_dict_in_generate=True,
      output_scores=True,
    )
    generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=False)[0]

    prediction, scores, beam_indices = generated_ids.sequences, generated_ids.scores, generated_ids.beam_indices
    transition_beam_scores = model.compute_transition_scores(
        sequences=prediction,
        scores=scores,
        beam_indices=beam_indices,
    )

    parsed_answer = processor.post_process_generation(sequence=generated_ids.sequences[0], 
        transition_beam_score=transition_beam_scores[0],
        task=task_prompt, image_size=(image.width, image.height)
    )

    print(parsed_answer)

prompt = "<OD>"
run_example_with_score(prompt)

```

for More detailed examples, please refer to [notebook](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb)
</details>

# Benchmarks

## Florence-2 Zero-shot performance
  
The following table presents the zero-shot performance of generalist vision foundation models on image captioning and object detection evaluation tasks. These models have not been exposed to the training data of the evaluation tasks during their training phase.  
  
| Method | #params | COCO Cap. test CIDEr | NoCaps val CIDEr | TextCaps val CIDEr | COCO Det. val2017 mAP |  
|--------|---------|----------------------|------------------|--------------------|-----------------------|
| Flamingo | 80B | 84.3 | - | - | - | 
| Florence-2-base| 0.23B | 133.0 | 118.7 | 70.1 | 34.7 | 
| Florence-2-large| 0.77B | 135.6 | 120.8 | 72.8 | 37.5 |

  
The following table continues the comparison with performance on other vision-language evaluation tasks.  
  
| Method | Flickr30k test R@1 | Refcoco val Accuracy | Refcoco test-A Accuracy | Refcoco test-B Accuracy | Refcoco+ val Accuracy | Refcoco+ test-A Accuracy | Refcoco+ test-B Accuracy | Refcocog val Accuracy | Refcocog test Accuracy | Refcoco RES val mIoU |  
|--------|----------------------|----------------------|-------------------------|-------------------------|-----------------------|--------------------------|--------------------------|-----------------------|------------------------|----------------------|  
| Kosmos-2 | 78.7 | 52.3 | 57.4 | 47.3 | 45.5 | 50.7 | 42.2 | 60.6 | 61.7 | - |  
| Florence-2-base | 83.6 | 53.9 | 58.4 | 49.7 | 51.5 | 56.4 | 47.9 | 66.3 | 65.1 | 34.6 |  
| Florence-2-large | 84.4 | 56.3 | 61.6 | 51.4 | 53.6 | 57.9 | 49.9 | 68.0 | 67.0 | 35.8 |  



## Florence-2 finetuned performance 

We finetune Florence-2 models with a collection of downstream tasks, resulting two generalist models *Florence-2-base-ft* and *Florence-2-large-ft* that can conduct a wide range of downstream tasks. 
  
The table below compares the performance of specialist and generalist models on various captioning and Visual Question Answering (VQA) tasks. Specialist models are fine-tuned specifically for each task, whereas generalist models are fine-tuned in a task-agnostic manner across all tasks. The symbol "▲" indicates the usage of external OCR as input.  
  
| Method         | # Params | COCO Caption Karpathy test CIDEr | NoCaps val CIDEr | TextCaps val CIDEr | VQAv2 test-dev Acc | TextVQA test-dev Acc | VizWiz VQA test-dev Acc |  
|----------------|----------|-----------------------------------|------------------|--------------------|--------------------|----------------------|-------------------------|  
| **Specialist Models**   |          |                                   |                  |                    |                    |                      |                         |  
| CoCa           | 2.1B     | 143.6                              | 122.4            | -                  | 82.3               | -                    | -                       |  
| BLIP-2         | 7.8B     | 144.5                              | 121.6            | -                  | 82.2               | -                    | -                       |  
| GIT2           | 5.1B     | 145.0                              | 126.9            | 148.6              | 81.7               | 67.3                 | 71.0                    |  
| Flamingo       | 80B      | 138.1                              | -                | -                  | 82.0               | 54.1                 | 65.7                    |  
| PaLI           | 17B      | 149.1                              | 127.0            | 160.0▲             | 84.3               | 58.8 / 73.1▲         | 71.6 / 74.4▲            |  
| PaLI-X         | 55B      | 149.2                              | 126.3            | 147.0 / 163.7▲     | 86.0               | 71.4 / 80.8▲         | 70.9 / 74.6▲            |  
| **Generalist Models**   |          |                                   |                  |                    |                    |                      |                         |  
| Unified-IO     | 2.9B     | -                                  | 100.0            | -                  | 77.9               | -                    | 57.4                    |  
| Florence-2-base-ft | 0.23B  | 140.0                              | 116.7            | 143.9              | 79.7               | 63.6                 | 63.6                    |  
| Florence-2-large-ft | 0.77B  | 143.3                              | 124.9            | 151.1              | 81.7               | 73.5                 | 72.6                    |  
  
  
| Method               | # Params | COCO Det. val2017 mAP | Flickr30k test R@1 | RefCOCO val Accuracy | RefCOCO test-A Accuracy | RefCOCO test-B Accuracy | RefCOCO+ val Accuracy | RefCOCO+ test-A Accuracy | RefCOCO+ test-B Accuracy | RefCOCOg val Accuracy | RefCOCOg test Accuracy | RefCOCO RES val mIoU |  
|----------------------|----------|-----------------------|--------------------|----------------------|-------------------------|-------------------------|------------------------|---------------------------|---------------------------|------------------------|-----------------------|------------------------|  
| **Specialist Models** |          |                       |                    |                      |                         |                         |                        |                           |                           |                        |                       |                        |  
| SeqTR                | -        | -                     | -                  | 83.7                 | 86.5                    | 81.2                    | 71.5                   | 76.3                      | 64.9                      | 74.9                   | 74.2                  | -                      |  
| PolyFormer           | -        | -                     | -                  | 90.4                 | 92.9                    | 87.2                    | 85.0                   | 89.8                      | 78.0                      | 85.8                   | 85.9                  | 76.9                   |  
| UNINEXT              | 0.74B    | 60.6                  | -                  | 92.6                 | 94.3                    | 91.5                    | 85.2                   | 89.6                      | 79.8                      | 88.7                   | 89.4                  | -                      |  
| Ferret               | 13B      | -                     | -                  | 89.5                 | 92.4                    | 84.4                    | 82.8                   | 88.1                      | 75.2                      | 85.8                   | 86.3                  | -                      |  
| **Generalist Models** |          |                       |                    |                      |                         |                         |                        |                           |                           |                        |                       |                        |  
| UniTAB               | -        | -                     | -                  | 88.6                 | 91.1                    | 83.8                    | 81.0                   | 85.4                      | 71.6                      | 84.6                   | 84.7                  | -                      |  
| Florence-2-base-ft | 0.23B    | 41.4                  | 84.0                | 92.6                 | 94.8                    | 91.5                   | 86.8                   | 91.7                      | 82.2                      | 89.8                   | 82.2                  | 78.0                  |  
| Florence-2-large-ft| 0.77B    | 43.4                  | 85.2               | 93.4                 | 95.3                    | 92.0                    | 88.3                   | 92.9                      | 83.6                      | 91.2                   | 91.7                  | 80.5                   |  
  

## BibTex and citation info

```
@article{xiao2023florence,
  title={Florence-2: Advancing a unified representation for a variety of vision tasks},
  author={Xiao, Bin and Wu, Haiping and Xu, Weijian and Dai, Xiyang and Hu, Houdong and Lu, Yumao and Zeng, Michael and Liu, Ce and Yuan, Lu},
  journal={arXiv preprint arXiv:2311.06242},
  year={2023}
}
```