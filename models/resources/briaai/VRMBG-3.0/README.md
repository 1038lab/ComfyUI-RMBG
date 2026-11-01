---
license: other
license_name: bria-vrmbg-3.0
license_link: LICENSE
pipeline_tag: image-segmentation
tags:
  - video
  - background-removal
  - video-matting
  - temporal-consistency
  - realtime
  - autoregressive
  - pytorch
  - vision
extra_gated_description: >-
  Use of this model requires a commercial agreement with BRIA AI. Academic
  access will be granted upon request — please fill in this form and indicate
  your academic affiliation, and the BRIA team will follow up to grant access.
extra_gated_heading: Request access — commercial use requires a BRIA AI agreement; academic access granted upon request
extra_gated_fields:
  Name: text
  Email: text
  Company/Org name: text
  Company Website URL: text
  Discord user: text
  Intended use:
    type: select
    options:
      - "Commercial"
      - "Academic - Research"
      - "Personal"
  Academic affiliation (institution & department, if applicable): text
  I understand that commercial use of this model requires a separate commercial agreement with BRIA AI, and that academic access is granted on request and is limited to non-commercial research and teaching: checkbox
  I agree to BRIA's Privacy policy and Terms & conditions: checkbox
---

# BRIA Video Background Removal v3.0 (VRMBG-3.0)

VRMBG-3.0 improves both temporal consistency and per-frame accuracy over VRMBG-2.0 while maintaining a lightweight design that enables **real-time** video background removal. The model achieves an attractive trade-off between efficiency and state-of-the-art performance — both in matte quality and in temporal stability — and was carefully trained on a proprietary video dataset spanning a diverse range of settings, subjects, and scene conditions.

For still-image background removal, see RMBG-2.0.

## Model Details

- **Developed by:** BRIA AI
- **Model type:** Video background removal / alpha matting
- **Parameters:** ~220M
- **Inference resolution:** 1024 × 1024
- **Input:** Current RGB video frame, paired with the previous frame's RGB multiplied by the previous frame's predicted alpha matte
- **Output:** Single-channel alpha matte for the current frame, in the range `[0, 1]`
- **Latency:** Real-time inference
- **License:** BRIA VRMBG-3.0 License — non-commercial use only. Commercial use requires a commercial agreement with BRIA AI.

## How it works

VRMBG-3.0 is **autoregressive** along the time axis. At each step the model consumes the current RGB frame together with the previous frame's RGB masked by the previous frame's predicted alpha, and emits the alpha matte for the current frame:

```
α_t = VRMBG3(RGB_t, RGB_{t-1} · α_{t-1})
```

For the first frame of a clip (no temporal prior), zero tensors are passed for both the previous-frame RGB and the previous-frame alpha. Conditioning on the previous frame's masked foreground provides a strong temporal prior that stabilises matte boundaries across frames and substantially reduces flicker compared with per-frame inference.

## Inference

### Minimal example

```python
import torch
import cv2
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# 1. Load the model
model = AutoModelForImageSegmentation.from_pretrained(
    "briaai/VRMBG-3.0", trust_remote_code=True,
)
model = model.eval().half().cuda()

# 2. Pre-processing.
INFER_SIZE = 1024
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()
device = torch.device("cuda")
dtype  = next(model.parameters()).dtype  # likely torch.float16

# 3. Initialise temporal state with zeros for the first frame.
prev_rgb_t = torch.zeros(3, INFER_SIZE, INFER_SIZE, device=device, dtype=dtype)
prev_alpha = torch.zeros(1, INFER_SIZE, INFER_SIZE, device=device, dtype=dtype)

cap = cv2.VideoCapture("input.mp4")
mattes = []

while True:
    ok, bgr = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    rgb_resized = cv2.resize(rgb, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_LINEAR)
    current_t = normalize(to_tensor(rgb_resized)).to(device=device, dtype=dtype)

    # Build the paired input: [current RGB, previous RGB * previous alpha].
    paired = torch.cat([current_t, prev_rgb_t * prev_alpha], dim=0).unsqueeze(0)
    paired = paired.contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        pred = model(paired)[-1].sigmoid().squeeze(0)  # (1, H, W) in [0, 1]

    # Resize the matte back to native resolution.
    alpha_native = cv2.resize(
        pred[0].float().cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR
    )
    mattes.append(alpha_native)

    # Update temporal state for the next frame.
    prev_rgb_t = current_t
    prev_alpha = pred

cap.release()
```

## Intended Use

- Real-time video background removal for production content (people, objects, products) where temporal stability matters.
- Autoregressive inference along the time axis: the model consumes the current frame together with the previous frame's predicted alpha at each step.
- For still-image background removal, use RMBG-2.0.

## Files

| File | Description |
|---|---|
| `config.json` | HF config with `auto_map` for `trust_remote_code` loading |
| `vrmbg3_config.py` | `PretrainedConfig` subclass referenced by `config.json` |
| `model.py` | Model architecture (`BiRefNet`, a `PreTrainedModel`) |
| `model.safetensors` | Trained weights in safetensors format, 885 MB |
| `pytorch_model.bin` | Same weights as a PyTorch `state_dict` |
| `README.md` | This model card |

## License

Released under the BRIA VRMBG-3.0 License. This model is not open source at the moment. Commercial use is subject to a commercial agreement with BRIA AI — please contact the BRIA team to request access or arrange a commercial agreement.
