---
license: openrail++
pipeline_tag: image-to-image
library_name: diffusers
base_model:
- stabilityai/stable-diffusion-2
---

# SDMatte - SafeTensors Models for Interactive Matting

This repository provides **SafeTensors** versions of the SDMatte models for **interactive image matting**, optimized for seamless use with **ComfyUI**.

---

## 🔍 About SDMatte

**SDMatte: Grafting Diffusion Models for Interactive Matting** is a state-of-the-art model that leverages the power of **diffusion priors** to achieve high-precision matting — especially around fine details and complex edges.

### ✨ Key Features

- **Diffusion-Powered**: Uses strong priors from diffusion models to extract high-fidelity details  
- **Interactive Matting**: Visual prompt-driven control for intuitive editing  
- **Edge & Texture Focus**: Excels in handling challenging edge regions and fine textures  
- **Coordinate & Opacity Awareness**: Improves matting accuracy with spatial and opacity context  

---

## 📦 Available Models

- `SDMatte.safetensors` – Standard version for interactive matting  
- `SDMatte_plus.safetensors` – Enhanced version with improved performance  

---

## 🧩 Built for ComfyUI: `ComfyUI-RMBG`

These models are designed for use with our **ComfyUI custom node**:  
➡️ [ComfyUI-RMBG on GitHub](https://github.com/1038lab/ComfyUI-RMBG)

This custom node integrates SDMatte into ComfyUI workflows, enabling high-quality interactive matting inside a visual pipeline.

### 🔄 Latest Update  
**Version:** `v2.9.0`  
**Date:** `2025-08-18`  
📄 [Read the update changelog](https://github.com/1038lab/ComfyUI-RMBG/blob/main/update.md#v290-20250818)

---

## 🙌 Credits and Attribution

### 📚 Original Work

- **Authors**: vivoCameraResearch Team  
- **Model Repository**: [Hugging Face – LongfeiHuang/SDMatte](https://huggingface.co/LongfeiHuang/SDMatte)  
- **Official Code**: [GitHub – vivoCameraResearch/SDMatte](https://github.com/vivoCameraResearch/SDMatte)  
- **Paper**: *SDMatte: Grafting Diffusion Models for Interactive Matting*

### 📝 Abstract (from the original paper)

> Recent interactive matting methods have shown satisfactory performance in capturing the primary regions of objects, but they fall short in extracting fine-grained details in edge regions. Diffusion models trained on billions of image-text pairs demonstrate exceptional capability in modeling highly complex data distributions and synthesizing realistic texture details, while exhibiting robust text-driven interaction capabilities — making them an attractive solution for interactive matting.

---