# ComfyUI-RMBG Update Log

## V3.0.0 (2026/01/01)  
### New nodes
- Added `Florence2 Segmentation` node for Florence-2 tasks: polygon masks, phrase grounding (boxes), and region proposals.
- Added `Florence2 To Coordinates` tool node to convert Florence-2 JSON into center coordinates, bounding boxes, and masks.
![v3.0.0_Florence2](example_workflows/florence2_node.jpg)
- Added `YoloV8` / `YoloV8Adv` nodes for YOLOv8 detection, producing annotated images, merged masks, and mask lists.
![v3.0.0_Yolo](example_workflows/YOLO_Node.jpg)
- Added `ColorToMask` node to generate masks from a target color with threshold and invert options.
![colorToMask](https://github.com/user-attachments/assets/1a70878e-2bf2-449a-8bda-a85951e1f357)
- Added `ImageToList` Node: Combines up to 6 images into a batch with optional resize modes: off, fit, crop.
- Added `MaskToList` Node: Converts a batch of masks into a mask list.
- Added `ImageMaskToList` Node: Converts a batch of images and masks into an image and mask list.
![List](https://github.com/user-attachments/assets/119f2988-776c-428e-8e4d-85696c4e84de)
- Added `ImageResize` Node: Comprehensive all-in-one image resizing tool with robust handling for most scenarios. Supports custom width and height, megapixel constraints, longest/shortest side resizing, padding, cropping, and additional flexible options.
- Enhanced the `Compare` node by adding support for bg_color and text_color properties. These improvements are now applicable for both side-by-side image comparison and video comparison.
![image_compare_resize](https://github.com/user-attachments/assets/4c43b00e-8bba-44d7-a465-7398dcd7050e)
- Updated `SAM3 Segmentation` node: added output mode (merged/separate), max segment, segment_pick, and device controls.  
![image_compare_resize](https://github.com/user-attachments/assets/f270471b-ff33-4a27-9abb-cecb735b534d)  

### 🔧 PyTorch JIT Compatibility Fix  
- Removed global torch.load override; TorchScript handled locally in SAM2.
- TorchScript is handled via a local fallback to avoid interfering with other nodes.
- Improves overall compatibility and stability in mixed ComfyUI environments.

### 📦 Dependency Update
- `triton-windows` for proper SAM2, SAM3 model execution on Windows platforms.
- `ultralytics` Required for YOLO node.
> [!NOTE]
> YOLO nodes require the optional `ultralytics` package. Install it only if you need YOLO to avoid dependency conflicts: `./ComfyUI/python_embeded/python -m pip install ultralytics --no-deps`.

## V2.9.6 (2025/12/09)
### ImageCompare Node Rebuilt
- Rebuilt ImageCompare node with enhanced features
- Added support for 3 images (previously 2)
- Added size_base parameter: choose largest, smallest, or specific image as reference
- Added customizable text_color and bg_color parameters

![v2.9.6_Image Compare](https://github.com/user-attachments/assets/e4ee824d-207e-4f46-b2db-0110e99c84c7)

### 🔧 ComfyUI New Schema Compatibility
- Updated our nodes to match the latest ComfyUI V3 schema changes.
- Fixed compatibility issues affecting multiple nodes, including `ImageCompositeMasked`.
- Thanks to reports in https://github.com/1038lab/ComfyUI-RMBG/issues/132 and https://github.com/1038lab/ComfyUI-RMBG/issues/146

### 🧹 SAM3Segment – Automatic Model Unload
- Added model unload for SAM3 segmentation.
- Helps free memory after each run and improves long-session stability.
- Thanks to contribution and feedback from https://github.com/1038lab/ComfyUI-RMBG/issues/147
## V2.9.5 (2025/10/15)
- Bug fix: SAM3 Segmentation CPU mode no longer crashes from mixed cuda/cpu tensors when a GPU is present. (https://github.com/1038lab/ComfyUI-RMBG/issues/135)
- Added missing dependency `decord` to requirements.txt. (https://github.com/1038lab/ComfyUI-RMBG/issues/136)
## V2.9.4 (2025/10/24)
- Added `SAM3 segmentation` node with Meta’s latest SAM3 segmentation model  
![v2.9.4_sam3](https://github.com/user-attachments/assets/70409f85-8814-47c4-8679-1e2389e5c78a)
- `SAM3Segment`: RMBG-focused text segmentation using the official `SAM3` Model checkpoint
  - Sharper edges and faster inference versus SAM2 in our tests; supports FP32/FP16 autocast on CUDA
  - Alpha/Color background output, mask blur/offset/invert, plus RGB mask image for quick compositing

https://github.com/user-attachments/assets/05cc101b-57a6-408d-b4ad-78e56dd927d6

## V2.9.3 (2025/10/05)
- Bug Fix: The latest ComfyUI update caused an issue with the `color` widget. We have addressed the problem and updated all related nodes. The widget now functions correctly.  **(User Reported [#118](https://github.com/1038lab/ComfyUI-RMBG/issues/118) )**
![v2.9._color](https://github.com/user-attachments/assets/422a7ad2-1522-4ea4-98d4-34fe1989f4e8)

## V2.9.2 (2025/09/30)
- Added `BiRefNet_toonOut` general purpose model (balanced performance) **(User request [#110](https://github.com/1038lab/ComfyUI-RMBG/issues/110) )**
![v2.9.2_BiRefNet_toonOut](https://github.com/user-attachments/assets/17dc7268-5017-415f-b0b5-4184c02bdaf2)
- `ImageStitch` Node Updates: Migrated to the latest architecture. Now supports 4-image input with a new 2x2 stitching mode. Automatically applies smart kontext_mode when 4 images are provided. Output layout configured as 3 images on the left and 1 on the right. Added support for magepixel and new upscaling methods.
![v2.9.2_imagestitch](https://github.com/user-attachments/assets/07bd919e-3ddf-4526-af6a-e4fa3e9f69ab)

## V2.9.1 (2025/09/12)
![v2.9.1](https://github.com/user-attachments/assets/9b6c3e6c-5866-4807-91ba-669eb7efc52b)
- Refactored `LoadImage` & `LoadImageAdvanced` Nodes
  - Reworked resizing logic for more powerful and intuitive control.
  - New execution priority: `megapixels` has the highest priority, otherwise `size` and `scale_by` now work together in a pipeline.
  - Improved image quality by calculating the final target size first and performing only a single `resize` operation to prevent quality loss.

- Added `METADATA_TEXT` Output to `LoadImageAdvanced`
  - The `LoadImageAdvanced` node now outputs the embedded generation parameters from AI-generated PNG files (e.g., prompts, model, seed).
  - This allows for easy workflow replication by connecting the metadata directly to text inputs.

- Enhanced the RMBG node to optimize batch processing of images and videos, **(User request [#100](https://github.com/1038lab/ComfyUI-RMBG/issues/100) )**

- Reconstructed the `ColorWidget` to improve stability and prevent potential freezes in certain ComfyUI configurations.
## V2.9.0 (2025/08/18)
- Added `SDMatte Matting` node **(User request [#99](https://github.com/1038lab/ComfyUI-RMBG/issues/99) )**
![v2 9 0](https://github.com/user-attachments/assets/05a5d41e-a73c-40cc-a4cc-c10380ecc425)
![v2 9 0](https://github.com/user-attachments/assets/5e74657d-8fa7-4987-8f8a-949b0f7aaa24)

- Optional `mask` input; if omitted and the input image has an alpha channel, the alpha is used as the mask
- Unified explicit bilinear resizing for inputs/outputs; improved consistency with other nodes
- Inference optimizations: `torch.inference_mode`, CUDA FP16 autocast, memory cleanup, and explicit GPU fallback messaging

## V2.8.0 (2025/08/11)
- Added SAM2 segmentation nodes with latest Facebook Research SAM2 technology
![v2 8 0](https://github.com/user-attachments/assets/16c5a67c-1aec-4def-9aa2-db9dcf2354a8)
  - `SAM2Segment`: Text-prompted segmentation with 4 model variants (Tiny/Small/Base Plus/Large)
  - Improved accuracy and faster processing compared to SAM V1
  - FP16/FP32 precision support and better edge detection
- Enhanced color widget support across all nodes
- Fixed color picker functionality and improved color handling consistency
- Updated SAM2 model integration with optimized memory usage and batch processing
- Bug Fixed

## V2.7.1 (2025/08/06)
- Bug fixes and improved code compatibility
## V2.7.0 (2025/07/27)
- Enhanced LoadImage node with direct URL and path support
![v2.7.0_ImageStitch](https://github.com/user-attachments/assets/cf3b5ab3-31e7-40f8-b941-23675d0a295e)
  - Added image_path_or_URL parameter for loading images from local paths or URLs
  - Improved URL handling with User-Agent support for better compatibility
  - Maintained compatibility with traditional file selection
  - Simplified workflow for external image sources
  - Three different LoadImage nodes for different purposes and needs:
    - `LoadImage`: Standard image loader with commonly used options, suitable for most workflows
    - `LoadImageSimple`: Minimalist image loader for quick and basic image loading
    - `LoadImageAdvanced`: Advanced image loader with extended configuration for power users

- Completely redesigned `ImageStitch` node with advanced features
![v2.7.0_ImageStitch](https://github.com/user-attachments/assets/3f31fe25-a453-4f86-bf3d-dc12a8affd39)
  - Compatible with ComfyUI's native image stitch functionality
  - Added support for 3-image stitching with kontext_mode
  - Improved spacing and background color options
  - Added maximum size constraints for output images
  - Enhanced image matching and padding options
  - Better handling of different image sizes and aspect ratios
  - Included commonly requested user settings for more flexibility

### Demo workflow (Flux kontext + Nunchaku + ImageStitch)
![v2.7.0_ImageStitch](https://github.com/user-attachments/assets/d73531a2-afbe-4a38-9459-86d8d55fcc91)

- Fixed background color handling across all nodes
  - Resolved errors reported by users when using color picker
  - Fixed color application in segmentation and background removal nodes
  - Improved color consistency across different operations

## V2.6.0 (2025/07/15)
![ReferenceLatentMaskr](https://github.com/user-attachments/assets/8eba03be-d139-4694-9ec4-7d99bace4a20)

- Added the first RMBG inpainting tool for the Flux Kontext model: the `ReferenceLatentMask` node, which leverages a reference latent and mask for precise region conditioning. (Stay tuned, more tools will be released in future updates.)
- Updated RMBG `LoadImage` node: added an upscaling method for improved output quality, refined image output to RGB format, and optimized the alpha channel in the mask output..

![ReferenceLatentMaskr2](https://github.com/user-attachments/assets/125925ca-c1c6-496e-8b6b-9c5bcd07749e)

## V2.5.2 (2025/07/11)
- Model repository Bug Fix
![V 2 5 2](https://github.com/user-attachments/assets/4b41887a-0d8a-4a5a-9128-1e866f410b60)

## V2.5.1 (2005/07/07)
- Fixed the missing BiRefNet Models

## V2.5.0 (2025/07/01)
- Introduced the `MaskOverlay` node, enabling mask overlays directly on images.
![mask_overlay](https://github.com/user-attachments/assets/d82abb5a-9702-4d21-a5cf-e6776c7b4c06)
- Added `ImageMaskResize` node for resizing image and mask with various options.
![image_mask_resize](https://github.com/user-attachments/assets/32afcd17-29a2-42bf-a31c-9ea2b604d5df)
- Implemented the LamaRemover node for object removal using the LaMa model. For a more advanced object removal solution, see our companion project: [ComfyUI-MiniMax-Remover](https://github.com/1038lab/ComfyUI-MiniMax-Remover)
![lamaRemover](https://github.com/user-attachments/assets/7230b39c-c443-44bb-bc6e-3e9ecfd03a3d)
- Added 2 BiRefNet models: `BiRefNet_lite-matting` and `BiRefNet_dynamic`
- Added batch image support for `Segment_v1` and `Segment_V2` nodes

## v2.4.0 (2025/06/01)
- Added `CropObject` node for cropping to object based on mask or alpha channel **(User request [#61](https://github.com/1038lab/ComfyUI-RMBG/issues/61) )**
- Added `ImageCompare` node for side-by-side image comparison with annotations
- Added `ColorInput` node pick preset color or input RGB color code in #000000 or #000 format **(User request [#62](https://github.com/1038lab/ComfyUI-RMBG/issues/62) )**
- Updated `MaskExtractor` node added color picker and support RGBA images by extracting and using the alpha channel as mask
- Updated `ImageCombiner` node added WIDTH and HEIGHT output

![ComfyUI-RMBG_V2 4 0 new nodes](https://github.com/user-attachments/assets/7ab023e7-70b4-4b97-910a-e608c03841cf)

### New Segment V2 (Recommended)
- Uses Hugging Face transformers library
- Better compatibility with newer PyTorch (2.x) and CUDA versions
- Recommended for users with modern GPU setups
- No groundingdino-py dependency required

**(User request [#66](https://github.com/1038lab/ComfyUI-RMBG/issues/66) )**
### Segment V1 (Legacy)
- Uses original groundingdino-py implementation
- May have compatibility issues with newer PyTorch/CUDA versions
- Consider using V2 if you encounter installation issues

### Installation
Choose the appropriate version based on your setup:
- For modern systems (PyTorch 2.x, CUDA 12.x+), use Segment V2
- For legacy systems or if you specifically need groundingdino-py, use Segment V1

## v2.3.2 (2025/05/15)
- Added support for more segmentation models in Segment node:
  - SAM HQ models (vit_h, vit_l, vit_b)
- Changed background color input to color picker for better color selection
- Updated and standardized `i18n` format for all nodes, improving multilingual compatibility and fixing some translation display issues
- Added node appearance style options, allowing customization of node appearance in the ComfyUI graph for better visual distinction and user experience

![v 2 3 2](https://github.com/user-attachments/assets/fc852183-6796-4ef7-a41a-499dbe6a4519)

## v2.3.1 (2025/05/02)
- Enhanced ICLoRA Concat node to fully support the native ComfyUI Load Image node, addressing previous limitations with mask scaling. ICLoRA Concat is now compatible with both the RMBG and native image loaders.
  
## v2.3.0 (2025/05/01)
- Added `Image Crop` node: Flexible cropping tool for images, supporting multiple anchor positions, offsets, and split output for precise region extraction.
- Added `ICLoRA Concat` node: Enables mask-based image concatenation with customizable direction (left-right or top-bottom), size, and region, suitable for advanced image composition and layout.
- Added resizing options for Load Image: Longest Side, Shortest Side, Width, and Height, enhancing flexibility.
- Fixed an issue where the preview node did not display images on Ubuntu.

![v2 3 0_node](https://github.com/user-attachments/assets/f53be704-bb53-4fdf-9e7f-fad00dcd5add)

## v2.2.1 (2025/04/05)
- Bug Fixed

## v2.2.0 (2025/04/05)
- Added the following nodes:
  - `Image Combiner`: Image Combiner, used to merge two images into one with various blending modes and positioning options.
  - `Image Stitch`: Image Stitch, used to stitch multiple images together in different directions (top, bottom, left, right).
  - `Image/Mask Converter`: used for converting between images and masks.
  - `Mask Enhancer`: an independent node for enhancing mask output.
  - `Mask Combiner`: Mask Combiner, used to combine multiple masks into one.
  - `Mask Extractor`: Mask Extractor, used to extract masks from images.

![Comfyu-rmbg_v2 2 1_node_sample](https://github.com/user-attachments/assets/68f4233c-b992-473e-aa30-ca32086f5221)

### Bug Fixes
- Fixed compatibility issues with transformers version 4.49+ dependencies.
- Fixed i18n translation errors in multiple languages.

### Improvements
- Added mask image output to each segment nodes, making mask output as images more convenient.

## V2.1.1 (2025/03/21)
Enhanced compatibility with Transformers
  - Added support for higher versions of the transformers library (≥ 4.49.0)
  - Resolved conflicts with other models requiring higher version transformers
  - Improved error handling and more user-friendly error messages
  - If you encounter issues, you can still revert to the recommended version: `pip install transformers==4.48.3`

## V2.1.0 (2025/03/19)
### New Features
The integration of internationalization (`i18n`) support significantly enhances ComfyUI-RMBG, enabling users worldwide to utilize background removal features in their preferred languages. This update fosters a more tailored and efficient workflow within ComfyUI-RMBG. The user interface has been improved to facilitate dynamic language switching according to user preferences. All newly introduced features are designed to be fully translatable, thereby improving accessibility for users who do not speak English.

# Supported Languages
| Custom Nodes `i18n` UI |
| ---------- |
| English, 中文, 日本語, Русский, 한국어, Français | 

 https://github.com/user-attachments/assets/62b80465-ba51-4c8f-b257-e3653ada0dc2

## v2.0.0 (2025/03/13)
### New Features
- Added Load Image, Preview Image, Preview Mask, and a node that previews both the image and the mask simultaneously. This is the first phase of our toolset, with more useful tools coming in future updates.
- Reorganized the code structure for better maintainability, making it easier to navigate and update.
- Renamed certain node classes to prevent conflicts with other repositories.
- Improved category organization with a new structure: 🧪AILab/🛠️UTIL/🖼️IMAGE, making tools easier to find and use.
- Integrated predefined workflows into the ComfyUI Browse Template section, allowing users to quickly load and understand each custom node’s functionality.

![image_mask_preview](https://github.com/user-attachments/assets/5e2b2679-4b63-4db1-a6c1-3b26b6f97df3)

### Technical Improvements
- Optimized utility functions for image and mask conversion
- Improved error handling and code robustness
- Updated and changed some variable names for consistency
- Enhanced compatibility with the latest ComfyUI versions

## v1.9.3 (2025/02/24)
- Clean up the code and fix the transformers version issue `transformers>=4.35.0,<=4.48.3`

## v1.9.2 (2025/02/21)
![RMBG_V1 9 2](https://github.com/user-attachments/assets/aaf51bff-931b-47ef-b20b-0dabddc49873)
### New Features
- Added Fast Foreground Color Estimation feature
  - New `refine_foreground` option for optimizing transparent backgrounds
  - Improved edge quality and detail preservation
  - Better handling of semi-transparent regions

### Technical Improvements
- Added OpenCV dependency for advanced image processing
- Enhanced foreground refinement algorithm
- Optimized memory usage for large images
- Improved edge detection accuracy

## v1.9.1 (2025/02/20)
### Technical Updates
- Changed repository for model management to the new repository
- Reorganized models files structure for better maintainability

## v1.9.0 (2025/02/19)
![rmbg_v1 9 0](https://github.com/user-attachments/assets/a7649781-42c9-4af4-94c7-6841e9395f5a)
Add and group all BiRefNet models collections into BiRefNet node.

### New BiRefNet Models Adds
- Added `BiRefNet` general purpose model (balanced performance)
- Added `BiRefNet_512x512` model (optimized for 512x512 resolution)
- Added `BiRefNet-portrait` model (optimized for portrait/human matting)
- Added `BiRefNet-matting` model (general purpose matting)
- Added `BiRefNet-HR model` (high resolution up to 2560x2560)
- Added `BiRefNet-HR-matting` model (high resolution matting)
- Added `BiRefNet_lite` model (lightweight version for faster processing)
- Added `BiRefNet_lite-2K` model (lightweight version for 2K resolution)

### Technical Improvements
- Added FP16 (half-precision) support for better performance
- Optimized for high-resolution image processing
- Enhanced memory efficiency
- Maintained compatibility with existing workflows
- Simplified model loading through Transformers pipeline

## v1.8.0 (2025/02/07)
![BiRefNet-HR](https://github.com/user-attachments/assets/c27bf3c5-92b9-472d-b097-5fed0f182d47)
** (To ensure compatibility with the old V1.8.0 workflow, we have replaced this image with the new BiRefNet Node) (2025/03/01)

### New Model Added: BiRefNet-HR
  - Added support for BiRefNet High Resolution model
  - Trained with 2048x2048 resolution images
  - Superior performance metrics (maxFm: 0.925, MAE: 0.026)
  - Better edge detection and detail preservation
  - FP16 optimization for faster processing
  - MIT License for commercial use

![BiRefNet-HR-2](https://github.com/user-attachments/assets/12441891-0330-4972-95c2-b211fce07069)
** (To ensure compatibility with the old V1.8.0 workflow, we have replaced this image with the new BiRefNet Node) (2025/03/01)

### Technical Improvements
- Added FP16 (half-precision) support for better performance
- Optimized for high-resolution image processing
- Enhanced memory efficiency
- Maintained compatibility with existing workflows
- Simplified model loading through Transformers pipeline

### Performance Comparison
- BiRefNet-HR vs other models:
  - Higher resolution support (up to 2048x2048)
  - Better edge detection accuracy
  - Improved detail preservation
  - Optimized for high-resolution images
  - More efficient memory usage with FP16 support

## v1.7.0 (2025/02/05)
![rmbg_v1 7 0](https://github.com/user-attachments/assets/22053105-f3db-4e24-be66-ae0ad2cc248e)
### New Model Added: BEN2
- Added support for BEN2 (Background Elimination Network 2)
  - Improved performance over original BEN model
  - Better edge detection and detail preservation
  - Enhanced batch processing capabilities (up to 3 images per batch)
  - Optimized memory usage and processing speed

### Model Changes
- Updated model repository paths for BEN and BEN2
- Switched to 1038lab repositories for better maintenance and updates
- Maintained full compatibility with existing workflows

### Technical Improvements
- Implemented efficient batch processing for BEN2
- Optimized memory management for large batches
- Enhanced error handling and model loading
- Improved model switching and resource cleanup

### Comparison with Previous Models
![rmbg_v1 7 0](https://github.com/user-attachments/assets/5370305e-1b31-47ad-a1b4-852991b38f45)
- BEN2 vs BEN:
  - Better edge detection
  - Improved handling of complex backgrounds
  - More efficient batch processing
  - Enhanced detail preservation
  - Faster processing speed

## v1.6.0 (2025/01/22)

### New Face Segment Custom Node
- Added a new custom node for face parsing and segmentation
  - Support for 19 facial feature categories (Skin, Nose, Eyes, Eyebrows, etc.)
  - Precise facial feature extraction and segmentation
  - Multiple feature selection for combined segmentation
  - Same parameter controls as other RMBG nodes
  - Automatic model downloading and resource management
  - Perfect for portrait editing and facial feature manipulation

![RMBG_v1 6 0](https://github.com/user-attachments/assets/9ccefec1-4370-4708-a12d-544c90888bf2)

## v1.5.0 (2025/01/05)

### New Fashion and accessories Segment Custom Node
- Added a new custom node for fashion and accessories segmentation.
  - Capable of identifying and segmenting various fashion items such as dresses, shoes, and accessories.
  - Utilizes advanced machine learning techniques for accurate segmentation.
  - Supports real-time processing for enhanced user experience.
  - Ideal for fashion-related applications, including virtual try-ons and outfit recommendations.
  - Support for gray background color.

![RMBGv_1 5 0](https://github.com/user-attachments/assets/a250c1a6-8425-4902-b902-a6e1a8bfe959)

## v1.4.0 (2025/01/02)

### New Clothes Segment Node
- Added intelligent clothes segmentation functionality
  - Support for 18 different clothing categories (Hat, Hair, Face, Sunglasses, Upper-clothes, etc.)
  - Multiple item selection for combined segmentation
  - Same parameter controls as other RMBG nodes (process_res, mask_blur, mask_offset, background options)
  - Automatic model downloading and resource management

![rmbg_v1 4 0](https://github.com/user-attachments/assets/978c168b-03a8-4937-aa03-06385f34b820)

## v1.3.2 (2024/12/29)

### Updates
- Enhanced background handling to support RGBA output when "Alpha" is selected.
- Ensured RGB output for all other background color selections.

## v1.3.1 (2024/12/25)

### Bug Fixes
- Fixed an issue with mask processing when the model returns a list of masks.
- Improved handling of image formats to prevent processing errors.

## v1.3.0 (2024/12/23)

### New Segment (RMBG) Node
- Text-Prompted Intelligent Object Segmentation
  - Use natural language prompts (e.g., "a cat", "red car") to identify and segment target objects
  - Support for multiple object detection and segmentation
  - Perfect for precise object extraction and recognition tasks

![rmbg v1.3.0](https://github.com/user-attachments/assets/7607546e-ffcb-45e2-ab90-83267292757e)

### Supported Models
- SAM (Segment Anything Model)
  - sam_vit_h: 2.56GB - Highest accuracy
  - sam_vit_l: 1.25GB - Balanced performance
  - sam_vit_b: 375MB - Lightweight option
- GroundingDINO
  - SwinT: 694MB - Fast and efficient
  - SwinB: 938MB - Higher precision

### Key Features
- Intuitive Parameter Controls
  - Threshold: Adjust detection precision
  - Mask Blur: Smooth edges
  - Mask Offset: Expand or shrink selection
  - Background Options: Alpha/Black/White/Green/Blue/Red
- Automatic Model Management
  - Auto-download models on first use
  - Smart GPU memory handling

### Usage Examples
1. Tag-Style Prompts
   - Single object: "cat"
   - Multiple objects: "cat, dog, person"
   - With attributes: "red car, blue shirt"
   - Format: Use commas to separate multiple objects (e.g., "a, b, c")

2. Natural Language Prompts
   - Simple sentence: "a person wearing a red jacket"
   - Complex scene: "a woman in a blue dress standing next to a car"
   - With location: "a cat sitting on the sofa"
   - Format: Write a natural descriptive sentence

3. Tips for Better Results
   - For Tag Style:
     - Separate objects with commas: "chair, table, lamp"
     - Add attributes before objects: "wooden chair, glass table"
     - Keep it simple and clear
   - For Natural Language:
     - Use complete sentences
     - Include details like color, position, action
     - Be as descriptive as needed
   - Parameter Adjustments:
     - Threshold: 0.25-0.35 for broad detection, 0.45-0.55 for precision
     - Use mask blur for smoother edges
     - Adjust mask offset to fine-tune selection

## v1.2.2 (2024/12/12)
![RMBG1 2 2](https://github.com/user-attachments/assets/cb7b1ad0-a2ca-4369-9401-54957af6c636)

### Improvements
- Changed INSPYRENET model format from .pth to .safetensors for:
  - Better security
  - Faster loading speed (2-3x faster)
  - Improved memory efficiency
  - Better cross-platform compatibility
- Simplified node display name for better UI integration

## v1.2.1 (2024/12/02)

### New Features
- ANPG (animated PNG), AWEBP (animated WebP) and GIF supported.

https://github.com/user-attachments/assets/40ec0b27-4fa2-4c99-9aea-5afad9ca62a5

### Bug Fixes
- Fixed video processing issue

### Performance Improvements
- Enhanced batch processing in RMBG-2.0 model
- Added support for proper batch image handling
- Improved memory efficiency by optimizing image size handling

### Technical Details
- Added original size preservation for maintaining aspect ratios
- Implemented proper batch tensor processing
- Improved error handling and code robustness
- Performance gains:
  - Single image processing: ~5-10% improvement
  - Batch processing: up to 30-50% improvement (depending on batch size and GPU)

## v1.2.0 (2024/11/29)

### Major Changes
- Combined three background removal models into one unified node
- Added support for RMBG-2.0, INSPYRENET, and BEN models
- Implemented lazy loading for models (only downloads when first used)

### Model Introduction
- RMBG-2.0 ([Homepage](https://huggingface.co/briaai/RMBG-2.0))
  - Latest version of RMBG model
  - Excellent performance on complex backgrounds
  - High accuracy in preserving fine details
  - Best for general purpose background removal

- INSPYRENET ([Homepage](https://github.com/plemeri/InSPyReNet))
  - Specialized in human portrait segmentation
  - Fast processing speed
  - Good edge detection capability
  - Ideal for portrait photos and human subjects

- BEN (Background Elimination Network) ([Homepage](https://huggingface.co/PramaLLC/BEN))
  - Robust performance on various image types
  - Good balance between speed and accuracy
  - Effective on both simple and complex scenes
  - Suitable for batch processing

### Features
- Unified interface for all three models
- Common parameters for all models:
  - Sensitivity adjustment
  - Processing resolution control
  - Mask blur and offset options
  - Multiple background color options
  - Invert output option
  - Model optimization toggle

### Improvements
- Optimized memory usage with model clearing
- Enhanced error handling and user feedback
- Added detailed tooltips for all parameters
- Improved mask post-processing

### Dependencies
- Updated all package dependencies to latest stable versions
- Added support for transparent-background package
- Optimized dependency management

## v1.1.0 (2024/11/21)

### New Features
- Added background color options
  - Alpha (transparent background)
  - Black, White, Green, Blue, Red

![RMBG_v1 1 0](https://github.com/user-attachments/assets/b7cbadff-5386-4d96-bc34-a19ad34efb4b)

- Improved mask processing
  - Better detail preservation
  - Enhanced edge quality
  - More accurate segmentation
    
![rmbg version compare](https://github.com/user-attachments/assets/8339aa8e-46db-4f11-aa7b-0a710f0a1711)

- Added video batch processing
  - Support for video file background removal
  - Maintains original video framerate and resolution
  - Multiple output format support (with Alpha channel)
  - Efficient batch processing for video frames

https://github.com/user-attachments/assets/259220d3-c148-4030-93d6-c17dd5bccee1

- Added model cache management
  - Cache status checking
  - Model memory cleanup
  - Better error handling

### Parameter Updates
- Renamed 'invert_mask' to 'invert_output' for clarity
- Added sensitivity adjustment for mask strength
- Updated tooltips for better clarity

### Technical Improvements
- Optimized image processing pipeline
- Added proper model cache verification
- Improved memory management
- Better error handling and recovery
- Enhanced batch processing performance for videos

### Dependencies
- Added timm>=0.6.12,<1.0.0 for model support
- Updated requirements.txt with version constraints

### Bug Fixes
- Fixed mask detail preservation issues
- Improved mask edge quality
- Fixed memory leaks in model handling

### Usage Notes
- The 'Alpha' background option provides transparent background
- Sensitivity parameter now controls mask strength
- Model cache is checked before each operation
- Memory is automatically cleaned when switching models
- Video processing supports various formats and maintains quality




















