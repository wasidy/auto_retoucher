## AutoRetoucher - SDXL-based application for fast and quality retouching high-resolution photos
### Requirements:
- Python 3.10
- Videocard with CUDA and minimum 12Gb VRAM
  
### Installation:
- Copy files to folder and run `pip install -r requirements.txt` (venv recommended)

### Usage
This application is mainly used for retouching portraits with a lot of exposed body areas that require careful retouching, such as shootings in lingerie, swimsuits or explicit erotic.
Manual retouching with frequency separation requires considerable time and high qualification of a retoucher.
Base SDXL Img2Img pipeline allows you to get quality results, but it limited to optimal size of image, 1024х1024 for example.
AutoRetoucher creates a perfect result with the following steps:
- creating mask for human figure on image (using Mask2Former with fusing masks if more than one person)
- splitting source image with mask to tiles size of 1024x1024 px (or you can select another value, but SDXL trained on this size)
- generating new tiles with specified parameters of generation (denoise steps, and generation steps) and common prompt
- compiling result tiles with blending and paste new image to original background with generated mask
- after generation you can regenerate any part of image with custom prompt and paste it again
- also batching is supported

### Interface
