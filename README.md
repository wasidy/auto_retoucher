## AutoRetoucher - SDXL-based application for fast and quality retouching high-resolution photos
### Requirements:
- Python 3.10
- Videocard with CUDA and minimum 12Gb VRAM
  
### Installation:
- copy files to folder and run `install.bat`
- copy SDXL checkpoints to 'models' folder
- launch `run.bat` and open 127.0.0.1:7860 in your browser 

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
- the application is also suitable for retouching portraits, but I recommend to resize it smaller for best result
- for the final you can load PNG file into Photoshop, as a new layer over original image, and apply mask for some areas

### Interface
![image](https://github.com/wasidy/auto_retoucher/assets/122546017/64003051-4b47-47f2-89a1-787be22cadee)

1. Input image. Click for load source file. If file too small, it will be upscaled.
2. Main control buttons. <kbd>GENERATE</kbd> start generation of all tiles, <kbd>STOP</kbd> stops generation without errors, <kbd>Reset all parameters</kbd> resets all sliders to default values.
3. Positive and negative prompt for whole image. Be carefully with specified prompt. Some tiles do not correspondent prompt, especially at large denoising values.
4. Image processing. You can <kbd>ROTATE</kbd> your image, <kbd>RESIZE</kbd> for short side value. <kbd>COPY INPUT TO OUTPUT</kbd> allows you to generate and paste custom tiles without whole generation. <kbd>RESIZE BATCH</kbd> checkbox means resizing all batch images to specified size. <kbd>MASK BLUR</kbd> determines value for blurring mask before final paste with alpha on background. <kbd>MASK EXPAND</kbd> is size of expansion mask before blurring. <kbd>FILL MASK</kbd> fills whole image with mask. For examle, you want to proceed background, or humag figure was not recognized. <kbd>REMAKE FIGURE MASK</kbd> regenerates mask with new blur and expand parameters.
5. Grid processing. <kbd>Minimum overlap</kbd> is minimum overlap between tiles. Too small value decreases amount of tiles, but can worse quality of mixing tiles. <kbd>Tile size</kbd> is size of grid's tile. 1024x1024 optiman for SDXL, but another values is allowed. <kbd>Minimum density</kbd> is value for grid computation. If too smal part of masked image in tile, there is no necessary to gerenate it. Value in percent. <kbd>REMESH GRID</kbd> recalculates grid with new values.
6. Generation settings. <kbd>Denoise strength</kbd> is denoising strength for Img2Img pipeline. For skin retouching is fine value between 5-10. Higher values can do more affect for regenerating, but it may couse mismatch lines, countoures and texture in different tiles. <kbd>Batch size</kbd> is number of simultaneously generated tiles. It depends of your videocard memory and tile size. <kbd>Steps</kbd> is amount of generation steps. More steps - better quality. 50 is enough.
7. Output image. After generation you can save PNG file with clicking RMB.
8. Regenerate custom tile. As source for regenerating you can click LMB on any area input OR output image. Tile will be loaded in window.
9. <kbd>Tile masking</kbd> is value for box alpha. <kdb>Apply generated mask</kbd> means that the generation result will be applied only to the area on which the base mask was generated. <kbd>APPLY TO IMAGE</kbd> will paste generated tile to original coordinates onto output image, with blur settings. <kbd>Prompt</kbd> and <kbd>Negative prompt</kdb> is custom prompt for single tile generation.
---
#### Sample images are copyrighted by Vasiliy Stepanov.
