# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:36:49 2024

@author: Vasiliy Stepanov
"""

import gc
import os
import re
from tkinter import Tk, filedialog
import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter
import gradio as gr
import torch
from PIL import Image, ImageDraw
from transformers import Mask2FormerForUniversalSegmentation
from transformers import Mask2FormerImageProcessor
from diffusers import StableDiffusionXLImg2ImgPipeline


MIN_WIDTH = 1024
MIN_HEIGHT = 1024
PREVIEW_SIZE = 1536  # Longest size of image
MASK_SIZE = 1536  # Resize image for masking
MODELS_PATH = 'models'
OUTPUT_PATH = 'outputs'
DEFAULT_PROMPT = 'good skin, beauty, wealth, sport body, young'
DEFAULT_NEGATIVE_PROMPT = 'bad skin, ugly, creases, defects, old'
DEFAULT_TILE_SIZE = 1024
DEFAULT_MINIMUM_OVERLAP = 64
DEFAULT_MINIMUM_DENSITY = 5
DEFAULT_MASK_BLUR = 32
DEFAULT_MASK_EXPAND = 10
DEFAULT_DENOISE_STRENGTH = 10
DEFAULT_STEPS = 50
DEFAULT_BATCH_SIZE = 1
MASK_MODEL = 'facebook/mask2former-swin-base-coco-panoptic'
MASK_PROCESSOR = 'facebook/mask2former-swin-small-coco-panoptic'


def blending_mask(image, mask_size, side=''):
    ''' This function adds black gradient (0...255) for each for sides
        sides = l, r, b, t or any combination
        input: numpy RGB image (x,y,3)
        output: numpy RGB image (x,y,3) with gradient
    '''
    image = image/255.0
    height, width = image.shape[0:2]
    temp_img = np.ones(image.shape)

    for s in side:
        match s:
            case 'r':
                gradient = np.linspace(1, 0, mask_size)
                gradient = np.tile(gradient, (height, 1))[:,:,np.newaxis]
                x, y, x1, y1 = -mask_size, 0, width, height
            case 'l':
                gradient = np.linspace(0, 1, mask_size)
                gradient = np.tile(gradient, (height, 1))[:,:,np.newaxis]
                x, y, x1, y1 = 0, 0, mask_size, height
            case 't':
                gradient = np.linspace(0, 1, mask_size)
                gradient = np.repeat(gradient[:,None], width, axis=1)[:,:,np.newaxis]
                x, y, x1, y1 = 0, 0, width, mask_size
            case 'b':
                gradient = np.linspace(1, 0, mask_size)
                gradient = np.repeat(gradient[:,None], width, axis=1)[:,:,np.newaxis]
                x, y, x1, y1 = 0, height-mask_size, width, height
            case _:
                raise ValueError('Unsupported value')
        temp_img[y:y1, x:x1, :] = temp_img[y:y1, x:x1, :]*gradient

    return (image*temp_img*255).astype(np.uint8)


class AutoRetoucher:
    '''
    Class for processing high resolution image with people (for example, 40 megapixels from DSLR)
    For example, fast skin and face retouching full-sized or medium-sized portraits.

    Steps are:
        1. Detect human figure and generating mask
        2. Split image within mask in square size sprites (for SDXL up to 1024x1024px)
        3. Processing every sprite with SDXL Img2Img, with prompt.
        4. Compile sprite with overlappings in whole image
        5. Applying new image onto original background with figure' mask and blurred edges
    '''

    def __init__(self):
        self.source = None
        self.mask = None
        self.q_sx = None
        self.q_sy = None
        self.overlap_x = None
        self.overlap_y = None
        self.sprites = None
        self.sprites_table = None
        self.source = None
        self.generated_imgs = []
        self.pipe = None
        self.stop_button_pressed = False

    def get_checkpoints_list(self, path=MODELS_PATH):
        ''' Returns checkpoint's list in specified folder '''
        flist = os.listdir(path+'/')
        extentions = ['.safetensors', '.ckpt']
        checkpoints = [file for file in flist if os.path.splitext(file)[1] in extentions]
        if len(checkpoints) == 0:
            raise gr.Error('Put Stable Diffusion cpkt or safetensors checkpoints to folder!')
        self.current_checkpoint = checkpoints[0]
        return checkpoints

    def resize(self, image, max_size=1600):
        ''' Resize image to preview (mask, composite, grid and e.t.c) with value of longest size '''
        width, height = image.size[0:2]
        ratio = min(max_size/width, max_size/height)
        return image.resize(((int(width*ratio), int(height*ratio))))

    def resize_source(self, image, new_size):
        ''' Resize source image with value of shortest side'''
        image = Image.fromarray(image)
        width, height = image.size[0:2]
        ratio = max(new_size/width, new_size/height)
        image = image.resize(((int(width*ratio), int(height*ratio))))
        image = np.asanyarray(image, dtype=np.uint8)
        self.source = image
        return image

    def rotate(self, angle):
        ''' Routate source image for specified angle '''
        if self.source is not None:
            self.source = np.asarray(Image.fromarray(self.source).rotate(angle, expand=True),
                                     dtype = np.uint8)
        else:
            return None
        return self.source

    def load_image(self, image):
        ''' Functions does not returns value because updating input image is event.
            Image already loaded in gr.Image, returns only size '''

        # Copying image to self.source variable
        if type(image) is np.ndarray:
            if np.array_equal(self.source, image):
                # Image already loaded
                return image.shape[0:2]
            else:
                if image.shape[0] < MIN_HEIGHT or image.shape[1] < MIN_WIDTH:
                    gr.Warning(f'Image size should be at least {MIN_WIDTH}x{MIN_HEIGHT}\
                               pixels. Image upscaled')
                    image = self.resize_source(image, MIN_WIDTH)
                else:
                    self.source = image
            return image.shape[0:2]
        else:
            return None

    def generate_mask(self, image, mode, mask_blur, mask_expand):
        ''' Generating mask of human's figures with Mask2Former
            'figure' for human figure with Mask2Former
            'fill' for filling whole image '''
            
        self.mask = np.zeros(image.shape[0:2], dtype=np.uint8)
        id_person = None
        if mode == 'figure':
            model = Mask2FormerForUniversalSegmentation.from_pretrained(MASK_MODEL)
            processor = Mask2FormerImageProcessor.from_pretrained(MASK_PROCESSOR)
            inputs = processor(self.source, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
            pred = processor.post_process_panoptic_segmentation(outputs, label_ids_to_fuse=[0],
                                                                target_sizes=[[self.source.shape[0],
                                                                               self.source.shape[1]]])[0]
            for key in pred['segments_info']:
                if key['label_id'] == 0:
                    id_person = key['id']
            if id_person is not None:
                self.mask = np.array([pred['segmentation'] == id_person], dtype=np.uint8)[0,::]*255
            else:
                gr.Warning('Figure not found. Use fill mask or another image')
            del model
            del processor

        elif mode == 'fill':
            self.mask = np.ones(self.source.shape[0:2], dtype=np.uint8)*255

        if mask_expand > 0:
            self.mask = self.mask_dilate(self.mask, mask_expand)
        if mask_blur > 0:
            self.mask = self.mask_blur(self.mask, mask_blur)

        # Generate transparent mask over image
        composite_image = np.array(self.source/2+self.mask[...,np.newaxis]/2, dtype=np.uint8)
        composite_image = Image.fromarray(composite_image)
        preview_mask = Image.fromarray(self.mask, mode='L')
        return self.resize(composite_image), self.resize(preview_mask)

    def mask_blur(self, image, radius):
        ''' Blurring mask with gaussian filter '''
        image = gaussian_filter(image, sigma=radius/2, truncate=1)
        return image

    def mask_dilate(self, image, size, iterations=1):
        ''' Expanding mask with cv.dilate function '''
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
        image = cv.dilate(image, kernel, iterations=1)
        return image

    def generate_grid(self, min_overlap, tile_size, min_density):
        ''' Generate list of overlapped tiles '''
        
        self.original_imgs=[]  # Clearing previous tiles
        mask = self.mask.copy()
        image = self.source.copy()

        height, width = mask.shape[0:2]

        sx = width // (tile_size - min_overlap) + 1    # Num of X tiles
        sy = height // (tile_size - min_overlap) + 1   # Num of Y tiles
        overlap_x = tile_size - (width - tile_size) // (sx - 1)
        overlap_y = tile_size - (height - tile_size) // (sy - 1)
        step_width = tile_size - overlap_x
        step_height = tile_size - overlap_y

        # Generate empty table of sprites. Table format is [(bool, (y,x))]
        tiles_table = []
        tiles_table = [[(False, (0, 0))] * (sx+1) for i in range(sy+1)]
        tiles_coords = []

        # add mask covered tiles to list for processing
        for y in range(sy):
            for x in range(sx):
                xc = x*step_width
                yc = y*step_height
                lookup = mask[yc:yc+tile_size, xc:xc+tile_size]
                if lookup.sum() / (tile_size**2) > min_density:
                    tiles_coords.append((xc, yc, xc + tile_size, yc + tile_size))
                    tiles_table[y][x] = (True, (xc, yc))

        self.q_sx = sx
        self.q_sy = sy
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.sprites = tiles_coords
        self.sprites_table = tiles_table
        #self.sprite_size = tile_size # Нужен ли? Если есть список с тайлами

        grid_preview = Image.fromarray(image, mode='RGB')
        draw = ImageDraw.Draw(grid_preview)
        for s in tiles_coords:
            # Append tiles to list
            self.original_imgs.append(image[s[1]:s[1]+tile_size, s[0]:s[0]+tile_size, :])
            # Drawing for preview purposes only
            draw.rectangle(s, width=int(width/200), outline='red')
        print(f'Q of tiles:{len(tiles_coords)}')
        return self.resize(grid_preview)

    def get_tile_with_coords(self, image, tile_size, coordinates):
        x, y = coordinates
        height, width = image.shape[0:2]
        lx = int(x - tile_size/2)
        ly = int(y - tile_size/2)
        lx = 0 if lx < 0 else lx
        ly = 0 if ly < 0 else ly
        lx = width - tile_size if lx + tile_size > width else lx
        ly = height - tile_size if ly + tile_size > height else ly
        lx2 = lx+tile_size
        ly2 = ly+tile_size
        tile = image[ly:ly2, lx:lx2, :]
        return tile, lx, ly, lx2, ly2

    def load_sdxl_pipe(self):
        if self.pipe is None:
            gr.Info('Loading SDXL pipeline and model')
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                MODELS_PATH+'/'+self.current_checkpoint, torch_dtype=torch.float16,
                use_safetensors=True)
            self.pipe = self.pipe.to('cuda')
            self.pipe.enable_xformers_memory_efficient_attention()
        return

    def delete_pipe(self):
        if self.pipe is not None:
            #self.pipe = self.pipe.to('cpu')
            del self.pipe
            gc.collect()
            torch.cuda.empty_cache()
            self.pipe = None
        return

    def generate_single_tile(self, image, strength, prompt, negative_prompt, steps):
        ''' Generate a single tile '''
        self.load_sdxl_pipe()
        generated_image = np.array(self.pipe(prompt=prompt, negative_prompt=negative_prompt,
                                             image=Image.fromarray(image),
                                             strength=strength/100.0).images[0])
        return generated_image.astype(np.uint8)

    def generate(self, strength, prompt='', negative_prompt='', steps=50, batch_size=1):
        ''' Generate new images from all tiles, compiling and paste onto background with alpha '''
        self.load_sdxl_pipe()
        self.generated_imgs = []
        self.stop_button_pressed = False

        # Generating with batch num
        strength = strength/100.0
        iter_img = iter(self.original_imgs)
        next_elem = True
        temp_images_list = []

        if self.original_imgs:
            next_elem = True
            tile_size = self.original_imgs[0].shape[0]
        count = 0
        total = len(self.original_imgs)
        while next_elem:
            if self.stop_button_pressed:
                return None
            input_batch_imgs = []
            for i in range(batch_size):
                try:
                    t_i = Image.fromarray(next(iter_img))
                    input_batch_imgs.append(t_i)
                except StopIteration:
                    next_elem = False
                    break

            if input_batch_imgs:
                count = count + i + 1
                gr.Info(f'Generating {count} of {total} tiles')
                q = len(input_batch_imgs)
                gen_out = self.pipe(prompt=prompt,
                                                     negative_prompt=negative_prompt,
                                                     image=input_batch_imgs,
                                                     num_images_per_prompt=q,
                                                     num_inference_steps=steps,
                                                     strength=strength).images

                for j in gen_out:
                    self.generated_imgs.append(np.array(j))

        # Compile tiles
        height, width = self.source.shape[0:2]
        new_image = np.zeros((height, width, 3), dtype=np.uint8)
        count = 0
        for y in range(self.q_sy):
            
            temp_ir = np.zeros((tile_size, width, 3), dtype=np.uint8) # Temp image row
            for x in range(self.q_sx):
                if self.sprites_table[y][x][0]:
                    x_t, y_t = self.sprites_table[y][x][1]
                    sides = ''
                    if x>0 and x<self.q_sx-1:
                        if self.sprites_table[y][x-1][0] is True:
                            sides=sides+'l'
                        if self.sprites_table[y][x+1][0] is True:
                            sides=sides+'r'
                    else:
                        sides='r' if x==0 else 'l' if x==self.q_sx-1 else ''
                    t = blending_mask(self.generated_imgs[count], self.overlap_x, side=sides)
                    temp_ir[0:tile_size,x_t:x_t+tile_size,:]+=t
                    count = count +1

            sides = ''
            if y > 0:
                sides += 't'
            if y < self.q_sy-1:
                sides += 'b'

            if count > 0:
                new_image[y_t:y_t+tile_size,0:width,:]+=blending_mask(temp_ir,
                                                                             self.overlap_y,
                                                                             side=sides)

        # Applying new image to original background
        alpha_channel = Image.fromarray(self.mask.astype(dtype=np.uint8), mode='L')
        final_img = Image.fromarray(self.source, mode='RGB')
        processed_img = Image.fromarray(new_image, mode='RGB')
        final_img.paste(processed_img, (0, 0), alpha_channel)

        return final_img

    def paste_tile(self, image, tile, coords, blend_size, blend_mask=False):
        ''' Paste tile to image with coordinates.
            Blend mask parameter means combine figure mask with border mask '''
        x1, y1, x2, y2 = coords
        n = blend_size // 2
        border_mask = np.zeros(tile.shape[0:2], dtype=np.uint8)
        border_mask[n:tile.shape[0]-n, n:tile.shape[1]-n] = 255
        border_mask = gaussian_filter(border_mask, sigma=n-1, truncate=1)
        border_mask = Image.fromarray(border_mask, mode='L')

        if blend_mask:
            image_mask = Image.fromarray(self.mask[y1:y2, x1:x2], mode='L')
            final_mask = Image.fromarray(np.zeros(tile.shape[0:2], dtype=np.uint8), mode='L')
            final_mask.paste(image_mask, border_mask)
        else:
            final_mask = border_mask

        image = Image.fromarray(image, mode='RGB')
        tile = Image.fromarray(tile, mode='RGB')
        image.paste(tile, (coords[0:2]), final_mask)

        return np.array(image, dtype=np.uint8)

    def clear(self):
        self.sprites_table,
        self.original_imgs,
        self.mask,
        self.source = None
        return None, None, None


retoucher = AutoRetoucher()
css = '''
footer {visibility: hidden}
#generate {background-color: #DB8633; border-radius: 5px}
#blue {background-color: #2F4985; border-radius: 5px}
#stop {background-color: #960000; border-radius: 5px}
'''

theme = gr.themes.Monochrome(
    spacing_size="sm",
    radius_size="sm",
).set(
    background_fill_primary_dark='*neutral_950',
    background_fill_secondary_dark='*neutral_900'
)

with gr.Blocks(theme=theme, css=css, title='Auto Retoucher SDXL') as demo:
    '''
    Input image, mask, and sliced srites saved in AutoRetoucher class for speed and memory
    optimization, because Gradio works slow with high-resolution images.
    Class sends back resized images, except generation output.
    You can change size of preview image in global variable.
    '''
    with gr.Row():
        # Input panel
        with gr.Column():
            # Images
            with gr.Tabs('Preprocess') as input_tabs:

                with gr.TabItem('Input image', id=0):
                    input_image = gr.Image(sources='upload',
                                           show_download_button=False, container=False,
                                           label='test', show_label=True)
                with gr.TabItem('Composite mask', id=1):
                    mask_preview = gr.Image(sources='upload', label='Composite mask',
                                          show_download_button=False, container=False,
                                          show_label=False, interactive=False)
                with gr.TabItem('Mask'):
                    mask = gr.Image(visible = True, sources='upload', label='Mask',
                                          show_download_button=False, container=False,
                                          show_label=False, interactive=False)
                '''
                with gr.TabItem('Edit mask'):
                    edit_mask = gr.ImageEditor(sources='upload', transforms=(),
                                               eraser=gr.Eraser(default_size=20),
                                               crop_size=None,
                                               brush=gr.Brush(default_size=20,
                                                              colors=["#FFFFFF"],
                                                              color_mode="fixed"))
                '''
                with gr.TabItem('Grid', id=2):
                    grid_image = gr.Image(sources='upload', label='Grid',
                                          show_download_button=False, container=False,
                                          show_label=False, interactive=False)
                with gr.TabItem('Batch processing'):
                    batch_files = gr.File(file_count='multiple', file_types=['image'])
                    batch_files_bt = gr.Button('Batch processing with current setting',
                                               elem_id='generate')

                    with gr.Row():
                        with gr.Column():
                            out_file_format = gr.Radio(choices = ['JPG', 'PNG'], value='PNG',
                                                   show_label=False, interactive=True)
                        with gr.Column():
                            select_folder = gr.Button('Select outputs folder')
                            cwd = re.sub('\\\\','/', os.getcwd() + OUTPUT_PATH)
                        select_folder_show = gr.Text(label='Outputs folder:',
                                                     value = cwd,
                                                     interactive=False, show_label=True)

            with gr.Row():
                with gr.Column():
                    generate = gr.Button('GENERATE', interactive=False, elem_id='generate')
                    stop_button = gr.Button('STOP', interactive=True, elem_id='stop')
                with gr.Column():
                    reset_all_bt = gr.Button('RESET ALL', elem_id='clear')

            with gr.Accordion(label='Image and mask processing', open=True):
                with gr.Row():
                    rotate_left = gr.Button('◀ Rotate left', elem_id='blue')
                    rotate_right = gr.Button('Rotate Right ▶', elem_id='blue')
                with gr.Row():
                    resize_image = gr.Button('Resize Image', elem_id='blue')
                    copy_input_to_output = gr.Button('Copy input to output', elem_id='blue')
                with gr.Row():
                    resize_image_size = gr.Slider(minimum = 1024, maximum = 8192, value=2048,
                                                  step=64, label='Shortest side',
                                                  show_label=True, interactive=True)
                    current_size = gr.Text(show_label=False, interactive=False)
                    resize_batch = gr.Checkbox(label='Resize batch', value=False)
                with gr.Row():
                    with gr.Column():
                        mask_blur = gr.Slider(minimum=0, maximum=256,
                                              value=DEFAULT_MASK_BLUR, step=1,
                                              label='Mask blur', show_label=True,
                                              interactive=True)
                    with gr.Column():
                        mask_expand = gr.Slider(minimum=0, maximum=100, value=DEFAULT_MASK_EXPAND,
                                                label='Mask expand', show_label=True,
                                                interactive=True)
                with gr.Row():
                    with gr.Column():
                        fill_mask = gr.Button('Fill mask', elem_id='blue')
                    with gr.Column():
                        make_mask = gr.Button('Body mask', elem_id='blue')

            with gr.Accordion(label='Grid processing', open=False):
                with gr.Row():
                    with gr.Column():
                        min_overlap = gr.Slider(minimum=32, maximum=256,
                                                value=DEFAULT_MINIMUM_OVERLAP,
                                                step=8, label='Minimum overlap',
                                                show_label=True, interactive=True)
                        min_density = gr.Slider(minimum=0, maximum=100,
                                                value=DEFAULT_MINIMUM_DENSITY,
                                                step=1, label='Minumum density',
                                                show_label=True, interactive=True)
                    with gr.Column():
                        sprite_size = gr.Slider(minimum=512, maximum=1536,
                                                value=DEFAULT_TILE_SIZE,
                                                step=128, label='Tile size',
                                                show_label=True, interactive=True)
                        remesh_grid = gr.Button('Remesh grid', elem_id='blue')

            with gr.Accordion(label='Generation settings', open=True):
                with gr.Row():
                    with gr.Column():
                        strength = gr.Slider(minimum=0, maximum=100, value=10,
                                             label='Denoise strength', show_label=True,
                                             interactive=True)
                        steps = gr.Slider(minimum=1, maximum=100, value=50, step=1,
                                          label='Steps', show_label=True, interactive=True)

                    with gr.Column():
                        batch_size = gr.Slider(minimum=1, maximum=16, value=1, step=1,
                                               label='Batch size', show_label=True,
                                               interactive=True)
                        opacity = gr.Slider(minimum=1, maximum=100, value=100, step=1,
                                            label='Opacity', show_label=True, interactive=True)
                        debug_mode = gr.Checkbox(label='Debug mode (do not generate)', value=False)

                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(label='Prompt',
                                            lines=2,
                                            value=DEFAULT_PROMPT)
                        # use_prompt_embed_bt = gr.Checkbox(label='Use embeddings', value=False)

                    with gr.Column():
                        negative_prompt = gr.Textbox(label='Negative prompt',
                                                     lines=2,
                                                     value=DEFAULT_NEGATIVE_PROMPT)
                        # use_neg_prompt_embed_bt = gr.Checkbox(label='Use negative embeddings', value=False)
                with gr.Row():
                    checkpoints_list = retoucher.get_checkpoints_list()
                    checkpoints = gr.Dropdown(choices=checkpoints_list,
                                              value=checkpoints_list[0],
                                              label='SDXL checkpoint',
                                              interactive=True)

        # Output panel
        with gr.Column():
            with gr.Tabs('Output') as out_tabs:
                with gr.TabItem('Output image', id=3):
                    out = gr.Image(show_label=False, interactive=False, show_download_button=True)
                    with gr.Accordion('ReGEN custom tile'):
                        with gr.Tabs() as single_tile:
                            with gr.TabItem('Before', id=18):
                                tile_for_regen = gr.Image(show_label=False, sources=None,
                                                          interactive=False)
                            with gr.TabItem('After', id=19):
                                tile_after = gr.Image(show_label=False, sources=None,
                                                          interactive=False)
                        with gr.Row():
                            with gr.Column():
                                apply_mask_for_tile = gr.Checkbox(label='Apply generated mask',
                                                                  value=True)
                            with gr.Column():
                                custom_tile_mask_value = gr.Slider(label='Tile masking', value=64,
                                                                   minimum=0, maximum=256,
                                                                   show_label=True,
                                                                   interactive=True)
                        with gr.Row():
                            with gr.Column():
                                regen_custom_tile = gr.Button('GENERATE', elem_id='generate',
                                                              )
                                tile_coords = gr.State()
                            with gr.Column():
                                apply_custom_tile = gr.Button('APPLY', elem_id='blue')

                            with gr.Row():
                                with gr.Column():
                                    ct_prompt = gr.Textbox(label='Prompt', lines=2,
                                                           value=DEFAULT_PROMPT)
                                with gr.Column():
                                    ct_negative_prompt = gr.Textbox(label='Negative prompt',
                                                                    lines=2,
                                                                    value=DEFAULT_NEGATIVE_PROMPT)

    # add visualization of process (tqdm)
    mask_mode = gr.State(value='figure')
    mask_exist = gr.State(value=False)
    grid_exist = gr.State(value=False)

    # Image processing -------------------------
    def rotate_left_fn():
        image = retoucher.rotate(angle=90)
        return image
    rotate_left.click(fn=rotate_left_fn, outputs=[input_image], concurrency_id='fn')

    def rotate_right_fn():
        image = retoucher.rotate(angle=270)
        return image
    rotate_right.click(fn=rotate_right_fn, outputs=[input_image], concurrency_id='fn')

    def resize_image_fn(image, size):
        if type(image) is np.ndarray:
            image = retoucher.resize_source(image, size)
            gr.Warning('Multiple resizing can worse image quality')
            return image
        return None
    resize_image.click(fn=resize_image_fn, inputs=[input_image, resize_image_size],
                       outputs=[input_image], concurrency_id='fn')

    copy_input_to_output.click(fn=lambda x: x, inputs=[input_image], outputs=[out])

    # Mask and grid processing ---------------------------
    def fill_mask_fn(input_image, min_overlap, sprite_size, min_density):
        gr.Info('Fill mask')
        composite_mask_preview, mask_preview = retoucher.generate_mask(input_image, mode='fill',
                                                                       mask_blur=0, mask_expand=0)
        gr.Info('Generating grid')
        grid_image = retoucher.generate_grid(min_overlap, sprite_size, min_density)
        return composite_mask_preview, mask_preview, grid_image

    fill_mask.click(fn=fill_mask_fn, inputs=[input_image, min_overlap, sprite_size, min_density],
                    outputs=[mask_preview, mask, grid_image], concurrency_id='fn')

    def grid_generate(min_overlap, sprite_size, min_density):

        gr.Info('Generating grid')
        return retoucher.generate_grid(min_overlap, sprite_size, min_density)

    def process_mask_and_grid(input_image, mask_preview, min_overlap, sprite_size, min_density,
                              mask_blur, mask_expand):
        input_image_size = (0, 0)
        if type(input_image) is not np.ndarray:
            gr.Info('Mask cleared')
            return None, None, None, gr.Text('Image not loaded'), gr.Button(interactive=False)
        else:
            input_image_size = retoucher.load_image(input_image)

        gr.Info('Generating mask')
        mask_mode = 'figure'
        mask_preview, mask = retoucher.generate_mask(input_image, mask_mode, mask_blur, mask_expand)
        mask_preview = np.asanyarray(mask_preview, dtype=np.uint8)
        grid_image = grid_generate(min_overlap, sprite_size, min_density)

        return mask_preview, mask, grid_image, gr.Text(input_image_size), gr.Button(interactive=True)

    # Event for loading or clearing input image
    input_image.change(fn=process_mask_and_grid,
                       inputs=[input_image, mask_preview, min_overlap, sprite_size, min_density,
                               mask_blur, mask_expand],
                       outputs=[mask_preview, mask, grid_image, current_size, generate],
                       concurrency_id='fn')

    # Remesh grid
    remesh_grid.click(fn=grid_generate, inputs=[min_overlap, sprite_size, min_density],
                      outputs=[grid_image], concurrency_id='fn')

    # Generating image
    def generate_fn(strength, prompt, negative_prompt, steps, batch_size):
        # disable generate button during generate
        output = retoucher.generate(strength, prompt, negative_prompt, steps, batch_size)
        return output

    # add checking mask and grid
    generate.click(fn=generate_fn, inputs=[strength, prompt, negative_prompt, steps, batch_size],
                   outputs=[out], concurrency_id='fn')

    def stop_fn():
        gr.Warning('Stopping generation')
        retoucher.stop_button_pressed = True
        return None

    stop_button.click(fn=stop_fn)

    def load_checkpoint(checkpoint_name):
        retoucher.delete_pipe()
        retoucher.current_checkpoint = checkpoint_name
        retoucher.load_sdxl_pipe()
        return checkpoint_name

    checkpoints.input(fn=load_checkpoint, inputs=[checkpoints], outputs=[checkpoints],
                      concurrency_id='fn')

    # Batch files processing ----------------------
    def select_folder_fn():
        root = Tk()
        root.attributes("-topmost", True)
        root.withdraw()
        save_file_path = filedialog.askdirectory()
        root.destroy()
        if os.path.isdir(save_file_path):
            return str(save_file_path)
        else:
            return None
    select_folder.click(fn=select_folder_fn, outputs=[select_folder_show])

    def batch_generation(batch_files, strength, prompt, negative_prompt, steps, batch_size,
                         opacity, min_overlap, sprite_size, min_density, out_file_format,
                         select_folder_show):

        for file in batch_files:
            # Loading image
            image = np.asarray(Image.open(file), dtype=np.uint8)
            retoucher.load_image(image)

            # Generate mask
            mask_mode = 'figure'
            mask_preview, mask = retoucher.generate_mask(input_image, mask_mode)

            # Generate Grid
            mask_preview = np.asanyarray(mask_preview, dtype=np.uint8)
            grid_image = grid_generate(mask_preview, min_overlap, sprite_size, min_density)

            # Process Image
            gr.Info(f'Processing {os.path.basename(file)}')
            output = retoucher.generate(strength, prompt, negative_prompt, steps, batch_size)

            # Save image
            fname, fext = os.path.splitext(os.path.basename(file))
            out_fname = select_folder_show + '/' + fname + '_processed.' + out_file_format.lower()

            try:
                output.save(out_fname)
            except gr.Error(f'Error writing {out_fname}'):
                return None
        return output

    batch_files_bt.click(fn=batch_generation, inputs=[batch_files, strength, prompt,
                                                      negative_prompt, steps, batch_size,
                                                      opacity, min_overlap, sprite_size,
                                                      min_density, out_file_format,
                                                      select_folder_show], outputs=[out],
                         concurrency_id='fn')

    def reset_all_fn():
        return None
    reset_all_bt.click(fn=reset_all_fn)

    # Custom tile section ----------------------
    def get_custom_tile(ev: gr.SelectData, image, sprite_size):
        tile, x1, y1, x2, y2 = retoucher.get_tile_with_coords(image,
                                                              sprite_size,
                                                              (ev.index[0], ev.index[1]))
        return tile, (x1, y1, x2, y2)

    def generate_custom_tile_fn(image, strength, prompt, negative_prompt, steps):
        if type(image) is np.ndarray:
            gr.Info('Generating single tile')
            image = retoucher.generate_single_tile(image, strength, prompt,
                                                   negative_prompt, steps)
        else:
            gr.Warning('Select tile from source or generated image!')
        return image

    def apply_custom_tile_fn(image, tile, coords, custom_tile_mask_value, apply_base_mask):
        if type(tile) is not np.ndarray:
            gr.Warning('Nothing to apply')
            return image
        if type(image) is not np.ndarray:
            gr.Warning('Output image not generated')
            return None
        image = retoucher.paste_tile(image, tile, coords,
                                     custom_tile_mask_value, apply_base_mask)
        return image

    apply_custom_tile.click(fn=apply_custom_tile_fn, inputs=[out,
                                                             tile_after,
                                                             tile_coords,
                                                             custom_tile_mask_value,
                                                             apply_mask_for_tile],
                            outputs=[out])

    regen_custom_tile.click(fn=lambda: gr.Tabs(selected=19), outputs=[single_tile]).\
        then(fn=generate_custom_tile_fn, inputs=[tile_for_regen, strength,
                                                 ct_prompt,
                                                 ct_negative_prompt,
                                                 steps],
             outputs=[tile_after], show_progress='full', concurrency_id='fn')

    out.select(fn=get_custom_tile, inputs=[out, sprite_size],
               outputs=[tile_for_regen, tile_coords]).then(fn=lambda: gr.Tabs(selected=18),
                                                           outputs=[single_tile])

    input_image.select(fn=get_custom_tile, inputs=[input_image, sprite_size],
                       outputs=[tile_for_regen, tile_coords]).then(fn=lambda: gr.Tabs(selected=18),
                                                                   outputs=[single_tile])

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1)
    demo.launch()
