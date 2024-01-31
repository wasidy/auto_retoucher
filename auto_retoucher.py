# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:36:49 2024

@author: Vasiliy Stepanov
"""

import gradio as gr
import sys
import torch
from PIL import Image, ImageDraw
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from diffusers.utils import load_image
import numpy as np
import matplotlib.patches as patches
import cv2 as cv
from transformers import Mask2FormerForUniversalSegmentation
from transformers import AutoImageProcessor
from transformers import Mask2FormerModel, Mask2FormerImageProcessor

from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from scipy.ndimage import gaussian_filter


MIN_WIDTH = 1024
MIN_HEIGHT = 1024
PREVIEW_SIZE = 1600 #Longest size of image


def blending_mask(image, mask_size, side=''):
    ''' This function adds black gradient (0...1) for each for sides
        sides = l, r, b, t or any combination
        input: numpy RGB image (x,y,3)
        output: numpy RGB image (x,y,3) with gradient
    '''
    image = image/255.0
    width = image.shape[1]
    height = image.shape[0]
    temp_img = np.ones((height, width, 3))

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


class MaskGenerator:
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
#        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
#        self.processor = Mask2FormerImageProcessor.from_pretrained('facebook/mask2former-swin-small-coco-instance')
        self.mask = ''
        #self.pipe = ''
        #self.pipe = pipe.to("cuda")

    def generate_mask(self, image):
        model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
        processor = Mask2FormerImageProcessor.from_pretrained('facebook/mask2former-swin-small-coco-instance')
        self.mask = ''

        #image = Image.open(input_image)
        inputs = processor(image, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        #print(image.shape[0], image.shape[1])
        pred = processor.post_process_instance_segmentation(outputs,
                                                            target_sizes=[[image.shape[0],
                                                                           image.shape[1]]])[0]

        for key in pred['segments_info']:
            if key['label_id']==0:
                id_person = key['id']

        self.mask = np.array([pred['segmentation'] == id_person], dtype=np.int8)[0,::]*255
        #print(type(image), image.shape, self.mask.shape)
        #image = np.array(image, dtype=np.uint8)
        show_image = np.array(image/2+self.mask[...,np.newaxis]/2, dtype=np.uint8)

        return show_image, self.mask


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
        self.positive_prompt = ''
        self.negative_prompt = ''
        self.strength = 0.05
        self.source = None
        self.mask = None
        self.q_sx = None
        self.q_sy = None
        self.overlap_x = None
        self.overlap_y = None
        self.sprites = None
        self.sprites_table = None
        #self.original_imgs=[]
        self.source = None
        self.generated_imgs = []
        self.height = 0
        self.width = 0
        self.current_generation = None
        self.pipe = None
        pass
    def preview_grid(self):
        pass

    def resize(self, image, max_size=1600):
        width, height = image.size[0:2]
        ratio = min(max_size/width, max_size/height)
        return image.resize(((int(width*ratio), int(height*ratio))))

    def resize_source(self, image, new_size):
        image = Image.fromarray(image)
        width, height = image.size[0:2]
        ratio = max(new_size/width, new_size/height)
        image = image.resize(((int(width*ratio), int(height*ratio))))
        print(f'New size W:{image.size[0]}, H:{image.size[1]}')
        image = np.asanyarray(image, dtype=np.uint8)
        self.source = image
        return image

    def process(self):
        pass

    def rotate(self, angle):
        if self.source is not None:
            self.source=np.asarray(Image.fromarray(self.source).rotate(angle, expand=True),
                                   dtype=np.uint8)
        else:
            return None
        return self.resize(Image.fromarray(self.source))

    def load_image(self, image):
        if np.array_equal(self.source, image):
            # Image already loaded
            return image.shape[0:2]
        else:
            self.source = image
            if image.shape[0] < MIN_HEIGHT or image.shape[1] < MIN_WIDTH:
                gr.Warning(f'Too small image. Size should be at least {MIN_WIDTH}x{MIN_HEIGHT} pixels. Image not loaded')
                return None
        # Functions does not returns value because updating input image is event
        return image.shape[0:2]
        #return self.resize(Image.fromarray(image))

    def process_all(self, ):
        if self.mask is None:
            self.generate_mask(image)

        if self.sprites_table is None:
            self.split_sprites()
        self.generate(strength, prompt, debug)
        return None

    def generate_mask(self, image, mode='figure'):
        # mode 0: figure with Mask2Former
        # mode 1: whole image mask
        #print(mode, type(mode))
        # self.source = image
        self.mask = None # or np.zeroz??
        
        if mode == 'figure':
            model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
            processor = Mask2FormerImageProcessor.from_pretrained('facebook/mask2former-swin-small-coco-instance')
            inputs = processor(self.source, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
            pred = processor.post_process_instance_segmentation(outputs,
                                                                target_sizes=[[self.source.shape[0],
                                                                               self.source.shape[1]]])[0]
            for key in pred['segments_info']:
                if key['label_id']==0:
                    id_person = key['id']

            self.mask = np.array([pred['segmentation'] == id_person], dtype=np.uint8)[0,::]*255
            del model
            del processor
        elif mode == 'fill':
            self.mask = np.ones(self.source.shape[0:2], dtype=np.uint8)*255
            print(self.mask.shape)

        # Generate transparent mask over image
        preview_image = np.array(self.source/2+self.mask[...,np.newaxis]/2, dtype=np.uint8)
        preview_image = Image.fromarray(preview_image)
        preview_mask = Image.fromarray(self.mask, mode='L')

        return self.resize(preview_image), self.resize(preview_mask)


    def generate_grid(self, img_for_grid, min_overlap, sprite_size, min_density):
        # This function split image to sprites with masked areas.

        self.original_imgs=[]

        
        width = self.mask.shape[1]
        height = self.mask.shape[0]     # Неправильный код!!!! Неверно рассчитывается перекрытие
                                        # Добавить проверку кратности 8
        sx = width // (sprite_size - min_overlap) + 1    # Num of X sprites
        sy = height // (sprite_size - min_overlap) + 1   # Num of Y sprites
        overlap_x = sprite_size - (width - sprite_size) // (sx - 1)
        overlap_y = sprite_size - (height - sprite_size) // (sy - 1)

        #print(f'Overlap X:{overlap_x}, overlap Y:{overlap_y}, Sprites X:{sx}, Sprites Y:{sy}')
        step_width = sprite_size - overlap_x
        step_height = sprite_size - overlap_y

        # Generate empty table of sprites. Table format is [(bool, (y,x))]
        sprites_table = []
        sprites_table = [[(False, (0,0))] * (sx+1) for i in range(sy+1)]

        sprites = []

        for y in range(sy):
            for x in range(sx):
                xc = x*step_width
                yc = y*step_height
                lookup = self.mask[yc:yc+sprite_size, xc:xc+sprite_size]
                if lookup.sum() / (sprite_size**2) > min_density:
                    sprites.append((xc, yc, xc + sprite_size, yc + sprite_size))
                    sprites_table[y][x] = (True, (xc,yc))
        self.q_sx = sx
        self.q_sy = sy
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.sprites = sprites
        self.sprites_table = sprites_table
        self.sprite_size = sprite_size

        grid = Image.fromarray(self.source, mode='RGB')
        draw = ImageDraw.Draw(grid)
        for s in sprites:
            self.original_imgs.append(self.source[s[1]:s[1]+sprite_size, s[0]:s[0]+sprite_size,:])
            draw.rectangle(s, width=int(width/200), outline='red')
# При повторной генерации спрайтов предыдущие удалить! Реализовать клик по спрайту для повторной генерации
        print(type(sprites), len(sprites))
        return self.resize(grid)


    def get_sprites(self):
        sprites_tuple = []
        for n, s in enumerate(self.original_imgs):
            sprites_tuple.append((s,str(n)))
        return sprites_tuple

    def safety_checker(self, images, clip_input):
        return images, [False]

    def show_mask(self):
        return self.mask

    def show_source(self):
        return self.source


    def load_sdxl_pipe(self):
        if self.pipe is None:
            gr.Info('Loading SDXL pipeline and model')
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                "models/juggernautXL_v7FP16VAEFix.safetensors", torch_dtype=torch.float16,
                use_safetensors=True)
            self.pipe = self.pipe.to("cuda")
            self.pipe.enable_xformers_memory_efficient_attention()
        return

    def get_current_generation(self):
        # Only for showing process
        print('Beep!')
        return self.current_generation

    def process_img2img(self, num_image, strength, prompt):
        # Generating a single image/batch
        self.load_sdxl_pipe()
        image = self.original_imgs[int(num_image)]
        print(type(image))

        generated_image = np.array(self.pipe(prompt, image=Image.fromarray(image),
                                        strength=strength/100.0).images[0])
        return generated_image.astype(np.uint8)

    def generate(self, strength, prompt, debug=False):
        #

        if not debug:
            self.load_sdxl_pipe()
            #self.pipe = StableDiffusionImg2ImgPipeline.from_single_file('models/cyberrealistic_v40.safetensors',
            #                                                            torch_dtype=torch.float16)
            #self.pipe.safety_checker=self.safety_checker

        self.generated_imgs = []
        negative_prompt='tatoo'
        strength = strength/100.0
        for count, temp_img in enumerate(self.original_imgs):
            if not debug:
                gr.Info(f'{count+1} of {len(self.original_imgs)+1} processed')
                generated_image = np.array(self.pipe(prompt=prompt, negative_prompt=negative_prompt, image=Image.fromarray(temp_img),
                                                strength=strength).images[0])
                self.current_generation = generated_image.copy()
            else:
                generated_image = np.ones((self.sprite_size,self.sprite_size,3), dtype=np.uint8)*127
            self.generated_imgs.append(generated_image)
        ################ Compile
        height, width = self.source.shape[0:2]
        new_image = np.zeros((height, width, 3), dtype=np.uint8)
        #gr.Info(f'{len(sprites_table)} Sprites')
        count = 0
        for y in range(self.q_sy):
            temp_ir = np.zeros((self.sprite_size, width, 3), dtype=np.uint8) # Temp image row
            for x in range(self.q_sx):
                if self.sprites_table[y][x][0]:
                    x_t, y_t = self.sprites_table[y][x][1]

                    sides = ''
                    if x>0 and x<self.q_sx-1:
                        if self.sprites_table[y][x-1][0]==True: sides=sides+'l'
                        if self.sprites_table[y][x+1][0]==True: sides=sides+'r'
                    else:
                        sides='r' if x==0 else 'l' if x==self.q_sx-1 else ''
                    #print(x_t, y_t)

                    t = blending_mask(self.generated_imgs[count], self.overlap_x, side=sides)

                    temp_ir[0:self.sprite_size,x_t:x_t+self.sprite_size,:]+=t

                    count = count +1
            sides=''
            if y>0: sides+='t'
            if y<self.q_sy-1: sides+='b'
            #sides = 't' if y>0 else 'b' if y<q_sy-1 else 'bt'
            if count>0:
                new_image[y_t:y_t+self.sprite_size,0:width,:]+=blending_mask(temp_ir,self.overlap_y, side=sides)

        # Blur mask and convert it to PIL alpha
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        mask = cv.dilate(self.mask, kernel, iterations=1)
        blurred_mask = gaussian_filter(mask, sigma=15)
        alpha_channel = Image.fromarray(blurred_mask.astype(dtype=np.uint8), mode='L')

        # Applying new image to original background
        final_img = Image.fromarray(self.source, mode='RGB')
        processed_img = Image.fromarray(new_image, mode='RGB')
        final_img.paste(processed_img, (0, 0), alpha_channel)

        return final_img

    def prepare_output(self):
        pass

    def fill_mask(self, image):
        self.mask = np.ones((image.shape[0],image.shape[1]))*255


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
'''

with gr.Blocks(theme='gradio/monochrome', css=css, title='Auto Retoucher SDXL') as demo:

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
                with gr.TabItem('Edit mask'):
                    edit_mask = gr.ImageEditor(sources='upload', transforms=(),
                                               eraser=gr.Eraser(default_size=20),
                                               crop_size=None,
                                               brush=gr.Brush(default_size=20,
                                                              colors=["#FFFFFF"],
                                                              color_mode="fixed"))
                with gr.TabItem('Grid', id=2):
                    grid_image = gr.Image(sources='upload', label='Grid',
                                          show_download_button=False, container=False,
                                          show_label=False, interactive=False)
                with gr.TabItem('Batch processing'):
                    i = gr.File(file_count='multiple', file_types=['image'])
            with gr.Row():
                
                generate = gr.Button('GENERATE BEAUTY', interactive=False, elem_id='generate')
                
                clear_all = gr.Button('CLEAR ALL', elem_id='clear')
                #with gr.Column():


                #with gr.Column():

            with gr.Accordion(label='Image and mask processing', open=True):
                with gr.Row():
                    
                    
                    rotate_left = gr.Button('◀ Rotate left', elem_id='blue')
                    rotate_right = gr.Button('Rotate Right ▶', elem_id='blue')
                    
                with gr.Row():
                    resize_image = gr.Button('Resize Image', elem_id='blue')
                with gr.Row():
                    resize_image_size = gr.Slider(minimum = 1024, maximum = 8192, value=2048,
                                                  step=64, label='Shortest side',
                                                  show_label=True, interactive=True)
                    
                    current_size = gr.Text(show_label=False, interactive=False)
                    resize_batch = gr.Checkbox(label='Resize batch', value=False)
                with gr.Row():
                    fill_mask = gr.Button('Fill mask', elem_id='blue')
    
                    edit_mask = gr.Button('Edit mask', elem_id='blue', interactive=False)
            with gr.Accordion(label='Grid processing'):
                    min_overlap = gr.Slider(minimum=32, maximum=256, value=64,
                                            step=8, label='Minimum overlap',
                                            show_label=True, interactive=True)

                    min_density = gr.Slider(minimum=0, maximum=100, value=5, step=1,
                                            label='Minumum density',
                                            show_label=True, interactive=True)
                    sprite_size = gr.Slider(minimum=512, maximum=1536, value=1024,
                                            step=128, label='Sprite size',
                                            show_label=True, interactive=True)
                    mask_blur = gr.Slider(minimum=9, maximum=256, value=15, step=1,
                                          label='Mask blur', show_label=True,
                                          interactive=True)
                    mask_expand = gr.Slider(minimum=0, maximum=100, value=10,
                                            label='Mask expand', show_label=True,
                                            interactive=True)
                    remesh_grid = gr.Button('Remesh grid')

            with gr.Accordion(label='Generation settings', open=True):
                strength = gr.Slider(minimum=0, maximum=100, value=10,
                                     label='Denoise strength', show_label=True,
                                     interactive=True)


                steps = gr.Slider(minimum=1, maximum=100, value=50, step=1,
                                  label='Steps', show_label=True, interactive=True)
                batch_size = gr.Slider(minimum=1, maximum=16, value=1, step=1,
                                       label='Batch size', show_label=True, interactive=True)
                opacity = gr.Slider(minimum=1, maximum=100, value=1, step=1,
                                    label='Opacity', show_label=True, interactive=True)
                debug_mode = gr.Checkbox(label='Debug mode (do not generate)', value=False)
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(label='Prompt')
                    with gr.Column():
                        negative_prompt = gr.Textbox(label='Negative prompt')

            temp_image = gr.Image(visible=False)

        # Output panel
        with gr.Column():
            with gr.Tabs('Output') as out_tabs:
                with gr.TabItem('Output image', id=3):
                    out = gr.Image(show_label=False, interactive=False, show_download_button=True)

                with gr.TabItem('Sprites', id=5):
                    sprites_gallery = gr.Gallery(columns=4, allow_preview=False)
                    get_sprites = gr.Button('Get sprites')
                    with gr.Accordion('Generation'):
                        sprite_test_num = gr.Text(label='Sprite for generation:', value='0', show_label='True')
                        with gr.Tabs('Preview') as preview:
                            with gr.TabItem('Before'):
                                test_img = gr.Image(show_label=False, interactive=False)
                            with gr.TabItem('After'):
                                processed_test_img = gr.Image(show_label=False, interactive=False)
                        gen_once = gr.Button('GENERATE ONCE', elem_id='generate', interactive=False)
                        gen_once.click(fn=retoucher.process_img2img,
                                       inputs=[sprite_test_num, strength, prompt],
                                       outputs=[processed_test_img], concurrency_id='fn',
                                       scroll_to_output=True)
                with gr.TabItem('ReGen selected'):
                    pass

            #gen_sprites = gr.Button('Generate')
            make_grid = gr.Button('Preview grid', visible=False)
            info_text = gr.Textbox(show_label=False, interactive=False)

    
    # add visualization of process (tqdm)
    mask_mode = gr.State(value='figure') # 'figure', 'fill'

    def show_me(ev: gr.SelectData):
        #print(f'Ev selected:{ev.selected}\nEv data:{ev._data}\nEv Value:{ev.value}\nEv index:{ev.index}\nEv target:{ev.target}')
        #print(ev.value['image']['path'])
        return ev.index, ev.value['image']['path']
    sprites_gallery.select(fn=show_me, outputs=[sprite_test_num, test_img])

    

    def rotate_left_fn():
        image=retoucher.rotate(angle=90)
        return image
    rotate_left.click(fn=rotate_left_fn, outputs=[input_image], concurrency_id='fn')

    def rotate_right_fn():
        image=retoucher.rotate(angle=270)
        return image
    rotate_right.click(fn=rotate_right_fn, outputs=[input_image], concurrency_id='fn')

    def resize_image_fn(image, size):
        if type(image) is np.ndarray:
            image = retoucher.resize_source(image, size)
            return image
        return None
    resize_image.click(fn=resize_image_fn, inputs=[input_image, resize_image_size],
                       outputs=[input_image], concurrency_id='fn')
    
    #def remesh_grid_fn(mask_preview, min_overlap, sprite_size, min_density):
    #    if type(mask_preview) is not np.ndarray:
    #        return None
    #    return retoucher.generate_grid(mask_preview, min_overlap, sprite_size, min_density)
    
    #remesh_grid.click(fn=remesh_grid_fn, inputs=[mask_preview, min_overlap, sprite_size, min_density],
    #                  outputs=[grid_image], concurrency_id='fn')


    '''
    fill_mask.click(fn=lambda: 'fill', outputs=[mask_mode]).\
        then(fn=retoucher.generate_mask, inputs=[input_image, mask_mode], outputs=[mask_image]).\
        then(fn=retoucher.split_sprites, inputs=[input_image, mask,
                                                  mask_image, min_overlap,
                                                  sprite_size, min_density],
                                          outputs=[grid_image])
    '''
    #re_mesh.click(fn=lambda: gr.Image(visible=False), outputs=[cur_gen])                                                                                                #show_mask.click(fn=retoucher.show_mask, outputs=temp_image)
    #show_source.click(fn=retoucher.show_source, outputs=temp_image)
    # Imput image events
 
    # Autofocus on tabs
    get_sprites.click(fn=retoucher.get_sprites, outputs=[sprites_gallery])
    #mask_image.change(fn=lambda: gr.Tabs(selected=1), inputs=[], outputs=[input_tabs])
    #grid_image.change(fn=lambda: gr.Tabs(selected=2), inputs=[], outputs=[input_tabs])


    def grid_generate(mask_preview, min_overlap, sprite_size, min_density):
        gr.Info('Generating grid')
        return retoucher.generate_grid(mask_preview, min_overlap, sprite_size, min_density)

    # Delete Mask_preview from sending to makegrid
    def process_mask_and_grid(input_image, mask_preview, min_overlap, sprite_size, min_density):
        input_image_size=(0,0)
        if type(input_image) is not np.ndarray:
            gr.Info('Mask cleared')
            return None, None, None, gr.Text('Image not loaded'), gr.Button(interactive=False)
        else:
            input_image_size = retoucher.load_image(input_image)
        
        gr.Info('Generating mask')
        mask_mode = 'figure'
        mask_preview, mask = retoucher.generate_mask(input_image, mask_mode)
        mask_preview = np.asanyarray(mask_preview, dtype=np.uint8)
        grid_image=grid_generate(mask_preview, min_overlap, sprite_size, min_density)
        #button_enable = gr.Button(interactive=True)
        return mask_preview, mask, grid_image, gr.Text(input_image_size), gr.Button(interactive=True)

    # Event for loading or clearing input image        
    input_image.change(fn=process_mask_and_grid,
                       inputs=[input_image, mask_preview, min_overlap, sprite_size, min_density],
                       outputs=[mask_preview, mask, grid_image, current_size, generate], concurrency_id='fn')
    
    # Remesh grid
    remesh_grid.click(fn=grid_generate, inputs=[mask_preview, min_overlap,
                                                sprite_size, min_density],
                      outputs=[grid_image], concurrency_id='fn')

    # Auto generate mask, grid and output
    def check_mask_and_grid(input_image, mask_preview, mask, grid_image, strength,
                           prompt, negative_prompt,
                           steps, batch_size, opacity,
                           min_overlap, sprite_size, min_density):
        
        if type(input_image) is not np.ndarray:
            gr.Warning('Image not loaded')
            return None

        if type(mask_preview) is not np.ndarray:
            gr.Info('Creating mask and grid')
            mask_mode = 'figure'
            #mask_preview, mask = retoucher.generate_mask(input_image, mask_mode)
            #mask_preview = np.asanyarray(mask_preview, dtype=np.uint8)
            mask_preview, mask = mask_generate(input_image)
        if type(grid_image) is not np.ndarray:
            
            grid_image = retoucher.generate_grid(mask_preview, min_overlap, sprite_size, min_density)
        output = retoucher.generate(strength, prompt)

        return output, mask_preview, mask, grid_image

    def generate_fn(strength, prompt):
        # disable generate button during generate
        output = retoucher.generate(strength, prompt)
        return output
    
    #add checking mask and grid
    
    generate.click(fn=generate_fn, inputs=[strength, prompt], outputs=[out], concurrency_id='fn')
    
    #test_gen.click(fn=auto_process_image,
    #               inputs=[input_image, mask_preview, mask, grid_image, 
    #                       strength, prompt, negative_prompt,
    #                       steps, batch_size, opacity,
    #                       min_overlap, sprite_size, min_density],
    #               outputs=[out, mask_preview, mask, grid_image], concurrency_id='fn')

    #gen_sprites.click(fn=retoucher.generate, inputs=[strength, prompt,debug_mode],
    #                  outputs=[out])


if __name__ == '__main__':
    demo.queue(default_concurrency_limit=1)
    demo.launch()
