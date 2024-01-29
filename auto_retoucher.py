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

    def process(self):
        pass
    def load_image(self, image):
        self.source = image
        if image.shape[0] < MIN_HEIGHT or image.shape[1] < MIN_WIDTH:
            raise gr.Error(f'Too small image. Size should be at least {MIN_WIDTH}x{MIN_HEIGHT} pixels')
            return None
        return image

    def process_all(self, ):
        if self.mask is None:
            self.generate_mask(image)

        if self.sprites_table is None:
            self.split_sprites()
        self.generate(strength, prompt, debug)

    def generate_mask(self, image, mode='figure'):
        # mode 0: figure with Mask2Former
        # mode 1: whole image mask
        print(mode, type(mode))
        #self.source = image
        self.mask = None # or np.zeroz??
        gr.Info('Generating mask...')
        if mode == 'figure':
            model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
            processor = Mask2FormerImageProcessor.from_pretrained('facebook/mask2former-swin-small-coco-instance')
            inputs = processor(image, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
            pred = processor.post_process_instance_segmentation(outputs,
                                                                target_sizes=[[image.shape[0],
                                                                               image.shape[1]]])[0]
            for key in pred['segments_info']:
                if key['label_id']==0:
                    id_person = key['id']

            self.mask = np.array([pred['segmentation'] == id_person], dtype=np.int8)[0,::]*255
            del model
            del processor
        elif mode == 'fill':
            self.mask = np.ones(image.shape[0:2])*255
            print(self.mask.shape)

        preview_image = np.array(image/2+self.mask[...,np.newaxis]/2, dtype=np.uint8)
        max_preview_size = 1024

        #ratio = min(preview_image[0]/max_preview_size, preview_image[1]/max_preview_size)

        preview_image = Image.fromarray(preview_image)
        #.thumbnail(max_preview_size)

        return preview_image

    def compile_sprites(self):
        pass
    def split_sprites(self, source, mask, img_for_grid, min_overlap=64, sprite_size=768, min_density=5):
        # This function split image to sprites with masked areas. 
        #self.mask = mask[:,:,0]
        #print(f'Mask shape{self.mask.shape}')
        #self.source = source
        #self.sprites_table = None
        self.original_imgs=[]
        gr.Info('Splitting mask...')
        width = self.mask.shape[1]
        height = self.mask.shape[0]     # Неправильный код!!!! Неверно рассчитывается перекрытие
                                        # Добавить проверку кратности 8
        sx = width // (sprite_size - min_overlap) + 1    # Num of X sprites
        sy = height // (sprite_size - min_overlap) + 1   # Num of Y sprites
        overlap_x = sprite_size - (width - sprite_size) // (sx - 1)
        overlap_y = sprite_size - (height - sprite_size) // (sy - 1)
        print(overlap_x, overlap_y)
        #print(f'Overlap X:{overlap_x}, overlap Y:{overlap_y}, Sprites X:{sx}, Sprites Y:{sy}')
        step_width = sprite_size - overlap_x
        step_height = sprite_size - overlap_y

        # Generate empty table of sprites
        sprites_table = []
        sprites_table = [[(False, (0,0))] * (sx+1) for i in range(sy+1)]

        sprites = []
        # Adding sprites to list with density criterion
        # LATER! Change list to np.array

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

        grid = Image.fromarray(img_for_grid, mode='RGB')
        draw = ImageDraw.Draw(grid)
        for s in sprites:
            self.original_imgs.append(self.source[s[1]:s[1]+sprite_size, s[0]:s[0]+sprite_size,:])
            draw.rectangle(s, width=10, outline='red')
# При повторной генерации спрайтов предыдущие удалить! Реализовать клик по спрайту для повторной генерации
        print(type(sprites), len(sprites))
        return grid
    
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
        generated_image = np.array(self.pipe(prompt, image=Image.fromarray(image),
                                        strength=strength/100.0).images[0])
        return generated_image.astype(np.uint8)

    def generate(self, strength, prompt, debug):
        #

        if not debug:
            self.load_sdxl_pipe()
            #self.pipe = StableDiffusionImg2ImgPipeline.from_single_file('models/cyberrealistic_v40.safetensors',
            #                                                            torch_dtype=torch.float16)
            #self.pipe.safety_checker=self.safety_checker

        self.generated_imgs = []
        strength = strength/100.0
        for count, temp_img in enumerate(self.original_imgs):
            if not debug:
                gr.Info(f'{count+1} of {len(self.original_imgs)+1} processed')
                generated_image = np.array(self.pipe(prompt, image=Image.fromarray(temp_img),
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

        return f'{count+1} sprites generated! {mask.shape, self.mask.shape, blurred_mask.shape, alpha_channel.size}', final_img, alpha_channel

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

with gr.Blocks(theme='gradio/monochrome', css="footer {visibility: hidden}", title='Auto Retoucher SDXL') as demo:
    with gr.Row():
        # Input panel
        with gr.Column():
            # Images
            with gr.Tabs('Preprocess') as input_tabs:
                with gr.TabItem('Input image', id=0):
                    input_image = gr.Image(sources='upload',
                                           show_download_button=False, container=False,
                                           )
                with gr.TabItem('Mask', id=1):
                    mask_image = gr.Image(sources='upload', label='Mask',
                                          show_download_button=False, container=False,
                                          show_label=False, interactive=False)
                with gr.TabItem('Mask with grid', id=2):
                    grid_image = gr.Image(sources='upload', label='Grid',
                                          show_download_button=False, container=False,
                                          show_label=False)
            with gr.Row():
                #with gr.Column():
                gen_mask = gr.Button('Make mask and grid', size='lg')
                clear_all = gr.ClearButton()
                #with gr.Column():
                gen_sprites = gr.Button('Generate', variant='primary')
                test_gen = gr.Button('Total test generate')
                #with gr.Column():

            with gr.Accordion(label='Mask and grid settings', open=True):
                with gr.Row():
                    with gr.Column():
                        fill_mask = gr.Button('Fill')
                    with gr.Column():
                        expand_mask = gr.Button('Expand mask')
                    with gr.Column():
                        concate_mask = gr.Button('Concate mask')
                    with gr.Column():
                        edit_mask = gr.Button('Edit mask')

                    min_overlap = gr.Slider(minimum=32, maximum=256, value=64,
                                            step=8, label='Minimum overlap',
                                            show_label=True, interactive=True)

                    min_density = gr.Slider(minimum=0, maximum=100, value=5, step=1,
                                            label='Minumum density',
                                            show_label=True, interactive=True)
                    sprite_size = gr.Slider(minimum=512, maximum=1536, value=768,
                                            step=128, label='Sprite size',
                                            show_label=True, interactive=True)
                    mask_blur = gr.Slider(minimum=9, maximum=256, value=15, step=1,
                                          label='Mask blur', show_label=True,
                                          interactive=True)
                    mask_expand = gr.Slider(minimum=0, maximum=100, value=10,
                                            label='Mask expand', show_label=True,
                                            interactive=True)
                    re_mesh = gr.Button('Remesh grid')

            with gr.Accordion(label='Generation settings', open=True):
                strength = gr.Slider(minimum=0, maximum=100, value=15,
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
                with gr.TabItem('Current generation', id=4):
                    cur_gen = gr.Image(value=retoucher.get_current_generation,
                                       show_label=False, interactive=False, every=None)
                with gr.TabItem('Sprites', id=5):
                    sprites_gallery = gr.Gallery(columns=4, allow_preview=False)
                    get_sprites = gr.Button('Get sprites')
                    with gr.Accordion('Generation'):
                        sprite_test_num = gr.Text(label='Sprite for generation:', value='0', show_label='True')
                        test_img = gr.Image()
                        gen_once = gr.Button('Generate once')
                        gen_once.click(fn=retoucher.process_img2img,
                                       inputs=[sprite_test_num, strength, prompt],
                                       outputs=[test_img])

            #gen_sprites = gr.Button('Generate')
            make_grid = gr.Button('Preview grid', visible=False)
            info_text = gr.Textbox(show_label=False, interactive=False)
            mask = gr.Image(visible = False)
    #input_image.clear(fn=lambda: gr.Button(interactive=False), outputs=[gen_mask])
    # add visualization of process (tqdm)
    mask_mode = gr.State(value='figure') # 'figure', 'fill'

    def show_me(ev: gr.SelectData):
        #print(f'Ev selected:{ev.selected}\nEv data:{ev._data}\nEv Value:{ev.value}\nEv index:{ev.index}\nEv target:{ev.target}')
        #print(ev.value['image']['path'])
        return ev.index, ev.value['image']['path']

    sprites_gallery.select(fn=show_me, outputs=[sprite_test_num, test_img])

    gen_mask.click(fn=retoucher.generate_mask,
                   inputs=[input_image, mask_mode],
                   outputs=[mask_image]).then(fn=retoucher.split_sprites,
                                              inputs=[input_image, mask, mask_image,
                                              min_overlap, sprite_size, min_density],
                                              outputs=grid_image)

    make_grid.click(fn=retoucher.split_sprites, inputs=[input_image, mask, out, min_overlap, sprite_size, min_density], outputs=[out])

    gen_sprites.click(fn=retoucher.generate, inputs=[strength, prompt,debug_mode],
                      outputs=[info_text, out, temp_image])

    fill_mask.click(fn=lambda: 'fill', outputs=[mask_mode]).\
        then(fn=retoucher.generate_mask, inputs=[input_image, mask_mode], outputs=[mask_image]).\
        then(fn=retoucher.split_sprites, inputs=[input_image, mask,
                                                  mask_image, min_overlap,
                                                  sprite_size, min_density],
                                          outputs=[grid_image])
    
    re_mesh.click(fn=lambda: gr.Image(visible=False), outputs=[cur_gen])                                                                                                #show_mask.click(fn=retoucher.show_mask, outputs=temp_image)
    #show_source.click(fn=retoucher.show_source, outputs=temp_image)
    # Imput image events
    input_image.clear(fn=retoucher.clear,
                      outputs=[input_image,
                               mask_image,
                               grid_image]).then(fn=lambda: gr.Tabs(selected=0),
                                                 inputs=[],outputs=[input_tabs])
    input_image.upload(fn=retoucher.load_image, inputs=[input_image], outputs=[input_image])

    # Autofocus on tabs
    get_sprites.click(fn=retoucher.get_sprites, outputs=[sprites_gallery])
    mask_image.change(fn=lambda: gr.Tabs(selected=1), inputs=[], outputs=[input_tabs])
    grid_image.change(fn=lambda: gr.Tabs(selected=2), inputs=[], outputs=[input_tabs])

    #test_gen.click(fn=retoucher.process_all, inputs=[])

if __name__ == '__main__':
    demo.queue(max_size=10)
    demo.launch()
