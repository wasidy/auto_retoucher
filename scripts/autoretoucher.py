# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:26:38 2024

@author: Pussy
"""
import numpy as np
import os
from PIL import Image
from PIL import ImageEnhance
from typing import List, Tuple

from scripts.imageutils import compose_image_and_mask, make_grid, draw_grid
from scripts.imageutils import image_blur, image_dilate, split_mask_to_blobs
from scripts.imageutils import image_ss_resize, image_ls_resize, get_custom_tile
from scripts.imageutils import get_tiles_from_image, blending_mask, gradient_box
from scripts.imageutils import blend_with_alpha, skin_mask_generate

import cv2

class Canvas():
    ''' Class describing canvas for retoucher
        image: source np.array image
        mask: generated mask
        tiles: tuple (X,Y) with size of grid
        overlaps: value of tile overlaps for X and Y
        tiles_coords: list of tuples (X,Y,X1,Y1), with coordinates tile with mask
        tiles_boolean: List of tuples with boolean values()
    '''

    def __init__(self):
        image: np.ndarray = None
        mask: np.ndarray = None
        face_mask = None
        skin_mask: np.ndarray = None
        overlaps = None
        tiles = None
        tiles_bool = None
        tiles_coords = None

class FaceCoords():
    def __init__(self, center_x, center_y, width, height, area):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.area = area

def process_all():
    ''' steps of auto proceeding:
    loading image
    Change image size
    
    generate mask with settings
    generate grid with settings
    regenerate mask
    regenerate grid
    
    '''
    pass

class AutoRetoucher():
    def __init__(self, config, mask_predict, clip_predict, sdxl_pipe, face_detector):
        self.config = config
        self.mask_generator = mask_predict
        self.clip_generator = clip_predict
        self.sdxl_pipe = sdxl_pipe
        self.face_detector = face_detector
        self.canvas = Canvas()

    def step_clear_image(self):
        self.canvas.image = None
        self.canvas.mask = None
        self.canvas.face_mask = None
        self.canvas.skin_mask = None
        self.overlaps = None
        self.tiles = None
        self.tiles_bool = None
        self.tiles_coords = None
        print('Image and masks cleared')
        return None

    # Load image step
    def load_image(self,
                   image,
                   resize_image_at_load=False,
                   image_size=None):
        ''' Loading image to canvas, resize if smaller or bigger '''

        if not isinstance(image, np.ndarray):
            print('Image not loaded')
            return None, None, None, None
        current_image_size = image.shape[0:2]
        # Check ratio
        if max(current_image_size)/min(current_image_size) > self.config.image_maximum_aspect_ratio:
            print('Aspect ration exceed config value!')
            return None, None, None, None
        # Check minimum and maximum size
        if resize_image_at_load:
            # Правильно ли условие?
            image_size = self.config.image_default_size if None else image_size
            print(f'Resizing image to {image_size} at shortest side')
            image = image_ss_resize(image, image_size)
        else:
            if max(current_image_size) > self.config.image_maximum_size:
                print(f'Image too large. Resized to {self.config.image_maximum_size}')
                image = image_ls_resize(image, self.config.image_maximum_size)
            if min(current_image_size) < self.config.image_minimum_size:
                print(f'Image too small. Resized to {self.config.image_minimum_size}')
                image = image_ss_resize(image, self.config.image_minimum_size)
        current_image_size = image.shape[0:2]
        self.canvas.image = image
        return image, current_image_size

    # Generate mask step
    def generate_mask(self,
                      mask_mode,
                      label_id,
                      mask_smooth,
                      figure_mask_expand,
                      figure_mask_blur,
                      face_mask_threshold=127,
                      apply_skin_mask=True,
                      use_standart_skin_tones=False,
                      skin_mask_hue_threshold=2,
                      skin_mask_sat_threshold=2,
                      skin_mask_val_threshold=2,
                      skin_mask_expand=0,
                      skin_mask_blur=0,
                      preview=True
                      ):
        if not isinstance(self.canvas.image, np.ndarray):
            print('Image not loaded!')
            return None
        
        # Generating mask
        print(f'Mask mode: {mask_mode}')
        current_image_size = self.canvas.image.shape[0:2]
        if mask_mode == 'Mask2Former':
            mask = self.mask_generator.predict(self.canvas.image, label_id)
            if mask is None:
                print('Object not fount, maks is None. Generated fill mask')
                mask = np.zeros(current_image_size, dtype=np.uint8)
            else:
                pass
                #mask = image_dilate(mask, mask_expand)
                #mask = image_blur(mask, mask_blur)
            if apply_skin_mask:
                pass
        
        elif mask_mode == 'Fill':
            mask = np.ones(current_image_size, dtype=np.uint8)*255
        elif mask_mode == 'Faces':
            mask, faces = self.face_detector.detect_faces(self.canvas.image,
                                                          expand_value=1.4,
                                                          threshold=face_mask_threshold,
                                                          mask_smooth=mask_smooth,
                                                          mask_expand=figure_mask_expand)
            # face_mask = self.clip_generator.predict(image=self.canvas.image,
            #                                    prompt='face',
            #                                    threshold=face_mask_threshold)
            # mask = self.process_blobs(face_mask)
        else:
            mask = None
            
        composed_mask = compose_image_and_mask(self.canvas.image, mask) if preview else None
        return composed_mask, mask
    
    
    
    def process_blobs(self, image):
        faces, _ = split_mask_to_blobs(image)
        kernel_size = 10
        smooth_size=50
        blur_size=30
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        fin_image = np.zeros(image.shape[0:2], dtype=np.uint8)
        for i in faces:
            i = cv2.dilate(i, kernel, iterations=1)
            i = cv2.blur(i, (smooth_size, smooth_size))
            _, i = cv2.threshold(i, 127, 255, cv2.THRESH_BINARY)
            i = cv2.blur(i, (blur_size, blur_size))
            fin_image = fin_image + i
        
        return fin_image
    
    def old_load_image(self,
                   image,
                   tile_size,
                   minimum_overlap,
                   minimum_density,
                   mask_blur,
                   mask_expand,
                   label_id=0,
                   mask_mode='Mask2Former',
                   apply_skin_mask=True,
                   resize_image_size=None,
                   resize_image_at_load=True,
                   skin_mask_hue_threshold=2,
                   skin_mask_sat_threshold=2,
                   skin_mask_val_threshold=2,
                   preview=True):
        ''' generate mask and grid if image was changed. If 'preview' parameter is True,
            returns resized images of composed mask, mask and grid '''
        if not isinstance(image, np.ndarray):
            print('Image not loaded')
            return None, None, None, None

        current_image_size = image.shape[0:2]
        # Check ratio
        if max(current_image_size)/min(current_image_size) > self.config.image_maximum_aspect_ratio:
            print('Aspect ration exceed config value!')
            return None, None, None, None

        # Check minimum and maximum size
        if resize_image_at_load:
            # Правильно ли условие?
            resize_image_size = self.config.image_default_size if None else resize_image_size
            print(f'Resizing image to {resize_image_size} at shortest side')
            image = image_ss_resize(image, resize_image_size)
        else:
            if max(current_image_size) > self.config.image_maximum_size:
                print(f'Image too large. Resize to {self.config.image_maximum_size}')
                image = image_ls_resize(image, self.config.image_maximum_size)
            if min(current_image_size) < self.config.image_minimum_size:
                print(f'Image too small. Resize to {self.config.image_minimum_size}')
                image = image_ss_resize(image, self.config.image_minimum_size)

        current_image_size = image.shape[0:2]
        print(f'Mask mode: {mask_mode}')
        if mask_mode == 'Mask2Former':
            mask = self.mask_generator.predict(image, label_id)
            if mask is None:
                print('Object not fount, maks is None. Use fill mask')
                mask = np.zeros(current_image_size, dtype=np.uint8)
            else:
                pass
                #mask = image_dilate(mask, mask_expand)
                #mask = image_blur(mask, mask_blur)
        elif mask_mode == 'Fill':
            mask = np.ones(image.shape[0:2], dtype=np.uint8)*255
        else:
            mask = None
        # Detect faces

        
        if apply_skin_mask:
            
            face, face_mask, coords = self.generate_clip_mask(image,
                                                              'face',
                                                              tile_size,
                                                              scale_to_tile=True)
            skin_mask_image = skin_mask_generate(image,
                                                 face,
                                                 face_mask,
                                                 skin_mask_hue_threshold,
                                                 skin_mask_sat_threshold,
                                                 skin_mask_val_threshold)
            
            mask = np.logical_and(skin_mask_image, mask).astype(np.uint8)
            mask = mask * 255
            #mask = skin_mask_image*255
        
        

        self.canvas.image = image
        self.canvas.mask = mask

        q_f = len(self.detect_faces())
        print("Num of faces: ", q_f)                

        composed_mask = compose_image_and_mask(image, mask) if preview else None

        grid_preview = self.generate_grid(minimum_overlap, tile_size, minimum_density,
                                          image=self.canvas.image, preview=preview)

        if preview:
            return (self.canvas.image,
                    image_ls_resize(composed_mask, self.config.image_preview_size),
                    image_ls_resize(mask, self.config.image_preview_size),
                    
                    grid_preview,
                    current_image_size
                    )
        return None
    

    def resize_image(self, size, minimum_ovelrap, tile_size, minimum_density):
        ''' resize image and mask '''
        self.canvas.image = image_ss_resize(self.canvas.image, size)
        self.canvas.mask = image_ss_resize(self.canvas.mask, size)
        preview_grid = self.generate_grid(minimum_ovelrap, tile_size,
                                          minimum_density, self.canvas.image, True)

        return self.canvas.image, preview_grid, self.canvas.image.shape[0:2]

    
    def generate_grid(self, minimum_overlap, tile_size, minimum_density, image=None, preview=True):
        ''' If preview is false, image will not be generated '''

        (self.canvas.tiles,
         self.canvas.overlaps,
         self.canvas.tiles_bool,
         self.canvas.tiles_coords) = make_grid(self.canvas.image, self.canvas.mask,
                                               minimum_overlap,
                                               tile_size,
                                               minimum_density)

        if preview:
            grid_preview = draw_grid(image, self.canvas.tiles_coords)
            return image_ls_resize(grid_preview, self.config.image_preview_size)

        return None

    def generate_image(self, image, strength, prompt, negative_prompt, steps, batch_size,
                       save_file=None, save_format=None, add_modelname=None):

        # Заменить self.tiles_coords на данные из компонента градио?
        source_tiles = get_tiles_from_image(image, self.canvas.tiles_coords)
        generated_tiles = self.sdxl_pipe.generate_batch_images(source_tiles, strength, prompt,
                                                               negative_prompt, steps, batch_size)

        # Compile tiles
        qsx, qsy = self.canvas.tiles  # Amount of tiles in image, X,Y
        overlap_x, overlap_y = self.canvas.overlaps
        tiles_table = self.canvas.tiles_bool

        height, width = image.shape[0:2]
        new_image = np.zeros((height, width, 3), dtype=np.uint8)


        # Merging cell in fucntion!
        count = 0
        if len(generated_tiles) > 0:
            tile_size = generated_tiles[0].shape[0]
            for y in range(qsy):
                temp_row = np.zeros((tile_size, width, 3), dtype=np.uint8)
                for x in range(qsx):
                    if tiles_table[y][x][0] is True:
                        x_t, y_t = tiles_table[y][x][1]  # If tiles is generated, add coords
                        sides = ''
                        if 0 < x < qsx-1:
                            if tiles_table[y][x-1][0] is True:
                                sides = sides+'l'
                            if tiles_table[y][x+1][0] is True:
                                sides = sides+'r'
                        else:
                            sides = 'r' if x == 0 else 'l' if x == qsx-1 else ''
                        t = blending_mask(generated_tiles[count], overlap_x, side=sides)
                        temp_row[0:tile_size, x_t:x_t+tile_size, :] += t
                        count = count + 1

                sides = ''
                if y > 0:
                    sides += 't'
                if y < qsy-1:
                    sides += 'b'

                if count > 0:
                    new_image[y_t:y_t+tile_size, 0:width, :] += blending_mask(temp_row,
                                                                              overlap_y,
                                                                              side=sides)


        # Applying new image to original background
        alpha_channel = Image.fromarray(self.canvas.mask.astype(dtype=np.uint8), mode='L')
        final_img = Image.fromarray(image, mode='RGB')
        processed_img = Image.fromarray(new_image, mode='RGB')
        final_img.paste(processed_img, (0, 0), alpha_channel)
        final_img = np.array(final_img, dtype=np.uint8)
        
        face_process_required = True
        faces = self.detect_faces()
        print("Num of faces: ", len(faces))    
        final_img = self.generate_faces(final_img, faces, tile_size)
        
        t_image = Image.fromarray(final_img)
        enhancer = ImageEnhance.Sharpness(t_image)
        sharped = enhancer.enhance(2)
        final_img = np.array(sharped, dtype=np.uint8)
        
        return final_img

    #

    def generate_faces(self, image, faces, tile_size, prompt='', negative_prompt='', strength=10, steps=50):
        
        result = image
        for i, face in enumerate(faces):
            print(f'Generating face #{i+1}')
            face_size = max(face.width, face.height)
            print(f'Get face from {face.center_x, face.center_y}')
            img, x1, y1, x2, y2 = get_custom_tile(self.canvas.image,
                                                  face_size,
                                                  (face.center_x, face.center_y))
            print(f'Get face from {x1, y1, x2, y2}')
            q = face_size/tile_size
            img_for_gen = image_ss_resize(img, tile_size)
            face_tile_size = self.generate_custom_tile(img_for_gen,
                                                       strength,
                                                       prompt,
                                                       negative_prompt,
                                                       steps,
                                                       tile_size)
            
            generated_face = image_ss_resize(face_tile_size, face_size)
            result = self.apply_custom_tile(result,
                                            generated_face,
                                            (x1,y1,x2,y2),
                                            custom_tile_border_mask_value=64,
                                            apply_base_mask=False)
        return result
            
            
            

    def generate_custom_tile(self, image, strength, prompt, negative_prompt,
                             steps, tile_size, additional_mask=None):
        
        image_size = image.shape[0]
        
        resized = False
        if image_size != tile_size:
            temp_image = image_ss_resize(image, tile_size)
            resized = True
        else:
            temp_image = image
        
        result = self.sdxl_pipe.generate_single_image(
            temp_image,
            strength,
            prompt,
            negative_prompt,
            steps)
        
        if resized:
            result = image_ss_resize(result, image_size)
        
        if additional_mask:
            if additional_mask.shape[0:2] == result.shape[0:2]:
                result = blend_with_alpha(image, result, additional_mask)
        
        return result

    def apply_custom_tile(self, image, tile, coords,
                          custom_tile_border_mask_value, apply_base_mask):
        
        if not isinstance(image, np.ndarray):
            return None
        x1, y1, x2, y2 = coords
        border_mask = gradient_box(tile.shape[0], custom_tile_border_mask_value)
        border_mask = Image.fromarray(border_mask, mode='L')
        
        if apply_base_mask:
            image_mask = Image.fromarray(self.canvas.mask[y1:y2, x1:x2], mode='L')
            final_mask = Image.fromarray(np.zeros(tile.shape[0:2], dtype=np.uint8), mode='L')
            final_mask.paste(image_mask, border_mask)
        else:
            final_mask = border_mask

        image = Image.fromarray(image, mode='RGB')
        tile = Image.fromarray(tile, mode='RGB')
        image.paste(tile, (coords[0:2]), final_mask)
        image = np.array(image, dtype=np.uint8)
        
        return image

    def batch_generation(self, images_file_list, strength, prompt, negative_prompt,
                         steps, batch_size,
                         minimum_overlap, tile_size, minimum_density, file_format,
                         output_folder, resize_batch, resize_batch_size, mask_blur,
                         mask_expand, label_id, mask_batch_mode, fill_mask_if_not_detected):
        ''' Batch generation with specified parameters '''
        for file in images_file_list:
            # Loading image
            try:
                image = Image.open(file)
                print(f'{file} loaded')
            except Warning:
                print(f'Error loading {file}')
                continue

            image = np.array(image, dtype=np.uint8)
            if resize_batch:
                image = image_ss_resize(image, resize_batch_size)
                print(f'Image resized to {image.shape[0:2]}')

            if mask_batch_mode == 'Mask2Former':
                print('Generating mask')
                self.generate_mask_and_grid(image, minimum_overlap, tile_size, minimum_density,
                                            mask_blur, mask_expand, label_id, preview=False)
                if not self.canvas.mask.any() > 0 and fill_mask_if_not_detected:
                    print('Object not detected. Using fill mask')
                    self.canvas.mask.fill
            else:
                self.canvas.mask.fill(255)
            
            print('Generating image')
            output = self.generate_image(self.canvas.image, strength, prompt,
                                         negative_prompt, steps, batch_size)

            # Save image
            fname, fext = os.path.splitext(os.path.basename(file))
            out_fname = output_folder + fname + '_.' + file_format.lower()

            try:
                output.save(out_fname)
                print(f'Image saved to {out_fname}')
            except Warning:
                print(f'Error saving {out_fname}')

        return output

    def fill_mask(self, preview=True):
        if not isinstance(self.canvas.mask, np.ndarray):
            return None
        self.canvas.mask.fill(255)
        return image_ls_resize(self.canvas.mask, self.config.image_preview_size)

    def detect_faces(self):
        mask = self.clip_generator.predict(self.canvas.image, 'face')
        faces = []
        if mask.max() != 255:
            print('No face detected')
            return None
        else:
            face_masks, face_coords = split_mask_to_blobs(mask)
            for i, m in enumerate(face_masks):
                faces.append(FaceCoords(face_coords[i][1][0],  # Center X
                                        face_coords[i][1][1],  # Center Y
                                        face_coords[i][0][2],  # Width
                                        face_coords[i][0][3],  # Height
                                        face_coords[i][0][4]   # Area
                                        ))
            print(f'Detected {len(faces)} faces')
            self.canvas.face_mask = mask
        return faces
        
       

    def generate_clip_mask(self, image, keyword, tile_size, scale_to_tile):
        
        labeled_mask = self.clip_generator.predict(image, keyword)
        
        masks, coords = split_mask_to_blobs(labeled_mask)
        # Add processing mask for dilating and smooting with parameters
        
        mask_size = 0
        for i, m in enumerate(masks):
            print('Face width, height, sqr: ', coords[i][0][2:5])
            if m.sum() > mask_size:
                
                max_mask_idx = i
        
                    
        x_center = int(coords[max_mask_idx][1][1])
        y_center = int(coords[max_mask_idx][1][0])
        
        #print(x_center, y_center, tile_size)
        crop_tile_size = tile_size
        # добавить проверку на минимальный размер
        
        if scale_to_tile:
            x, y, w, h, _ = coords[max_mask_idx][0]
            print(f'Width, height of clip mask: {w, h}')
            if ( w > tile_size or h > tile_size) or (w < tile_size/2 or h <tile_size/2):
                crop_tile_size = int(max(w, h)*1.2)
                #print(f'Tile of clipmask size: {crop_tile_size}')
                #if crop_tile_size > tile_size:
                #    crop_tile_size = tile_size
        
        blur_size = crop_tile_size//32
                    
        tile, lx, ly, lx2, ly2 = get_custom_tile(image, crop_tile_size, (y_center, x_center))
        tile_mask = masks[max_mask_idx][ly:ly2, lx:lx2]
        tile_mask = image_blur(tile_mask, 32)
        #print(f'Tile mask shape: {tile_mask.shape}')
        return tile, tile_mask, (lx, ly, lx2, ly2)

    def get_custom_tile_and_mask(self, tile_size, coordinates):
        
        tile, x1, y1, x2, y2 = get_custom_tile(self.canvas.image,
                                               tile_size,
                                               coordinates)
        mask = self.canvas.mask[y1:y2, x1:x2]
        return tile, mask, x1, y1, x2, y2


    def invert_mask(self):
        pass

    def stop_generation(self):
        pass
    def load_checkpoint(self):
        pass

    def reset_all(self):
        pass
    def save_output_image(self):
        pass
    
    def test_cfg(self, data):
        print(data)
        

