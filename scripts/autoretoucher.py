# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:26:38 2024

@author: Pussy
"""
import numpy as np
from PIL import Image
from typing import List, Tuple

from scripts.imageutils import compose_image_and_mask, make_grid, draw_grid
from scripts.imageutils import image_blur, image_dilate
from scripts.imageutils import image_ss_resize, image_ls_resize
from scripts.imageutils import get_tiles_from_image, blending_mask

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
        overlaps = None
        tiles = None
        tiles_bool = None
        tiles_coords = None



class AutoRetoucher():
    def __init__(self, config, mask_predict, sdxl_pipe):
        self.config = config
        self.mask_generator = mask_predict
        self.sdxl_pipe = sdxl_pipe
        self.canvas = Canvas()

    def generate_mask_and_grid(self, image, minimum_overlap, tile_size, minimum_density,
                               mask_blur, mask_expand, label_id=0, preview=True):
        ''' generate mask and grid if image was changed. If 'preview' parameter is True,
            returns resized images of composed mask, mask and grid '''
        if not isinstance(image, np.ndarray):
            return None, None, None, None

        current_image_size = image.shape[0:2]
        # Check ratio
        if max(current_image_size)/min(current_image_size) > self.config.image_maximum_aspect_ratio:
            return None, None, None, None

        # Check minimum and maximum size
        if max(current_image_size) > self.config.image_maximum_size:
            image = image_ls_resize(image, self.config.image_maximum_size)
            current_image_size = image.shape[0:2]

        if min(current_image_size) < tile_size:
            image = image_ss_resize(image, tile_size)
            current_image_size = image.shape[0:2]

        mask = self.mask_generator.predict(image, label_id)
        mask = image_blur(mask, mask_blur)
        mask = image_dilate(mask, mask_expand)

        self.canvas.image = image
        self.canvas.mask = mask

        composed_mask = compose_image_and_mask(image, mask) if preview else None

        grid_preview = self.generate_grid(minimum_overlap, tile_size, minimum_density,
                                          preview=preview, image=self.canvas.image)
        if preview:
            return (image_ls_resize(composed_mask, self.config.image_preview_size),
                    image_ls_resize(mask, self.config.image_preview_size),
                    image_ls_resize(grid_preview, self.config.image_preview_size),
                    current_image_size
                    )
        return None

    def generate_grid(self, minimum_overlap, tile_size, minimum_density, preview=True, image=None):
        (self.canvas.tiles,
         self.canvas.overlaps,
         self.canvas.tiles_bool,
         self.canvas.tiles_coords) = make_grid(self.canvas.image, self.canvas.mask,
                                               minimum_overlap,
                                               tile_size,
                                               minimum_density)
        if preview:
            grid_preview = draw_grid(image, self.canvas.tiles_coords)
            return grid_preview

        return None

    def generate_image(self, image, strength, prompt, negative_prompt, steps, batch_size,
                       save_file, save_format, add_modelname):

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
        return final_img

    def fill_mask(self):
        if not isinstance(self.canvas.mask, np.ndarray):
            return None
        self.canvas.mask.fill(255)
        return image_ls_resize(self.canvas.mask, self.config.image_preview_size)

    def invert_mask(self):
        pass

    def generate_mask(self):
        pass
    def remesh_grid(self):
        pass

    def stop_generation(self):
        pass
    def load_checkpoint(self):
        pass
    def select_folder_for_batch(self):
        pass
    def generate_batch_files(self):
        pass
    def apply_custom_tile(self):
        pass
    def generate_custom_tile(self):
        pass
    def reset_all(self):
        pass
    def save_output_image(self):
        pass


