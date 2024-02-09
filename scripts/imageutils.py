import numpy as np
import cv2
from typing import List


def split_mask_to_parts(image):
    ''' Input - labeled 2D array
        Output - list of 2D ndarrays with splitted masks, list of tuples with stats and centers '''

    masks: List = None
    coords: List = None
    blobs, labels, stats, centers = cv2.connectedComponentsWithStats(image)

    for i in range(1, blobs):
        l_mask = np.array(labels == i).astype(np.uint8)
        masks.append(l_mask*255)
        coords.append((stats[i], centers[i]))
    return masks, coords


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
                gradient = np.tile(gradient, (height, 1))[:, :, np.newaxis]
                x, y, x1, y1 = -mask_size, 0, width, height
            case 'l':
                gradient = np.linspace(0, 1, mask_size)
                gradient = np.tile(gradient, (height, 1))[:, :, np.newaxis]
                x, y, x1, y1 = 0, 0, mask_size, height
            case 't':
                gradient = np.linspace(0, 1, mask_size)
                gradient = np.repeat(gradient[:, None], width, axis=1)[:, :, np.newaxis]
                x, y, x1, y1 = 0, 0, width, mask_size
            case 'b':
                gradient = np.linspace(1, 0, mask_size)
                gradient = np.repeat(gradient[:, None], width, axis=1)[:, :, np.newaxis]
                x, y, x1, y1 = 0, height-mask_size, width, height
            case _:
                raise ValueError('Unsupported value')
        temp_img[y:y1, x:x1, :] = temp_img[y:y1, x:x1, :]*gradient

    return (image*temp_img*255).astype(np.uint8)


def add_black_gradient(image, gradient_size, side=''):
    ''' This function adds black gradient (0...255) for each for sides
        sides = l, r, b, t or any combination
        input: numpy RGB image (x,y,1)
        output: numpy RGB image (x,y,1) with gradient
    '''
    image = image/255.0
    height, width = image.shape[0:2]
    temp_img = np.ones(image.shape)

    for s in side:
        match s:
            case 'r':
                gradient = np.linspace(1, 0, gradient_size)
                gradient = np.tile(gradient, (height, 1))[:, :]
                x, y, x1, y1 = -gradient_size, 0, width, height
            case 'l':
                gradient = np.linspace(0, 1, gradient_size)
                gradient = np.tile(gradient, (height, 1))[:, :]
                x, y, x1, y1 = 0, 0, gradient_size, height
            case 't':
                gradient = np.linspace(0, 1, gradient_size)
                gradient = np.repeat(gradient[:, None], width, axis=1)[:, :]
                x, y, x1, y1 = 0, 0, width, gradient_size
            case 'b':
                gradient = np.linspace(1, 0, gradient_size)
                gradient = np.repeat(gradient[:, None], width, axis=1)[:, :]
                x, y, x1, y1 = 0, height-gradient_size, width, height
            case _:
                raise ValueError('Unsupported value')
        temp_img[y:y1, x:x1] = temp_img[y:y1, x:x1]*gradient
    return (image*temp_img*255).astype(np.uint8)


def image_ls_resize(image, longside_size):
    ''' Resize image to preview (mask, composite, grid and e.t.c) with value of LONGEST size
        Input: numpy array, longest_size. Interpolation is BICUBIC '''
    height, width = image.shape[0:2]
    ratio = min(longside_size/height, longside_size/width)
    resized_image = cv2.resize(image, (int(width*ratio), int(height*ratio)), cv2.INTER_CUBIC)
    return resized_image


def image_ss_resize(image, shortest_size):
    ''' Resize image to preview (mask, composite, grid and e.t.c) with value of SHORTEST size
        Input: numpy array, longest_size. Interpolation is BICUBIC '''
    height, width = image.shape[0:2]
    ratio = max(shortest_size/height, shortest_size/width)
    resized_image = cv2.resize(image, (int(width*ratio), int(height*ratio)), cv2.INTER_CUBIC)
    return resized_image


def image_rotate(image, angle):
    ''' Rotate image clockwise. Angle must be 90, 180, 270 degrees
        Inputs: numpy image and angle '''
    match angle:
        case 90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        case 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        case 270:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        case _:
            rotated_image = image
    return rotated_image


def image_blur(image, kernel):
    ''' Blur image with normalized box filter. Inputs: image (nparray), kernel '''
    ''' Perhaps, it is not necessary. '''
    blurred_image = cv2.blur(image, (kernel, kernel))
    return blurred_image

def image_dilate(image, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    image = cv2.dilate(image, kernel, iterations=1)
    return image

def blurred_box(box_size, blur_size):
    ''' Creates box mask with blurred borders. Size is amout pixels from border to max value '''
    box = np.zeros((box_size, box_size), dtype=np.uint8)
    indent = blur_size // 2
    box[indent:box_size-indent, indent:box_size-indent] = 255
    blur_size = indent*2-1
    blurred_box = cv2.blur(box, (blur_size, blur_size))
    return blurred_box


def compose_image_and_mask(image, mask):
    ''' Paste mask over image with transparency '''
    composite = np.array(image/2 + mask[..., np.newaxis]/2, dtype=np.uint8)
    return composite


def make_grid(image, mask, minimum_overlap, tile_size, minimum_density):
    ''' Creates grid of tiles with specified parameters.
        Returns amount of XY tiles, overlapping and tables of coordinates and presence'''
    height, width = mask.shape[0:2]
    if width == tile_size:
        x_tiles = 1
        overlap_x = 0
    else:
        x_tiles = width // (tile_size - minimum_overlap) + 1    # Num of X tiles
        overlap_x = tile_size - (width - tile_size) // (x_tiles - 1)
    if height == tile_size:
        y_tiles = 1
        overlap_y = 0
    else:
        y_tiles = height // (tile_size - minimum_overlap) + 1   # Num of Y tiles
        overlap_y = tile_size - (height - tile_size) // (y_tiles - 1)

    step_width = tile_size - overlap_x
    step_height = tile_size - overlap_y

    tiles_boolean_table = [[(False, (0, 0))] * (x_tiles+1) for i in range(y_tiles+1)]
    tiles_coords = []

    for y in range(y_tiles):
        for x in range(x_tiles):
            xc = x*step_width
            yc = y*step_height
            lookup = mask[yc:yc+tile_size, xc:xc+tile_size]
            if lookup.sum() / (tile_size**2) > minimum_density:
                tiles_coords.append((xc, yc, xc + tile_size, yc + tile_size))
                tiles_boolean_table[y][x] = (True, (xc, yc))

    return (x_tiles, y_tiles), (overlap_x, overlap_y), tiles_boolean_table, tiles_coords


def draw_grid(image, tile_coordinates):
    ''' draw grid with different colors '''
    grid = image.copy()
    pad = image.shape[1]//300
    for t in tile_coordinates:
        x1, y1, x2, y2 = t
        grid = cv2.rectangle(grid, (x1+pad, y1+pad), (x2-pad, y2-pad),
                             (np.random.randint(0, 255), np.random.randint(0, 255),
                              np.random.randint(0, 255)), pad)
    return grid


def get_tiles_from_image(image, tiles_coordinates):
    ''' return list of nparrays with tiles '''
    tiles = []
    for s in tiles_coordinates:
        x1, y1, x2, y2 = s
        tiles.append(image[y1:y2, x1:x2, :])
    return tiles


def get_custom_tile(image, tile_size, coordinates):
    ''' Returns square tile inside image with center coordinates.
        If coords outside borders, it will be corrected to be inside image'''
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


if __name__ == '__main__':
    from PIL import Image
    img = Image.open('C:/images/010.jpg')
    img = np.array(img)
    msk = np.ones(img.shape, dtype=np.uint8)*255
    tiles, overlaps, booltable, coords = make_grid(img, msk, 64, 1024, 5)
    print(tiles)
    pass