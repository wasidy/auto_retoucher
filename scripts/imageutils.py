import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from typing import List


def split_mask_to_blobs(image):
    ''' Input - labeled 2D array
        Output - list of 2D ndarrays with splitted masks, list of tuples with stats and centers '''

    masks = []
    coords = []
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
    if kernel > 0:
        image = cv2.blur(image, (kernel, kernel))
    return image

def image_dilate(image, size):
    if size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        image = cv2.dilate(image, kernel, iterations=1)
    return image

def gradient_box(box_size, blur_size):
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
    ''' draw transparent grid with coordinates '''
    grid = image.copy().astype(np.float32)
    alpha = 0.25
    beta = 0.75

    for t in tile_coordinates:
        x1, y1, x2, y2 = t
        red_box = np.zeros((y2-y1, x2-x1, 3), dtype=np.float32)
        red_box[:, :, 0] = 255  # Filling red
        grid[y1:y2, x1:x2, :] *= beta
        red_box *= alpha
        grid[y1:y2, x1:x2, :] = grid[y1:y2, x1:x2, :] + red_box

    grid = grid.astype(np.uint8)
    return grid


def get_tiles_from_image(image, tiles_coordinates):
    ''' return list of nparrays with tiles '''
    tiles = []
    for s in tiles_coordinates:
        x1, y1, x2, y2 = s
        tiles.append(image[y1:y2, x1:x2, :])
    return tiles

def smooth_mask(image, smooth_size=10, expand=10):
    ''' smooth and expand in percents of image size'''
    img_size = max(image.shape[0:2])
    expand = round(img_size/100*expand)
    if expand > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand, expand))
        image = cv2.dilate(image, kernel, iterations=1)
    smooth_size = round(img_size/100*smooth_size)
    if smooth_size > 0:
        image = cv2.blur(image, (smooth_size, smooth_size))
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return image

def get_custom_tile(image, tile_size, coordinates):
    ''' Returns square tile inside image with center coordinates.
        If coords outside borders, it will be corrected to be inside image'''
    x, y = coordinates
    height, width = image.shape[0:2]
    
   
    lx = int(x - tile_size/2)
    ly = int(y - tile_size/2)
    
    lx = 0 if lx < 0 else lx
    ly = 0 if ly < 0 else ly
    
    lx = width - tile_size if (lx + tile_size) > width else lx
    ly = height - tile_size if (ly + tile_size) > height else ly
    lx2 = lx+tile_size
    ly2 = ly+tile_size
    tile = image[ly:ly2, lx:lx2, :]
    print(x, y, tile_size, lx, ly, lx2, ly2)
    # В функции ошибка при равенстве размеров тайла и изображения по какой-либо из сторон
    return tile, lx, ly, lx2, ly2


def blend_with_alpha(image, target, alpha):
    alpha = alpha/255.0
    beta = 1 - alpha
    image = image/255.0 * beta[:,:,np.newaxis]
    target = target/255.0 * alpha[:,:,np.newaxis]
    
    result = (image+target)*255
    result = result.astype(np.uint8)
    
    return result
    #result = cv2.addWeighted(image, alpha, target, beta, 0.0)


def skin_mask_generate(image, face_img, face_mask,
                       hue_threshold=1, sat_threshold=2, val_threshold=2,
                       kernel_size=20, sigma=7, blur=5):
    ''' Shift tone shifting H value for centering skin tones
        Inputs:
            image: numpy RGB image
            face: cropped face
            face_mask: mask of face
        Returns binary mask '''

    # Convert to HSV and shift hue
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:, :, 0] = (image[:, :, 0] + 90) % 180
    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2HSV)
    face_img[:, :, 0] = (face_img[:, :, 0] + 90) % 180

    # Building histogram and smooth it
    hue_hist = cv2.calcHist([face_img], [0], face_mask, [179], [0, 180])
    hue_hist = hue_hist * (100 / hue_hist.max())
    hue_hist = hue_hist.squeeze()
    hue_hist = gaussian_filter1d(hue_hist, 1)

    sat_hist = cv2.calcHist([face_img], [1], face_mask, [255], [0, 255])
    sat_hist = sat_hist * (100 / sat_hist.max())
    sat_hist = sat_hist.squeeze()
    sat_hist = gaussian_filter1d(sat_hist, 1)

    val_hist = cv2.calcHist([face_img], [2], face_mask, [255], [0, 255])
    val_hist = val_hist * (100 / val_hist.max())
    val_hist = val_hist.squeeze()
    val_hist = gaussian_filter1d(val_hist, 1)

    # Find low and high values on histogram
    h_min = np.argmax(hue_hist > hue_threshold)
    h_max = hue_hist.shape[0] - np.argmax(hue_hist[::-1] > hue_threshold)

    s_min = np.argmax(sat_hist > sat_threshold)
    s_max = sat_hist.shape[0] - np.argmax(sat_hist[::-1] > sat_threshold)

    v_min = np.argmax(val_hist > val_threshold)
    v_max = val_hist.shape[0] - np.argmax(val_hist[::-1] > sat_threshold)

    # Creating color range
    color1 = np.asarray([h_min, s_min, v_min])
    color2 = np.asarray([h_max, s_max, v_max])
    
    image = cv2.medianBlur(image, blur)
    mask = cv2.inRange(image, color1, color2)*255
    
    
    #kernel_size = 4
    #sigma = 3
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #print(mask.max())
    #mask = cv2.GaussianBlur(mask, (sigma, sigma), 0)
    #_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def combine_masks(*masks):
    
    pass

if __name__ == '__main__':
    from PIL import Image
    img = Image.open('C:/images/010.jpg')
    img = np.array(img)
    msk = np.ones(img.shape, dtype=np.uint8)*255
    tiles, overlaps, booltable, coords = make_grid(img, msk, 64, 1024, 5)
    print(tiles)
    pass