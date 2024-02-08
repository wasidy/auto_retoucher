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


def image_resize(image, longside_size):
    ''' Resize image to preview (mask, composite, grid and e.t.c) with value of longest size
        Input: numpy array, longest_size. Interpolation is BICUBIC '''
    height, width = image.shape[0:2]
    ratio = min(longside_size/height, longside_size/width)
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


def blurred_box(box_size, blur_size):
    ''' Creates box mask with blurred borders. Size is amout pixels from border to max value '''
    box = np.zeros((box_size, box_size), dtype=np.uint8)
    indent = blur_size // 2
    box[indent:box_size-indent, indent:box_size-indent] = 255
    blur_size = indent*2-1
    blurred_box = cv2.blur(box, (blur_size, blur_size))
    return blurred_box
