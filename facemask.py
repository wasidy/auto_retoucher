import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter1d
from PIL import Image
from scripts.pipelines import ClipSegmentation
from scripts.imageutils import split_mask_to_blobs
from scripts.imageutils import get_custom_tile


model = ClipSegmentation("CIDAS/clipseg-rd64-refined",
                         "CIDAS/clipseg-rd64-refined")


def skin_mask_generate(image, face_img, face_mask,
                       hue_threshold=1, sat_threshold=2, val_threshold=2):
    ''' Shift tone shifting H value for centering skin tones
        Inputs:
            image: numpy RGB image
            face: cropped face
            face_mask: mask of face
        Returns L mask '''

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

    skin_mask = cv2.inRange(image, color1, color2)

    return skin_mask


if __name__ == '__main__':
    img2 = Image.open('C:/images/004.jpg')
    img2 = np.array(img2)
    keyword = 'face'

    labeled_mask = model.predict(img2, keyword)

    masks, coords = split_mask_to_blobs(labeled_mask)

    mask_size = 0
    for i, m in enumerate(masks):
        if m.sum() > mask_size:
            max_mask_idx = i

    x_center = int(coords[max_mask_idx][1][1])
    y_center = int(coords[max_mask_idx][1][0])
    x, y, w, h, _ = coords[max_mask_idx][0]

    crop_tile_size = int(max(w, h)*1.2)

    tile, lx, ly, lx2, ly2 = get_custom_tile(img2, crop_tile_size, (y_center, x_center))
    face_mask = masks[max_mask_idx][ly:ly2, lx:lx2]
    face_img = img2[ly:ly2, lx:lx2]

    skin_mask = skin_mask_generate(img2, face_img, face_mask, 2, 2, 2)
    test_mask = np.zeros(img2.shape[0:2], dtype=np.uint8)
    test_mask[300:900, 300:600] = 255

    final_mask = np.logical_and(skin_mask, test_mask)
    #final_mask = skin_mask - test_mask
    
    plt.imshow(final_mask)