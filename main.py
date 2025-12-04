from pillow_heif import register_heif_opener
from PIL import Image
import numpy as np
from datetime import datetime
from scipy import ndimage
from typing import Literal
from matplotlib.colors import rgb_to_hsv

register_heif_opener()

def rgb_to_luminance(rgb):
    # luminance values go from 0 to 255
    r, g, b = rgb[...,0].astype(float), rgb[...,1].astype(float), rgb[...,2].astype(float)
    return 0.2126*r + 0.7152*g + 0.0722*b

def rgb_to_gray(rgb):
    # standard luminance transform
    return (0.2126*rgb[...,0] +
            0.7152*rgb[...,1] +
            0.0722*rgb[...,2]).astype(np.float32)

def rgb_to_hue(arr):
    # arr: uint8, shape (H, W, 3)
    rgb = arr.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(rgb)
    hue = hsv[..., 0]
    return hue

sorting_handler = {
    "by_luminance": rgb_to_luminance,
    "by_hue": rgb_to_hue
}

def generate_sorting(sort_type: Literal["by_hue", "by_luminance"], array):
    sorting_function = sorting_handler[sort_type]
    return sorting_function(array)

def sobel_edges(array):
    gray = rgb_to_gray(array)

    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)

    mag = np.hypot(gx, gy)
    mag = (mag / mag.max() * 255).astype(np.uint8)
    return mag

def generate_mask(mask_type: Literal["luminance", "sobel_edges"], array, threshold):
    if mask_type == "luminance":
        lum_array = rgb_to_luminance(array)
        return lum_array > threshold
    elif mask_type == "sobel_edges":
        edge_array = sobel_edges(array)
        return edge_array < threshold


def pixel_sort_columns(array, mask_type, sort_type,
                       threshold=100, min_segment_len=5, reverse=False,
                       pre_loaded_mask=None):
    """
    array: numpy array shape (H, W, 3), dtype uint8
    """
    height, width, colour = arr.shape
    output = array.copy()

    if pre_loaded_mask is not None:
        mask_full = pre_loaded_mask
    else:
        mask_full = generate_mask(mask_type, output, threshold)

    for x in range(width):
        column = output[:,x,:]

        mask = mask_full[:,x]

        i = 0
        while i < height:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < height and mask[j]:
                j += 1
            if (j - i) >= min_segment_len:
                segment = column[i:j]
                key = generate_sorting(sort_type, segment)
                order = np.argsort(key)
                if reverse:
                    order = order[::-1]
                column[i:j] = segment[order]
            i = j
    return output

def pixel_sort_rows(array, mask_type, sort_type,
                    threshold=100, min_segment_len=5, reverse=False,
                    pre_loaded_mask=None):
    """
    array: numpy array shape (H, W, 3), dtype uint8
    """
    height, width, colour = arr.shape
    output = array.copy()

    if pre_loaded_mask is not None:
        mask_full = pre_loaded_mask
    else:
        mask_full = generate_mask(mask_type, output, threshold)

    for y in range(height):
        row = output[y,:,:]

        mask = mask_full[y,:]

        i = 0
        while i < width:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < width and mask[j]:
                j += 1
            if (j - i) >= min_segment_len:
                segment = row[i:j]
                key = generate_sorting(sort_type, segment)
                order = np.argsort(key)
                if reverse:
                    order = order[::-1]
                row[i:j] = segment[order]
            i = j
    return output

def pixel_sort_diagonals(array, mask_type, sort_type,
                         threshold=100, min_segment_len=5, reverse=False,
                         pre_loaded_mask=None):
    H, W, _ = array.shape
    out = array.copy()

    if pre_loaded_mask is not None:
        mask_full = pre_loaded_mask
    else:
        mask_full = generate_mask(mask_type, out, threshold)

    for d in range(-(H - 1), W):
        coords = []
        for y in range(H):
            x = y + d
            if 0 <= x < W:
                coords.append((y, x))

        if not coords:
            continue

        diag_pixels = np.array([out[y, x] for (y, x) in coords])
        diag_mask   = np.array([mask_full[y, x] for (y, x) in coords])

        i = 0
        L = len(coords)
        while i < L:
            if not diag_mask[i]:
                i += 1
                continue

            j = i
            while j < L and diag_mask[j]:
                j += 1

            if (j - i) >= min_segment_len:
                segment = diag_pixels[i:j]
                key = generate_sorting(sort_type, segment)
                order = np.argsort(key)
                if reverse:
                    order = order[::-1]
                diag_pixels[i:j] = segment[order]

            i = j

        for k, (y, x) in enumerate(coords):
            out[y, x] = diag_pixels[k]

    return out

if __name__ == "__main__":
    output_image_name = "pixel_sorted_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"

    img = Image.open("inputs/IMG_3911.heic").convert("RGB")
    arr = np.asarray(img)

    res = pixel_sort_rows(
        arr,
        mask_type="sobel_edges",
        sort_type="by_luminance",
        threshold=100,
        min_segment_len=20,
        reverse=False
    )

    res = pixel_sort_diagonals(
        res,
        mask_type="luminance",
        sort_type="by_hue",
        threshold=100,
        min_segment_len=20,
        reverse=True
    )

    out_img = Image.fromarray(res)
    out_img.save(output_image_name)
    print(output_image_name)