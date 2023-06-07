import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def show_bbox_cv2(img, bbox):
    x, y, x2, y2 = bbox
    w = x2 - x
    h = y2 - y
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return img

def show_bbox_plt(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                                facecolor=(0,0,0,0), lw=2))

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25) 

def show_mask(mask, ax, random_color=False):
    mask = mask.detach().cpu()
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def visualize_sam(img, masks, bboxes=None, points=None):
    """Does not have points functionality yet. Need to add if desired."""

    if bboxes is None and points is None:
        print('Error: Must pass bboxes or points to visualize a sam output!')
        return
    
    if bboxes is None:
        print('Error! Must pass a bbox to visualize a sam output!')
        return

    plt.figure('original image')
    plt.imshow(img)

    plt.figure('visualize sam')
    plt.imshow(img)

    for mask in masks:
        show_mask(mask, plt.gca())
    for bbox in bboxes:
        show_bbox_plt(bbox, plt.gca())

    plt.axis('off')
    plt.show()

def get_rgb_mask_from_binary_mask(mask):
    """Goes from (h, w, 1) -> (h, w, 3) where all channels are 255 (white)
    Parameters
        mask (numpy arr): mask to convert
    """

    h, w, _ = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype='uint8')
    for i in range(3):
        mask_rgb[:, :, i] = mask[:, :, 0]
    return mask_rgb * 255

def cut_out_image(img, mask):
    """Turns every pixel in the image not in the mask to (0, 0, 0) (black)"""

    assert mask.shape == img.shape, "Error: Image and mask must be the same shape"
    h, w, _ = mask.shape
    mask = mask.astype('uint8')

    for r in range(h):
        for c in range(w):
            if mask[r, c, 0] == False:
                img[r, c, :] *= 0
    
    return img

def resize_and_pad(img_path,
                   resized_img_save_path,
                   downsample_size,
                   final_size):
    """Function not finished, but sample code for something similar is below."""
    resized_img = cv2.imread('/home/grayson/Desktop/Code/ForkGAN/datasets/alderley/testA/day_mask_4112.png')
    resized_img = cv2.resize(resized_img, (128, 128))
    left_add = np.zeros((128, 128, 3))
    resized_img = np.concatenate((left_add, resized_img), axis=1)

    bottom_add = np.zeros((128, 256, 3))
    resized_img = np.concatenate((resized_img, bottom_add), axis=0)

    cv2.imshow('what', resized_img)
    cv2.waitKey(5000)

def refine_thermal_mask(img, mask):
    """Takes in an image of a thermal cutout on a black background and it's 
    respective binary mask. Returns a refined version of that thermal cutout 
    and an updated binary mask on a black background.

    Params:
        img (numpy arr): image to refine
        mask (numpy arr): mask (can be binary or rgb)

    Returns:
        img (numpy arr): refined image
    """

    h, w, _ = img.shape
    _, _, mask_channels = mask.shape

    # if mask is in rgb form, move to binary form
    if mask_channels != 1:
        mask = get_binary_from_rgb_mask(mask)

    # get the average pixel value and define the threshold
    total_num_pxls = np.sum(mask)
    average_pxl_value = np.sum(img) // total_num_pxls
    threshold = .6 * average_pxl_value

    # loop through every pixel. If the sum of that pixel's rgb values are less
    # than a threshold, set that pixel to black (0, 0, 0)
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 0:
                continue
            if np.sum(img[i, j, :]) < threshold:
                img[i, j, :] = np.zeros((1, 1, 3))
                mask[i, j] = 0

    return img, mask

def get_binary_from_rgb_mask(rgb_mask):
    """Takes in an image that is rgb where each pixel is either (0, 0, 0) black
    or (255, 255, 255) white."""

    h, w, _ = rgb_mask.shape

    rgb_mask = np.max(rgb_mask, axis=2)

    ones = np.ones((h, w))
    binary_mask = ones <= rgb_mask
    
    binary_mask = np.expand_dims(binary_mask, axis=2)
   

    return binary_mask

def ooi_fuse_v1(ooi,
             global_img):
    """Uses linear interpolation to fuse the ooi with the global image"""

    h, w, c = ooi.shape
    new_img = np.zeros_like(global_img)

    for i in range(h):
        for j in range(w):
            # apply weight based on how bright the ooi pixel is
            pixel_brightness = (np.sum(ooi[i,j]) // 3) / 255 # between 0 and 1
            new_img[i,j] = pixel_brightness*ooi[i,j]+(1-pixel_brightness)*global_img[i,j]
    
    return new_img

def ooi_fuse_v2(ooi,
                global_img):
    """Uses linear interpolation to fuse the ooi with the global image, but
    first calculates the brightness for each pixel, and normalizes the entire
    image to have a min of zero and max of 1"""

    h, w, c = ooi.shape
    new_img = np.zeros_like(global_img)

    """Currently, a large portion of the image is zeros. This does not count as
    a minimum value, as this is a mask, so we must remove these from the np.min
    calculation."""

    pixel_brightness = (np.sum(ooi, axis=2) // 3) / 255
    min_val = 255
    for i in range(h):
        for j in range(w):
            if pixel_brightness[i,j] < min_val and pixel_brightness[i,j] != 0:
                min_val = pixel_brightness[i,j]

    for i in range(h):
        for j in range(w):
            if pixel_brightness[i,j] == 0:
                pixel_brightness[i,j] = min_val

    # normalize for min at 0 and max at 1
    pixel_brightness = ((pixel_brightness-min_val) 
                         / (np.max(pixel_brightness) - min_val))
    
    # pixel_brightness = ((pixel_brightness-np.min(pixel_brightness)) 
    #                     / (np.max(pixel_brightness) - np.min(pixel_brightness)))

    for i in range(h):
        for j in range(w):
            # apply weight based on how bright the ooi pixel is
            new_img[i,j] = (pixel_brightness[i,j]*ooi[i,j]
                            +(1-pixel_brightness[i,j])*global_img[i,j])
    
    return new_img

def ooi_fuse_v3(ooi,
                global_img):
    """Uses linear interpolation to fuse the ooi with the global image, but
    first calculates the brightness for each pixel, and normalizes the entire
    image to have a min of zero and max of 1"""

    h, w, c = ooi.shape
    new_img = np.zeros_like(global_img)

    """Currently, a large portion of the image is zeros. This does not count as
    a minimum value, as this is a mask, so we must remove these from the np.min
    calculation."""

    # use the increased contrast image to get pixel brightness, but use the
    # unprocessed image when combining

    ooi_increased_brightness = increase_contrast(ooi)

    pixel_brightness = (np.sum(ooi_increased_brightness, axis=2) // 3) / 255
    min_val = 255
    for i in range(h):
        for j in range(w):
            if pixel_brightness[i,j] < min_val and pixel_brightness[i,j] != 0:
                min_val = pixel_brightness[i,j]

    for i in range(h):
        for j in range(w):
            if pixel_brightness[i,j] == 0:
                pixel_brightness[i,j] = min_val

    # normalize for min at 0 and max at 1
    pixel_brightness = ((pixel_brightness-min_val) 
                         / (np.max(pixel_brightness) - min_val))
    
    # pixel_brightness = ((pixel_brightness-np.min(pixel_brightness)) 
    #                     / (np.max(pixel_brightness) - np.min(pixel_brightness)))

    for i in range(h):
        for j in range(w):
            # apply weight based on how bright the ooi pixel is
            new_img[i,j] = (pixel_brightness[i,j]*ooi[i,j]
                            +(1-pixel_brightness[i,j])*global_img[i,j])
    
    return new_img

def ooi_fuse_v4(ooi,
                global_img):
    """Same as v2, but adds gaussian blur at the beginning."""

    ooi = cv2.GaussianBlur(ooi, (7, 7), cv2.BORDER_DEFAULT)
    h, w, c = ooi.shape
    new_img = np.zeros_like(global_img)

    """Currently, a large portion of the image is zeros. This does not count as
    a minimum value, as this is a mask, so we must remove these from the np.min
    calculation."""

    pixel_brightness = (np.sum(ooi, axis=2) // 3) / 255
    min_val = 255
    for i in range(h):
        for j in range(w):
            if pixel_brightness[i,j] < min_val and pixel_brightness[i,j] != 0:
                min_val = pixel_brightness[i,j]

    for i in range(h):
        for j in range(w):
            if pixel_brightness[i,j] == 0:
                pixel_brightness[i,j] = min_val

    # normalize for min at 0 and max at 1
    pixel_brightness = ((pixel_brightness-min_val) 
                         / (np.max(pixel_brightness) - min_val))
    
    # pixel_brightness = ((pixel_brightness-np.min(pixel_brightness)) 
    #                     / (np.max(pixel_brightness) - np.min(pixel_brightness)))

    for i in range(h):
        for j in range(w):
            # apply weight based on how bright the ooi pixel is
            new_img[i,j] = (pixel_brightness[i,j]*ooi[i,j]
                            +(1-pixel_brightness[i,j])*global_img[i,j])
    
    return new_img

def ooi_fuse_v5(ooi, global_img):
    pass
    
def increase_contrast(img):
    # define the alpha and beta
    alpha = 1.9 # Contrast control
    beta = 1 # Brightness control

    # call convertScaleAbs function
    adjusted = cv2.convertScaleAbs(img.copy(), alpha=alpha, beta=beta)

    h, w, _ = img.shape
    for i in range(h):
        for j in range(w):
            if np.sum(img[i,j]) == 0:
                adjusted[i,j] = 0

    return adjusted


