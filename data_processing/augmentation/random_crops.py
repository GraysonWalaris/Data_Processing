import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from .. import sam_functions
from .. import view

"""A collection of methods for extracting random crops from the images in
the Tarsier_Main_Dataset.
"""

DATA_FOLDER_BASE_PATH = os.environ.get('WALARIS_MAIN_DATA_PATH')
IMAGES_BASE_PATH = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'),
                                'Images')
LABELS_BASE_PATH = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'), 
                                'Labels_NEW')

def random_crop(img,
                crop_size,
                ):
    """
    Randomly crop an image to create a new data sample.

        Args:
            img (numpy arr): Numpy array of image file in the form (h, w, 3)
            crop_size (tuple): (h, w) of the cropped image

        Returns:
            cropped_image (np.array): Cropped image
    """
    cropped_height, cropped_width = crop_size
    img_height, img_width, _ = img.shape

    assert cropped_height % 2 == 0
    assert cropped_width % 2 == 0
    assert img_width % 2 == 0
    assert img_height % 2 == 0

    if img_width < cropped_width or img_height < cropped_height:
        return None
    
    print(img_width, cropped_width)
    print(img_height, cropped_height)

    x_start = np.random.randint(0, img_width-cropped_width-1) if img_width > cropped_height else 0
    y_start = np.random.randint(0, img_height-cropped_height-1) if img_height > cropped_height else 0

    cropped_img = img[y_start:y_start+cropped_height, x_start:x_start+cropped_width, :]

    return cropped_img

def random_crop_around_bbox(img,
                bbox,
                crop_size,
                space_from_edge=10,
                return_masks=False,
                sam_model=None,
                bbox_w_lim=(0, 10000),
                bbox_h_lim=(0, 10000),
                bbox_area_lim=(0, 1000000)
                ):
    """
    Randomly crop an image to create a new data sample, translate bounding box
      to new image size.

        Parameters:
            img (numpy arr): Numpy array of image file in the form (h, w, 3)
            crop_size (tuple): (h, w) of the cropped image
            space_from_edge (int): Number of pixels from the edge the bounding
             box must be after the random crop.
            return_masks (bool): If true, will utilize the segment anything
             model to produce masks for the object of interest and return them.
            bbox_w_lim (tuple): (min bbox width, max bbox width)
            bbox_h_lim (tuple): (min bbox height, max bbox height)
            bbox_area_lim (tuple): (min bbox area, max bbox area)

        Returns:
            cropped_image (np.array): Cropped image
            new_bbox (list): Bbox translated to fit the cropped image
            masks (tensor): None if return_masks==False. Else, Tensor of masks 
             from the crop in the image.
    """
    cropped_height, cropped_width = crop_size

    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    bbox_center = (y1+y2)//2, (x1+x2)//2

    # make sure the box fits within the limits
    if bbox_width > bbox_w_lim[1] or bbox_width < bbox_w_lim[0]:
        print('bbox width outside limit.. skipping')
        print('bbox width', bbox_width)
        return None, None, None
    if bbox_height > bbox_w_lim[1] or bbox_height < bbox_h_lim[0]:
        print('bbox height', bbox_height)
        print('bbox height outside limit.. skipping')
        return None, None, None
    if bbox_area > bbox_area_lim[1] or bbox_area < bbox_area_lim[0]:
        print('bbox area', bbox_area)
        print('bbox area outside limit.. skipping')
        return None, None, None
    
    # ensure bounding box is smaller than cropped image size (with space from
    # edge taken into account)
    if (bbox_height > cropped_height - space_from_edge 
        or bbox_width > cropped_width - space_from_edge):
        print('bbox too large.. skipping')
        return None, None, None
    
    # ensure the image is larger than the cropped image size
    img_height, img_width, _ = img.shape
    if img_height < cropped_height or img_width < cropped_width:
        print('image size too small.. skipping')
        return None, None, None
    
    # find the max distance between the center of the crop and center of bbox
    # that keeps the entire bbox in the image

    assert cropped_height % 2 == 0
    assert cropped_width % 2 == 0
    assert img_width % 2 == 0
    assert img_height % 2 == 0

    max_distance_x = cropped_width//2 - bbox_width//2 - space_from_edge
    max_distance_y = cropped_height//2 -  bbox_height//2 - space_from_edge

    if max_distance_x == 0 or max_distance_y == 0:
        print('image size too small.. skipping')
        return None, None, None

    # randomly find the new center distance of the crop
    x_center_distance = np.random.randint(-max_distance_x, max_distance_x)
    y_center_distance = np.random.randint(-max_distance_y, max_distance_y)

    # since the cropped image dimensions are always even, the center point of
    # crop will be the top left of the 4 pixel center of the cropped image (there
    # is no true center with even numbers)

    crop_center = (bbox_center[0] + y_center_distance,
                   bbox_center[1] + x_center_distance)
    
    # ensure crop is not outside image size
    crop_center = (max(cropped_height//2-1, crop_center[0]),
                      max(cropped_width//2-1, crop_center[1]))
    crop_center = (min(img_height-cropped_width//2-1, crop_center[0]),
                   min(img_width-cropped_width//2-1, crop_center[1]))

    crop_y1 = crop_center[0]-cropped_height//2+1
    crop_y2 = crop_center[0]+cropped_height//2+1
    crop_x1 = crop_center[1]-cropped_width//2+1
    crop_x2 = crop_center[1]+cropped_width//2+1

    crop_y1, crop_y2, crop_x1, crop_x2 = (int(crop_y1), int(crop_y2), 
                                          int(crop_x1), int(crop_x2))

    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # get the location of the bounding box relative to the cropped image
    new_bbox = [x1-crop_x1, y1-crop_y1, x2-crop_x1, y2-crop_y1]

    masks = None

    if return_masks:

        img_for_sam = cv2.cvtColor(cropped_img.copy(), cv2.COLOR_BGR2RGB)

        predictor = sam_functions.SamPredictor(sam_model)

        predictor.set_image(img_for_sam)

        input_boxes = torch.tensor([
            [new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3]]
        ], device=predictor.device)
        
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes,
                                                                  img_for_sam.shape[:2])
        
        masks, scores, logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )

    return cropped_img, new_bbox, masks

def random_crop_around_bbox_v2(img,
                bbox,
                crop_size,
                space_from_edge=10,
                return_masks=False,
                sam_model=None,
                bbox_w_lim=(0, 10000),
                bbox_h_lim=(0, 10000),
                bbox_area_lim=(0, 1000000)
                ):
    """
    V2: Applies preprocessing before taking mask to increase performance.

    Randomly crop an image to create a new data sample, translate bounding box
      to new image size.

        Args:
            img (numpy arr): Numpy array of image file in the form (h, w, 3)
            crop_size (tuple): (h, w) of the cropped image
            space_from_edge (int): Number of pixels from the edge the bounding
                box must be after the random crop.
            return_masks (bool): If true, will utilize the segment anything
                model to produce masks for the object of interest and return them.
            bbox_w_lim (tuple): (min bbox width, max bbox width)
            bbox_h_lim (tuple): (min bbox height, max bbox height)
            bbox_area_lim (tuple): (min bbox area, max bbox area)

        Returns:
            cropped_image (np.array): Cropped image
            new_bbox (list): Bbox translated to fit the cropped image
            masks (tensor): None if return_masks==False. Else, Tensor of masks 
                from the crop in the image.
    """
    cropped_height, cropped_width = crop_size

    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    bbox_center = (y1+y2)//2, (x1+x2)//2

    # make sure the box fits within the limits
    if bbox_width > bbox_w_lim[1] or bbox_width < bbox_w_lim[0]:
        print('bbox width outside limit.. skipping')
        print('bbox width', bbox_width)
        return None, None, None
    if bbox_height > bbox_w_lim[1] or bbox_height < bbox_h_lim[0]:
        print('bbox height', bbox_height)
        print('bbox height outside limit.. skipping')
        return None, None, None
    if bbox_area > bbox_area_lim[1] or bbox_area < bbox_area_lim[0]:
        print('bbox area', bbox_area)
        print('bbox area outside limit.. skipping')
        return None, None, None
    
    # ensure bounding box is smaller than cropped image size (with space from
    # edge taken into account)
    if (bbox_height > cropped_height - space_from_edge 
        or bbox_width > cropped_width - space_from_edge):
        print('bbox too large.. skipping')
        return None, None, None
    
    # ensure the image is larger than the cropped image size
    img_height, img_width, _ = img.shape
    if img_height < cropped_height or img_width < cropped_width:
        print('image size too small.. skipping')
        return None, None, None
    
    # find the max distance between the center of the crop and center of bbox
    # that keeps the entire bbox in the image

    assert cropped_height % 2 == 0
    assert cropped_width % 2 == 0
    assert img_width % 2 == 0
    assert img_height % 2 == 0

    max_distance_x = cropped_width//2 - bbox_width//2 - space_from_edge
    max_distance_y = cropped_height//2 -  bbox_height//2 - space_from_edge

    # randomly find the new center distance of the crop
    x_center_distance = np.random.randint(-max_distance_x, max_distance_x)
    y_center_distance = np.random.randint(-max_distance_y, max_distance_y)

    # since the cropped image dimensions are always even, the center point of
    # crop will be the top left of the 4 pixel center of the cropped image (there
    # is no true center with even numbers)

    crop_center = (bbox_center[0] + y_center_distance,
                   bbox_center[1] + x_center_distance)
    
    # ensure crop is not outside image size
    crop_center = (max(cropped_height//2-1, crop_center[0]),
                      max(cropped_width//2-1, crop_center[1]))
    crop_center = (min(img_height-cropped_width//2-1, crop_center[0]),
                   min(img_width-cropped_width//2-1, crop_center[1]))

    crop_y1 = crop_center[0]-cropped_height//2+1
    crop_y2 = crop_center[0]+cropped_height//2+1
    crop_x1 = crop_center[1]-cropped_width//2+1
    crop_x2 = crop_center[1]+cropped_width//2+1

    crop_y1, crop_y2, crop_x1, crop_x2 = (int(crop_y1), int(crop_y2), 
                                          int(crop_x1), int(crop_x2))

    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # get the location of the bounding box relative to the cropped image
    new_bbox = [x1-crop_x1, y1-crop_y1, x2-crop_x1, y2-crop_y1]

    masks = None

    if return_masks:

        img_for_sam = cv2.cvtColor(cropped_img.copy(), cv2.COLOR_BGR2RGB)

        # increase contrast within the bounding box
        bbox_img = img_for_sam[new_bbox[1]:new_bbox[3], 
                               new_bbox[0]:new_bbox[2],
                               :]

        bbox_img_processed = view.increase_contrast(bbox_img)

        processed_img_for_sam = img_for_sam.copy()
        processed_img_for_sam[new_bbox[1]:new_bbox[3], 
                              new_bbox[0]:new_bbox[2],
                              :] = bbox_img_processed
        
        plt.subplot(1, 2, 1)
        plt.imshow(processed_img_for_sam)
        plt.subplot(1, 2, 2)
        plt.imshow(img_for_sam)
        plt.show()

        predictor = sam_functions.SamPredictor(sam_model)

        predictor.set_image(img_for_sam)

        input_boxes = torch.tensor([
            [new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3]]
        ], device=predictor.device)
        
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes,
                                                                  img_for_sam.shape[:2])
        
        masks, scores, logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )

        predictor_processed = sam_functions.SamPredictor(sam_model)

        predictor_processed.set_image(img_for_sam)

        input_boxes = torch.tensor([
            [new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3]]
        ], device=predictor_processed.device)
        
        transformed_boxes = predictor_processed.transform.apply_boxes_torch(input_boxes,
                                                                  img_for_sam.shape[:2])
        
        masks_processed, scores_processed, logits_processed = predictor_processed.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )

        processed_img_for_sam = cv2.cvtColor(processed_img_for_sam, 
                                             cv2.COLOR_RGB2BGR)

    return cropped_img, new_bbox, masks, processed_img_for_sam, masks_processed
        
if __name__=='__main__':
    pass




