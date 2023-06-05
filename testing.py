import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from modules import labels, random_crops, sam_functions, view

def testing(save_path):
    """Same as diffusion_OOI_1, but applies preprocessing before taking the 
    mask for improved results.
    """

    # get images from specified dataset
    images = glob.glob('/home/grayson/Documents/Experiment_Datafolders/curated_256x256_diffusion_object_of_interest/*')
     
    sam_model = sam_functions.load_sam_model()
    for count, img_path in enumerate(images):
        img_info = labels.get_img_info_from_img_name(img_path.split('/')[-1])
        if img_info is None:
            continue
        img = labels.get_img_from_img_info(img_info)
        bbox = img_info['labels'][0]['bbox']
        cropped_img, new_bbox, masks, cropped_img_processed, masks_processed = random_crops.random_crop_around_bbox_v2(img, 
                                                                                                                      bbox, 
                                                                                                                      crop_size=(256, 256), 
                                                                                                                      return_masks=True, 
                                                                                                                      sam_model=sam_model
                                                                                                                      )
        if cropped_img is None:
            continue

        mask = np.transpose(masks[0].cpu().numpy(), (1, 2, 0))
        mask_processed = np.transpose(masks_processed[0].cpu().numpy(), (1, 2, 0))
        mask = view.get_rgb_mask_from_binary_mask(mask)
        mask_processed = view.get_rgb_mask_from_binary_mask(mask_processed)
        cutout_img = view.cut_out_image(cropped_img, mask)
        coutout_img_processed = view.cut_out_image(cropped_img_processed, mask_processed)

        new_cutout, new_mask = view.refine_thermal_mask(cutout_img.copy(),
                                                        mask.copy())
        new_cutout_processed, new_mask_processed = view.refine_thermal_mask(coutout_img_processed.copy(),
                                                                            mask_processed.copy())
        new_mask = view.get_rgb_mask_from_binary_mask(new_mask)
        new_mask_processed = view.get_rgb_mask_from_binary_mask(new_mask_processed)

        plt.subplot(1, 4, 1)
        plt.imshow(cropped_img)
        plt.subplot(1, 4, 2)
        plt.imshow(mask)
        plt.subplot(1, 4, 3)
        plt.imshow(cropped_img_processed)
        plt.subplot(1, 4, 4)
        plt.imshow(mask_processed)
        plt.show()

        # cv2.imwrite(f'{save_path}/cropped_img_{str(count).zfill(4)}.png', 
        #             new_cutout)
        count += 1
        break
