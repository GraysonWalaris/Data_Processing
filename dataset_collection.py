import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from modules import labels, random_crops, sam_functions, view

"""This holds the scrips used to collect data for various experiments. The
experiment log can be found in this google drive."""

def blob2therm_1(save_path, num_samples):
    """Gets pairs of binary masks and cutout images from the thermal dataset
    of all classes.

    This script uses the curated dataset found here:
    /home/grayson/Documents/Experiment_Datafolders/curated_for_mask2therm

    It randomly goes through the curated dataset and randomly crops around the
    bounding box. It then gets a binary mask of the object of interest and cuts
    the object of interest from the image and places it on a black background.

    The following data augmentation techniques are applied randomly to increase
    dataset variability:

    Number of samples generated: 1000
    """

    # create save paths in the save directory
    masks_save_path = os.path.join(save_path, 'masks')
    cropped_imgs_save_path = os.path.join(save_path, 'cropped_imgs')
    if not os.path.exists(masks_save_path):
        os.mkdir(masks_save_path)
    if not os.path.exists(cropped_imgs_save_path):
        os.mkdir(cropped_imgs_save_path)

    # if there are already images in the folder, add specified number of
    # samples more to the folder
    imgs_already_in_folder = glob.glob(masks_save_path+'/*')
    imgs_already_in_folder.sort()
    count = int(imgs_already_in_folder[-1].split('_')[-1][:-4]) if len(imgs_already_in_folder) > 0 else 0

    # get images from specified dataset
    images = glob.glob('/home/grayson/Documents/Experiment_Datafolders/curated_for_mask2therm/*')
     
    start_count = count
    sam_model = sam_functions.load_sam_model()
    while count < num_samples + start_count:
        # randomly get img_name from the list of images in the target dataset
        idx = np.random.randint(0, len(images))
        img_info = labels.get_img_info_from_img_name(images[idx].split('/')[-1])
        if img_info is None:
            continue
        img = labels.get_img_from_img_info(img_info)
        bbox = img_info['labels'][0]['bbox']
        cropped_img, new_bbox, masks = random_crops.random_crop_around_bbox(img, 
                                                   bbox, 
                                                   crop_size=(256, 256), 
                                                   return_masks=True, 
                                                   sam_model=sam_model
                                                   )
        if cropped_img is None:
            continue

        mask = np.transpose(masks[0].cpu().numpy(), (1, 2, 0))

        mask = view.get_rgb_mask_from_binary_mask(mask)
        cutout_img = view.cut_out_image(cropped_img, mask)

        new_cutout, new_mask = view.refine_thermal_mask(cutout_img.copy(),
                                                        mask.copy())
        new_mask = view.get_rgb_mask_from_binary_mask(new_mask)

        cv2.imwrite(f'{masks_save_path}/mask_{str(count).zfill(4)}.png', 
                    new_mask)
        cv2.imwrite(f'{cropped_imgs_save_path}/cropped_img_{str(count).zfill(4)}.png', 
                    new_cutout)
        count += 1

def blob2therm_1_validation(save_path, num_samples):
    """Get binary masks of objects of interest in the visible light domain."""

    sam_model = sam_functions.load_sam_model()

    count = 0
    seen_images = set()
    while count < num_samples:
        img_info = labels.get_random_label(data_type='day', class_type='all', data_split='whole')
        if img_info is None:
            continue
        img_name = img_info['image_path'].split('/')[-1][:-4]
        if img_name in seen_images:
            continue
        seen_images.add(img_name)
        img = labels.get_img_from_img_info(img_info)
        if img is None:
            continue
        bbox = img_info['labels'][0]['bbox']

        # put a mininum limit on the bounding box size
        bbox_width = bbox[2]-bbox[0]
        bbox_height = bbox[3]-bbox[1]

        if bbox_width < 20 or bbox_height < 20:
            continue

        cropped_img, new_bbox, masks = random_crops.random_crop_around_bbox(img, 
                                                   bbox, 
                                                   crop_size=(256, 256), 
                                                   return_masks=True, 
                                                   sam_model=sam_model
                                                   )
        if cropped_img is None:
            continue

        mask = np.transpose(masks[0].cpu().numpy(), (1, 2, 0))

        mask = view.get_rgb_mask_from_binary_mask(mask)
        cutout_img = view.cut_out_image(cropped_img.copy(), mask)

        view.visualize_sam(cropped_img, masks, [new_bbox])

        cv2.imwrite(f'{save_path}/mask_{str(count).zfill(4)}.png', 
                    mask)
        count += 1

def diffusion_OOI_1(save_path):
    """Gets cutout images from the thermal uav dataset .

    This script uses the curated dataset found here:
    /home/grayson/Documents/Experiment_Datafolders/curated_256x256_diffusion_object_of_interest

    It goes through the curated dataset oonce and randomly crops around the
    bounding box. It then gets a binary mask of the object of interest and cuts

    Number of samples generated: Size of curated dataset
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
        cropped_img, new_bbox, masks = random_crops.random_crop_around_bbox(img, 
                                                   bbox, 
                                                   crop_size=(256, 256), 
                                                   return_masks=True, 
                                                   sam_model=sam_model
                                                   )
        if cropped_img is None:
            continue

        mask = np.transpose(masks[0].cpu().numpy(), (1, 2, 0))

        mask = view.get_rgb_mask_from_binary_mask(mask)
        cutout_img = view.cut_out_image(cropped_img, mask)

        new_cutout, new_mask = view.refine_thermal_mask(cutout_img.copy(),
                                                        mask.copy())
        new_mask = view.get_rgb_mask_from_binary_mask(new_mask)
        cv2.imwrite(f'{save_path}/cropped_img_{str(count).zfill(4)}.png', 
                    new_cutout)
        count += 1

def diffusion_OOI_2(save_path, num_samples):
    """ Get center cropped images of thermal UAVs from the dataset to train
    a latent diffusion model.

    Function makes sure that once an image was used, it will not be used again 
    to reduce the possibility of duplicates in the training data. This is
    implemented in a naive way -- will need to be optimized if num_samples is
    close to the total amount of thermal data images.

    The images are center cropped to allow us to test if the diffusion model
    is simply "memorizing" the training data or producing novel images.
    """

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    images = glob.glob(save_path+'/*.jpg')
    images.sort()
    if len(images) != 0:
        start_count = int(images[-1].split('_')[-1][:-4])
    else:
        start_count = 0
    count = start_count
    seen_images = set()
    while count < num_samples + start_count:
        img_info = labels.get_random_label(data_type='thermal', class_type='all', data_split='whole')
        img_name = img_info['image_path'].split('/')[-1][:-4]
        if img_name in seen_images:
            continue
        seen_images.add(img_name)
        img = labels.get_img_from_img_info(img_info)
        bbox = img_info['labels'][0]['bbox']

        # put a mininum limit on the bounding box size
        bbox_width = bbox[2]-bbox[0]
        bbox_height = bbox[3]-bbox[1]

        if bbox_width < 20 or bbox_height < 20:
            continue

        cropped_img = random_crops.random_crop(img, crop_size=(256, 256))
        if cropped_img is None:
            continue
        
        cv2.imwrite(f'{save_path}/cropped_img_{str(count).zfill(4)}.jpg', 
                        cropped_img)
        
        count += 1

        if count % 100 == 0:
            print(count)

if __name__=='__main__':
    # run the experiments here
    # diffusion_OOI_2('/home/grayson/Documents/Experiment_Datafolders/diffusion_OOI_2', 10000)
    blob2therm_1_validation('/home/grayson/Desktop/Code/Data_Processing/test_dataset', 50)
