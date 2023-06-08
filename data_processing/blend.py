import os
import json
import glob
import numpy as np
import cv2
from random import randint
import matplotlib.pyplot as plt
from fpie.io import read_image
from segment_anything import SamPredictor, sam_model_registry
from data_processing.sam_functions import load_sam_model, get_mask_from_sam
from data_processing.labels import get_random_label
from fpie.process import EquProcessor

import io

class OOI_Blender:
    def __init__(self, **kwargs):
        self.data_type = kwargs.get('data_type', 'thermal')
        self.class_type = kwargs.get('class_type', 'all')
        self.data_split = kwargs.get('data_split', 'whole')
        self.multi_object_param = kwargs.get('multi_object_param', .3)

    def blend_ooi_to_backgrounds(self, 
                                 num_samples,
                                 background_folder,
                                 experiment_name,
                                 save_path=None,
                                 save=False,
                                 visualize=True,
                                 multi_object_param=None):
        """Blends objects of interest into background images.
        
        Parameters
            num_samples (int): number of new images to create and save
            background_folder (str): file path to the folder of backgrounds to
             blend ooi into
            experiment_name (str): this is the name of the folder that the images
             will be saved to within the Tarsier_Main_Dataset
            save_path (str): folder to save blended images to (if not specified,
             will automatically add images to the correct locations in the 
             Tarsier_Main_Dataset)
            visualize (bool): if true, visualizes the process (helps with debugging)
            multi_object_param (float b/w 0 and 1): percentage chance to add
             another object to a background (add multiple ooi in one background
             image)
        Returns

        """
        if multi_object_param is None:
            multi_object_param = self.multi_object_param

        STEPS = 5000
        IMAGES_BASE_PATH = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'),
                                    'Images')
        
        background_img_folder_path = background_folder

        # define sam model and fpie solver
        sam = load_sam_model()
        solver = EquProcessor("max", "numpy", 1, 0, 1024) # src - avg - max
        mask_predictor = SamPredictor(sam)

        # get possible background images from background img folder
        background_img_paths = glob.glob(background_img_folder_path+'/*')
        count = 0

        # sample *num_samples* new images
        while count < num_samples:

            # set this to true to begin multi-object loop
            add_another_object = True

            # get random background
            idx_background = np.random.randint(0, len(background_img_paths))
            background = read_image(background_img_paths[idx_background])
            
            # define probability to add another object in the image at each loop
            prob = multi_object_param
            bboxes = []
            
            # dict that keeps track of the number of different classes of object
            # in each new img
            classes = {'uav': 0, 'airplane': 0, 'bird': 0, 'helicopter': 0}
            labels = []

            while add_another_object:

                # pull the img info from a random label in the main dataset
                img_info = get_random_label(data_type=self.data_type,
                                            data_split=self.data_split,
                                            class_type=self.class_type)

                # get relevant information from img_info
                # get random idx from labels
                labels_idx = np.random.randint(0, len(img_info['labels']))
                label = img_info['labels'][labels_idx]
                bbox = label['bbox']
                src_img_path = os.path.join(IMAGES_BASE_PATH, 
                                            img_info['image_path'])
                src = read_image(src_img_path)

                # use segment anything to generate a mask from the src ooi img
                mask = get_mask_from_sam(src, bbox, mask_predictor)

                # check if the mask is too close to the edge, if so, skip this 
                # img. this is done to prevent introducing ooi that are cut off
                # at the edges
                too_close = self._is_too_close_to_edge(mask, buffer=10)
                if too_close:
                    continue

                # increment class label in classes dict
                category_name = label['category_name']
                # if the class is 'ufo' skip it
                if category_name not in classes:
                    continue
                classes[category_name] = classes[category_name] + 1

                # randomize the location to place ooi within the background img
                (src_random, 
                mask_random, 
                bbox) = self._random_placement(src.copy(), mask.copy(), 
                                               background, buffer=10)
                # if src_random is None, that means an invalid bbox was returned.
                # Thus, we skip this object of interest
                if src_random is None:
                    continue
                
                # add to bboxes and labels lists
                label['bbox'] = bbox
                bboxes.append(bbox)
                labels.append(label)

                # blend the ooi into the background img using fpie library
                try:
                    n_random = solver.reset(src_random, 
                                            mask_random, 
                                            background, 
                                            (0, 0), 
                                            (0, 0))
                    new_img_random, err = solver.step(STEPS)
                except:
                    print(bbox)
                    print(background.shape)
                    print('Exception on fpie solver! continuing...')
                    continue
                

                # create img with bboxes showing in case you wish to visualize
                new_img_w_bbox_random = new_img_random.copy()
                for bbox in bboxes:
                    start_point = int(bbox[0]), int(bbox[1])
                    end_point = int(bbox[2]), int(bbox[3])
                    new_img_w_bbox_random = cv2.rectangle(new_img_w_bbox_random, 
                                                          start_point, 
                                                          end_point, 
                                                          color=(255,0,0))

                # determine if will add another object to this background img
                add_another_object = (np.random.randint(0, 101) / 100) < prob

                # if adding another object, decrease the prob that will add
                # another object on the next loop iteration (decreases the 
                # chances of seeing 3, 4, 5, etc objects)
                if add_another_object:
                    background = new_img_random
                    prob *= prob

            # show the synthetic images with bboxes
            if visualize:
                final_img = np.concatenate((new_img_random, 
                                            new_img_w_bbox_random), axis=1)
                plt.figure('final img')
                plt.imshow(final_img)
                plt.show()

            # save the new img
            if save:
                if save_path is None:
                    img_save_path = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'),
                                                            'Images')
                else:
                    img_save_path = save_path
                category_name = max(classes, key=classes.get)
                path_to_img_folder = os.path.join(img_save_path, 
                                                  category_name,
                                                  experiment_name)
                # if the folder does not exist, create it
                if not os.path.exists(path_to_img_folder):
                    os.mkdir(path_to_img_folder)

                # get the current number of images in that folder
                num_imgs = len(glob.glob(path_to_img_folder+'/*.png'))
                img_name = f'thermal_{category_name}_{str(num_imgs).zfill(6)}.png'
                img_save_path = os.path.join(img_save_path,
                                             category_name,
                                             experiment_name,
                                             img_name)
                cv2.imwrite(img_save_path, new_img_random)

                # save the img label
                # need to update the img_info
                img_info['height'], img_info['width'], _ = new_img_random.shape
                img_info['video_id'] = experiment_name
                base_folder_path_length = len(os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'),
                                                            'Images'))
                img_info['image_path'] = img_save_path[base_folder_path_length+1:]
                img_info['labels'] = labels
                img_info['resolution'] = str(min(img_info['height'], 
                                                 img_info['width']))+'p'
                
                if save_path is None:
                    label_base_save_path = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'),
                                                        'Labels_NEW')
                    label_save_path = os.path.join(label_base_save_path,
                                                   experiment_name,
                                                   'train',
                                                   category_name,
                                                   f'{experiment_name}_{category_name}.json')
                self._save_label(label_save_path, img_info, 
                                 img_save_path.split('/')[-1][:-4])
                
            count += 1
            print(f'Generated {count} images..')
                
    def test_saved_data(self, labels_file):
        """Used to test the data generated by the 'blend_ooi_to_backgrounds_script.
        """

        # load the labels file
        with open(labels_file) as file:
            data = json.load(file)
            data = data

        for label in data:
            img_info = list(label.values())[0]
            img_path = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'),
                                                                'Images',
                                                                img_info['image_path'])
            
            # load image
            img = cv2.imread(img_path)

            # draw bboxes
            for label in img_info['labels']:
                bbox = label['bbox']
                start_point = int(bbox[0]), int(bbox[1])
                end_point = int(bbox[2]), int(bbox[3])
                img = cv2.rectangle(img,
                                    start_point,
                                    end_point,
                                    color=(255, 0, 0))
                
            plt.figure()
            plt.imshow(img)
            plt.show()
                
    def _random_placement(self, src, mask, background, buffer):
        """Get the bounding box around the mask"""

        def get_bbox_from_mask(mask, buffer):
            x1 = y1 = x2 = y2 = -1       

            for i in range(mask.shape[0]-buffer):
                if np.sum(mask[i+buffer, :]) > 0:
                    y1 = i
                    break

            for i in range(mask.shape[1]-buffer):
                if np.sum(mask[:, i+buffer]) > 0:
                    x1 = i
                    break

            for i in reversed(range(mask.shape[0] - buffer)):
                if np.sum(mask[i-buffer, :]) > 0:
                    y2 = i
                    break
            
            for i in reversed(range(mask.shape[1] - buffer)):
                if np.sum(mask[:, i-buffer]) > 0:
                    x2 = i
                    break

            if x2 < x1 or y2 < y1:
                plt.figure()
                plt.imshow(mask)
                plt.show()

                # not sure what causes this error, but if x2 < x1 or y2 < y1 skip
                return None, None, None, None

                for i in range(mask.shape[0]-buffer):
                    if np.sum(mask[i+buffer, :]) > 0:
                        y1 = i
                        break

                for i in range(mask.shape[1]-buffer):
                    if np.sum(mask[:, i+buffer]) > 0:
                        x1 = i
                        break

                for i in reversed(range(mask.shape[0] - buffer)):
                    if np.sum(mask[i-buffer, :]) > 0:
                        y2 = i
                        break
                
                for i in reversed(range(mask.shape[1] - buffer)):
                    if np.sum(mask[:, i-buffer]) > 0:
                        x2 = i
                        break

            return x1, y1, x2, y2
        
        x1, y1, x2, y2 = get_bbox_from_mask(mask, buffer)

        # skip if invalid bbox from mask
        if x1 is None:
            return None, None, None

        # get just the new bbox from the src and mask. This will be resized and
        # randomly located within the background by padding the edges of each img
        src = src[y1:y2, x1:x2, :]
        mask = mask[y1:y2, x1:x2]

        tgt_h, tgt_w, _ = background.shape

        current_w = x2-x1
        current_h = y2-y1

        # we can pad the left side with anywhere from 0 to (tgt_w - current_w) and
        # the top with anywhere from 0 to (tgt_h - current_h). We will then pad
        # the right and top sides to fill in the target shape

        start_x = np.random.randint(0, tgt_w-current_w)
        start_y = np.random.randint(0, tgt_h-current_h)

        new_bbox = (start_x+buffer, start_y+buffer, 
                    start_x+current_w-buffer, start_y+current_h-buffer)

        padded_src = np.zeros_like(background)
        padded_mask = np.zeros_like(background[...,0])

        padded_src[start_y:start_y+current_h,
                start_x:start_x+current_w, :] = src
        padded_mask[start_y:start_y+current_h,
                    start_x:start_x+current_w] = mask

        return padded_src, padded_mask, new_bbox
    
    def _is_too_close_to_edge(self, mask, buffer):
        """If object is within the buffer pixels from the edge, skip it."""
        h, w = mask.shape
        for i in range(buffer):
            if (np.sum(mask[i, :]) > 0 or np.sum(mask[:, i]) > 0
                or np.sum(mask[h-i-1, :]) > 0 or np.sum(mask[:, w-i-1]) > 0):
                return True
            
        return False
    
    def _save_label(self, label_file_path: str, img_info: dict , img_name: str) -> None:
        """Adds label to a json file. Creates a new file if it does not exist.

        Parameters:
            label_file_path (str): path to label file
            label (dict): label dictionary to add to the json file
            in the form:
            label = {
                'bbox': bbox,
                'image_path': image_path,
                'original_image_path': original_image_path
            }
        """

        label = {img_name.split('/')[-1] : img_info}

        # ensure the folder that holds the label.json file exists
        label_file_name = label_file_path.split('/')[-1]
        label_folder_path = label_file_path[:-len(label_file_name)-1]
        if not os.path.exists(label_folder_path):
            os.makedirs(label_folder_path)

        if os.path.exists(label_file_path):
            with open(label_file_path) as f:
                json_object = json.load(f)
            if type(json_object) is dict:
                json_object = [json_object]
            json_object.append(label)
        else:
            json_object = label

        with open(label_file_path, 'w') as outfile:
            json.dump(json_object, outfile)
                