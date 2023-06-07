import json
import os
import glob
import numpy as np
import cv2

IMAGES_BASE_PATH = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'),
                                'Images')
LABELS_BASE_PATH = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'), 
                                'Labels_NEW')

def get_img_info(label_path, img_name):
    # if img_name is pass with .png or .jpg extension, remove it
    img_name = img_name.split('.')[0]

    with open(label_path, 'r') as f:
        info_list = json.load(f)
        if img_name in info_list:
            img_info = info_list[img_name]
        else:
            return None

    return img_info

def get_labels_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data

def get_random_label_helper(data_type_folder, data_split, class_type):
    """Helper function for the get_random_label function. This returns None 
    if a data folder doesnt exist as well as the data_type_folder."""

    # make sure the selected folder has the data split folder specified
    data_split_folders = glob.glob(data_type_folder+'/*')
    data_split_folder_exists = False
    for data_split_folder in data_split_folders:
        if data_split_folder.split('/')[-1]  == data_split:
            data_folder = data_split_folder
            data_split_folder_exists = True
            break
    
    if data_split_folder_exists == False:
        print('skip')
        return None, data_type_folder

    # make sure the selected folder has the data class folder specified
    data_class_folders = glob.glob(data_folder+'/*')
    if class_type == 'all':
        idx = np.random.randint(0, len(data_class_folders))
        data_folder = data_class_folders[idx]
    else:
        data_class_folder_exists = False
        for data_class_folder in data_class_folders:
            if data_class_folder.split('/')[-1] == class_type:
                data_folder = data_class_folder
                data_class_folder_exists = True
                break
        
        if data_class_folder_exists == False:
            print('skip')
            return None, data_type_folder

    label_files = glob.glob(data_folder+'/*')
    idx = np.random.randint(0, len(label_files))
    labels_json_file = label_files[idx]

    # get dictionary of labels
    labels = get_labels_from_json(labels_json_file)

    # get random image name
    labels_list = list(labels)
    idx = np.random.randint(0, len(labels_list))
    img_name = labels_list[idx]

    # get image information
    img_info = get_img_info(labels_json_file, img_name)
   
    return img_info, data_type_folder

def get_random_label(data_type='all',
                     data_split='whole',
                     class_type='all',
                     include_thermal_synthetic=False):
    """Provide the image information from a random image from the 
    Tarsier_Main_Dataset.

    Parameters:
        data_type (str): Specifies the type of data (ie thermal).
        data_split (str): Specifies train, val, or whole data split.
        class_type (str): Specifies which class of object to include in random
         search.
        include_thermal_synthetic (bool): If true, will include the thermal
         synthetic data type folder in random search.

    Returns:
        img info (dict):
    """

    # assertions
    possible_data_types = ['all', 'day', 'night',
                            'thermal', 'thermal_synthetic']
    possible_data_splits = ['train', 'val', 'whole']
    assert data_type in possible_data_types, "Make sure specified data_type is \
        valid ('all', 'day', 'night', 'thermal', 'thermal_synthetic')"
    assert data_split in possible_data_splits, "Make sure data split is valid \
        ('train', 'val', or 'whole')"

    if data_type == 'all':
        data_type_folders = glob.glob(LABELS_BASE_PATH+'/*')
        data_type_folders.sort()
        if include_thermal_synthetic == False:
            del data_type_folders[-1]
    else:
        data_type_folders = [(os.path.join(LABELS_BASE_PATH, data_type))]

    # get random data_type folder
    while True:
        idx = np.random.randint(0, len(data_type_folders))
        data_type_folder = data_type_folders[idx]
        if data_type == 'all':
            img_info, _ = get_random_label_helper(data_type_folder, data_split,
                                                  class_type)
            if img_info is None:
                continue
            else:
                return img_info
        else:
            break

    # make sure the selected folder has the data split folder specified
    data_split_folders = glob.glob(data_type_folder+'/*')
    data_split_folder_exists = False
    for data_split_folder in data_split_folders:
        if data_split_folder.split('/')[-1]  == data_split:
            data_folder = data_split_folder
            data_split_folder_exists = True
            break
    
    assert data_split_folder_exists, """Make sure your specified data_split
        exists in the subfolder"""

    # make sure the selected folder has the data class folder specified
    data_class_folders = glob.glob(data_folder+'/*')
    if class_type == 'all':
        idx = np.random.randint(0, len(data_class_folders))
        data_folder = data_class_folders[idx]
    else:
        data_class_folder_exists = False
        for data_class_folder in data_class_folders:
            if data_class_folder.split('/')[-1] == class_type:
                data_folder = data_class_folder
                data_class_folder_exists = True
                break
        
        assert data_class_folder_exists, """Make sure your specified data_split
            exists in the subfolder"""

    label_files = glob.glob(data_folder+'/*')
    idx = np.random.randint(0, len(label_files))
    labels_json_file = label_files[idx]

    # get dictionary of labels
    labels = get_labels_from_json(labels_json_file)

    # get random image name
    labels_list = list(labels)
    idx = np.random.randint(0, len(labels_list))
    img_name = labels_list[idx]

    # get image information
    img_info = get_img_info(labels_json_file, img_name)
   
    return img_info

def get_img_from_img_info(img_info):
    img = cv2.imread(os.path.join(IMAGES_BASE_PATH, img_info['image_path']))

    return img

def save_label_to_file(label_file_path, label):
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

    return

def get_img_info_from_img_name(img_name):
    """Given a string of the name of the image (ie. thermal_uav_574_000058.png)
    returns a dictionary with the image information.
     
    Parameters:
        img_name (str): name of the image to get the label for
     
    Returns:
        img_info (dict):
    
    """
    # if img_name is pass with .png or .jpg extension, remove it
    img_name = img_name.split('.')[0]

    img_name_info = img_name.split('_')

    if len(img_name_info) == 4:
        data_type = img_name_info[0]
        object_type = img_name_info[1]
        video_number = int(img_name_info[2])
        image_number = int(img_name_info[3])
    else:
        data_type = 'day'
        object_type = img_name_info[0]
        video_number = int(img_name_info[1])
        image_number = int(img_name_info[2])

    if data_type == 'blank':
        data_type = 'day'

    labels_json_file = os.path.join(LABELS_BASE_PATH, 
                                    data_type, 
                                    'whole',object_type,
                                    img_name[:-7]+'.json')

    # get image information
    img_info = get_img_info(labels_json_file, img_name)
   
    return img_info

if __name__=='__main__':
    pass
