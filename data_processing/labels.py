import json
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

IMAGES_BASE_PATH = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'),
                                'Images')
LABELS_BASE_PATH = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'), 
                                'Labels_NEW')

# the COCO paper uses the below classes. the coco dataset uses an expanded version
COCO_CLASSES_PAPER = [
   "person",
   "bicycle",
   "car",
   "motorcycle",
   "airplane",
   "bus",
   "train",
   "truck",
   "boat",
   "traffic_light",
   "fire_hydrant",
   "stop_sign",
   "parking_meter",
   "bench",
   "bird",
   "cat",
   "dog",
   "horse",
   "sheep",
   "cow",
   "elephant",
   "bear",
   "zebra",
   "giraffe",
   "backpack",
   "umbrella",
   "handbag",
   "tie",
   "suitcase",
   "frisbee",
   "skis",
   "snowboard",
   "sports_ball",
   "kite",
   "baseball_bat",
   "baseball_glove",
   "skateboard",
   "surfboard",
   "tennis_racket",
   "bottle",
   "wine_glass",
   "cup",
   "fork",
   "knife",
   "spoon",
   "bowl",
   "banana",
   "apple",
   "sandwich",
   "orange",
   "broccoli",
   "carrot",
   "hot_dog",
   "pizza",
   "donut",
   "cake",
   "chair",
   "couch",
   "potted_plant",
   "bed",
   "dining_table",
   "toilet",
   "tv",
   "laptop",
   "mouse",
   "remote",
   "keyboard",
   "cell_phone",
   "microwave",
   "oven",
   "toaster",
   "sink",
   "refrigerator",
   "book",
   "clock",
   "vase",
   "scissors",
   "teddy_bear",
   "hair_drier",
   "toothbrush",
]

# these are the coco classes used by the official coco dataset
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic_light",
    "fire_hydrant",
    "street_sign",
    "stop_sign",
    "parking_meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye_glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports_ball",
    "kite",
    "baseball_bat",
    "baseball_glove",
    "skateboard",
    "surfboard",
    "tennis_racket",
    "bottle",
    "plate",
    "wine_glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot_dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted_plant",
    "bed",
    "mirror",
    "dining_table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell_phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy_bear",
    "hair_drier",
    "toothbrush",
    "hairbrush"
]

# get the dict name2num for the coco dataset classes
COCO_CLASSES_DICT_NAME2NUM = {}
COCO_CLASSES_DICT_NUM2NAME = {}
for num, class_type in enumerate(COCO_CLASSES):
    COCO_CLASSES_DICT_NAME2NUM[class_type] = num+1
    COCO_CLASSES_DICT_NUM2NAME[num+1] = class_type

# get the dict name2num for the coco paper classes
COCO_CLASSES_DICT_NAME2NUM_PAPER_VERSION = {}
COCO_CLASSES_DICT_NUM2NAME_PAPER_VERSION = {}
for num, class_type in enumerate(COCO_CLASSES_PAPER):
    COCO_CLASSES_DICT_NAME2NUM_PAPER_VERSION[class_type] = num+1
    COCO_CLASSES_DICT_NUM2NAME_PAPER_VERSION[num+1] = class_type

COCO_CATEGORIES = [
    {"supercategory": "person","id": 1,"name": "person"},
    {"supercategory": "vehicle","id": 2,"name": "bicycle"},
    {"supercategory": "vehicle","id": 3,"name": "car"},
    {"supercategory": "vehicle","id": 4,"name": "motorcycle"},
    {"supercategory": "vehicle","id": 5,"name": "airplane"},
    {"supercategory": "vehicle","id": 6,"name": "bus"},
    {"supercategory": "vehicle","id": 7,"name": "train"},
    {"supercategory": "vehicle","id": 8,"name": "truck"},
    {"supercategory": "vehicle","id": 9,"name": "boat"},
    {"supercategory": "outdoor","id": 10,"name": "traffic light"},
    {"supercategory": "outdoor","id": 11,"name": "fire hydrant"},
    {"supercategory": "outdoor","id": 13,"name": "stop sign"},
    {"supercategory": "outdoor","id": 14,"name": "parking meter"},
    {"supercategory": "outdoor","id": 15,"name": "bench"},
    {"supercategory": "animal","id": 16,"name": "bird"},
    {"supercategory": "animal","id": 17,"name": "cat"},
    {"supercategory": "animal","id": 18,"name": "dog"},
    {"supercategory": "animal","id": 19,"name": "horse"},
    {"supercategory": "animal","id": 20,"name": "sheep"},
    {"supercategory": "animal","id": 21,"name": "cow"},
    {"supercategory": "animal","id": 22,"name": "elephant"},
    {"supercategory": "animal","id": 23,"name": "bear"},
    {"supercategory": "animal","id": 24,"name": "zebra"},
    {"supercategory": "animal","id": 25,"name": "giraffe"},
    {"supercategory": "accessory","id": 27,"name": "backpack"},
    {"supercategory": "accessory","id": 28,"name": "umbrella"},
    {"supercategory": "accessory","id": 31,"name": "handbag"},
    {"supercategory": "accessory","id": 32,"name": "tie"},
    {"supercategory": "accessory","id": 33,"name": "suitcase"},
    {"supercategory": "sports","id": 34,"name": "frisbee"},
    {"supercategory": "sports","id": 35,"name": "skis"},
    {"supercategory": "sports","id": 36,"name": "snowboard"},
    {"supercategory": "sports","id": 37,"name": "sports ball"},
    {"supercategory": "sports","id": 38,"name": "kite"},
    {"supercategory": "sports","id": 39,"name": "baseball bat"},
    {"supercategory": "sports","id": 40,"name": "baseball glove"},
    {"supercategory": "sports","id": 41,"name": "skateboard"},
    {"supercategory": "sports","id": 42,"name": "surfboard"},
    {"supercategory": "sports","id": 43,"name": "tennis racket"},
    {"supercategory": "kitchen","id": 44,"name": "bottle"},
    {"supercategory": "kitchen","id": 46,"name": "wine glass"},
    {"supercategory": "kitchen","id": 47,"name": "cup"},
    {"supercategory": "kitchen","id": 48,"name": "fork"},
    {"supercategory": "kitchen","id": 49,"name": "knife"},
    {"supercategory": "kitchen","id": 50,"name": "spoon"},
    {"supercategory": "kitchen","id": 51,"name": "bowl"},
    {"supercategory": "food","id": 52,"name": "banana"},
    {"supercategory": "food","id": 53,"name": "apple"},
    {"supercategory": "food","id": 54,"name": "sandwich"},
    {"supercategory": "food","id": 55,"name": "orange"},
    {"supercategory": "food","id": 56,"name": "broccoli"},
    {"supercategory": "food","id": 57,"name": "carrot"},
    {"supercategory": "food","id": 58,"name": "hot dog"},
    {"supercategory": "food","id": 59,"name": "pizza"},
    {"supercategory": "food","id": 60,"name": "donut"},
    {"supercategory": "food","id": 61,"name": "cake"},
    {"supercategory": "furniture","id": 62,"name": "chair"},
    {"supercategory": "furniture","id": 63,"name": "couch"},
    {"supercategory": "furniture","id": 64,"name": "potted plant"},
    {"supercategory": "furniture","id": 65,"name": "bed"},
    {"supercategory": "furniture","id": 67,"name": "dining table"},
    {"supercategory": "furniture","id": 70,"name": "toilet"},
    {"supercategory": "electronic","id": 72,"name": "tv"},
    {"supercategory": "electronic","id": 73,"name": "laptop"},
    {"supercategory": "electronic","id": 74,"name": "mouse"},
    {"supercategory": "electronic","id": 75,"name": "remote"},
    {"supercategory": "electronic","id": 76,"name": "keyboard"},
    {"supercategory": "electronic","id": 77,"name": "cell phone"},
    {"supercategory": "appliance","id": 78,"name": "microwave"},
    {"supercategory": "appliance","id": 79,"name": "oven"},
    {"supercategory": "appliance","id": 80,"name": "toaster"},
    {"supercategory": "appliance","id": 81,"name": "sink"},
    {"supercategory": "appliance","id": 82,"name": "refrigerator"},
    {"supercategory": "indoor","id": 84,"name": "book"},
    {"supercategory": "indoor","id": 85,"name": "clock"},
    {"supercategory": "indoor","id": 86,"name": "vase"},
    {"supercategory": "indoor","id": 87,"name": "scissors"},
    {"supercategory": "indoor","id": 88,"name": "teddy bear"},
    {"supercategory": "indoor","id": 89,"name": "hair drier"},
    {"supercategory": "indoor","id": 90,"name": "toothbrush"}
]

WALARIS_CLASS_LABELS_NAME2NUM = {
    "uav": 1,
    "airplane": 2, 
    "bicycle": 3,
    "bird": 4,
    "boat": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "cow": 9,
    "dog": 10,
    "horse": 11,
    "motorcycle": 12,
    "person": 13,
    "traffic_light": 14,
    "train": 15,
    "truck": 16,
    "ufo": 17,
    "helicopter": 18,
    "phantom": 19,
    "mavic": 20,
    "spark": 21,
    "inspire": 22
}

WALARIS_CLASS_LABELS_NUM2NAME = {
    1: "uav",
    2: "airplane", 
    3: "bicycle",
    4: "bird",
    5: "boat",
    6: "bus",
    7: "car",
    8: "cat",
    9: "cow",
    10: "dog",
    11: "horse",
    12: "motorcycle",
    13: "person",
    14: "traffic_light",
    15: "train",
    16: "truck",
    17: "ufo",
    18: "helicopter",
    19: "phantom",
    20: "mavic",
    21: "spark",
    22: "inspire"
}

MAPPING_WALARIS_TO_COCO = {
    1: 5,   # uav (1) -> airplance (5)
    2: 5,   # airplane (2) -> airplane (5)
    3: 2,   # bicycle (3) -> bicycle (2)
    4: 16,  # bird (4) -> bird (16)
    5: 9,   # boat (5) -> boat (9)
    6: 6,   # bus (6) -> bus (6)
    7: 3,   # car (7) -> car (3)
    8: 17,  # cat (8) -> cat (17)
    9: 21,  # cow (9) -> cow (21)
    10: 18, # dog (10) -> dog (18)
    11: 19, # horse (11) -> horse (19)
    12: 4,  # motorcycle (12) -> motorcycle (4)
    13: 1,  # person (13) -> person (1)
    14: 10, # traffic_light (14) -> traffic light (10)
    15: 7,  # train (15) -> train (7)
    16: 8,  # truck (16) -> truck (8)
    17: 16, # ufo (17) -> bird (16)
    18: 5   # helicopter (18) -> airplane (5)
}

MAPPING_COCO_TO_WALARIS = {
    # in progress
}

# Functions for labels and datasets in the Walaris format

def get_img_paths_walaris_format(exclude_synthetic=True):
    """Get a python set of all relative image paths for the entire dataset from
    the walaris standard format labels.
    
    Args:
        exclude_synthetic (bool): if true, will not include the image paths to 
            synthetic images

    Returns:

    """
    
    img_paths = set()

    # get a list of all image paths in labels
    for data_type_folder in tqdm(glob.glob(LABELS_BASE_PATH+'/*')):

        # skip synthetic data if specified
        if 'synthetic' in data_type_folder and exclude_synthetic:
            continue

        # loop through each object type folder in the 'whole' directory
        for object_type_folder in glob.glob(data_type_folder+'/whole/*'):

            # loop through each label and add the image path to set
            for label_json_file in glob.glob(object_type_folder+'/*'):

                # load the label information
                with open(label_json_file, 'r') as file:
                    label_info_dict = json.load(file)

                # loop through all images in the label info dictionary and add 
                # the image path to the set
                for img_label in label_info_dict:
                    img_paths.add(label_info_dict[img_label]['image_path'])

        # the day data has a unique label directory called 'test'. Loop through
        # this if the data_type_folder is 'day'
        if 'day' in data_type_folder:
            for object_type_folder in glob.glob(data_type_folder+'/test_(not_included_in_whole)/*'):

                # loop through each label and add the image path to set
                for label_json_file in glob.glob(object_type_folder+'/*'):

                    # load the label information
                    with open(label_json_file, 'r') as file:
                        label_info_dict = json.load(file)

                    # loop through all images in the label info dictionary and add 
                    # the image path to the set
                    for img_label in label_info_dict:
                        img_paths.add(label_info_dict[img_label]['image_path'])

    return img_paths

def get_img_info(label_path, img_name):
    """Return the img_info dictionary corresponding to a img_name in a label
    file.
    
    Args:
        label_path (str): path to the label .json file
        img_name (str): name of the image to get the img info for

    Returns:
        img_info (dict): walaris img info dictionary
    """

    # if img_name is pass with .png or .jpg extension, remove it
    img_name = img_name.split('.')[0]

    with open(label_path, 'r') as f:
        info_list = json.load(f)
        if img_name in info_list:
            img_info = info_list[img_name]
        else:
            return None

    return img_info

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
    with open(labels_json_file, 'r') as f:
        labels = json.load(f)

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

    Args:
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
    with open(labels_json_file, 'r') as f:
        labels = json.load(f)

    # get random image name
    labels_list = list(labels)
    idx = np.random.randint(0, len(labels_list))
    img_name = labels_list[idx]

    # get image information
    img_info = get_img_info(labels_json_file, img_name)
   
    return img_info

def save_dict_to_json(json_file_path, dictionary, delete=False):
    """Adds dictionary to a json file. Creates a new file if it does not exist.

    Args:
        label_file_path (str): path to label file
        label (dict): label dictionary to add to the json file
        delete (bool): If true, deletes any information that was already stored
            in the json file.
    Returns:

    """

    if os.path.exists(json_file_path):
        with open(json_file_path) as f:
            json_object = json.load(f)
        if type(json_object) is dict:
            json_object = [json_object]
        json_object.append(dictionary)
        if delete:
            os.remove(json_file_path)
            json_object = dictionary
    else:
        json_object = dictionary

    with open(json_file_path, 'w') as outfile:
        json.dump(json_object, outfile)

    return

def create_labels(bboxes, class_labels):
    """ Creates a list of bbox labels with class_label information

    Args:
        bboxes (list): list of bounding boxes in an image
        class_labels (list): list of class labels in an image that correspond
         in order to the list of bounding boxes

    Returns:
        label (list): 
            (example)
            [
                {'bbox': [x1, y1, x2, y2], 'category_id': x, 'category_name': xxxx},
                {'bbox': [x1, y1, x2, y2], 'category_id': x, 'category_name': xxxx},
                {'bbox': [x1, y1, x2, y2], 'category_id': x, 'category_name': xxxx},
            ]
    """
    assert len(bboxes) == len(class_labels), "Error: Number of bounding boxes \
        does not match number of class labels."
    labels = []
    for idx in range(len(bboxes)):
        bbox = bboxes[idx]
        category_name = class_labels[idx]
        category_id = WALARIS_CLASS_LABELS_NAME2NUM[category_name]

        labels.append({
            'bbox': bbox,
            'category_name': category_name,
            'category_id': category_id
        })

    return labels

def create_walaris_img_info(img_path, bboxes, class_labels):
    """Creates an image info dictionary in the Walaris standard format.

    Args:
        img_path (str): file path of the image
        bboxes (list): list of bounding boxes for the objects detected in the
         image
        class_labels (list): list of class labels that correspond to the list
         of bounding boxes

    Returns:
        label (dict): A label dictionary that follows the standard Walaris
         label format
    """
    img_path = os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'),
                            'Images',
                            ('/').join(img_path.split('/')[-3:]))
    img = cv2.imread(img_path)
    global_class_label, video_id, img_name = img_path.split('/')[-3:]
    height, width, _ = img.shape
    labels = create_labels(bboxes, class_labels)
    resolution = f'{height}p'

    img_info = {
        'height': height,
        'width': width,
        'resolution': resolution,
        'video_id': video_id,
        'image_path': ('/').join(img_path.split('/')[-3:]),
        'labels': labels
    }

    return img_info, img_name.split('.')[0]

def get_walaris_img_info_from_img_name(img_name):
    """Given a string of the name of the image (ie. thermal_uav_574_000058.png)
    returns a dictionary with the image information.
     
    Args:
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

def get_img_paths_coco_format(exclude_synthetic=True):
    """Get a python set of all relative image paths for the entire dataset from
    the walaris standard format labels.
    
    *note: you MUST have the following directory structure and naming 
    convention. any labels folder (val, train, whole) MUST have a sister coco
    folder with _COCO appended to the end of the folder name

    |--Tarsier_Main_Dataset
        |--Labels_NEW
            |--day
                |-- train
                |--train_COCO
                |--val
                |--val_COCO
                |--whole
                |--whole_COCO
            |--night
            |--thermal
            |--thermal_synthetic
    
    Args:
        exclude_synthetic (bool): if true, will not include the image paths to 
            synthetic images

    Returns:
    """

    img_paths = set()

    # get a list of all image paths in labels
    for data_type_folder in tqdm(glob.glob(LABELS_BASE_PATH+'/*')):

        # skip synthetic data if specified
        if 'synthetic' in data_type_folder and exclude_synthetic:
            continue

        # loop through each object type folder in the 'whole' directory
        for label_json_file in glob.glob(data_type_folder+'/whole_COCO/*.json'):

            # load the label information
            with open(label_json_file, 'r') as file:
                label_info_dict = json.load(file)

            # loop through all img dicts and add the relative file name to 
            # the set
            for img in label_info_dict['images']:
                img_paths.add(img['file_name'])

        # the day data has a unique label directory called 'test'. Loop through
        # this if the data_type_folder is 'day'
        if 'day' in data_type_folder:
            for label_json_file in glob.glob(data_type_folder+'/test_(not_included_in_whole)_COCO/*.json'):

                # load the label information
                with open(label_json_file, 'r') as file:
                    label_info_dict = json.load(file)

                # loop through all img dicts and add the relative file name to 
                # the set
                for img in label_info_dict['images']:
                    img_paths.add(img['file_name'])

    return img_paths

def get_rand_sample_from_coco_json(original_json_file,
                                     new_json_file,
                                     sample_size,
                                     seed=None,
                                     include_unlabelled_images=False):
    """ Randomly sample a coco format dataset from a coco format dataset.

    Args:
        original_json_file (str): file path of json file to sample from
        new_json_file (str): file path to the new random sample json file
        sample_size (int): number of samples in the random sample
        seed (int): pass a seed for reproducability
        include_unlabelled_images (bool): If False, check to make sure that
            sampled images have annotations in them before adding them to the
            sample.

    Returns:

    """

    def get_annotations_by_img_id(annotations,
                                  target_img_id):
        """ Gets a list of annotations that correspond to a specific img_id.
        
        Args:
            annotations (list(dict)): a list of annotation dicts from a coco
                format dataset
            img_id (int or str): image id to match annotations to
            
        Returns:
            matching_annotations (list(dict)): a list of annotations with
                img_ids that match the target_img_id
        """

        # binary search to find annotation
        l_ptr = 0
        r_ptr = len(annotations)

        target_img_id = int(target_img_id)

        idx = -1
        while l_ptr <= r_ptr:
            mid = int(r_ptr - l_ptr) // 2 + l_ptr
            current_img_id = annotations[mid]['image_id']
            if current_img_id == target_img_id:
                idx = mid
                break
            elif current_img_id > target_img_id:
                r_ptr = mid - 1
            else:
                l_ptr = mid + 1

        if idx == -1:
            return None

        # look to the left and to the right to get all of the annotations with
        # the same image id
        matching_annotations = []
        ptr = idx
        while(ptr >= 0
            and annotations[ptr]['image_id'] == target_img_id):
            matching_annotations.append(annotations[ptr])
            ptr -= 1
        ptr = idx+1
        while(ptr < len(annotations)
            and annotations[ptr]['image_id'] == target_img_id):
            matching_annotations.append(annotations[ptr])
            ptr += 1

        return matching_annotations
            

    with open(original_json_file, 'r') as file:
        data =json.load(file)

    images = data['images']
    annotations = data['annotations']

    # sort the annotations by image id for binary search later in algorithm
    annotations = sorted(annotations, key=lambda x: x['image_id'])

    if not include_unlabelled_images:
        # remove images without annotations (there are no detections on some images)
        l_ptr = r_ptr = 0
        while r_ptr < len(images):
            target_img_id = images[r_ptr]['id']
            matching_annotations = get_annotations_by_img_id(annotations,
                                                             target_img_id)
            if matching_annotations is not None:
                images[l_ptr], images[r_ptr] = images[r_ptr], images[l_ptr]
                l_ptr += 1
            
            r_ptr += 1
        
        images = images[:l_ptr]
    
    # use a seed for reproducability if specified
    if seed is not None:
        random.seed(seed)
    
    random.shuffle(images)

    sampled_images = images[:sample_size]
    sampled_img_annotations = []
    for img_dict in sampled_images:
        target_img_id = img_dict['id']

        # binary search for image id match in annotations
        l_ptr = 0
        r_ptr = len(annotations)

        idx = -1
        while l_ptr < r_ptr:
            mid = int(r_ptr - l_ptr - 1) // 2 + l_ptr
            current_img_id = annotations[mid]['image_id']
            if current_img_id == target_img_id:
                idx = mid
                break
            elif current_img_id > target_img_id:
                r_ptr = mid - 1
            else:
                l_ptr = mid + 1

        if idx == -1:
            print("Error: No annotations found for this image. Continuing..")
            continue

        # look to the left and to the right to get all of the annotations with
        # the same image id
        ptr = idx
        while(ptr >= 0
              and annotations[ptr]['image_id'] == target_img_id):
            sampled_img_annotations.append(annotations[ptr])
            ptr -= 1
        ptr = idx+1
        while(ptr < len(annotations)
              and annotations[ptr]['image_id'] == target_img_id):
            sampled_img_annotations.append(annotations[ptr])
            ptr += 1

    data['images'] = sampled_images
    data['annotations'] = sampled_img_annotations

    with open(new_json_file, 'w') as file:
        json.dump(data, file)

    return

def get_category_info_coco_format(json_file,
                                  label_convention,
                                  isPrint=False):
    """Print out information in the terminal regarding the categories found and 
    number of categories in a coco ground truth or results dataset.
    
    Args:
        json_file (str): file path to the json folder to read from
        label_convention (str): specify which label convention the data is
            using

    Returns:

    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    annotations = data['annotations']
    class_labels_present = dict()

    for annotation in annotations:
        if annotation['category_id'] not in class_labels_present:
            class_labels_present[annotation['category_id']] = 1
        else:
            class_labels_present[annotation['category_id']] += 1
    if isPrint:
        num_images = len(data['images'])
        print(f'{num_images} images in dataset...')
        for class_label in class_labels_present:
            if label_convention == 'walaris':
                print(f'{class_label}: {WALARIS_CLASS_LABELS_NUM2NAME[class_label]} - {class_labels_present[class_label]}')
            elif label_convention == 'coco':
                print(f'{class_label}: {COCO_CLASSES_DICT_NUM2NAME[class_label]} - {class_labels_present[class_label]}')

    class_labels_by_name = {}
    for class_label in class_labels_present:
        num_object_present = class_labels_present[class_label]
        if label_convention == 'walaris':
                class_labels_by_name[WALARIS_CLASS_LABELS_NUM2NAME[class_label]] = num_object_present
        elif label_convention == 'coco':
            class_labels_by_name[COCO_CLASSES_DICT_NUM2NAME[class_label]] = num_object_present

    return class_labels_by_name