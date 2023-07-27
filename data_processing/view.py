import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
import random

from data_processing.labels import COCO_CLASSES_DICT_NUM2NAME, COCO_CLASSES_DICT_NAME2NUM, WALARIS_CLASS_LABELS_NUM2NAME, get_random_label

BASE_DATASET_PATH = os.environ.get('WALARIS_MAIN_DATA_PATH')
BASE_IMAGE_PATH = os.path.join(BASE_DATASET_PATH, 'Images')
BASE_LABELS_PATH = os.path.join(BASE_DATASET_PATH, 'Labels_NEW')

def show_bboxes(boxes, ax, bbox_format: str, labels=None):
    """Displays an image with bounding boxes and labels drawn.
    
    Args:
        boxes (list): list of bounding boxes
        ax (plt.ax object): axis object from matplotlib.pyplot
        bbox_format (str): specify the format for the bboxes (xyxy | xywh)
        labels (list): a list of labels in string format
        
    Returns:
    
    """
    def show_bbox(box, ax, bbox_format: str, label=None):
        """Draw one bbox and corresponding label to a plt.ax object."""
        if bbox_format == 'xyxy':
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                                        facecolor=(0,0,0,0), lw=2))
        elif bbox_format == 'xywh_min' or bbox_format == 'xywh_coco':
            x0, y0 = box[0], box[1]
            w, h = box[2], box[3]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                                        facecolor=(0,0,0,0), lw=1))
        elif bbox_format == 'xywh_center' or bbox_format == 'xywh_yolo':
            x0, y0 = box[0], box[1]
            w, h = box[2], box[3]
            ax.add_patch(plt.Rectangle((x0-w/2, y0-h/2), w, h, edgecolor='green', 
                                        facecolor=(0,0,0,0), lw=1))
        if label is not None:
            if bbox_format == 'xyxy':
                ax.text(int(x0), int(y0)-10, label, color='red', fontsize=10)
            elif bbox_format == 'xywh_min' or bbox_format == 'xywh_coco':
                ax.text(int(x0), int(y0)-10, label, color='red', fontsize=10)
            elif bbox_format == 'xywh_center' or bbox_format == 'xywh_yolo':
                ax.text(int(x0)-w/2, int(y0)-h/2-10, label, color='red', fontsize=10)

    if labels is not None:
        for idx in range(len(boxes)):
            show_bbox(boxes[idx], ax, 
                          bbox_format, label=labels[idx])
    else:
        for idx in range(len(boxes)):
            show_bbox(boxes[idx], ax, bbox_format)
    
def show_masks(masks, ax, random_color=True):
    """Displays an image with bounding boxes and labels drawn.
    
    Args:
        masks (list): list of masks in np binary format (1 channel)
        ax (plt.ax object): axis object from matplotlib.pyplot
        random_color (bool): specifies whether a random color will be used
        
    Returns:
    
    """
    def show_mask(mask, ax, random_color=False):
        """Draw one mask to a plt.ax object."""
        try:
            mask = mask.detach().cpu().numpy()
        except:
            pass
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    for mask in masks:
        show_mask(mask, ax, random_color)

def convert_mask_binary2rgb(binary_mask):
    """Goes from binary (h, w, 1) -> rgb (h, w, 3) where all channels are 255 
    (white) or 0 (black)
    
    Args:
        binary_mask (numpy arr): binary mask to convert

    Returns:
        rgb_mask (np array): converted rgb mask
    """

    h, w, _ = binary_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype='uint8')
    for i in range(3):
        rgb_mask[:, :, i] = binary_mask[:, :, 0]
    return rgb_mask * 255

def convert_mask_rgb2binary(rgb_mask):
    """Goes from rgb (h, w, 3) -> binary (h, w, 1) where each pixel is 1 or 0.
    
    Args:
        rgb_mask (numpy arr): rgb mask to convert

    Returns:
        binary_mask (np array): binary rgb mask
    """

    h, w, _ = rgb_mask.shape

    rgb_mask = np.max(rgb_mask, axis=2)

    ones = np.ones((h, w))
    binary_mask = ones <= rgb_mask
    
    binary_mask = np.expand_dims(binary_mask, axis=2)
   

    return binary_mask

def cut_out_image(img, rgb_mask):
    """Turns every pixel in the image not in the mask to (0, 0, 0) (black).
    
    Args:
        img (np.array): img to process
        mask (np.array): rgb format mask
        
    Returns:
        img (np.array): img with all pixels not in the mask set to black"""

    assert rgb_mask.shape == rgb_mask.shape, "Error: Image and mask must be the same shape"
    h, w, _ = rgb_mask.shape
    rgb_mask = rgb_mask.astype('uint8')

    for r in range(h):
        for c in range(w):
            if rgb_mask[r, c, 0] == False:
                img[r, c, :] *= 0
    
    return img

def visualize_walaris_dataset(data_domain='all',
                              data_split='whole',
                              class_type='all',
                              include_thermal_synthetic=False):
    """Will randomly select images and display them with the bounding boxes.
    
    Args:
        data_domain (str): day | thermal | night | all
        data_split (str): val | train | whole
        class_type (str): airplane | uav | helicopter | bird | all
        include_thermal_synthetic (bool): If true, will include the thermal
            synthetic data type folder in random search.

    Returns:
    
    """

    while 1:
        random_label_dict = get_random_label(data_domain,
                                        data_split,
                                        class_type,
                                        include_thermal_synthetic)
        
        visualize_walaris_label(random_label_dict)


def visualize_walaris_label(label_dict):
    """Given the image name and label file, show the image on the screen with
    the bounding boxes superimposed on the image. Will be used to verify the 
    labels. The label file is assumed to be in the Walaris standard format.
    
    Args:
        label_dict (dict): dictionary for a label in the walaris format
        
        Example Label Dict:
        {"video_id": "day_airplane_0", 
        "width": 848, 
        "height": 480, 
        "resolution": "480p", 
        "labels": [{"category_name": "airplane", 
                    "bbox": [306, 259, 322, 266], 
                    "category_id": 2}], 
        "image_path": "airplane/day_airplane_0/day_airplane_0_000050.png"}

    Returns:
        
    """

    # read the image
    img_path = os.path.join(BASE_IMAGE_PATH, label_dict['image_path'])
    img = cv2.imread(img_path)

    # get a list of bboxes from the label
    bboxes = []
    class_labels = []

    for label in label_dict['labels']:
        bboxes.append(label['bbox'])
        class_labels.append(label['category_name'])

    # loop through each bounding box and superimpose it on the image
    for idx in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[idx]
        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))
        cv2.rectangle(img, start_point, end_point, 
                      color=(0, 0, 255), thickness=1)
        cv2.putText(
            img,
            class_labels[idx],
            (int(x1), int(y1)-10),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.6,
            color = (0, 255, 0),
            thickness=2
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    plt.figure('label test')
    plt.imshow(img)
    plt.show()

def visualize_img_using_walaris_label(img_path=None):
    """Shows the img on screen with the bboxes from the walaris labels.
    
    Args:
        img_path (str): path to the img you wish to visualize. if None, the
            user will be prompted to manually enter an image path
    
    Returns:

    """

    if img_path is None:
        img_path = input('Enter an image path to visualize: ')

    # if the path is the relative path, get the full path
    img_path_list = img_path.split('/')

    if len(img_path_list) <= 3:
        relative_img_path = img_path
        full_img_path = os.path.join(BASE_IMAGE_PATH, relative_img_path)
    else:
        full_img_path = img_path
        relative_img_path = ('/').join(img_path_list[-3:])

    # get object type of the image
    if 'airplane' in img_path_list[-1]:
        object_type = 'airplane'
    elif 'helicopter' in img_path_list[-1]:
        object_type = 'helicopter'
    elif 'uav' in img_path_list[-1]:
        object_type = 'uav'
    elif 'bird' in img_path_list[-1]:
        object_type = 'bird'
    else:
        object_type = None

    # get data domain of the image
    if 'day' or 'blank' in img_path_list[-1]:
        data_domain = 'day'
    elif 'night' in img_path_list[-1]:
        data_domain = 'night'
    elif 'thermal' in img_path_list[-1]:
        data_domain = 'thermal'
    else:
        data_domain = None

    if data_domain is not None:
        domain_folders = [os.path.join(BASE_LABELS_PATH, data_domain)]
    else:
        domain_folders = glob.glob(BASE_LABELS_PATH+'/*')
    
    # create placeholder variable
    label_file_match = None
    # loop through domain type folders
    for domain_folder in domain_folders:
        # get the object type folder
        if object_type is not None:
            object_type_folders = [os.path.join(domain_folder, 
                                                'whole',
                                                object_type)]
        else:
            object_type_folders = glob.glob(domain_folder+'/whole/*')

        # loop through object type folders
        for object_type_folder in object_type_folders:
            # get all of the possible label files
            label_file_paths = glob.glob(object_type_folder+'/*')
            for label_file_path in label_file_paths:
                img_name = label_file_path.split('/')[-1][:-5]
                if img_name == img_path_list[-2]:
                    label_file_match = label_file_path
                    break

        if label_file_match is not None:
            break

    # load the data from label_file_match
    with open(label_file_match, 'r') as file:
        label_dicts = json.load(file)

    for label_dict in label_dicts.values():
        if label_dict['image_path'] == relative_img_path:
            label_dict_match = label_dict
            break
    
    visualize_walaris_label(label_dict_match)
    
def visualize_coco_labelled_img(img_path,
                                annotations,
                                label_convention):
    """Show the bounding boxes of a labelled coco image. Assumes the
    annotations are in coco format (meaning bboxes are xywh).
    
    Args:
        img_path (str): path to the labelled image
        annotations (list): list of annotation dictionaries corresponding to
            the labelled img
        label_convention (str): what label conventions are the annotations
            using? (walaris or coco supported)
         
    Returns
        None
    """

    # ensure the img_path is only the img file extension
    img_path = img_path.split('/')[-3:]

    base_path = os.environ.get('WALARIS_MAIN_DATA_PATH')

    full_img_path = os.path.join(base_path,
                                 'Images',
                                 img_path[0],
                                 img_path[1],
                                 img_path[2])
    
    # TODO: Add more general visualization capability
    # make sure the image can be found on the machine. check any active yolo
    # dataset directories if it cannot be found in the default location
    if not os.path.exists(full_img_path):
        pass

    assert os.path.exists(full_img_path), "Error: Image not found on machine."

    # read img
    img = cv2.imread(full_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get the bboxes
    bboxes = []
    class_labels = []
    for annotation in annotations:
        bboxes.append(annotation['bbox'])
        if label_convention == 'coco':
            class_labels.append(COCO_CLASSES_DICT_NUM2NAME[annotation['category_id']])
        elif label_convention == 'walaris':
            class_labels.append(WALARIS_CLASS_LABELS_NUM2NAME[annotation['category_id']])
        else:
            class_labels.append(label_convention[annotation['category_id']])

    fig, ax = plt.subplots()
    ax.imshow(img)

    # plot bboxes on image
    show_bboxes(bboxes, ax, bbox_format='xywh_coco', labels=class_labels)
    plt.show()

    return

def visualize_coco_ground_truth_dataset(json_file,
                                        label_convention):
    """Randomly visualize images and labels from a dataset in the format of the
    ground truth coco dataset. This can be used to test and visualize sampled
    datasets.

    Args:
        json_file (str): path to ground truth dataset .json file
        label_convention (str): what label conventions are the annotations
            using? (walaris or coco supported)

    Returns:
    
    """
    supported_label_conventions = set({
        'walaris',
        'coco'
    })

    # assert (type(label_convention) is dict), "Error: Label "\
    #     "convention must be a supported type or a custom dictionary."

    with open(json_file, 'r') as file:
        data = json.load(file)

    images = data['images']
    annotations = data['annotations']

    # sort the annotations to easily collect all annotations with a specific
    # instance ID
    annotations = sorted(annotations, key=lambda x: x['image_id'])

    base_image_path = os.environ.get('WALARIS_MAIN_DATA_PATH', )

    while 1:
        # get image information to predict
        idx = np.random.randint(0, len(images))
        img_path_ext = images[idx]['file_name']

        # if image is not saved on machine, continue
        full_path = os.path.join(base_image_path, 
                                 'Images',
                                 img_path_ext)
        if not os.path.exists(full_path):
            print(f'Skipping: Image not found on machine..')
            continue
        target_img_id = images[idx]['id']

        # get all of the annotations with this image id

        # binary search to find annotation
        l_ptr = 0
        r_ptr = len(annotations)-1

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
            print("No annotations found for this image. Continuing..")
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.show()
            continue

        # look to the left and to the right to get all of the annotations with
        # the same image id
        curr_img_annotations = []
        ptr = idx
        while(ptr >= 0
              and annotations[ptr]['image_id'] == target_img_id):
            curr_img_annotations.append(annotations[ptr])
            ptr -= 1
        ptr = idx+1
        while(ptr < len(annotations)
              and annotations[ptr]['image_id'] == target_img_id):
            curr_img_annotations.append(annotations[ptr])
            ptr += 1

        # visualize the image
        visualize_coco_labelled_img(full_path, curr_img_annotations, label_convention)

def visualize_coco_results(results_json_file,
                           id2img_path_json_file,
                           label_convention):
    """Randomly visualize images and labels from a dataset in the coco results 
    format. This can be used to test and visualize sampled datasets.

    Args:
        json_file (str): path to ground truth dataset .json file
        id2img_path_json_file (str): path to the json file that holds a
            a dictionary of img_ids and their corresponding img paths
        label_convention (str): what label conventions are the annotations
            using? (walaris or coco supported)

    Returns:
    
    """
    with open(results_json_file, 'r') as file:
        data = json.load(file)

    with open(id2img_path_json_file, 'r') as file:
        id2img_path = json.load(file)
    
    annotations = data
    print(len(id2img_path))
    # sort the annotations to easily collect all annotations with a specific
    # instance ID
    annotations = sorted(annotations, key=lambda x: x['image_id'])

    base_image_path = os.environ.get('WALARIS_MAIN_DATA_PATH', )

    while 1:
        # get image information to predict
        idx = np.random.randint(0, len(annotations))
        img_id = str(annotations[idx]['image_id'])
        if img_id not in id2img_path:
            print('ID not in id2img_path...')
            continue
        img_path_ext = id2img_path[img_id]

        # if image is not saved on machine, continue
        full_path = os.path.join(base_image_path, 
                                 'Images',
                                 img_path_ext)
        if not os.path.exists(full_path):
            print(f'Skipping: Image not found on machine..')
            continue
        target_img_id = annotations[idx]['image_id']

        # get all of the annotations with this image id

        # look to the left and to the right to get all of the annotations with
        # the same image id
        curr_img_annotations = []
        ptr = idx
        while(ptr > 0 and annotations[ptr]['image_id'] == target_img_id):
            curr_img_annotations.append(annotations[ptr])
            ptr -= 1
        ptr = idx+1
        while(ptr < len(annotations) and annotations[ptr]['image_id'] == target_img_id):
            curr_img_annotations.append(annotations[ptr])
            ptr += 1

        # visualize the image
        visualize_coco_labelled_img(full_path, curr_img_annotations, label_convention)

def visualize_yolo_labelled_img(img_path,
                                annotations,
                                label_convention):
    """Show the bounding boxes of a labelled coco image. Assumes the
    annotations are in coco format (meaning bboxes are xywh).
    
    Args:
        img_path (str): path to the labelled image
        annotations (list): list of annotations corresponding to
            the labelled img in the yolo format (class_id x y w h)
        label_convention (str): what label conventions are the annotations
            using? (walaris or coco supported)
         
    Returns

    """
    
    # make sure the image can be found on the machine
    # check for .jpg version of image too
    if not os.path.exists(img_path):
        img_path = img_path.replace('png', 'jpg')

    assert os.path.exists(img_path), "Error: Image not found on machine."

    # read img
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get the bboxes and class labels
    bboxes = []
    class_labels = []
    for annotation in annotations:
        bboxes.append(annotation[1:])
        if label_convention == 'coco':
            class_labels.append(COCO_CLASSES_DICT_NUM2NAME[int(annotation[0])+1])
        elif label_convention == 'walaris':
            class_labels.append(WALARIS_CLASS_LABELS_NUM2NAME[int(annotation[0])+1])

    fig, ax = plt.subplots()
    ax.imshow(img)

    # the yolo bboxes come in a normalized format. unnormalize them before
    # rendering the bboxes
    height, width, _ = img.shape

    for bbox in bboxes:
        bbox[0], bbox[2] = int(bbox[0]*width), int(bbox[2]*width)
        bbox[1], bbox[3] = int(bbox[1]*height), int(bbox[3]*height)

    # plot bboxes on image
    show_bboxes(bboxes, ax, bbox_format='xywh_yolo', labels=class_labels)
    plt.show()

    return

def visualize_yolo_ground_truth_dataset(yolo_labels_folder: str,
                                        label_convention):
    """Randomly visualize images and labels from a dataset in the format of the
    ground truth yolo dataset. This can be used to test and visualize sampled
    datasets.

    Args:
        yolo_labels_folder (str): path to the folder containing the label .txt
            files
        label_convention (str): what label conventions are the annotations
            using? (walaris or coco supported)

    Returns:
    
    """

    # get a list of all of the labels for each image
    label_file_paths = glob.glob(yolo_labels_folder+'/*')

    # get a random label from the label file path
    while 1:
        label_file_path = random.choice(label_file_paths)

        # get the image file path from the label name
        label_file_name = label_file_path.split('/')[-1]
        img_file_name = label_file_name.replace('txt', 'png')
        img_file_path = label_file_path.split('/')
        img_file_path[-2] = 'images'
        img_file_path[-1] = img_file_name
        img_file_path = ('/').join(img_file_path)
        
        # get a list of annotations from the label file
        with open(label_file_path, 'r') as file:
            labels = []
            for line in file:
                line = line.strip()
                label = [float(x) for x in line.split(" ")]
                labels.append(label)
        
        # display the image
        visualize_yolo_labelled_img(img_file_path,
                                    labels,
                                    label_convention=label_convention)
        















# ### These are more specific functions and likely won't have much use outside of
# ### their projects. Should remove eventually.

# def refine_thermal_mask(img, mask):
#     """Takes in an image of a thermal cutout on a black background and it's 
#     respective binary mask. Returns a refined version of that thermal cutout 
#     and an updated binary mask on a black background.

#     Params:
#         img (numpy arr): image to refine
#         mask (numpy arr): mask (can be binary or rgb)

#     Returns:
#         img (numpy arr): refined image
#     """

#     h, w, _ = img.shape
#     _, _, mask_channels = mask.shape

#     # if mask is in rgb form, move to binary form
#     if mask_channels != 1:
#         mask = convert_mask_binary2rgb(mask)

#     # get the average pixel value and define the threshold
#     total_num_pxls = np.sum(mask)
#     average_pxl_value = np.sum(img) // total_num_pxls
#     threshold = .6 * average_pxl_value

#     # loop through every pixel. If the sum of that pixel's rgb values are less
#     # than a threshold, set that pixel to black (0, 0, 0)
#     for i in range(h):
#         for j in range(w):
#             if mask[i, j] == 0:
#                 continue
#             if np.sum(img[i, j, :]) < threshold:
#                 img[i, j, :] = np.zeros((1, 1, 3))
#                 mask[i, j] = 0

#     return img, mask

# def ooi_fuse_v1(ooi,
#              global_img):
#     """Uses linear interpolation to fuse the ooi with the global image"""

#     h, w, c = ooi.shape
#     new_img = np.zeros_like(global_img)

#     for i in range(h):
#         for j in range(w):
#             # apply weight based on how bright the ooi pixel is
#             pixel_brightness = (np.sum(ooi[i,j]) // 3) / 255 # between 0 and 1
#             new_img[i,j] = pixel_brightness*ooi[i,j]+(1-pixel_brightness)*global_img[i,j]
    
#     return new_img

# def ooi_fuse_v2(ooi,
#                 global_img):
#     """Uses linear interpolation to fuse the ooi with the global image, but
#     first calculates the brightness for each pixel, and normalizes the entire
#     image to have a min of zero and max of 1"""

#     h, w, c = ooi.shape
#     new_img = np.zeros_like(global_img)

#     """Currently, a large portion of the image is zeros. This does not count as
#     a minimum value, as this is a mask, so we must remove these from the np.min
#     calculation."""

#     pixel_brightness = (np.sum(ooi, axis=2) // 3) / 255
#     min_val = 255
#     for i in range(h):
#         for j in range(w):
#             if pixel_brightness[i,j] < min_val and pixel_brightness[i,j] != 0:
#                 min_val = pixel_brightness[i,j]

#     for i in range(h):
#         for j in range(w):
#             if pixel_brightness[i,j] == 0:
#                 pixel_brightness[i,j] = min_val

#     # normalize for min at 0 and max at 1
#     pixel_brightness = ((pixel_brightness-min_val) 
#                          / (np.max(pixel_brightness) - min_val))
    
#     # pixel_brightness = ((pixel_brightness-np.min(pixel_brightness)) 
#     #                     / (np.max(pixel_brightness) - np.min(pixel_brightness)))

#     for i in range(h):
#         for j in range(w):
#             # apply weight based on how bright the ooi pixel is
#             new_img[i,j] = (pixel_brightness[i,j]*ooi[i,j]
#                             +(1-pixel_brightness[i,j])*global_img[i,j])
    
#     return new_img

# def ooi_fuse_v3(ooi,
#                 global_img):
#     """Uses linear interpolation to fuse the ooi with the global image, but
#     first calculates the brightness for each pixel, and normalizes the entire
#     image to have a min of zero and max of 1"""

#     h, w, c = ooi.shape
#     new_img = np.zeros_like(global_img)

#     """Currently, a large portion of the image is zeros. This does not count as
#     a minimum value, as this is a mask, so we must remove these from the np.min
#     calculation."""

#     # use the increased contrast image to get pixel brightness, but use the
#     # unprocessed image when combining

#     ooi_increased_brightness = increase_contrast(ooi)

#     pixel_brightness = (np.sum(ooi_increased_brightness, axis=2) // 3) / 255
#     min_val = 255
#     for i in range(h):
#         for j in range(w):
#             if pixel_brightness[i,j] < min_val and pixel_brightness[i,j] != 0:
#                 min_val = pixel_brightness[i,j]

#     for i in range(h):
#         for j in range(w):
#             if pixel_brightness[i,j] == 0:
#                 pixel_brightness[i,j] = min_val

#     # normalize for min at 0 and max at 1
#     pixel_brightness = ((pixel_brightness-min_val) 
#                          / (np.max(pixel_brightness) - min_val))
    
#     # pixel_brightness = ((pixel_brightness-np.min(pixel_brightness)) 
#     #                     / (np.max(pixel_brightness) - np.min(pixel_brightness)))

#     for i in range(h):
#         for j in range(w):
#             # apply weight based on how bright the ooi pixel is
#             new_img[i,j] = (pixel_brightness[i,j]*ooi[i,j]
#                             +(1-pixel_brightness[i,j])*global_img[i,j])
    
#     return new_img

# def ooi_fuse_v4(ooi,
#                 global_img):
#     """Same as v2, but adds gaussian blur at the beginning."""

#     ooi = cv2.GaussianBlur(ooi, (7, 7), cv2.BORDER_DEFAULT)
#     h, w, c = ooi.shape
#     new_img = np.zeros_like(global_img)

#     """Currently, a large portion of the image is zeros. This does not count as
#     a minimum value, as this is a mask, so we must remove these from the np.min
#     calculation."""

#     pixel_brightness = (np.sum(ooi, axis=2) // 3) / 255
#     min_val = 255
#     for i in range(h):
#         for j in range(w):
#             if pixel_brightness[i,j] < min_val and pixel_brightness[i,j] != 0:
#                 min_val = pixel_brightness[i,j]

#     for i in range(h):
#         for j in range(w):
#             if pixel_brightness[i,j] == 0:
#                 pixel_brightness[i,j] = min_val

#     # normalize for min at 0 and max at 1
#     pixel_brightness = ((pixel_brightness-min_val) 
#                          / (np.max(pixel_brightness) - min_val))
    
#     # pixel_brightness = ((pixel_brightness-np.min(pixel_brightness)) 
#     #                     / (np.max(pixel_brightness) - np.min(pixel_brightness)))

#     for i in range(h):
#         for j in range(w):
#             # apply weight based on how bright the ooi pixel is
#             new_img[i,j] = (pixel_brightness[i,j]*ooi[i,j]
#                             +(1-pixel_brightness[i,j])*global_img[i,j])
    
#     return new_img

# def ooi_fuse_v5(ooi, global_img):
#     pass
    
# def increase_contrast(img):
#     # define the alpha and beta
#     alpha = 1.9 # Contrast control
#     beta = 1 # Brightness control

#     # call convertScaleAbs function
#     adjusted = cv2.convertScaleAbs(img.copy(), alpha=alpha, beta=beta)

#     h, w, _ = img.shape
#     for i in range(h):
#         for j in range(w):
#             if np.sum(img[i,j]) == 0:
#                 adjusted[i,j] = 0

#     return adjusted

if __name__=='__main__':
    # visualize_img_using_walaris_label()
    pass
