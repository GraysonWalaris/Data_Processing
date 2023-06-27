import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from .labels import COCO_CLASSES_DICT_NUM2NAME, COCO_CLASSES_DICT_NAME2NUM, WALARIS_CLASS_LABELS_NUM2NAME

def show_bboxes(boxes, ax, bbox_format: str, labels=None):
    """Displays an image with bounding boxes and labels drawn.
    
    Args:
        boxes (list): list of bounding boxes
        ax (plt.ax object): axis object from matplotlib.pyplot
        bbox_format (str): specify the format for the bboxes
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
        elif bbox_format == 'xywh':
            x0, y0 = box[0], box[1]
            w, h = box[2], box[3]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                                        facecolor=(0,0,0,0), lw=1))
        if label is not None:
            ax.text(int(x0), int(y0)-10, label, color='red', fontsize=10)

    if labels is not None:
        for idx in range(len(boxes)):
            show_bbox(boxes[idx], ax, 
                          bbox_format='xywh', label=labels[idx])
    else:
        for idx in range(len(boxes)):
            show_bbox(boxes[idx], ax, bbox_format='xywh')
    
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
    
    # make sure the image can be found on the machine
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

    fig, ax = plt.subplots()
    ax.imshow(img)

    # plot bboxes on image
    show_bboxes(bboxes, ax, bbox_format='xywh', labels=class_labels)
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














### These are more specific functions and likely won't have much use outside of
### their projects. Should remove eventually.

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
        mask = convert_mask_binary2rgb(mask)

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
