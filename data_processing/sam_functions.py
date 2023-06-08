import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
from segment_anything import sam_model_registry, SamPredictor

"""A collection of common methods for the segment anything AI model."""

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                                facecolor=(0,0,0,0), lw=2))
    
def superimpose_masks_and_bboxes(image, masks, bboxes):
    bboxes = torch.tensor(
        bboxes
    , device='cpu')
     
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in bboxes:
        show_box(box.cpu().numpy(), plt.gca())
    plt.axis('off')
    plt.show()
    
def load_sam_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        print(f'Sam model loaded on {torch.cuda.get_device_name()}')
    else:
        print(f'Sam model loaded on CPU')

    sam_checkpoint = os.environ.get('SAM_CHECKPOINT_PATH')
    model_type = 'vit_h'

    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    return sam_model.to(device)

def get_mask_from_sam(img, bbox, sam_model):
    sam_model.set_image(img)
    bbox = np.array(bbox)
    masks, scores, logits = sam_model.predict(box=bbox, multimask_output=False)
    mask = (masks[0].astype("uint8")) * 255
    return mask[1:-1, 1:-1]