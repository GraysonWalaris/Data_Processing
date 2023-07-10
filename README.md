# Description
This repository is used to collect data for training of various AI algorithms at Walaris. It contains many useful functions for synthetic data generation, data augmentation, and data collection for the main dataset.

# Setup

## FPIE Dependencies
Some functions rely on the fpie package. In order to build the .whl file for the fpie package, you must first install the extenstions required. To do so, go to the official fpie package repository and ensure that all of the extensions specified in the README are installed.
* Make sure to install cmake and gcc, as these will be necessary to build fpie when pip installing the data_processing library.*

## Environment Setup
To install this, it is recommended to use virtual environments. (For this tutorial I will be using anaconda, but any virtual environment manager will do.)

Create a new anaconda environment: `conda create --name <env_name> python=3.10`

Download this repository and navigate to it in the terminal and install the package and dependencies with:

~~~bash
pip install -e .
~~~

Install PyTorch in your conda environment:

~~~bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
~~~

## Environment Variables
A few environment variables will need to be set to use this package.

First, download the segment_anything model default weights here (https://github.com/facebookresearch/segment-anything#model-checkpoints) and place them in a file location of your choice. Set the following environment variable to your weights file path. In a Linux environment, add the following to your .bashrc:

~~~bash
export SAM_CHECKPOINT_PATH='<path_to_sam_checkpoint_folder>/checkpoints/sam_vit_h.pth'
~~~

Next, set the following environment variable to the base folder of the dataset title "Tarsier_Main_Dataset". *Note, this entire repo is based of the dataset structure that was in place as of June 7, 2023. If this dataset structure changes, the code will need to be modified*

On Linux, add this to your .bashrc:

~~~bash
export WALARIS_MAIN_DATA_PATH='/home/grayson/Documents/Tarsier_Main_Dataset/'
~~~

For creating YOLO datasets from COCO json files, you will need to add one environment variable to your ~/.bashrc:

~~~bash
export YOLO_ADMIN_PATH='/home/$USER/yolo_admin_files'
~~~

Now you should be set up!

## COCO Training

([Official COCO Dataset Format](https://cocodataset.org/#format-data))

For training using a COCO dataset with Walaris training images, a few requirements must be met:
1. Each image must be located within the $WALARIS_MAIN_DATA_PATH/Images directory.
2. The file name for each image in the coco dataset must be the relative path to that image from the $WALARIS_MAIN_DATA_PATH/Images directory.

You can use the label converter to generate coco datasets from the walaris main dataset, and can write custom scripts to add other images to the coco dataset.

### Visualizing Dataset

To ensure that the dataset was created properly, you can visualize the dataset with the below example:

~~~python
from data_processing import view

view.visualize_coco_ground_truth_dataset('path/to/coco_file.json',
                                         label_convention='walaris')
~~~

## YOLO Training

([Official YOLO Dataset Format](https://docs.ultralytics.com/datasets/detect/))

For training a custom YOLO model using our data, we had to get creative. Our file structure for our dataset does not allow us to easily train on YOLO models, because YOLO datasets are set up with all of the images in one folder and all of the labels in another folder. This format does not work with our 'relative path' file structure for our images that we use for COCO datasets.

**WARNING**: Be aware that creating a YOLO dataset from our main dataset involves moving all of the image files from their original locations. If you do not follow the steps exactly as shown below, you risk losing some images and having to re-sync your dataset.

Below are the steps to train on a YOLO dataset using our data:

1. Create a training and validation json file for a COCO dataset.
2. Use the labels.get_yolo_dataset_from_coco_json_file() function to create your yolo dataset.
3. You can then train a custom yolo model using the standard ultralytics package.
4. Once you have finished your training, run the labels.yolo_clean() function to restore the original dataset.

Here is an example script to create a YOLO dataset from a coco dataset and train it using the ultralytics standard package:

~~~python
from data_processing import labels

coco_train_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/train_COCO/exp2_dino_fully_labelled_day_train.json.json'
coco_val_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/val_COCO/exp1_dino_fully_labelled_day_val_results.json'
yolo_base_folder_path = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/yolo_datasets/day_yolo_dataset'

labels.get_yolo_dataset_from_coco_json_file(coco_train_json,
                                            coco_val_json,
                                            yolo_base_folder_path,
                                            dataset_name='walaris_day_yolo',
                                            class_names='walaris')

from ultralytics import YOLO

results = model.train(data='/home/grayson/Documents/model_training/Tarsier_Main_Dataset/yolo_datasets/day_yolo_dataset/walaris_day_yolo.yaml', epochs=3)
~~~

### Visualizing Dataset

To ensure that the dataset was created properly, you can visualize the dataset with the below example:

~~~python
from data_processing import view

view.visualize_yolo_ground_truth_dataset('/path/to/yolo/labels/folder',
                                         label_convention='walaris')
~~~

# Classes Documentation

## blend

`from data_processing import blend`

### OOI_Blender

Use this class to generate novel images by taking objects of interest from the Tarsier_Main_Dataset and blending them into background images of your choice.
