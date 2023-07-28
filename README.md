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
conda install pytorch torchvision torchaudio pytorch-cuda=<your_installed_CUDA_version_here> -c pytorch -c nvidia
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

For creating YOLO datasets or COCO dataset directory structures (moving all training images into one folder) from COCO json files, you will need to add one environment variable to your ~/.bashrc:

~~~bash
export WALARIS_RESTORE_PATH='/home/$USER/walaris_dataset_restore_files'
~~~

**WARNING**: You can specify the WALARIS_RESTORE_PATH anywhere you like, but it is recommended to make sure that it is specified to a path that is very unlikely to be deleted. If this path is deleted when you have an active auxiliary dataset, you will lose the ability to restore your Walaris main dataset and will have to re-sync your dataset (download all lost images).

Now you should be set up!

## COCO Training

([Official COCO Dataset Format](https://cocodataset.org/#format-data))

For training using a COCO dataset with Walaris training images, a few requirements must be met:
1. Each image must be located within the $WALARIS_MAIN_DATA_PATH/Images directory.
2. The file name for each image in the coco dataset must be the relative path to that image from the $WALARIS_MAIN_DATA_PATH/Images directory.

You can use the walaris label converter to generate coco datasets from the walaris main dataset, and can write custom scripts to add other images to the coco dataset.

### Visualizing Dataset

To ensure that the dataset was created properly, you can visualize the dataset with the below example:

~~~python
from data_processing import view

view.visualize_coco_ground_truth_dataset('path/to/coco_file.json',
                                         label_convention='walaris')
~~~

## YOLO Training (Ultralytics Models)

([Official YOLO Dataset Format](https://docs.ultralytics.com/datasets/detect/))

For training a custom YOLO model (Ultralytics Version) using our data, we had to get creative. Our file structure for our dataset does not allow us to easily train on YOLO models, because YOLO datasets are set up with all of the images in one folder and all of the labels in another folder. This format does not work with our 'relative path' file structure for our images that we use for COCO datasets.

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

# Useful Methods

Use the following methods by importing the views and labels modules

~~~python
from data_processing import views, labels
~~~

You can find detailed doc string specifying how to use the functions, paramater information, and more by looking at the source code for each function.

## views module

~~~python
views.visualize_coco_ground_truth_dataset(json_file, label_convention)
~~~

Randomly visualize images and labels from a dataset in the format of a coco json file. This can be used to test and visualize sampled
datasets.

~~~python
views.visualize_yolo_ground_truth_dataset(yolo_labels_folder, label_convention)
~~~

Randomly visualize images and labels from a dataset in the format of an ultralytics yolo dataset. This can be used to test and visualize sampled
datasets.

~~~python
views.visualize_walaris_dataset(data_domain='all',
                              data_split='whole',
                              class_type='all', include_thermal_synthetic=False)
~~~

Randomly select images from our dataset and display them with their labels.

~~~python
views.visualize_img_using_walaris_label(img_path)
~~~

Display the labelled image from the walaris dataset at the specified image path.

~~~python
visualize_coco_labelled_img(img_path, annotations, label_convention)
~~~

Display an image using coco format annotations to show the labels.

~~~python
views.show_bboxes(boxes, ax, bbox_format, labels)
~~~

Displays an image with bounding boxes and labels drawn.

## labels module

**Note** There are a variety of dictionaries mapping numbers to label names for different standard formats including the walaris format, coco format and more. Explore the data_processing/labels.py file to see.

~~~python
labels.get_random_label(data_type='all',
                     data_split='whole',
                     class_type='all',
                     include_thermal_synthetic=False):
~~~

Provide the image information from a random image from the walaris datset.

### For COCO datasets..

The following are some of the most useful functions for working with COCO datasets.

~~~python
labels.get_annotations_by_img_id(annotations,
                              target_img_id,
                              anns_sorted=True)
~~~

Gets a list of annotations that correspond to a specific image id.

~~~python
labels.get_rand_sample_from_coco_json(original_json_file,
                                   new_json_file,
                                   sample_size,
                                   seed=None,
                                   include_unlabelled_images=False)
~~~

Randomly sample a coco format dataset from a coco format dataset.

~~~python
labels.remove_labels_by_class_coco(coco_json_file,
                                new_json_file,
                                label_convention='walaris',
                                classes_to_include=None,
                                classes_to_remove=None,
                                include_images_w_no_labels=False)
~~~

Users can specify which classes to keep or which classes to remove from a coco dataset. Helps to clean unwanted classes from a dataset.

~~~python
labels.get_category_info_coco_format(json_file,
                                  label_convention,
                                  isPrint=False)
~~~

Print out information in the terminal regarding the categories found and number of categories in a coco ground truth or results dataset.

~~~python
labels.get_bbox_info_from_coco_annotations(annotations,
                                           label_convention,
                                           class_name_to_show,
                                           bin_count = 1000)
~~~

Plot information regarding the distribution of bbox sizes for a class using a frequency plot histogram.

### For yolo datasets (Ultralytics models)..

**WARNING** Please know that the below functions will be moving images to and from your walaris dataset to create the directory structures necessary to train Ultralitic's YOLO models. If you follow the instructions, you will not lost data, but there is potential to lose your images if you delete the created yolo dataset or change the paths to or within the created dataset. As a good rule of thumb, always call the `labels.restore_walaris_dataset()` function when you are done training to restore the dataset and avoid the risk of losing data. 

~~~python
labels.get_yolo_dataset_from_coco_json_file(coco_train_json_file: str,
                                            coco_val_json_file: str,
                                            yolo_dataset_base_path: str,
                                            dataset_name: str = None,
                                            class_names = 'walaris_18')
~~~

Transform a coco dataset the references images from the walaris main dataset into a yolo formatted dataset. The images will need to be moved from their standard location in the Tarsier_Main_Dataset to a new folder in the yolo_dataset_base_path. These images can be moved back with the restore_walaris_dataset() function.

~~~python
labels.restore_walaris_dataset()
~~~

This file can be called to restore your walaris dataset as long as you have not modified the directory structure of any auxiliary datasets that you have created.