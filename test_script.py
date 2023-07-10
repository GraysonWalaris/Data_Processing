from data_processing import labels, view
import json
import os

# coco_json_file = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/val_COCO/exp1_dino_fully_labelled_day_val_results.json'
# yolo_label_folder = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/val_yolo/exp1_dino_fully_labelled_day_val_results'

# labels.get_file_name2relative_path_json_file()

# labels.move_imgs_walaris_to_yolo()

# labels.convert_labels_coco2yolo(coco_json_file,
#                                 yolo_label_folder)

# with open(coco_json_file, 'r') as file:
#     data = json.load(file)

# yes = 'wait'

# view.visualize_coco_ground_truth_dataset(coco_json_file,
#                                          'walaris')

# labels.get_category_info_coco_format(coco_json_file,
#                                      'walaris',
#                                      isPrint=True)

# view.visualize_yolo_ground_truth_dataset(yolo_label_folder,
#                                          label_convention='walaris')

# import glob

# label_paths = glob.glob(yolo_label_folder+'/*.txt')

# what = 'yes'
# info = {}
# info['current_image_folders'] = []

# with open('/home/grayson/Documents/model_training/yolo_dataset_log.json', 'r') as file:
#     # data = json.dump(info, file)
#     data = json.load(file)

# print(data)

coco_train_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/train_COCO/exp2_dino_fully_labelled_day_train.json.json'
coco_val_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/val_COCO/exp1_dino_fully_labelled_day_val_results.json'
yolo_base_folder_path = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/yolo_datasets/day_yolo_dataset'

# labels.get_yolo_dataset_from_coco_json_file(coco_train_json,
#                                             coco_val_json,
#                                             yolo_base_folder_path,
#                                             dataset_name='walaris_day_yolo',
#                                             class_names='walaris')

# view.visualize_yolo_ground_truth_dataset('/home/grayson/Documents/model_training/Tarsier_Main_Dataset/yolo_datasets/day_yolo_dataset/valid/labels',
#                                          'walaris')

# labels.yolo_clean()

# import yaml
# from yaml.loader import SafeLoader

# yaml_file = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/yolo_datasets/day_yolo_dataset/walaris_day_yolo.yaml'

# with open(yaml_file, 'r') as file:
#     data = yaml.load(file, Loader=SafeLoader)

# what = 'yes'

# import glob

# images = glob.glob('/home/grayson/Documents/model_training/Tarsier_Main_Dataset/yolo_datasets/day_yolo_dataset/train/images/*')

# print(len(images))
# with open(coco_train_json, 'r') as file:
#     data = json.load(file)

# print(len(data['images']))
# # for image in images:
# import tqdm

# data = []
# with open(labels.YOLO_DATASET_LOG, 'r') as file:
#     for line in file:
#         data.append(line.replace('\n', ''))

# for label in tqdm(data):
#     bbox = data.split(' ')[1:]
#     for num in bbox:
#         if num < 0 or num > 1:
#             print('error')

from data_processing import view

view.visualize_coco_ground_truth_dataset('path/to/coco_file.json',
                                         label_convention='walaris')


