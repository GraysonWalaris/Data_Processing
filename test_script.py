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

# coco_train_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/train_COCO/small_coco_train_sample.json'
# coco_val_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/val_COCO/small_coco_val_sample.json'
# yolo_base_folder_path = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/yolo_datasets/day_yolo_dataset_small'

labels.get_yolo_dataset_from_coco_json_file(coco_train_json,
                                            coco_val_json,
                                            yolo_base_folder_path,
                                            dataset_name='walaris_day_yolo',
                                            class_names='walaris')

view.visualize_yolo_ground_truth_dataset(yolo_base_folder_path+'/valid/labels',
                                         'walaris')

# labels.yolo_clean()

# import json
# import random

# coco_train_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/train_COCO/exp2_dino_fully_labelled_day_train.json.json'
# coco_val_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/val_COCO/exp1_dino_fully_labelled_day_val_results.json'

# new_coco_train_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/train_COCO/small_coco_train_sample.json'
# new_coco_val_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/val_COCO/small_coco_val_sample.json'

# train_samples = 200
# val_samples = 100

# with open(coco_train_json, 'r') as file:
#     data_train = json.load(file)

# with open(coco_val_json, 'r') as file:
#     data_val = json.load(file)

# new_images_train = []
# new_images_val = []

# train_img_set = set()
# for i in range(train_samples):
#     img_dict = random.choice(data_train['images'])
#     while img_dict['id'] in train_img_set:
#         img_dict = random.choice(data_train['images'])
        
#     train_img_set.add(img_dict['id'])
#     new_images_train.append(img_dict)

# val_img_set = set()
# for i in range(train_samples):
#     img_dict = random.choice(data_val['images'])
#     while img_dict['id'] in val_img_set:
#         img_dict = random.choice(data_val['images'])
        
#     val_img_set.add(img_dict['id'])
#     new_images_val.append(img_dict)

# # get the annotations for each image
# annotations_train = data_train['annotations']
# annotations_val = data_val['annotations']

# annotations_train = sorted(annotations_train, key=lambda x: x['image_id'])
# annotations_val = sorted(annotations_val, key=lambda x: x['image_id'])

# new_annotations_train = []
# for img_dict in new_images_train:
#     target_img_id = img_dict['id']

#     l_ptr = 0
#     r_ptr = len(annotations_train) - 1
#     idx = -1
#     while l_ptr <= r_ptr:
#         mid = (r_ptr - l_ptr) // 2 + l_ptr

#         if annotations_train[mid]['image_id'] == target_img_id:
#             new_annotations_train.append(annotations_train[mid])
#             idx = mid
#             break
#         elif annotations_train[mid]['image_id'] < target_img_id:
#             l_ptr = mid + 1
#         else:
#             r_ptr = mid - 1

#     if idx == -1:
#         print('No annotations found for image..')
#         continue
    
#     ptr = idx - 1
#     while ptr >= 0 and annotations_train[ptr]['image_id'] == target_img_id:
#         new_annotations_train.append(annotations_train[ptr])
#         ptr -= 1
    
#     ptr = idx + 1
#     while ptr < len(annotations_train) and annotations_train[ptr]['image_id'] == target_img_id:
#         new_annotations_train.append(annotations_train[ptr])
#         ptr += 1

# new_annotations_val = []
# for img_dict in new_images_val:
#     target_img_id = img_dict['id']

#     l_ptr = 0
#     r_ptr = len(annotations_val) - 1
#     idx = -1
#     while l_ptr <= r_ptr:
#         mid = (r_ptr - l_ptr) // 2 + l_ptr

#         if annotations_val[mid]['image_id'] == target_img_id:
#             new_annotations_val.append(annotations_val[mid])
#             idx = mid
#             break
#         elif annotations_val[mid]['image_id'] < target_img_id:
#             l_ptr = mid + 1
#         else:
#             r_ptr = mid - 1

#     if idx == -1:
#         print('No annotations found for image..')
#         continue
    
#     ptr = idx - 1
#     while ptr >= 0 and annotations_val[ptr]['image_id'] == target_img_id:
#         new_annotations_val.append(annotations_val[ptr])
#         ptr -= 1
    
#     ptr = idx + 1
#     while ptr < len(annotations_val) and annotations_val[ptr]['image_id'] == target_img_id:
#         new_annotations_val.append(annotations_val[ptr])
#         ptr += 1

# data_train['images'] = new_images_train
# data_train['annotations'] = new_annotations_train

# data_val['images'] = new_images_val
# data_val['annotations'] = new_annotations_val

# with open(new_coco_train_json, 'w') as file:
#     json.dump(data_train, file)

# with open(new_coco_val_json, 'w') as file:
#     json.dump(data_val, file)
    
import tqdm

# coco_train_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/train_COCO/exp2_dino_fully_labelled_day_train.json.json'
# coco_val_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/val_COCO/exp1_dino_fully_labelled_day_val_results.json'

# coco_train_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/train_COCO/small_coco_train_sample.json'
# coco_val_json = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/val_COCO/small_coco_val_sample.json'

# with open(coco_train_json, 'r') as file:
#     data = json.load(file)
# annotations = data['annotations']
# cat_ids = set()
# for annotation in annotations:
#     cat_ids.add(annotation['category_id'])

# print(cat_ids)