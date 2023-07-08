from data_processing import labels, view
import json

coco_json_file = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/train_COCO/exp2_dino_fully_labelled_day_train.json.json'
yolo_label_folder = '/home/grayson/Documents/model_training/Tarsier_Main_Dataset/Labels_NEW/day/train_yolo/exp2_dino_fully_labelled_day_train'

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

view.visualize_yolo_ground_truth_dataset(yolo_label_folder,
                                         label_convention='walaris')
