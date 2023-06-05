import cv2
import matplotlib.pyplot as plt
import numpy as np
from modules import view, random_crops

ooi = cv2.imread('/home/grayson/Desktop/Code/Data_Processing/play_dataset/cropped_img_0023.png')
adjusted = view.increase_contrast(ooi.copy())
h, w, _ = ooi.shape
ooi = adjusted

global_img = cv2.imread('/home/grayson/Documents/Experiment_Datafolders/FLIR_Dataset/FLIR_ADAS_v2/images_thermal_train/data/video-2SReBn5LtAkL5HMj2-frame-000571-JwSR5WhFK7hdNEhiJ.jpg')

# random crop global img
global_img = random_crops.random_crop(global_img, (256,256))

new_img_v1 = view.ooi_fuse_v1(ooi.copy(), global_img)
new_img_v2 = view.ooi_fuse_v2(ooi.copy(), global_img)
new_img_v3 = view.ooi_fuse_v3(ooi.copy(), global_img)
new_img_v4 = view.ooi_fuse_v4(ooi.copy(), global_img)

plt.figure('v2')
plt.imshow(new_img_v2)
plt.show()

# implant the ooi into the global image
new_img = np.zeros_like(global_img)

for i in range(h):
    for j in range(w):
        if np.sum(ooi[i,j]) != 0:
            new_img[i,j] = ooi[i,j]
        else:
            new_img[i,j] = global_img[i,j]

img_list = [
    ooi,
    global_img,
    new_img,
    new_img_v1,
    new_img_v2,
    new_img_v3,
    new_img_v4
]
# plt.subplot(1,6,1)
# plt.title('old_global')
# plt.imshow(global_img)
# plt.subplot(1,6,2)
# plt.title('ooi')
# plt.imshow(ooi)
# plt.subplot(1,6,3)
# plt.title('new_global')
# plt.imshow(new_img)
# plt.subplot(1,6,4)
# plt.title('new_global_v1')
# plt.imshow(new_img_v1)
# plt.subplot(1,6,5)
# plt.title('new_global_v2')
# plt.imshow(new_img_v2)

def plot_imgs(img_list):
    rows = len(img_list)//4+1
    cols = 4
    for idx, i in enumerate(img_list):
        plt.subplot(rows, cols, idx+1)
        plt.title(f'{idx}')
        img = cv2.resize(i, (512,512))
        plt.imshow(img)

    plt.show()

plot_imgs(img_list)

# new_img_v2 = cv2.resize(new_img_v2, (1024, 1024))
# plt.figure()
# plt.imshow(new_img_v2)
# plt.show()













# thresholding
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # blurred = cv2.GaussianBlur(gray, (7, 7), 0)
# blurred=gray

# thresh = cv2.adaptiveThreshold(blurred, 255,
#                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                cv2.THRESH_BINARY, 15, 5)

# plt.figure('figure1')
# plt.imshow(thresh)
# plt.figure('figure2')
# plt.imshow(image)
# thresh = np.expand_dims(thresh, axis=2)
# add_thresh = np.concatenate((thresh, thresh, thresh), axis=2)
# threshold_img = np.zeros_like(image)
# h, w, c = image.shape
# for i in range(h):
#     for j in range(w):
#         for x in range(c):
#             if add_thresh[i,j,x]==255:
#                 threshold_img[i,j,x] = image[i,j,x]

# plt.figure('figure3')
# plt.imshow(threshold_img)
# plt.show()
# what = 'yes'