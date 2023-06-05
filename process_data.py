from utils_old import *

def custom_func():
    process_data(5000,
                 '/home/grayson/Documents/Tarsier_Main_Dataset/Labels_NEW/thermal/whole/uav',
                 '/home/grayson/Desktop/Code/Diffusion-Models-pytorch/total_dataset/train',
                 'test',
                 50*50,
                 0,)

if __name__=='__main__':
    custom_func()