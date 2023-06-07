# Description
This repository is used to collect data for training of various AI algorithms at Walaris. It contains many useful functions for synthetic data generation, data augmentation, and data collection for the main dataset.

# Setup

## Environment Setup
To install this, it is recommended to use virtual environments. (For this tutorial I will be using anaconda, but any virtual environment manager will do.)

Create a new anaconda environment: `conda create --name <env_name> python=3.10`

Download this repository and navigate to it in the terminal and install the package and dependencies with:

`pip install -e .`

Install PyTorch in your conda environment:

`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

## Environment Variables
A few environment variables will need to be set to use this package.

First, download the segment_anything model default weights here (https://github.com/facebookresearch/segment-anything#model-checkpoints) and place them in a file location of your choice. Set the following environment variable to your weights file path. In a Linux environment, add the following to your .bashrc:

`export SAM_CHECKPOINT_PATH='<path_to_sam_checkpoint_folder>/checkpoints/sam_vit_h.pth'`

Next, set the following environment variable to the base folder of the dataset title "Tarsier_Main_Dataset". *Note, this entire repo is based of the dataset structure that was in place as of June 7, 2023. If this dataset structure changes, the code will need to be modified*

On Linux, add this to your .bashrc:

`export WALARIS_MAIN_DATA_PATH='/home/grayson/Documents/Tarsier_Main_Dataset/'`

Now you should be set up!
