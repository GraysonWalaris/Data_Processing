"""Test script for seeing what all classes we have labelled in our dataset."""

import glob
import os

labels = []
for data_type in glob.glob(os.path.join(os.environ.get('WALARIS_MAIN_DATA_PATH'), 'Labels_NEW'))