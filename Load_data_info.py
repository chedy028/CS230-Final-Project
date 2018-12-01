import os, sys
import numpy as np
import pandas as pd

'''input: load csv & file directory
   output: train_information'''

train_image_path = './input/train'
data = pd.read_csv('./input/train.csv')

train_dataset_info = []

for ID, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(train_image_path, ID),
        'labels':np.array([int(label) for label in labels])})

train_dataset_info = np.array(train_dataset_info)
