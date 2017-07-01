import numpy as np
import pandas as pd

import cv2

import pickle
import os
from os.path import join

from PIL import Image

from scripts.utils import DATA_DIR, TEST_DATA_DIR, ADD_DATA_DIR
from scripts.utils import CLASSES

def get_data_df(data_dir):
    img_names = []
    img_classes = []
    img_paths = []
    img_sizes = []
    for cls in CLASSES:
        cls_dir = join(data_dir, cls)
        for img_name in os.listdir(cls_dir):
            if img_name.endswith('jpg'):
                img_names.append(img_name)
                img_classes.append(cls)
                img_path = join(cls_dir, img_name)
                img_paths.append(img_path)
                img_sizes.append(Image.open(img_path).size)
            else:
                print(img_name, 'skipped')
    df = pd.DataFrame({'Name' : img_names,
                       'Path' : img_paths,
                       'Class' : img_classes,
                       'Width' : list(map(lambda x: x[0], img_sizes)),
                       'Height' : list(map(lambda x: x[1], img_sizes))})
    df = df[['Class', 'Name', 'Path', 'Width', 'Height']]
    return df.sort_values('Name')
    
def get_train_val_idx(df):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=10, random_state=42)
    for train_index, test_index in skf.split(df, df['Class']):
        return train_index, test_index
        
def get_train_val_df():
    data_df = get_data_df(DATA_DIR)
    train_index, val_index = get_train_val_idx(data_df)
    train_df = data_df.iloc[train_index]
    val_df = data_df.iloc[val_index]
    return train_df, val_df

def get_additional_df():
    return get_data_df(ADD_DATA_DIR)

def get_test_df():
    img_names = []
    img_paths = []
    img_sizes = []

    for img_name in os.listdir(TEST_DATA_DIR):
        if img_name.endswith('jpg'):
            img_names.append(img_name)
            img_path = join(TEST_DATA_DIR, img_name)
            img_paths.append(img_path)
            img_sizes.append(Image.open(img_path).size)
        else:
            print(img_name, 'skipped')
    df = pd.DataFrame({'Name' : img_names,
                       'Path' : img_paths,
                       'Width' : list(map(lambda x: x[0], img_sizes)),
                       'Height' : list(map(lambda x: x[1], img_sizes))})
    df = df[['Name', 'Path', 'Width', 'Height']]
    return df.sort_values('Name').reset_index(drop=True)
    
def load_train_val_df():
    path = '/workdir/data/train_val.pickle'
    if os.path.isfile(path):
        print('Load train val')
        with open(path, 'rb') as f:
            train_df, val_df = pickle.load(f)
    else:
        print('Dump train val')
        train_df, val_df = get_train_val_df()
        with open(path, 'wb') as f:
            pickle.dump((train_df, val_df), f)
    return train_df, val_df