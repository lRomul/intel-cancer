import numpy as np
import pandas as pd

import pickle
import os
from os.path import join

from PIL import Image

from scripts.utils import DATA_DIR, TEST_DATA_DIR, ADD_DATA_DIR
from scripts.utils import CLASSES, BBOX_FILES
from scripts.bboxes import get_bboxes_df

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
    return df.sort_values('Name').reset_index(drop=True)
    
def get_kfold_idx(df, k):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=12)
    fold_lst = []
    for train_index, test_index in skf.split(df, df.Class):
        fold_lst.append(test_index)
    return fold_lst
        
def get_kfold_df(data_dir, k=10):
    data_df = get_data_df(data_dir)
    data_df['Fold'] = k
    kfold_idx = get_kfold_idx(data_df, k)
    for i, idx in enumerate(kfold_idx):
        data_df.loc[idx, 'Fold'] = i
    return data_df


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
    
    
def load_train_add_kfold_df(k=10):
    path = '/workdir/data/%dfold.pickle'%k
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            train_df, add_df = pickle.load(f)
        print('Load kfold')
    else:
        train_df = get_kfold_df(DATA_DIR, k)
        train_df = train_df.merge(get_bboxes_df(), on='Path', how='inner')
        train_df['Additional'] = 0
        add_df = get_kfold_df(ADD_DATA_DIR, k)
        add_df['Additional'] = 1
        with open(path, 'wb') as f:
            pickle.dump((train_df, add_df ), f)
        print('Dump kfold')
    return train_df, add_df 

def load_detect_train_test_df(folder):
    file_path = join(folder, 'data_df_dict.pickle')
    with open(file_path, 'rb') as f:
        data_df_dict = pickle.load(f)
    return data_df_dict['train'], data_df_dict['test']