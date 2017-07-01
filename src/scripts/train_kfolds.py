import numpy as np
import pandas as pd
from PIL import Image

import os
from os.path import join

import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
sys.path.append('..')

from scripts.pandas_dataset import PandasDataset
from scripts.custom_transforms import train_transform, test_transform
from scripts.models import get_pretrained_model
from scripts.kfold_data import load_detect_train_test_df

from scripts.utils import mkdir, SAMPLE_PATH


def save_history(history, save_dir):
    save_info = dict()
    save_info['history'] = history
    save_info['best_val_loss'] = np.min(history['val'])
    save_info['best_epoch'] = np.argmin(history['val'])
    save_path = join(save_dir, 'history.txt')
    with open(save_path, 'w') as f:
        f.write(str(save_info))
        
def predict_test(model, test_df, save_dir):
    subm = pd.read_csv(SAMPLE_PATH, index_col=0)
    for i, row in test_df.iterrows():
        img = Image.open(row.Path)
        trans_img = test_transform()(img)
        trans_img = trans_img.unsqueeze(0)
        trans_img = Variable(trans_img)
        prob_pred = model.predict_proba(trans_img)
        subm.loc[row.Name] = prob_pred.cpu().data.numpy()
    subm_path = join(save_dir, 'subm.csv')
    subm.to_csv(subm_path)
        
def train_fold(kfold_df, test_df, params, fold):
    fold_save_path = join(params['save_dir'], 'fold_%d'%fold)
    model_save_path = join(fold_save_path, 'model.pth.tar')
    mkdir(fold_save_path)
    train_df = kfold_df[kfold_df.Fold != fold]
    val_df = kfold_df[(kfold_df.Fold == fold)&(kfold_df.Additional == 0)]

    train_loader = torch.utils.data.DataLoader(
        PandasDataset(train_df, train_transform()),
        batch_size=params['batch_size'], shuffle=True,
        num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        PandasDataset(val_df, test_transform()),
        batch_size=params['batch_size'], shuffle=False,
        num_workers=2, pin_memory=True)

    model = get_pretrained_model(params['arch'], params['lr'])
    model.model_path = model_save_path
    history = model.fit(train_loader, val_loader, n_epoch=params['n_epoch'])
    model.load_model(model.model_path)
    val_score = model.validate(val_loader)
    predict_test(model, test_df, fold_save_path)
    save_history(history, fold_save_path)
    del model, train_loader, val_loader

def train_kfolds(params):
    mkdir(params['save_dir'])
    kfold_df, test_df = load_detect_train_test_df(params['data_dir'])
    folds = sorted(kfold_df.Fold.unique())

    with open(join(params['save_dir'], 'params.txt'), 'w') as f:
        f.write(str(params))
    
    for fold in folds:
        print("Start train fold %d" % fold)
        train_fold(kfold_df, test_df, params, fold)
