import numpy as np
import pandas as pd
import cv2

import random
import os
from os.path import join

from scripts.utils import *
from scripts.train_val import load_train_val_df
from scripts.bboxes import get_bboxes_df

def load_bboxes_train_val_df():
    bbox_df = get_bboxes_df()
    train_df, val_df = load_train_val_df()
    train_df = train_df.merge(bbox_df, on='Path', how='inner')
    val_df = val_df.merge(bbox_df, on='Path', how='inner')
    return train_df, val_df

def check_poly(img, points):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    poly_img = img.copy().astype(np.uint32)
    cv2.fillConvexPoly(mask, points, 1)
    poly_img[mask.astype(bool)] += np.array((156, 12, 12), dtype=np.uint32)
    poly_img = np.clip(poly_img, 0, 255).astype(img.dtype)
    cv2.polylines(poly_img, [points], True, (12, 12, 200))
    return poly_img

def preproc_img(img):
    img = img / 128.0 - 1
    return img

def get_image_points(row):
    img = cv2.imread(row.Path)
    points = [
        [row.min_x, row.min_y],
        [row.min_x + row.bbox_width, row.min_y],
        [row.min_x + row.bbox_width, row.min_y + row.bbox_height],
        [row.min_x, row.min_y + row.bbox_height],
    ]
    return img, np.array(points)

def square_image(img, points=None):
    sqr_len = max(img.shape[:2])
    sqr_img = np.zeros((sqr_len, sqr_len, img.shape[2]), dtype=img.dtype)
    y_shift = (sqr_len - img.shape[0]) // 2
    x_shift = (sqr_len - img.shape[1]) // 2
    sqr_img[y_shift:y_shift+img.shape[0], x_shift:x_shift+img.shape[1]] = img
    if points is not None:
        sqr_points = points.copy()
        sqr_points[:, 0] += x_shift
        sqr_points[:, 1] += y_shift
        return sqr_img, sqr_points
    else:
        return sqr_img


def resize_image(img, points, new_shape):
    resize_img = cv2.resize(img, new_shape[::-1])
    resize_points = np.array(points, dtype=np.float64)
    resize_points[:, 0] *= new_shape[1] / img.shape[1]
    resize_points[:, 1] *= new_shape[0] / img.shape[0]
    return resize_img, resize_points.astype(np.int)


def rot90_image(img, points):
    rot_img = np.rot90(img)
    rot_points = points.copy()
    rot_points[:, 1] = img.shape[1] - points[:, 0]
    rot_points[:, 0] =  points[:, 1]
    return rot_img, rot_points


def flip_image(img, points, flip_code):
    flip_img = cv2.flip(img, flip_code)
    flip_points = points.copy()
    if flip_code:
        flip_points[:, 0] = flip_img.shape[0] - flip_points[:, 0]
    else:
        flip_points[:, 1] = flip_img.shape[1] - flip_points[:, 1]
    return flip_img, flip_points


def points2netout(points):
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    netout = np.array([x, y, width, height], dtype=float)
    return netout

def netout2points(netout):
    x, y, width, height = netout
    points = [
        [x-width/2, y-height/2],
        [x+width/2, y-height/2],
        [x+width/2, y+height/2],
        [x-width/2, y+height/2]
    ]
    return np.array(points, dtype=int)

def save_image(save_name, output_dir, img, points, check=True):
    save_img_path = join(output_dir, 'img', save_name+'.png')
    save_pts_path = join(output_dir, 'pts', save_name+'.npy')
    cv2.imwrite(save_img_path, img)
    netout = points2netout(points)
    np.save(save_pts_path, netout)
    if check:
        save_check_path = join(output_dir, 'check', save_name+'.png')
        points = netout2points(netout)
        cv2.imwrite(save_check_path, check_poly(img, points))
        

def augment_save_data_df(df, output_dir, net_size):
    mkdir(join(output_dir, 'img'))
    mkdir(join(output_dir, 'pts'))
    mkdir(join(output_dir, 'check'))
    
    for i, row in df.iterrows():
        img, points = get_image_points(row)
        img, points = square_image(img, points)
        img, points = resize_image(img, points, net_size)
        name = row.Name[:-len('.jpg')]+'_'+row.Class
        for r in ['0', '90', '180', '270']:
            save_image(name+"_"+r, output_dir, img, points)
            for fl in [0, 1, 'a']:
                if fl != 'a':
                    flip_img, flip_pts = flip_image(img, points, fl)
                    save_image(name+"_"+r+'_'+str(fl), output_dir, flip_img, flip_pts)
                else:
                    flip_img, flip_pts = flip_image(img, points, 0)
                    flip_img, flip_pts = flip_image(flip_img, flip_pts, 1)
                    save_image(name+"_"+r+'_'+fl, output_dir, flip_img, flip_pts)
            img, points = rot90_image(img, points)
            
        if i % 100 == 0:
            print('Processed %d images'%i)
            
    print("Save data to", output_dir)
    

def load_data(data_dir, img_size, N_max):
    img_dir = join(data_dir, 'img')
    pts_dir = join(data_dir, 'pts')
    img_names = os.listdir(img_dir)
    N_examples = min(len(img_names), N_max)

    data = np.zeros([N_examples, 3]+list(img_size), dtype=np.float32)
    target = np.zeros((N_examples, 4), dtype=np.float32)
    
    random.shuffle(img_names)
    for i, name in enumerate(img_names[:N_examples]):
        img = cv2.imread(join(img_dir, name))
        # img = preproc_img(img)
        pts = np.load(join(pts_dir, name[:-len('png')]+'npy'))
        data[i] = img.transpose((2, 0, 1))
        target[i] = pts
    return data, target

def load_dataset(data_dir, img_size, N_max=15000):
    if os.path.isfile(join(data_dir, 'X_train.npy')):
        X_train = np.load(join(data_dir, 'X_train.npy'))
        y_train = np.load(join(data_dir, 'y_train.npy'))
        X_val = np.load(join(data_dir, 'X_val.npy'))
        y_val = np.load(join(data_dir, 'y_val.npy'))
        print("Load data")
    else:
        X_train, y_train = load_data(join(data_dir, 'train'), img_size, N_max)
        X_val, y_val = load_data(join(data_dir, 'val'), img_size, N_max)
        np.save(join(data_dir, 'X_train.npy'), X_train)
        np.save(join(data_dir, 'y_train.npy'), y_train)
        np.save(join(data_dir, 'X_val.npy'), X_val)
        np.save(join(data_dir, 'y_val.npy'), y_val)
        print("Save data")
    return X_train, y_train, X_val, y_val