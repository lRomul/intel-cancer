import numpy as np
import pandas as pd

import os
from os.path import join

from scripts.utils import *


def type_bboxes(bbox_path):
    columns = 'Path number_of_rect min_x min_y bbox_width bbox_height'.split()
    df = pd.read_csv(bbox_path, sep=' ', header=None)
    df = df.iloc[:, :len(columns)]
    df.columns = columns
    df = df[df.number_of_rect == 1]
    df.Path = df.Path.map(lambda x:join(DATA_DIR, x.replace('\\', '/')))
    return df['Path min_x min_y bbox_width bbox_height'.split()]

def get_bboxes_df():
    bboxes_path = "/workdir/data/bboxes/%s_bbox.tsv"
    df_list = []
    for cls in CLASSES:
        df = type_bboxes(bboxes_path%cls)
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)