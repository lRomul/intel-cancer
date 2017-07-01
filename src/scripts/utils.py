import os
import matplotlib.pyplot as plt

DATA_DIR = '/workdir/data/train/'
TEST_DATA_DIR = '/workdir/data/test/'
ADD_DATA_DIR = '/workdir/data/additional/'
BBOX_FILES = '/workdir/data/bboxes/%s_bbox.tsv'
SAMPLE_PATH = '/workdir/data/sample_submission.csv'
CLASSES = ['Type_1', 'Type_2', 'Type_3']

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def show_bgr(img):
    plt.figure(figsize=(7, 7))
    plt.imshow(img[:, :, (2, 1, 0)])