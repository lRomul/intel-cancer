import sys
sys.path.append('..')

from scripts.train_kfolds import train_kfolds

data_dir = '/workdir/data/preproc_data/data002_kfold_val_detector002_size256_scale1.5'
save_dir = '/workdir/data/kfold_train/10fold_vgg16_001'

params = {
    'data_dir' : data_dir,
    'save_dir' : save_dir,
    'arch' : "vgg16",
    'batch_size' : 16,
    'lr' : 0.0001,
    'n_epoch' : 25
}

if __name__ == '__main__':
    train_kfolds(params)
