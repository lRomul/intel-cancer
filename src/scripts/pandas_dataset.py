import torch.utils.data as data

import pandas as pd
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(class_series):
    classes = class_series.unique()
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(df, class_to_idx):
    images = []
    for i, row in df.iterrows():
        if is_image_file(row.Path):
            item = (row.Path, class_to_idx[row.Class])
            images.append(item)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


def default_loader(path):
    return pil_loader(path)


class PandasDataset(data.Dataset):

    def __init__(self, df, transform=None, target_transform=None,
                 loader=default_loader):
        df = df[["Class", "Path"]]
        classes, class_to_idx = find_classes(df.Class)
        imgs = make_dataset(df, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in DataFrame"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.df = df
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)