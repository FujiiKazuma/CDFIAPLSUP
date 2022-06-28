from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from scipy.ndimage.interpolation import rotate
from pathlib import Path
from skimage import io


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, lossmask_dir, scale=1):
        self.imgs_dir = sorted(Path(imgs_dir).glob("*.*"))
        self.masks_dir = sorted(Path(masks_dir).glob("*.*"))
        self.lossmask_dir = sorted(Path(lossmask_dir).glob("*.*"))
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        assert len(self.imgs_dir) == len(self.masks_dir) == len(self.lossmask_dir), "file number error"

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        # logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = pil_img

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def random_crop_param(self, shape):
        c, h, w= shape
        top = np.random.randint(0, h - 256)
        left = np.random.randint(0, w - 256)
        bottom = top + 256
        right = left + 256
        return top, bottom, left, right

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = self.masks_dir[i]
        img_file = self.imgs_dir[i]
        lossmask_file = self.lossmask_dir[i]

        mask = io.imread(str(mask_file))
        img = io.imread(str(img_file))
        lossmask = io.imread(str(lossmask_file))

        mask = mask.astype("float64")
        img = img.astype("float64")

        img = img / img.max() if img.max() != 0 else img
        mask = mask / mask.max() if mask.max() != 0 else mask

        assert img.size == mask.size == lossmask.size, \
            f'Image and mask and lossmask {idx} should be the same size, but are {img.size} and {mask.size} and {lossmask.size}'

        img = self.preprocess(img)
        mask = self.preprocess(mask)
        lossmask = self.preprocess(lossmask)

        # data augumentation
        top, bottom, left, right = self.random_crop_param(img.shape)


        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'lossmask': torch.from_numpy(lossmask).type(torch.FloatTensor)
        }


class EvalDataset(Dataset):
    def __init__(self, imgs_dir):
        self.imgs_dir = sorted(Path(imgs_dir).glob("*.*"))
        
    def __len__(self):
        return len(self.imgs_dir)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = pil_img

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        img_file = self.imgs_dir[i]
        img = io.imread(str(img_file))
        
        img = img.astype("float64")
        img = img / img.max() if img.max() != 0 else img

        img = self.preprocess(img)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor)
        }
