import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn 

import argparse
import logging
import os
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from skimage import io
from PIL import Image
from torchvision import transforms

from .unet import UNet
from .utils.data_vis import plot_img_and_mask
from .utils.dataset import EvalDataset
from torch.utils.data import DataLoader, random_split


##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"
lap = 0
##


def make_paths(root_path, check_num, lap, seed):
    ps = []
    # ps.append(f"/home/fujii/hdd/BF-C2DL-HSC/02/ori")
    # ps.append(f"/home/fujii/hdd/C2C12P7/sequence/0318/sequ17/ori/exp1_F0017-00600.tif")
    ps.append(os.path.join(root_path, f"check{check_num}/testdata/cellimage"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/model_files/model_file-seed{seed:02}/CP_epoch30.pth"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_eval/predicted_image"))

    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.*")))
    
    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def predict_img(net,
                train_loader,
                device,
                paths):
    net.eval()

    with tqdm(total=len(train_loader), leave=False, position=0) as pbar0:
        for frame, batch in enumerate(train_loader):
            pbar0.set_description(f"frame:{frame:04}")
            pbar0.update(1)

            imgs = batch["image"]
            imgs = imgs.to(device=device, dtype=torch.float32)

            output = net(imgs)
            output = output.squeeze(0)
            output = output.squeeze().cpu().detach().numpy()

            save_path_vector = os.path.join(paths[-1], f"predicted-f{frame:04}.npz")
            np.savez_compressed(save_path_vector, output)
            pass

def main(root_path, check_num, lap):
    seed = np.loadtxt(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/peak/peak_num.txt")).argmax()

    paths = make_paths(root_path, check_num, lap, seed)
    ## path list ##
    # 0: cellimage
    # 1: model
    # -1: savepath
    files = load_files(paths)
    ## file list ##
    # 0: cellimage
    os.makedirs(paths[-1], exist_ok=True)

    set_seed(42 + seed)

    torch.manual_seed(25)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load(paths[1], map_location=device))

    dataset = EvalDataset(paths[0])
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    mask = predict_img(net=net,
                        train_loader=train_loader,
                        device=device,
                        paths=paths)


if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")
