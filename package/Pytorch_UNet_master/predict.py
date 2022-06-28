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
from .utils.dataset import BasicDataset

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"

lap = 0
##

def make_paths(root_path, check_num, lap, seed):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/cellimage"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/model_files/model_file-seed{seed:02}/CP_epoch30.pth"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/predicted_image/seed{seed:02}"))

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
                files,
                device):
    net.eval()
    relu = nn.ReLU()

    full_img = io.imread(str(files))

    if len(full_img.shape) == 3:
        full_img = full_img[:, :, 0]

    full_img = full_img.astype("float64")
    full_img = full_img / full_img.max() if full_img.max() != 0 else full_img
    img = BasicDataset.preprocess(full_img)

    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = relu(output)

        probs = probs.squeeze(0)

        full_mask = probs.squeeze().cpu().numpy()

    return full_mask

def main(root_path, check_num, lap, seed):
    paths = make_paths(root_path, check_num, lap, seed)
    ## path list ##
    # 0: cellimage
    # 1: model
    # -1: savepath
    files = load_files(paths)
    ## file list ##
    # 0: cellimage
    os.makedirs(paths[-1], exist_ok=True)

    set_seed(42+seed)
    # torch.manual_seed(13)
    # np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load(paths[1], map_location=device))

    for frame, fn in enumerate(files[0]):
        mask = predict_img(net=net,
                           files=fn,
                           device=device)

        save_path_vector = os.path.join(paths[-1], f"pre-f{frame:04}.npz")
        np.savez_compressed(save_path_vector, mask)

if __name__ == "__main__":
    main(root_path, check_num, lap, seed)
    print("finished")