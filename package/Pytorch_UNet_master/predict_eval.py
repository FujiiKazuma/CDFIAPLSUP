import argparse
import logging
import os
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from skimage import io
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/train_data"
check_num = 3
carry_over = False
##


def make_paths(root_path, check_num, carry_over):
    ps = []
    CO = "/carry_over" if carry_over else ""
    ps.append(f"/home/fujii/hdd/BF-C2DL-HSC/02/ori")
    ps.append(os.path.join(root_path, f"check{check_num}{CO}/model_file/CP_epoch50.pth"))

    ps.append(os.path.join(root_path, f"check{check_num}{CO}/predicted_image_eval"))

    return ps

def predict_img(net,
                files,
                frame,
                device,
                savepath,
                scale_factor=1):
    net.eval()
    relu = nn.ReLU()

    full_img = io.imread(str(files))

    if len(full_img.shape) == 3:
        full_img = full_img[:, :, 0]

    full_img = full_img.astype("float64")
    full_img = full_img / full_img.max() if full_img.max() != 0 else full_img
    img = BasicDataset.preprocess(full_img, scale_factor)

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

    save_path_vector = os.path.join(savepath, f"predicted-f{frame:04}.npz")
    np.savez_compressed(save_path_vector, full_mask)

    return full_mask


def main():
    paths = make_paths(root_path, check_num, carry_over)
    ## path list ##
    # 0: cellimage
    # 1: model
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    net = UNet(n_channels=1, n_classes=1)
    logging.info("Loading model {}".format(paths[1]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(paths[1], map_location=device))
    logging.info("Model loaded !")


    cellimage_files = sorted(Path(paths[0]).glob("*.tif"))

    with tqdm(total=len(cellimage_files), leave=False, position=0) as pbar0:
        for frame, fn in enumerate(cellimage_files):
            pbar0.set_description(f"frame:{frame:04}")
            pbar0.update(1)

            logging.info("\nPredicting image {} ...".format(fn))

            mask = predict_img(net=net,
                            files=fn,
                            frame=frame,
                            device=device,
                            savepath = paths[-1])


if __name__ == "__main__":
    main()
    print("finished")