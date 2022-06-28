from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch import nn

from .model import PUModel
from .model import AlexNet
import torchvision.models as models

from .loss import PULoss
from .dataset import PU_Dataset

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root2"

check_num = 1
lap = 0
PU_num = 1
##

def make_paths(root_path, check_num, lap, PU_num):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch/patch-r13.npz"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch/label.txt"))

    # 一番ロスの小さいモデルを採用
    model_file =    os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/PU-learning/check{PU_num}/model_file")
    loss_list =     np.loadtxt(os.path.join(model_file, "loss_list.txt"))
    loss_min =      np.where(loss_list == loss_list.min())[0][-1]
    ps.append(os.path.join(model_file, f"model_ep{loss_min:04}.bin"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/PU-learning/check{PU_num}"))

    return ps

def pred(model, device, train_loader):
    model.eval()

    pre_label = np.empty((0, 1))
    features = np.empty((0, 2304))
    with torch.autograd.detect_anomaly():  # nanが出たらエラー出る
        for (patch, label) in train_loader:
            patch, label = patch.to(device), label.to(device)

            output = model(patch)
            feature = model.features(patch)
            
            pre_label_tmp = output.to("cpu").detach().numpy().copy()
            features_tmp = feature.to("cpu").detach().numpy().copy().reshape(feature.shape[0], -1)  # 1次元目以外を平坦化
            pre_label = np.concatenate([pre_label, pre_label_tmp])
            features = np.concatenate([features, features_tmp])
            pass

    pre_label = np.array(pre_label).squeeze()
    features = np.array(features).squeeze()

    return pre_label, features

def main(root_path, check_num, lap, PU_num=1, batch_size=4000):
    paths = make_paths(root_path, check_num, lap, PU_num)
    ## path list ##
    # 0: patch
    # 1: labels
    # 2: model
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    torch.manual_seed(42)
    device = torch.device("cuda")

    train_set = PU_Dataset(images_path=paths[0], labels_path=paths[1], unlabeled_rate=0.5)

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, **kwargs)

    model = AlexNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(paths[2], map_location=device))

    # predict
    pre_label, features = pred(model, device, train_loader)

    save_path_vector = os.path.join(paths[-1], f"predict.txt")
    np.savetxt(save_path_vector, pre_label)

    save_path_vector = os.path.join(paths[-1], f"feature.txt")
    np.savetxt(save_path_vector, features)

if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")
