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

from .model import PUModel, AlexNet, PNet
import torchvision.models as models

from .loss import PULoss
from .dataset import PC_Dataset

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"
lap = 1
PC_num = 1

sr = 0.2  # select rate
##

def make_paths(root_path, check_num, lap, PC_num):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/Pclassifier/check{PC_num}/P/predict.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch/label.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/Pclassifier/check{PC_num}/P"))
    return ps

def main(root_path, check_num, lap, PC_num=1, sr=0.2):
    paths = make_paths(root_path, check_num, lap, PC_num)
    ## path list ##
    # 0: predict
    # 1: label
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)
    
    pre = np.loadtxt(paths[0])
    label = np.loadtxt(paths[1])
    lab = label[:, 3]

    # unlabeledデータのソートされたindexを取得
    pre_order = np.argsort(pre)
    tmp = lab[pre_order]
    tmp = np.where(tmp != 1)
    U_pre_order = pre_order[tmp]
    # 変更するラベルの数を、srとUnlabeledの数から決める
    sel_n = int(sr * np.sum(lab != 1))
    # 上位sel_n個のUnlabeledデータをPositiveに変更する
    label[U_pre_order[-sel_n:], 3] = 1
    
    save_path_vector = os.path.join(paths[-1], f"label.txt")
    np.savetxt(save_path_vector, label, fmt="%d")
        
    pass

if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")