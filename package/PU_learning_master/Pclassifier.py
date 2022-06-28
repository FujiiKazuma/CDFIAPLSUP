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
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root3"

lap = 0
push_direction = "P"
PU_num = 1
PC_num = 3
##

def make_paths(root_path, check_num, lap, PU_num, PC_num, push_direction):
    ps = []
    
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/PU-learning/check{PU_num}/feature.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch/label.txt"))

    model_file = (os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/Pclassifier/check{PC_num}/{push_direction}/modelfiles"))
    loss_list = np.loadtxt(os.path.join(model_file, "loss_list.txt"))[:, 0]
    loss_min = np.where(loss_list == loss_list.min())[0][-1]
    ps.append(os.path.join(model_file, f"model_ep{loss_min:04}.bin"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/Pclassifier/check{PC_num}/{push_direction}"))

    return ps

def pred(model, device, train_loader):
    model.eval()
    
    for feature, _label in train_loader:
        feature = feature.to(device=device)
        
        output = model(feature)

        predict = output.cpu().detach().numpy()
        pass

    return predict

def main(root_path, check_num, lap, push_direction, PU_num=1, PC_num=1):
    paths = make_paths(root_path, check_num, lap, PU_num, PC_num, push_direction)
    ## path list ##
    # 0: feature
    # 1: label
    # 2: model
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    torch.manual_seed(13)
    device = torch.device("cuda")

    train_set = PC_Dataset(paths[0], paths[1], push_direction)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=False, num_workers=1, pin_memory=True)

    model = PNet().to(device=device)
    model.load_state_dict(torch.load(paths[2], map_location=device))

    # predict
    predict = pred(model, device, train_loader)

    save_path_vector = os.path.join(paths[-1], f"predict.txt")
    np.savetxt(save_path_vector, predict)

    grad = model.state_dict()["fc1.weight"][0, :-1].cpu().detach().numpy()
    save_path_vector = os.path.join(paths[-1], f"gradient.txt")
    np.savetxt(save_path_vector, grad)

if __name__ == "__main__":
    main(root_path, check_num, lap, push_direction)
    print("finished")