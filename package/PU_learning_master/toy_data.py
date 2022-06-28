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

from model import PUModel
from model import AlexNet
import torchvision.models as models

from loss import PULoss
from dataset import PU_MNIST, PN_MNIST, ToyDataset

def savedata(n):
    plt.scatter(pdata[:, 0], pdata[:, 1], s=20, linewidths=0, c="red")
    plt.scatter(udata[:, 0], udata[:, 1], s=20, linewidths=0, c="blue", marker="+")
    save_path_vector = os.path.join(savepath, f"check_toy{n}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    save_path_vector = os.path.join(savepath, f"toy_coord{n}.txt")
    np.savetxt(save_path_vector, np.concatenate([pdata, udata]))

    save_path_vector = os.path.join(savepath, f"toy_label{n}.txt")
    np.savetxt(save_path_vector, np.concatenate([np.ones((1000)), np.zeros((1000))]), fmt="%d")


savepath = "/home/fujii/hdd/BF-C2DL-HSC/02/train_data/precheck3/PU-learning/check1/toy_data"
os.makedirs(savepath, exist_ok=True)

np.random.seed(seed=0)
pdata = np.random.multivariate_normal([0, 0], [[100, 0], [0, 10]], 1000)
udata = np.random.multivariate_normal([-30, 10], [[10, 0], [0, 40]], 1000)

savedata(1)

udata[1] = np.array([0, np.max(pdata, 0)[1]]) - 1
savedata(2)

udata[0] = np.max(pdata, 0) - 1
savedata(3)

print("finish")