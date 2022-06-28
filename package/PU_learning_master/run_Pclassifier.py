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
from torch.optim import SGD
from torch import nn

from .model import PUModel, AlexNet, PNet
import torchvision.models as models

from .loss import PULoss
from .dataset import PC_Dataset

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root3"

check_num = 1
lap = 0
PU_num = 1
PC_num = 3
push_direction = "P"

p = 4
lam = 1  # lambda
batch_size = 1000
epoch_num = 2000
lr = 1e-5  # 1e-5
##

def make_paths(root_path, check_num, lap, PU_num, PC_num, push_direction):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/PU-learning/check{PU_num}/feature.txt"))

    if push_direction == "P":
        ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch/label.txt"))
    if push_direction == "N":
        ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/Pclassifier/check{PC_num}/P/label.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/Pclassifier/check{PC_num}/{push_direction}/modelfiles"))

    return ps

def RPC(output, label, p):
    tmp = output[label == 1]
    tmp = torch.exp(-tmp)
    RPCp = torch.sum(tmp)

    tmp = output[label != 1]
    tmp = torch.exp(tmp * p)
    RPCn = torch.sum(tmp) / p

    RPC = RPCp + RPCn
    return RPC

def RPCL2(output, label, p, model, device, lam=1):
    rpc = RPC(output, label, p)

    w = model.fc1.weight[0, :-1].to(device=device)
    l2 = torch.tensor(lam, device=device) * torch.sum(w ** 2)
    l2 = l2.to(device=device)

    rpcl2 = rpc + l2

    return rpcl2, rpc, l2

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train(model, device, train_loader, optimizer, criterion, p, lam):
    model.train()
    
    epoch_loss = 0
    epoch_rpc = 0
    epoch_l2 = 0
    with torch.autograd.detect_anomaly():  # nanが出たらエラー出る
        for feature, label in train_loader:
            feature = feature.to(device=device)
            label = label.to(device=device)

            optimizer.zero_grad()
            output = model(feature)

            loss = criterion(output, label, p, model, device, lam=lam)
            epoch_loss += loss[0].item()
            epoch_rpc += loss[1].item()
            epoch_l2 += loss[2].item()

            loss[0].backward()
            optimizer.step()
            pass
    return epoch_loss, epoch_rpc, epoch_l2

def save_loss(loss_list, epoch_num, savepath):
    loss_list = np.array(loss_list)
    save_path_vector = os.path.join(savepath, f"loss_list.txt")
    np.savetxt(save_path_vector, loss_list)

    plt.plot(range(epoch_num), loss_list[:, 0], c="black", label="Rpc + L2")
    plt.plot(range(epoch_num), loss_list[:, 1], c="red", label="Rpc")
    plt.plot(range(epoch_num), loss_list[:, 2], c="blue", label="L2")
    plt.yscale("log")

    plt.title(f"losmin = {np.where(loss_list[:, 0] == loss_list[:, 0].min())}")
    plt.legend(loc='upper right')

    save_path_vector = os.path.join(savepath, f"loss_curve.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)
    plt.close()

def main(root_path, check_num, lap, push_direction, PU_num=1, PC_num=1, p=4, lam=0.1, batch_size=1000, epoch_num=2000, lr=1e-5):
    paths = make_paths(root_path, check_num, lap, PU_num, PC_num, push_direction)
    ## path list ##
    # 0: features
    # 1: labels
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    set_seed(42)
    # torch.manual_seed(13)
    # np.random.seed(0)

    device = torch.device("cuda")

    train_set = PC_Dataset(paths[0], paths[1], push_direction)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)

    model = PNet().to(device=device)
    criterion = RPCL2
    optimizer = SGD(model.parameters(), lr=lr)

    loss_list = []
    with tqdm(total=epoch_num, leave=False, position=0) as pbar0:
        for epoch in range(1, epoch_num + 1):
            pbar0.set_description("Epock{:04}".format(epoch))
            pbar0.update(1)

            loss = train(model, device, train_loader, optimizer, criterion, p, lam)
            loss_list.append(loss)
            pbar0.set_postfix({"loss": loss[0]})

            output_model_file = os.path.join(paths[-1], f"model_ep{epoch:04}.bin")
            torch.save(model.state_dict(), output_model_file)
            pass
    
    save_loss(loss_list, epoch_num, paths[-1])


if __name__ == "__main__":
    main(root_path, check_num, lap, push_direction)
    print("finished")