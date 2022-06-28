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
from dataset import PU_MNIST, PN_MNIST


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

##
# 入力してね
precheck_num = 3
check_num = 1

tmax = 10
p = 10
##
patch_path = f"/home/fujii/hdd/BF-C2DL-HSC/02/train_data/precheck{precheck_num}/patch/patch-r13.npz"
label_path = f"/home/fujii/hdd/BF-C2DL-HSC/02/train_data/precheck{precheck_num}/patch/label.txt"

model_file = f"/home/fujii/hdd/BF-C2DL-HSC/02/train_data/precheck{precheck_num}/PU-learning/check{check_num}/model_file"
loss_list = np.loadtxt(os.path.join(model_file, "loss_list.txt"))
loss_min = np.where(loss_list == loss_list.min())[0][-1]

model_path = model_file + f"/model_ep{loss_min}.bin"
savepath = f"/home/fujii/hdd/BF-C2DL-HSC/02/train_data/precheck{precheck_num}/PU-learning/check{check_num}"
os.makedirs(savepath, exist_ok=True)

def pnormpush_init(P, U, tmax):
    I = P.shape[0]
    K = U.shape[0]
    n = P.shape[1]

    L = np.zeros((tmax, n))  # L = lambda
    d = np.full((tmax, I, K), 1 / (I * K))
    M = np.empty((I, K, n))
    for k in range(K):
        M[:, k, :] = P - U[k]
        pass
    return L, d, M

def pnormpush_loop(L, d, M, tmax, p):
    I, K, n = M.shape
    d_ = np.empty((tmax, I, K))
    j = np.empty((tmax)).astype("int32")
    a = np.empty((tmax))
    e = np.zeros((tmax, n))
    z = np.empty((tmax))
    for t in range(tmax):
        tmp = np.sum(d[t], 0)
        tmp = tmp ** (p - 1)
        d_[t] = d[t] * tmp

        tmp = d_[t, ..., np.newaxis] * M
        tmp = np.sum(tmp, 0, keepdims=True)
        tmp = np.sum(tmp, 1)
        j[t] = int(np.argmax(tmp))

        # compute a[t] here

        e[t, j[t]] = 1
        L[t+1] = L[t] + a[t] * e[t]

        tmp = -a[t] * M[..., j[t]]
        tmp = np.exp(tmp)
        tmp = d[t] * tmp
        z[t] = np.sum(tmp)

        d[t+1] = tmp / z[t]
        pass
    return L[tmax]
        
def pred(args, model, device):
    model.eval()

    images = io.imread(patch_path)
    labels = np.loadtxt(label_path)[:, 1]
    assert images.shape[0] == labels.shape[0], "patch numer error"
    patch_num = images.shape[0]

    P_features = []
    U_features = []
    with tqdm(total=patch_num, leave=False, position=0) as pbar0:
        for patch_id in range(patch_num):
            pbar0.set_description("patch[{:04}/{:04} ({:.0f}%)]".format(
                patch_id, patch_num, 100. * patch_id / patch_num))
            pbar0.update(1)

            img = images[patch_id]
            label = labels[patch_id]

            img = img / images.max() if images.max() != 0 else img
            img = img * 255
            img = img.astype("float64")
            img = np.stack([img, img, img], 0)
            img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)
            label = torch.tensor(1) if label == 1 else torch.tensor(-1)

            img, label = img.to(device), label.to(device)

            feature = model.features(img)

            if label == 1:
                P_features.append(feature.to("cpu").detach().numpy().copy().flatten())
            else:
                U_features.append(feature.to("cpu").detach().numpy().copy().flatten())
                pass
            pass
            
    P_features = np.array(P_features)
    U_features = np.array(U_features)
    
    L, d, M = pnormpush_init(P_features, U_features, tmax)
    Ltmax = pnormpush_loop(L, d, M, tmax, p)

    print("stop")
    pass

def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--do_train", action='store_true', 
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--nnPU", action='store_true',
                        help="Whether to us non-negative pu-learning risk estimator.")
    parser.add_argument("--train_batch_size", default=30000, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=100, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=50, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    return parser.parse_args()

def main():

    args = get_args()


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = AlexNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    pred(args, model, device)
    print("finished")

if __name__ == "__main__":
    main()
