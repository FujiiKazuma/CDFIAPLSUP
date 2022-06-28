from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam

from model import PUModel
from model import AlexNet
import torchvision.models as models

from loss import PULoss
from dataset import PU_Dataset


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"

root_path = f"/home/fujii/hdd/C2C12P7/sequence/0303/sequ2/root1"

check_num = 11
lap = 0
PU_num = 1

unlabeled_rate = 0.5
##

def make_paths(root_path, check_num, lap, PU_num):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch/patch-r13.npz"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch/label.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/PU-learning/check{PU_num}/model_file"))

    return ps

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train(model, device, train_loader, optimizer, loss_fct):
    model.train()
    tr_loss = 0
    output_list = np.empty(0)
    with torch.autograd.detect_anomaly():  # nanが出たらエラー出る
        for (data, target) in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            output_list = np.hstack( [output_list, output.cpu().detach().numpy().flatten()] )

            loss = loss_fct(output.view(-1), target.type(torch.float))
            
            tr_loss += loss.item()
            loss.backward()
            optimizer.step()
            pass
    
    result = [output_list.min(), output_list.max(), output_list.mean()]
    return tr_loss, result


def main(root_path, check_num, lap, PU_num=1, unlabeled_rate=0.5, batch_size=4000, lr=1e-4, epoch_num=200):
    paths = make_paths(root_path, check_num, lap, PU_num)
    ## path list ##
    # 0: patch
    # 1: labels
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    # label = np.loadtxt(paths[1])
    # unlabeled_rate = np.sum(label[:, 3] == 0) / np.sum(label[:, 3] != 1)

    set_seed(42)
    # torch.manual_seed(42)
    # np.random.seed(0)

    device = torch.device("cuda")

    train_set = PU_Dataset(paths[0], paths[1], unlabeled_rate)
    prior = train_set.get_prior()

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs, worker_init_fn=worker_init_fn)

    model = AlexNet(num_classes=1).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.005)
    loss_fct = PULoss(prior=prior, nnPU=True)

    loss_list = []
    output_list = []
    with tqdm(total=epoch_num, leave=False, position=0) as pbar0:
        for epoch in range(1, epoch_num + 1):
            pbar0.set_description("Epock{:03}".format(epoch))
            pbar0.update(1)
            
            tmp = train(model, device, train_loader, optimizer, loss_fct)
            loss_list.append(tmp[:-1])
            output_list.append(tmp[-1])

            pbar0.set_postfix({"loss": loss_list[-1]})

            output_model_file = os.path.join(paths[-1], f"model_ep{epoch:04}.bin")
            torch.save(model.state_dict(), output_model_file)
            pass
    
    loss_list = np.array(loss_list)
    save_path_vector = os.path.join(paths[-1], f"loss_list.txt")
    np.savetxt(save_path_vector, loss_list)

    save_path_vector = os.path.join(paths[-1], f"output_list.txt")
    np.savetxt(save_path_vector, output_list)

    plt.plot(range(epoch_num), loss_list)
    plt.yscale('log')
    save_path_vector = os.path.join(paths[-1], f"loss_curve-check{check_num:02}-lap{lap:02}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)
    plt.close()

    # print(np.where(loss_list == loss_list.min()))

if __name__ == "__main__":
    main(root_path, check_num, lap, PU_num)
    print("finished")
