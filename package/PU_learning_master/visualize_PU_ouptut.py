from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
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

def matplotlib_init():
    matplotlib.font_manager._rebuild()

    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 25 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 15 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 12 # 軸だけ変更されます
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 0.6 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
    plt.rcParams["legend.labelspacing"] = 1. # 垂直方向（縦）の距離の各凡例の距離
    plt.rcParams["legend.handletextpad"] = 1.5 # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 2 # 点がある場合のmarker scale


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"

# root_path = f"/home/fujii/hdd/C2C12P7/sequence/0303/sequ2/root1"
root_path_list = [
                  f"/home/fujii/hdd/C2C12P7/sequence/0303/sequ2/root1",
                  f"/home/fujii/hdd/C2C12P7/sequence/0303/sequ6/root1",
                  f"/home/fujii/hdd/C2C12P7/sequence/0303/sequ9/root2",
                  f"/home/fujii/hdd/C2C12P7/sequence/0318/sequ17/root1",
                  f"/home/fujii/hdd/BF-C2DL-HSC/02/root4"]

check_num = 11
# lap = 0
lap_max = 5
PU_num = 1

unlabeled_rate = 0.5
##

def make_paths(root_path, check_num, lap_max, PU_num):
    ps = []
    for lap in range(lap_max):
        ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/PU-learning/check{PU_num}/model_file/output_list.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/PU_outputs"))
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


def main(root_path, check_num, lap_max, PU_num=1):
    paths = make_paths(root_path, check_num, lap_max, PU_num)
    ## path list ##
    # 0: output
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    PU_out = np.empty((0, 5))
    for lap in range(lap_max):
        tmp = np.loadtxt(paths[lap])
        tmp = np.hstack( [np.arange(200)[:, np.newaxis], tmp, np.full([len(tmp), 1], lap)] )
        PU_out = np.vstack( [PU_out, tmp])

    fig = px.line(x=PU_out[:, 0], y=[PU_out[:, 1], PU_out[:, 2], PU_out[:, 3]], animation_frame=PU_out[:, 4])
    fig.show()
    save_path_vector = os.path.join(paths[-1], f"PU_output.html")
    fig.write_html(save_path_vector)
    

if __name__ == "__main__":
    matplotlib_init()
    for root_path in root_path_list:
        main(root_path, check_num, lap_max, PU_num)
    print("finished")
