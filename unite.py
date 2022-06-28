import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm


from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, random_split
import cv2
import matplotlib
import matplotlib.pyplot as plt

def matplotlib_init():
    matplotlib.font_manager._rebuild()

    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 25 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 9 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
    plt.rcParams["legend.labelspacing"] = 1. # 垂直方向（縦）の距離の各凡例の距離
    plt.rcParams["legend.handletextpad"] = 2. # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 2 # 点がある場合のmarker scale

# import package

from package.Pytorch_UNet_master import train
from package.Pytorch_UNet_master import predict
from package.Pytorch_UNet_master import peak
from package.Pytorch_UNet_master import seed_checker
from package.Pytorch_UNet_master import carryover_peaks
from package.Pytorch_UNet_master import EvaluationMetric
from package.Pytorch_UNet_master import Evaluation_totrain
from package.Pytorch_UNet_master import make_patch

from package.PU_learning_master import run_classifier
from package.PU_learning_master import classifier
from package.PU_learning_master import run_Pclassifier
from package.PU_learning_master import Pclassifier
from package.PU_learning_master import select_PClas

from package.Pytorch_UNet_master import make_merged_anotattion


from package.Pytorch_UNet_master import check_predictimage
from package.Pytorch_UNet_master import check_trainingdata
from package.Pytorch_UNet_master import plot_pseudo
from package.tsne_python import PCA
from package.tsne_python import PCA2


from package.Pytorch_UNet_master import eval
from package.Pytorch_UNet_master import peaktoeval
from package.Pytorch_UNet_master import Evaltoeval
from package.Pytorch_UNet_master import plot_toeval
from package.Pytorch_UNet_master import plot_pred_toeval


##
root_path_list = []
# root_path_list.append("/home/fujii/hdd/C2C12P7/sequence/0303/sequ2/root1")  # control
# root_path_list.append("/home/fujii/hdd/C2C12P7/sequence/0303/sequ6/root1")  # FGF2
# root_path_list.append("/home/fujii/hdd/C2C12P7/sequence/0303/sequ9/root2")  # BMP2
# root_path_list.append("/home/fujii/hdd/C2C12P7/sequence/0318/sequ17/root1")  # FGF2 + BMP2
# root_path_list.append("/home/fujii/hdd/BF-C2DL-HSC/02/root4")  # BF-C2DL-HSC

# check_list = [2, 1, 1, 4, 1]
# check_list = [3, 2, 2, 5, 5]
# check_list = [2]
# prior_list = [0.95, 0.35, 0.30, 0.30, 0.30]
# prior_list = [0.5, 0.5, 0.5, 0.5, 0.5]

root_path_list = [
                  f"/home/fujii/hdd/C2C12P7/sequence/0303/sequ2/root1",
                  f"/home/fujii/hdd/C2C12P7/sequence/0303/sequ6/root1",
                  f"/home/fujii/hdd/C2C12P7/sequence/0303/sequ9/root2",
                  f"/home/fujii/hdd/C2C12P7/sequence/0318/sequ17/root1",
                  f"/home/fujii/hdd/BF-C2DL-HSC/02/root4"]

# root_path_list = [f"/home/fujii/hdd/BF-C2DL-HSC/02/root6"]

# root_path_list = [
#                   f"/home/fujii/hdd/C2C12P7/sequence/0303/sequ2",
#                   f"/home/fujii/hdd/C2C12P7/sequence/0303/sequ6",
#                   f"/home/fujii/hdd/C2C12P7/sequence/0303/sequ9",
#                   f"/home/fujii/hdd/C2C12P7/sequence/0318/sequ17",
#                   f"/home/fujii/hdd/BF-C2DL-HSC/02"
                #   ]

check_list = [
              12,
              12,
              12,
              12,
              12
              ]

prior_list = [
                0.9,
                0.9,
                0.9,
                0.9,
                0.9
                ]





##
# root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root4"
# root_path = "/home/fujii/hdd/C2C12P7/sequence/0318/sequ17/root1"

# check_num = 2
PU_num = 1
PC_num = 1

startlap = 0
lastlap = 5
##

def main():
    # startlap = 0
    # lastlap = 10
    with tqdm(total=lastlap - startlap, leave=False, position=0) as pbar0:
        for lap in range(startlap, lastlap):
            pbar0.set_description(f"lap:{lap:02}")
            pbar0.write(f"=========== lap{lap:02} start ============")
            
            if lap != startlap:
                pass

            for seed in range(5):
                pbar0.write(f"----------- cell detection {seed:02} -----------")
                train.main(root_path, check_num, lap, seed)
                # predict.main(root_path, check_num, lap)
                pbar0.write(f"----------- detection peak {seed:02}-----------")
                peak.main(root_path, check_num, lap, seed)

            seed_checker.main(root_path, check_num, lap)
            if lap > 0:
                carryover_peaks.main(root_path, check_num, lap)

            pbar0.write("----------- Evaluation tra -----------")
            EvaluationMetric.main(root_path, check_num, lap)
            pbar0.write("------------- maek patch -------------")
            make_patch.main(root_path, check_num, lap)

            pbar0.write("------------- PU-learning ------------")
            run_classifier.main(root_path, check_num, lap, PU_num=PU_num)
            classifier.main(root_path, check_num, lap, PU_num=PU_num)

            pbar0.write("---------- Pclassification P ---------")
            run_Pclassifier.main(root_path, check_num, lap, push_direction="P", PU_num=PU_num, PC_num=PC_num)
            Pclassifier.main(root_path, check_num, lap, push_direction="P", PU_num=PU_num, PC_num=PC_num)
            select_PClas.main(root_path, check_num, lap, PC_num=PC_num)

            pbar0.write("---------- Pclassification N ---------")
            run_Pclassifier.main(root_path, check_num, lap, push_direction="N", PU_num=PU_num, PC_num=PC_num)
            Pclassifier.main(root_path, check_num, lap, push_direction="N", PU_num=PU_num, PC_num=PC_num)

            pbar0.write("--------- make new traindata ---------")
            make_merged_anotattion.main(root_path, check_num, lap, PC_num=PC_num)

            pbar0.update(1)
        pass
    train.main(root_path, check_num, lastlap)
    pass

def main2(root_path, check_num):
    # startlap = 0
    # lastlap = 10
    with tqdm(total=lastlap - startlap, leave=False, position=0) as pbar0:
        for lap in range(startlap, lastlap):
            pbar0.set_description(f"lap:{lap:02}")
            pbar0.write(f"=========== lap{lap:02} start ============")

            if lap != startlap:
                pass

            for seed in range(5):
                pbar0.write(f"----------- cell detection {seed:02} -----------")
                train.main(root_path, check_num, lap, seed)
                predict.main(root_path, check_num, lap, seed)
                pbar0.write(f"----------- detection peak {seed:02}-----------")
                peak.main(root_path, check_num, lap, seed)

            seed_checker.main(root_path, check_num, lap)
            if lap > 0:
                carryover_peaks.main(root_path, check_num, lap)

            pbar0.write("----------- Evaluation tra -----------")
            EvaluationMetric.main(root_path, check_num, lap)
            Evaluation_totrain.main(root_path, check_num, lap)
            pbar0.write("---------- precheck ---------")
            check_predictimage.main(root_path, check_num, lap)
            pbar0.write("------------- maek patch -------------")
            make_patch.main(root_path, check_num, lap)

            pbar0.write("------------- PU-learning ------------")
            run_classifier.main(root_path, check_num, lap, PU_num=PU_num)
            classifier.main(root_path, check_num, lap, PU_num=PU_num)
            pbar0.write("---------- PCA_toPU ---------")
            PCA.main(root_path, check_num, lap, PU_num=PU_num)

            pbar0.write("---------- Pclassification P ---------")
            run_Pclassifier.main(root_path, check_num, lap, push_direction="P", PU_num=PU_num, PC_num=PC_num)
            Pclassifier.main(root_path, check_num, lap, push_direction="P", PU_num=PU_num, PC_num=PC_num)
            select_PClas.main(root_path, check_num, lap, PC_num=PC_num)
            pbar0.write("---------- PCA_PC_P ---------")
            PCA2.main(root_path, check_num, lap, "P", PU_num=PU_num, PC_num=PC_num)

            pbar0.write("---------- Pclassification N ---------")
            run_Pclassifier.main(root_path, check_num, lap, push_direction="N", PU_num=PU_num, PC_num=PC_num)
            Pclassifier.main(root_path, check_num, lap, push_direction="N", PU_num=PU_num, PC_num=PC_num)
            pbar0.write("---------- PCA_PC_N ---------")
            PCA2.main(root_path, check_num, lap, "N", PU_num=PU_num, PC_num=PC_num)

            pbar0.write("--------- make new traindata ---------")
            make_merged_anotattion.main(root_path, check_num, lap, PC_num=PC_num)
            pbar0.write("----------- pseudo ----------")
            plot_pseudo.main(root_path, check_num, lap, PC_num=PC_num)
            pbar0.write("---------- check tra ---------")
            check_trainingdata.main(root_path, check_num, lap+1)

            pbar0.update(1)
        pass
    for seed in range(5):
        train.main(root_path, check_num, lastlap, seed)
        predict.main(root_path, check_num, lastlap, seed)
        peak.main(root_path, check_num, lastlap, seed)

    seed_checker.main(root_path, check_num, lastlap)
    if lastlap > 0:
        carryover_peaks.main(root_path, check_num, lastlap)
    EvaluationMetric.main(root_path, check_num, lastlap)
    Evaluation_totrain.main(root_path, check_num, lastlap)

def check():
    # startlap = 0
    # lastlap = 1
    with tqdm(total=lastlap - startlap, leave=False, position=0) as pbar0:
        for lap in range(startlap, lastlap):
            pbar0.set_description(f"lap:{lap:02}")

            if lap != startlap:
                pass
            pbar0.write("---------- precheck ---------")
            check_predictimage.main(root_path, check_num, lap)
            pbar0.write("---------- PCA_toPU ---------")
            PCA.main(root_path, check_num, lap, PU_num=PU_num)
            pbar0.write("---------- PCA_PC_P ---------")
            PCA2.main(root_path, check_num, lap, "P", PU_num=PU_num, PC_num=PC_num)
            pbar0.write("---------- PCA_PC_N ---------")
            PCA2.main(root_path, check_num, lap, "N", PU_num=PU_num, PC_num=PC_num)
            pbar0.write("----------- pseudo ----------")
            plot_pseudo.main(root_path, check_num, lap, PC_num=PC_num)

            pbar0.write("---------- training ---------")
            check_trainingdata.main(root_path, check_num, lap+1)

            pbar0.update(1)

def evalation():
    # startlap = 0
    # lastlap = 10
    with tqdm(total=lastlap - startlap + 1, leave=False, position=0) as pbar0:
        for lap in range(startlap, lastlap + 1):
            pbar0.set_description(f"lap:{lap:02}")
            pbar0.write(f"=========== lap{lap:02} start ============")

            if lap != startlap:
                pass

            pbar0.write("---------- pred ---------")
            eval.main(root_path, check_num, lap)
            pbar0.write("---------- peak ---------")
            peaktoeval.main(root_path, check_num, lap)
            pbar0.write("---------- eval ---------")
            Evaltoeval.main(root_path, check_num, lap)
            pbar0.write("---------- plot ---------")
            plot_toeval.main(root_path, check_num, lap)
            plot_pred_toeval.main(root_path, check_num, lap)

            pbar0.update(1)

if __name__ == '__main__':
    matplotlib_init()

    with tqdm(total=5, leave=False, position=0) as pbar1:
        for cond, (root_path, check_num, prior) in enumerate(zip(root_path_list, check_list, prior_list)):
            pbar1.set_description(f"condtion:{cond:02}")
            pbar1.write(f"=========== condition{cond:02} start ============")

            # root_path = os.path.join( root_path, f"root{root_num}")
            main2(root_path, check_num)

            pbar1.update(1)

    # main2()
    # # check()
    # evalation()
    
    print("finished")