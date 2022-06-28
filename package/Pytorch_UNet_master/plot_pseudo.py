import cv2
import numpy as np
import argparse
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gc
import random
import glob
from skimage.feature import peak_local_max
from pathlib import Path
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image as pilimage
from skimage import io
from skimage.transform import rescale
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

import matplotlib
del matplotlib.font_manager.weight_dict['roman']

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

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root4"
# root_path = "/home/fujii/hdd/C2C12P7/sequence/0318/sequ17/root1"

check_num = 1
lap = 0
PC_num = 1
##
 
def make_paths(root_path, check_num, lap, PC_num):
    ps = []

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/Pclassifier/check{PC_num}/P/predict.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/Pclassifier/check{PC_num}/N/predict.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch/label.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/traindata/coordinate"))

    ps.append(os.path.join(root_path, f"check{check_num}/cellimage"))

    seed = np.loadtxt(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/peak/peak_num.txt")).argmax()
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/predicted_image/seed{seed:02}"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/check_Pseudo"))

    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[3]).glob("*.txt")))
    fs.append(sorted(Path(paths[4]).glob("*.*")))
    fs.append(sorted(Path(paths[5]).glob("*.*")))
    
    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def plot_pseudo(check_num, lap, high_peak, low_peak, all_peak, paths, files):
    id = 0
    for frame, (fn) in enumerate(zip(files[0], files[1], files[2])):
        gt = np.loadtxt(str(fn[0]))
        if len(gt.shape) == 1:
            gt = gt[np.newaxis, :]
        cellimage = io.imread(str(fn[1]))
        predimage = io.imread(str(fn[2]))

        h_p = high_peak[high_peak[:, 0] == frame][:, [1, 2]].astype("int32")
        l_p = low_peak[low_peak[:, 0] == frame][:, [1, 2]].astype("int32")
        a_p = all_peak[all_peak[:, 0] == frame][:, [1, 2]].astype("int32")

        fig = plt.figure(figsize=(20, 10))

        plt.subplot(1, 3, 1)
        plt.imshow(cellimage)
        plt.gray()
        plt.axis("off")

        left, right = plt.xlim()
        up, low = plt.ylim()
        plt.scatter(gt[:, 1], gt[:, 0], marker=".", s=20,linewidths=2, c="yellow", label="Grandtruth", alpha=0.7)
        # plt.scatter(a_p[:, 1], a_p[:, 0], marker=".", s=20,linewidths=2, c="green", label="Other peak", alpha=0.7)
        plt.scatter(h_p[:, 1], h_p[:, 0], marker=".", s=20,linewidths=2, c="red", label="True Pseudo")
        plt.scatter(l_p[:, 1], l_p[:, 0], marker=".", s=20,linewidths=2, c="blue", label="False Pseudo")
        plt.xlim(left, right)
        plt.ylim(up, low)


        plt.subplot(1, 3, 2)
        plt.imshow(predimage)
        plt.gray()
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(predimage)
        plt.gray()
        plt.axis("off")
        
        left, right = plt.xlim()
        up, low = plt.ylim()
        plt.scatter(gt[:, 1] , gt[:, 0],  marker="o", s=25, zorder=3, linewidths=2, c="yellow", label="GT", alpha=0.7)
        # plt.scatter(h_p[:, 1], h_p[:, 0], marker="o", s=25, zorder=2, linewidths=1, c="orange", edgecolors="red", label="True Pseudo")
        # plt.scatter(l_p[:, 1], l_p[:, 0], marker="o", s=25, zorder=2, linewidths=1, c="orange", edgecolors="blue", label="False Pseudo")
        plt.scatter(h_p[:, 1], h_p[:, 0], marker="o", s=25, zorder=2, linewidths=2, c="red", label="True Pseudo")
        plt.scatter(l_p[:, 1], l_p[:, 0], marker="o", s=25, zorder=2, linewidths=2, c="blue", label="False Pseudo")
        plt.scatter(a_p[:, 1], a_p[:, 0], marker="o", s=25, zorder=1, linewidths=2, c="green", label="Other Unlabeled", alpha=0.7)
        plt.xlim(left, right)
        plt.ylim(up, low)

        plt.legend(bbox_to_anchor=(0, 1), loc='upper right', borderaxespad=0, fontsize=15)
        
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)
        # plt.show()

        save_path_vector = os.path.join(paths[-1], f"check-f{frame:03}.png")
        plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)
        plt.close()


    pass

def main(root_path, check_num, lap, PC_num=1, sr=0.05):
    matplotlib_init()
    paths = make_paths(root_path, check_num, lap, PC_num)
    ## path list ##
    # 0: PCpred_P
    # 1: PCpred_N
    # 2: label
    # 3: GT
    # 4: cellimage
    # 5: predimage
    # -1: savepath
    files = load_files(paths)
    ## file list ##
    # 0: GT
    # 1: cellimage
    # 2: predimage
    os.makedirs(paths[-1], exist_ok=True)

    PCpred_P = np.loadtxt(paths[0])
    PCpred_N = np.loadtxt(paths[1])
    label = np.loadtxt(paths[2])

    # PPデータを除外する
    PCpred_P = PCpred_P[label[:, 3] != 1]
    PCpred_N = PCpred_N[label[:, 3] != 1] * -1
    label = label[label[:, 3] != 1]
    # PU_learningの結果の上下からsrずつindexを取得
    sn = int(sr * label.shape[0])
    high_index = np.argpartition(PCpred_P, -sn)[-sn:]
    low_index = np.argpartition(PCpred_N, sn)[:sn]
    # 共通の要素は除外する
    high_index2 = np.setdiff1d(high_index, low_index)
    low_index2 = np.setdiff1d(low_index, high_index)

    # 疑似ラベルとして選ばれたピーク
    high_peak = label[high_index2]
    low_peak = label[low_index2]

    plot_pseudo(check_num, lap, high_peak, low_peak, label, paths, files)

    pass

if __name__ == "__main__":
    main(root_path, check_num, lap, PC_num)
    print("finished")