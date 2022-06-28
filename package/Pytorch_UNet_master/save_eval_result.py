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

def matplotlib_init():
    matplotlib.font_manager._rebuild()

    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 15 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 20 # 軸だけ変更されます
    plt.rcParams["legend.fancybox"] = False # 丸角
    plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
    plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
    plt.rcParams["legend.labelspacing"] = 1. # 垂直方向（縦）の距離の各凡例の距離
    plt.rcParams["legend.columnspacing"] = 4. # 水平方向（横）の距離の各凡例の距離
    plt.rcParams["legend.handletextpad"] = 2. # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 2 # 点がある場合のmarker scale

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root4"

check_num = 1
startlap = 0
lastlap = 5
##

def make_paths(root_path, check_num, startlap, lastlap):
    ps = []
    for lap in range(startlap, lastlap+1):
        ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_eval/evaluations/evaluation_result.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/evaluations"))

    return ps

def load_files(paths, startlap, lastlap):
    fs = []
    for lap in range(startlap, lastlap+1):
        fs.append(sorted(Path(paths[lap]).glob("*.txt")))

    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def main(root_path, check_num, startlap, lastlap):
    matplotlib_init()
    paths = make_paths(root_path, check_num, startlap, lastlap)
    ## path list ##
    # 0~n: evaluation result
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    eval_list = np.empty((0, 3))
    for lap in range(startlap, lastlap+1):
        eval = np.loadtxt(paths[lap])[-1][np.newaxis, :]
        eval_list = np.concatenate([eval_list, eval])
        pass
    save_path_vector = os.path.join(paths[-1], f"eval_result_list-lap{startlap:02}-{lastlap:02}.txt")
    np.savetxt(save_path_vector, eval_list, header="precision recall F-measure")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, xlabel="Number of pseudo-labels added", ylabel="Accuracy")
    plt.ylim(0, 1.05)
    plt.plot(np.arange(startlap, lastlap+1), eval_list[:, 0], linewidth=3, c="blue", label="Precision")
    plt.plot(np.arange(startlap, lastlap+1), eval_list[:, 1], linewidth=3, c="green", label="Recall")
    plt.plot(np.arange(startlap, lastlap+1), eval_list[:, 2], linewidth=3, c="red", label="F-measure")

    plt.legend(bbox_to_anchor=(0, 1), loc='lower left', borderaxespad=0.1, fontsize=18, ncol=3)

    save_path_vector = os.path.join(paths[-1], f"eval_result-lap{startlap:02}-{lastlap:02}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0)

    plt.close()

if __name__ == "__main__":
    main(root_path, check_num, startlap, lastlap)
    print("finished")