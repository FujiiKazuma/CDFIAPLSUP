import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
import gc
import random
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

check_num = 1
lap = 5
##

def make_paths(root_path, check_num, lap):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_eval/evaluations/evaluation_result.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/evaluations"))

    return ps

def main(root_path, check_num, lap):
    matplotlib_init()
    paths = make_paths(root_path, check_num, lap)
    ## path list ##
    # 0: evaluation_result
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    evalres = np.loadtxt(paths[0])[:-1]
    frame = np.arange(evalres.shape[0])

    fig = plt.figure(figsize=(15, 8))
    plt.ylim(0, 1)
    plt.plot(frame, evalres[:, 0], c="blue", label="precision")
    plt.plot(frame, evalres[:, 1], c="green", label="recall")
    plt.plot(frame, evalres[:, 2], c="red", label="F-measure")

    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0, fontsize=18)

    # plt.show()

    save_path_vector = os.path.join(paths[-1], f"evalcheck-check{check_num:02}-lap{lap:02}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()

    pass

if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")