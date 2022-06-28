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
lap = 0
##

def make_paths(root_path, check_num, lap, seed):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/cellimage"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/predicted_image/seed{seed:02}"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/evaluations/eval/detection_result.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/evaluations/eval/gt_labels.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/evaluations/eval/detection_result-all.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/evaluations/eval/gt_labels-all.txt"))
    # ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/evaluations/detection_result.txt"))
    # ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/evaluations/eval/gt_labels-all.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/precheck"))

    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.*")))
    fs.append(sorted(Path(paths[1]).glob("*.npz")))
    
    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def minmax(n):
    n_min = n.min()
    n_max = n.max()
    n = (n - n_min) / (n_max - n_min) if n_max != n_min else n - n_min
    n = n * 255
    return n

def np2png(img):
    img = img - img.min()
    img = (img / img.max()) * 255


    img = np.stack([img, img, img], 2)
    img = img.astype("uint8")
    return img

def plot_predictresult(frame, cellimage, predictmask, detp, gtp, detp_allGT, gtp_allGT, savepath):
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(cellimage)
    plt.gray()
    plt.axis("off")

    left, right = plt.xlim()
    up, low = plt.ylim()
    
    gtp_tmp = gtp_allGT[gtp_allGT[:, 2] == 0]
    plt.scatter(gtp_tmp[:, 1], gtp_tmp[:, 0], marker=".", s=20,linewidths=2, c="orange", label="FN")
    gtp_tmp = gtp_allGT[gtp_allGT[:, 2] == 2]
    plt.scatter(gtp_tmp[:, 1], gtp_tmp[:, 0], marker=".", s=20,linewidths=2, c="green", label="TP")
    gtp_tmp = gtp[gtp[:, 2] == 0]
    plt.scatter(gtp_tmp[:, 1], gtp_tmp[:, 0], marker=".", s=20,linewidths=2, c="orange", label="FN")
    gtp_tmp = gtp[gtp[:, 2] == 2]
    plt.scatter(gtp_tmp[:, 1], gtp_tmp[:, 0], marker=".", s=20,linewidths=2, c="yellowgreen", label="TP")

    plt.xlim(left, right)
    plt.ylim(up, low)

    plt.subplot(1, 3, 2)
    plt.imshow(predictmask)
    plt.gray()
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(predictmask)
    plt.gray()
    plt.axis("off")

    left, right = plt.xlim()
    up, low = plt.ylim()
    
    detp_tmp = detp[detp[:, 2] == 0]
    plt.scatter(detp_tmp[:, 1], detp_tmp[:, 0], marker=".", s=20, linewidths=2, c="red", label="FP")
    detp_tmp = detp[detp[:, 2] == 1]
    plt.scatter(detp_tmp[:, 1], detp_tmp[:, 0], marker=".", s=20, linewidths=2, c="yellowgreen", label="TP")
    # detp_tmp = detp[detp[:, 2] == 2]
    # plt.scatter(detp_tmp[:, 1], detp_tmp[:, 0], marker=".", s=20, linewidths=2, c="orange", label="FN")

    detp_tmp = detp[(detp[:, 2] == 0) & (detp_allGT[:, 2] == 1)]
    plt.scatter(detp_tmp[:, 1], detp_tmp[:, 0], marker=".", s=20, linewidths=2, c="green", label="TP by true GT")
    plt.scatter([], [], marker=".", s=20, linewidths=2, c="orange", label="FN")

    plt.xlim(left, right)
    plt.ylim(up, low)

    
    plt.legend(bbox_to_anchor=(0, 1), loc='upper right', borderaxespad=0, fontsize=15)
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    # plt.show()

    save_path_vector = os.path.join(savepath, f"check-f{frame:03}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0)
    plt.close()

def plot_predictresult_only(frame, cellimage, predictmask, detp, gtp, detp_allGT, gtp_allGT, savepath):
    fig = plt.figure(figsize=(20, 10))

    plt.imshow(predictmask)
    plt.gray()
    plt.axis("off")

    left, right = plt.xlim()
    up, low = plt.ylim()
    
    detp_tmp = detp[detp[:, 2] == 1]
    plt.scatter(detp_tmp[:, 1], detp_tmp[:, 0], marker="o", s=15, linewidth=2, zorder=2, c="red", label="Positive")
    detp_tmp = detp[detp[:, 2] == 0]
    plt.scatter(detp_tmp[:, 1], detp_tmp[:, 0], marker="o", s=15, linewidth=2, zorder=1, c="orange", label="Unlabeled")

    plt.xlim(left, right)
    plt.ylim(up, low)

    
    # plt.legend(bbox_to_anchor=(0, 1), loc='upper right', borderaxespad=0, fontsize=15)
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    # plt.show()

    save_path_vector = os.path.join(savepath, f"check-f{frame:03}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)
    plt.close()

def plot_predictresult_only_all(frame, lap, cellimage, predictmask, detp, gtp, detp_allGT, gtp_allGT, savepath):
    fig = plt.figure(figsize=(20, 10))

    plt.imshow(predictmask)
    plt.gray()
    plt.axis("off")

    left, right = plt.xlim()
    up, low = plt.ylim()
    
    detp_tmp = detp_allGT[detp_allGT[:, 2] == 1]
    plt.scatter(detp_tmp[:, 1], detp_tmp[:, 0], marker="o", s=25, linewidth=2, zorder=2, c="green", label="TP")
    detp_tmp = detp_allGT[detp_allGT[:, 2] == 0]
    plt.scatter(detp_tmp[:, 1], detp_tmp[:, 0], marker="o", s=25, linewidth=2, zorder=1, c="red", label="FP")

    plt.xlim(left, right)
    plt.ylim(up, low)

    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.1)
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    # plt.show()

    save_path_vector = os.path.join(savepath, f"check-lap{lap:02}-f{frame:03}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)
    plt.close()

def main(root_path, check_num, lap):
    matplotlib_init()
    seed = np.loadtxt(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/peak/peak_num.txt")).argmax()
    paths = make_paths(root_path, check_num, lap, seed)
    ## path list ##
    # 0: cellimage
    # 1: predictimage
    # 2: detp
    # 3: gtp
    # 4: detp_all
    # 5: gtp_all
    # -1: savepath
    files = load_files(paths)
    ## file list ##
    # 0: cellimage
    # 1: predictimage
    os.makedirs(paths[-1], exist_ok=True)

    detps = np.loadtxt(paths[2])
    gtps = np.loadtxt(paths[3])
    detps_all = np.loadtxt(paths[4])
    gtps_all = np.loadtxt(paths[5])

    for frame, (fn) in enumerate(zip(files[0], files[1])):
        cellimage = io.imread(str(fn[0]))
        predictmask = io.imread(str(fn[1]))

        cellimage = minmax(cellimage)

        detp = detps[detps[:, 0] == frame][:, 1:]
        gtp = gtps[gtps[:, 0] == frame][:, 1:]
        detp_all = detps_all[detps_all[:, 0] == frame][:, 1:]
        gtp_all = gtps_all[gtps_all[:, 0] == frame][:, 1:]

        plot_predictresult(frame, cellimage, predictmask, detp, gtp, detp_all, gtp_all, paths[-1])
        # plot_predictresult_only(frame, cellimage, predictmask, detp, gtp, detp_all, gtp_all, paths[-1])
        # plot_predictresult_only_all(frame, lap, cellimage, predictmask, detp, gtp, detp_all, gtp_all, paths[-1])
        pass

if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")