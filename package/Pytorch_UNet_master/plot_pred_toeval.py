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

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"

check_num = 0
lap = 0
##

def make_paths(root_path, check_num, lap):
    ps = []
    # ps.append(f"/home/fujii/hdd/BF-C2DL-HSC/02/ori")
    ps.append(os.path.join(root_path, f"check{check_num}/testdata/cellimage"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_eval/predicted_image"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_eval/evaluations/detp"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_eval/evaluations/gtp"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_eval/precheck"))

    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.*")))
    fs.append(sorted(Path(paths[1]).glob("*.npz")))
    fs.append(sorted(Path(paths[2]).glob("*.txt")))
    fs.append(sorted(Path(paths[3]).glob("*.txt")))

    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def minmax(n):
    n_min = n.min()
    n_max = n.max()
    n = (n - n_min) / (n_max - n_min) if n_max != n_min else n - n_min
    n = n * 255
    return n

def plot_predictresult(check_num, lap, frame, cellimage, predictmask, detp, gtp, savepath):
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(cellimage)
    plt.gray()
    plt.axis("off")

    left, right = plt.xlim()
    up, low = plt.ylim()
    gtp_tmp = gtp[gtp[:, 2] == 0]
    plt.scatter(gtp_tmp[:, 1], gtp_tmp[:, 0], marker=".", s=20,linewidths=2, c="orange", label="FN")
    gtp_tmp = gtp[gtp[:, 2] == 2]
    plt.scatter(gtp_tmp[:, 1], gtp_tmp[:, 0], marker=".", s=20,linewidths=2, c="green", label="TP")
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
    detp_tmp = detp[detp[:, 2] == 1]
    plt.scatter(detp_tmp[:, 1], detp_tmp[:, 0], marker=".", s=20, zorder=2, linewidths=2, c="green", label="TP")
    detp_tmp = detp[detp[:, 2] == 0]
    plt.scatter(detp_tmp[:, 1], detp_tmp[:, 0], marker=".", s=20, zorder=1, linewidths=2, c="red", label="FP")
    plt.scatter([], [],                         marker=".", s=20, zorder=1, linewidths=2, c="pink", label="FN")
    plt.xlim(left, right)
    plt.ylim(up, low)
    
    plt.legend(bbox_to_anchor=(0, 1), loc='upper right', borderaxespad=0, fontsize=15)
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    # plt.show()

    save_path_vector = os.path.join(savepath, f"check-check{check_num:02}-lap{lap:02}-f{frame:04}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)
    plt.close()

def main(root_path, check_num, lap):
    paths = make_paths(root_path, check_num, lap)
    ## path list ##
    # 0: cellimage
    # 1: predictimage
    # 2: detp_all
    # 3: gtp_all
    # -1: savepath
    files = load_files(paths)
    ## file list ##
    # 0: cellimage
    # 1: predictimage
    # 2: detp_all
    # 3: gtp_all
    os.makedirs(paths[-1], exist_ok=True)

    # detps = np.loadtxt(paths[2])
    # gtps = np.loadtxt(paths[3])
    # detps_all = np.loadtxt(paths[4])
    # gtps_all = np.loadtxt(paths[5])

    for frame, (fn) in enumerate(zip(files[0], files[1], files[2], files[3])):
        if frame % 4 != 2:
            continue
        cellimage = io.imread(str(fn[0]))
        predictmask = io.imread(str(fn[1]))
        detp = np.loadtxt(str(fn[2]))
        gtp = np.loadtxt(str(fn[3]))

        cellimage = minmax(cellimage)
        predictmask = minmax(predictmask)

        plot_predictresult(check_num, lap, frame, cellimage, predictmask, detp, gtp, paths[-1])
        pass

if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")