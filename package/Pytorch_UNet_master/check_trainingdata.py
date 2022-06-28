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
# del matplotlib.font_manager.weight_dict['roman']

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
lap = 1
##
 
def make_paths(root_path, check_num, lap):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/cellimage"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/traindata/likelihoodmap"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/traindata/lossmask"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/traindata/coordinate"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/traindata/coordinate/N"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/traindata/check"))

    # ps.append(os.path.join(root_path, f"cellimage"))

    # ps.append(os.path.join(root_path, f"traindata_first/likelihoodmap"))
    # ps.append(os.path.join(root_path, f"traindata_first/lossmask"))
    # ps.append(os.path.join(root_path, f"traindata_first/coordinate"))

    # ps.append(os.path.join(root_path, f"traindata_first/check"))

    return ps

def load_files(paths, lap):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.*")))
    fs.append(sorted(Path(paths[1]).glob("*.npz")))
    fs.append(sorted(Path(paths[2]).glob("*.npz")))
    fs.append(sorted(Path(paths[3]).glob("*.txt")))
    if lap == 0:
        fs.append(sorted(Path(paths[3]).glob("*.txt")))
    else:
        fs.append(sorted(Path(paths[4]).glob("*.txt")))
    
    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def minmax(n):
    n_min = n.min()
    n_max = n.max()
    n = (n - n_min) / (n_max - n_min) if n_max != n_min else n - n_min
    n = n * 255
    return n

def check_training_data(check_num, lap, frame, cellimage, likelihoodmap, lossmask, coo_P, coo_N, savepath):
    fig = plt.figure(figsize=(20, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(cellimage)
    plt.gray()
    plt.axis("off")

    left, right = plt.xlim()
    up, low = plt.ylim()
    plt.scatter(coo_P[:, 1], coo_P[:, 0], marker="o", s=15, c="red", label="Positive")
    plt.scatter(coo_N[:, 1], coo_N[:, 0], marker="o", s=15, c="blue", label="Negative")
    plt.xlim(left, right)
    plt.ylim(up, low)

    # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.1)

    plt.subplot(1, 3, 2)
    plt.imshow(likelihoodmap)
    plt.gray()
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(lossmask)
    plt.gray()
    plt.axis("off")
    
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    # plt.show()

    # save_path_vector = os.path.join(savepath, f"check-check{check_num:02}-lap{lap:02}-f{frame:03}.png")
    save_path_vector = os.path.join(savepath, f"traindata_check-lap{lap:02}-f{frame:04}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0)
    plt.close()

def main(root_path, check_num, lap):
    matplotlib_init()
    paths = make_paths(root_path, check_num, lap)
    ## path list ##
    # 0: cellimage
    # 1: likelihoodmap
    # 2: lossmask
    # 3: coordinate
    # -1: savepath
    files = load_files(paths, lap)
    ## file list ##
    # 0: cellimage
    # 1: likelihoodmap
    # 2: lossmask
    # 3: coordinate
    os.makedirs(paths[-1], exist_ok=True)

    for frame, (fn) in enumerate(zip(files[0], files[1], files[2], files[3], files[4])):
        cellimage = io.imread(str(fn[0]))
        likelihoodmap = io.imread(str(fn[1]))
        lossmask = io.imread(str(fn[2]))
        coords = np.loadtxt(str(fn[3]), comments="%", dtype="int32")
        if (coords.size != 0) & (len(coords.shape) == 1):
            coords = coords[np.newaxis, :]
        if lap > 0:
            coords_N = np.loadtxt(str(fn[4]), comments="%", dtype="int32")
            if (coords_N.size != 0) & (len(coords_N.shape) == 1):
                coords_N = coords_N[np.newaxis, :]
        else:
            coords_N = np.empty((0, 2))
        
        cellimage = minmax(cellimage)
        likelihoodmap = minmax(likelihoodmap)
        lossmask = minmax(lossmask)

        check_training_data(check_num, lap, frame, cellimage, likelihoodmap, lossmask, coords, coords_N, paths[-1])
        pass

if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")