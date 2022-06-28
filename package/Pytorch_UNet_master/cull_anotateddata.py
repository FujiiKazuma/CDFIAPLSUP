import cv2
import numpy as np
import argparse
import os
import math
import matplotlib.pyplot as plt
import gc
import random
import shutil
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
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root4"
# root_path = "/home/fujii/hdd/C2C12P7/sequence/0318/sequ17/root1"

cull_rate = 0.1
lr = 24  # 尤度マップの半径
sigma = 6  # 尤度マップのsigma
mr = 12  # 損失反映マスクの半径
##

def make_paths(root_path):
    ps = []
    # ps.append(f"/home/fujii/hdd/BF-C2DL-HSC/02/ori")
    # ps.append(f"/home/fujii/hdd/BF-C2DL-HSC/02/gt_plots_02.txt")

    ps.append(os.path.join(root_path, f"cellimage"))
    ps.append(os.path.join(root_path, f"allGT"))

    ps.append(os.path.join(root_path, f"traindata_first/likelihoodmap"))
    ps.append(os.path.join(root_path, f"traindata_first/lossmask"))
    ps.append(os.path.join(root_path, f"traindata_first/coordinate"))

    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.*")))
    fs.append(sorted(Path(paths[1]).glob("*.txt")))
    
    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def rand_ints_nodup(min, max, k):
    assert k < max - min + 1, "number error"
    nl = []
    while len(nl) < k:
        n = random.randint(min, max)
        if not n in nl:
            nl.append(n)
    return nl

def minmax(n):
    n_min = n.min()
    n_max = n.max()
    n = (n - n_min) / (n_max - n_min) if n_max != n_min else n - n_min
    n = n * 255
    return n

def edge_check(mask, result, coord, r):
    xmin, xmax, ymin, ymax = coord[0]-r, coord[0]+r+1, coord[1]-r, coord[1]+r+1
    if xmin < 0:
        mask = mask[-xmin:]
        xmin = 0
    elif xmax > result.shape[0]:
        mask = mask[:-(xmax - result.shape[0])]
        xmax = result.shape[0]
    if ymin < 0:
        mask = mask[:, ymin:]
        ymin = 0
    elif ymax > result.shape[1]:
        mask = mask[:, :-(ymax - result.shape[1])]
        ymax = result.shape[1]
    return xmin, xmax, ymin, ymax, mask

def main(root_path):
    paths = make_paths(root_path)
    ## path list ##
    # 0: cellimage
    # 1: GT
    # -3: savepath(likelihoodmap)
    # -2: savepath(lossmask)
    # -1: savepath(coordinate)
    files = load_files(paths)
    ## file list ##
    # 0: cellimage
    # 1: GT
    os.makedirs(paths[-3], exist_ok=True)
    os.makedirs(paths[-2], exist_ok=True)
    os.makedirs(paths[-1], exist_ok=True)

    imagesize = io.imread(str(files[0][0])).shape
    # coords = np.loadtxt(paths[1] ,delimiter=",", comments="%", dtype="int32")

    # 尤度マップの元を作成
    lm = np.zeros((lr*2+1, lr*2+1))
    lm[lr, lr] = 255
    lm = gaussian_filter(lm, sigma=sigma, mode="constant")
    lm = minmax(lm)
    # マスクの元を作成
    mask = np.zeros((mr*2+1, mr*2+1))
    cv2.circle(mask, (mr, mr), mr, 1, -1)

    random.seed(42)
    np.random.seed(42)
    for frame, fn in enumerate(files[1]):
        coord = np.loadtxt(str(fn), dtype="int32")
        # 座標データをランダムにサンプリングする
        coord_num = coord.shape[0]  # 座標の数
        cull_num = math.ceil(coord_num * cull_rate)  # サンプリングする座標の数(小数点切り上げ)
        coord_cull = coord[rand_ints_nodup(0, coord_num-1, cull_num)]  # サンプリングされた座標

        result1 = np.zeros((imagesize[0], imagesize[1]))  # likelihoodmap
        result2 = np.zeros((imagesize[0], imagesize[1]))  # lossmask
        for coo in coord_cull:
            xmin, xmax, ymin, ymax, lm_tmp = edge_check(lm, result1, coo, lr)
            result1[xmin:xmax, ymin:ymax] = np.maximum(lm_tmp, result1[xmin:xmax, ymin:ymax])

            xmin, xmax, ymin, ymax, mask_tmp = edge_check(mask, result2, coo, mr)
            result2[xmin:xmax, ymin:ymax] = np.maximum(mask_tmp, result2[xmin:xmax, ymin:ymax])
            pass
        
        save_path_vector = os.path.join(paths[-3], f"likelihoodamp-lap00-f{frame:04}.npz")
        np.savez_compressed(save_path_vector, result1)
        save_path_vector = os.path.join(paths[-2], f"lossmask-lap00-f{frame:04}.npz")
        np.savez_compressed(save_path_vector, result2)
        save_path_vector = os.path.join(paths[-1], f"coordinates-lap00-f{frame:04}.txt")
        np.savetxt(save_path_vector, coord_cull, fmt="%d")
        
        pass

if __name__ == "__main__":
    main(root_path)
    print("finished")