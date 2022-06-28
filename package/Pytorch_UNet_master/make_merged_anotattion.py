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

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root2"

check_num = 1
lap = 0
PC_num = 1

sr = 0.05  # select range
sigma = 6
lmr = 24  # lm radius
##

def make_paths(root_path, check_num, now_lap, PC_num):
    ps = []
    # ps.append(f"/home/fujii/hdd/BF-C2DL-HSC/02/ori/t0000.tif")
    # ps.append(f"/home/fujii/hdd/C2C12P7/sequence/0318/sequ17/ori/exp1_F0017-00600.tif")

    ps.append(os.path.join(root_path, f"check{check_num}/cellimage"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{now_lap}/pred_to_train/patch/label.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{now_lap}/pred_to_train/Pclassifier/check{PC_num}/P/predict.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{now_lap}/pred_to_train/Pclassifier/check{PC_num}/N/predict.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{now_lap}/traindata/likelihoodmap"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{now_lap}/traindata/lossmask"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{now_lap}/traindata/coordinate"))
    if now_lap >= 1:
        ps.append(os.path.join(root_path, f"check{check_num}/lap{now_lap}/traindata/coordinate/N"))
    
    next_lap = now_lap + 1
    ps.append(os.path.join(root_path, f"check{check_num}/lap{next_lap}/traindata/likelihoodmap"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{next_lap}/traindata/lossmask"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{next_lap}/traindata/coordinate"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{next_lap}/traindata/coordinate/N"))

    return ps

def load_files(paths, now_lap):
    fs = []
    
    fs.append(sorted(Path(paths[4]).glob("*.npz")))
    fs.append(sorted(Path(paths[5]).glob("*.npz")))
    fs.append(sorted(Path(paths[6]).glob("*.txt")))
    if now_lap >= 1:
        fs.append(sorted(Path(paths[7]).glob("*.txt")))
    else:
        fs.append(sorted(Path(paths[6]).glob("*.txt")))
    fs.append(sorted(Path(paths[0]).glob("*.*")))

    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def minmax(n):
    n_min = n.min()
    n_max = n.max()
    n = (n - n_min) / (n_max - n_min) if n_max != n_min else n - n_min
    n = n * 255
    return n

def edge_check(tmp, result, coord, r):
    xmin, xmax, ymin, ymax = coord[0]-r, coord[0]+r+1, coord[1]-r, coord[1]+r+1
    if xmin < 0:
        tmp = tmp[-xmin:]
        xmin = 0
    elif xmax > result.shape[0]:
        tmp = tmp[:result.shape[0] - xmax]
        xmax = result.shape[0]
    if ymin < 0:
        tmp = tmp[:, -ymin:]
        ymin = 0
    elif ymax > result.shape[1]:
        tmp = tmp[:, :result.shape[1] - ymax]
        ymax = result.shape[1]
    return xmin, xmax, ymin, ymax, tmp

def make_chips(sigma, lmr):
    lm = np.zeros((lmr*2+1,lmr*2+1))
    lm2 = lm.copy()
    lm[lmr, lmr] = 255
    lm = gaussian_filter(lm, sigma=sigma, mode="constant")
    lm = minmax(lm)

    mask = np.zeros((4*sigma+1, 4*sigma+1))
    cv2.circle(mask, (2*sigma, 2*sigma), 2*sigma, 1, -1)
    return lm, lm2, mask

def make_training_data(now_lap, lm, lm2, mask, peaks, image_size, high_index, low_index, paths, files):
    id = 0
    for frame, (fn) in enumerate(zip(files[0], files[1], files[2], files[3])):
        ori_lm = io.imread(str(fn[0]))
        ori_mask = io.imread(str(fn[1]))
        # ori_coord = np.loadtxt(str(fn[2]))[:, [1, 0]]
        ori_coord = np.loadtxt(str(fn[2]))
        if len(ori_coord.shape) == 1:
            ori_coord = ori_coord[np.newaxis, :]

        if now_lap >= 1:
            ori_coord_N = np.loadtxt(str(fn[3]))  # [:, [0, 1]]
            if len(ori_coord_N.shape) == 1:
                ori_coord_N = ori_coord_N[np.newaxis, :]
            if ori_coord_N.size == 0:
                ori_coord_N = np.empty((0, 2))
        else:
            ori_coord_N = np.empty((0, 2))
        peak = peaks[peaks[:, 0] == frame]

        result1 = np.zeros(image_size)
        result2 = np.zeros(image_size)
        result3 = []
        result4 = []
        for coo in peak[:, 1:]:
            if id in high_index:
                xmin, xmax, ymin, ymax, lm_tmp = edge_check(lm, result1, coo, lmr)
                result3.append(coo)
            elif id in low_index:
                xmin, xmax, ymin, ymax, lm_tmp = edge_check(lm2, result1, coo, lmr)
                result4.append(coo)
            else:
                id += 1
                continue

            result1[xmin:xmax, ymin:ymax] = np.maximum(lm_tmp, result1[xmin:xmax, ymin:ymax])

            xmin, xmax, ymin, ymax, mask_tmp = edge_check(mask, result2, coo, 2*sigma)
            result2[xmin:xmax, ymin:ymax] = np.maximum(mask_tmp, result2[xmin:xmax, ymin:ymax])
            id += 1
            pass

        result1 = np.maximum(result1, ori_lm)
        result2 = np.maximum(result2, ori_mask)
        result3 = np.array(result3)
        result3 = np.concatenate([ori_coord, result3]) if result3.size != 0 else ori_coord
        result4 = np.array(result4)
        result4 = np.concatenate([ori_coord_N, result4]) if result4.size != 0 else ori_coord_N
        
        save_path_vector = os.path.join(paths[-4], f"likelihoodmap-f{frame:04}.npz")
        np.savez_compressed(save_path_vector, result1)
        save_path_vector = os.path.join(paths[-3], f"lossmask-f{frame:04}.npz")
        np.savez_compressed(save_path_vector, result2)
        save_path_vector = os.path.join(paths[-2], f"coordinates-f{frame:04}.txt")
        np.savetxt(save_path_vector, result3, fmt="%d")
        save_path_vector = os.path.join(paths[-1], f"coordinates-f{frame:04}.txt")
        np.savetxt(save_path_vector, result4, fmt="%d")
        pass


def main(root_path, check_num, lap, PC_num=1, sr=0.05, sigma=6, lmr=24):
    paths = make_paths(root_path, check_num, lap, PC_num)
    ## path list ##
    # 0: cellimage
    # 1: label
    # 2: PC_P_pred
    # 3: PC_N_pred
    # 4: now_likelihoodmap
    # 5: now_lossmask
    # 6: now_coordinate
    # 7: now_coordinate/N
    # -4: savepath(likelihoodmap)
    # -3: savepath(lossmask)
    # -2: savepath(coordinate)
    # -1: savepath(coordinate/N)
    files = load_files(paths, lap)
    ## file list ##
    # 0: now_likelihoodmap
    # 1: now_lossmask
    # 2: now_coordinate
    # 3: now_coordinate/N
    # 4: cellimage
    os.makedirs(paths[-4], exist_ok=True)
    os.makedirs(paths[-3], exist_ok=True)
    os.makedirs(paths[-2], exist_ok=True)
    os.makedirs(paths[-1], exist_ok=True)

    image_size = io.imread(str(files[4][0])).shape
    label = np.loadtxt(paths[1])  # PP = 1, UP = 0, UN = -1
    PC_Ppred = np.loadtxt(paths[2])
    PC_Npred = np.loadtxt(paths[3])

    peaks = label[:, :3].astype("uint32")
    label = label[:, 3]

    # 尤度マップとマスクの元を作成
    lm, lm2, mask = make_chips(sigma, lmr)

    # PPデータを除外する
    peaks = peaks[label != 1]
    PC_Ppred = PC_Ppred[label != 1]
    PC_Npred = PC_Npred[label != 1] * -1

    # PU_learningの結果の上下からsrずつindexを取得
    sn = int(sr * np.sum(label != 1))
    high_index = np.argpartition(PC_Ppred, -sn)[-sn:]
    low_index = np.argpartition(PC_Npred, sn)[:sn]
    # 共通の要素は除外する
    high_index2 = np.setdiff1d(high_index, low_index)
    low_index2 = np.setdiff1d(low_index, high_index)

    # 疑似ラベルを加えた学習データを作成
    make_training_data(lap, lm, lm2, mask, peaks, image_size, high_index2, low_index2, paths, files)

if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")