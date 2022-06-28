import cv2
import numpy as np
import argparse
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import gc
import random
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image as pilimage
from skimage import io
from skimage.transform import rescale
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from pathlib import Path

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root4"

check_num = 1
lap = 0
r = 13  # radius of patch
##

def make_paths(root_path, check_num, lap):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/cellimage"))
    
    if lap >= 1:
        ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/carry_over_peak/peak_carryed.txt"))
    else:
        seed = np.loadtxt(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/peak/peak_num.txt")).argmax()
        ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/peak/seed{seed:02}/peak.txt"))
        
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/evaluations/detection_result.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/evaluations/detection_result-all.txt"))
    if lap >= 1:
        ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/evaluations/detection_result_N.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch/check"))

    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.*")))
    
    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def make_patch_bound(peak, img, r):
    xmin, xmax, ymin, ymax = peak[0]-r, peak[0]+r+1, peak[1]-r, peak[1]+r+1
    if xmin < 0:
        xmin = 0
        xmax = 2*r+1
    elif xmax > img.shape[0]:
        xmax = img.shape[0]
        xmin = img.shape[0] - (2*r+1)
    if ymin < 0:
        ymin = 0
        ymax = 2*r+1
    elif ymax > img.shape[1]:
        ymax = img.shape[1]
        ymin = img.shape[1] - (2*r+1)
    return int(xmin), int(xmax), int(ymin), int(ymax)

def plot_patch(frame, peak, image, label, patch, id, left, right, top, bottom, savepath):
    if label == 1: PUN = "PP"
    elif label == 0: PUN = "UP"
    elif label == -1: PUN = "UN"

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    plt.gray()
    plt.axis("off")
    rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=1, edgecolor="red", facecolor='none')
    ax.add_patch(rect)

    if image.shape[0]/2 < peak[1]:
        tx, ha = right, "right"
    else:
        tx, ha = left, "left"
    if image.shape[1]/2 < peak[0]:
        ty, va = top, "bottom"
    else:
        ty, va = bottom, "top"

    bbox_props = dict(boxstyle="square,pad=0", linewidth=1, facecolor="red", edgecolor="red")
    text = f"frame{frame:02}:id{id:04}, " + PUN
    ax.text(tx, ty, text, ha=ha, va=va, rotation=0, size=10, bbox=bbox_props)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(patch)
    plt.gray()
    plt.axis("off")

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    
    # plt.show()

    save_path_vector = os.path.join(savepath, f"check/patch_check-f{frame:02}-id{id:04}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()

def plot_patch_only(patch, id, savepath):
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(patch)
    plt.gray()
    plt.axis("off")

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    
    # plt.show()

    save_path_vector = os.path.join(savepath, f"patch-id{id:04}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()


def main(root_path, check_num, lap, r=13):
    paths = make_paths(root_path, check_num, lap)
    ## path list ##
    # 0: cellimage
    # 1: peaks
    # 2: detp
    # 3: detp_allGT
    # 4: detp_N
    # -2: savepath(patch)
    # -1: savepath(check)
    files = load_files(paths)
    ## file list ##
    # 0: cellimage
    os.makedirs(paths[-2], exist_ok=True)
    os.makedirs(paths[-1], exist_ok=True)

    peaks = np.loadtxt(paths[1])
    detps = np.loadtxt(paths[2])
    detps_all = np.loadtxt(paths[3])
    if lap != 0:
        detps_N = np.loadtxt(paths[4])

        # Nの疑似ラベルとマッチするピークを除外
        if detps_N.size > 0:
            peaks = np.delete(peaks, np.where(detps_N[:, 3] == 1), axis=0)
            detps = np.delete(detps, np.where(detps_N[:, 3] == 1), axis=0)
            detps_all = np.delete(detps_all, np.where(detps_N[:, 3] == 1), axis=0)

    # PP = 1, UP = 0, UN = -1 になるようにラベルを作成
    # detp, detp_all -> 0:FP, 1:TP
    lab_tmp = detps[:, 3] - (1 - detps_all[:, 3])
    label = np.concatenate([peaks, lab_tmp[:, np.newaxis]], axis=1)
    save_path_vector = os.path.join(paths[-2], f"label.txt")
    np.savetxt(save_path_vector, label, fmt="%d")

    # patchの器を作成
    patch = np.empty((0, r*2+1, r*2+1))
    for frame, fn in enumerate(files[0]):
        # いろいろ読み込み
        cellimage = io.imread(str(fn))
        peak = peaks[peaks[:, 0] == frame][:, 1:]
        
        # patchを作成
        for p in peak:
            xmin, xmax, ymin, ymax = make_patch_bound(p, cellimage, r)
            patch = np.concatenate([patch, cellimage[np.newaxis, xmin:xmax, ymin:ymax]])
            pass
        pass

    patch = np.array(patch)
    save_path_vector = os.path.join(paths[-2], f"patch-r{r:02}.npz")
    np.savez_compressed(save_path_vector, patch)

    print("saved patch and label")

    # patch = io.imread("/home/fujii/hdd/BF-C2DL-HSC/02/root4/check1/lap0/pred_to_train/patch/patch-r13.npz")
    # for id in range(100, 120):
    #     plot_patch_only(patch[id], id, paths[-1])
        


    # id = 0
    # with tqdm(total=patch.shape[0], leave=False, position=0) as pbar0:
    #     for frame, fn in enumerate(files[1]):
    #         peaks = np.loadtxt(str(fn))

    #         for p in peaks:
    #             pbar0.set_description(f"id:{id:04}")
    #             pbar0.update(1)

    #             xmin, xmax, ymin, ymax = make_patch_bound(p, cellimage, r)
    #             plot_patch(frame, p, cellimage, label[id, 1], patch[id], id, ymin, ymax, xmin, xmax, paths[-1])

    #             id += 1
    #             pass
    #         pass

if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")