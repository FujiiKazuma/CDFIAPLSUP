import os
import logging
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from skimage import io
import glob
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"

lap = 1
PU_num = 1
prior = 0.5
##

def make_paths(root_path, lap, PU_num):
    ps = []
    ps.append(os.path.join(root_path, f"cellimage"))

    ps.append(os.path.join(root_path, f"lap{lap}/pred_to_train/PU-learning/check{PU_num}/predict.txt"))
    ps.append(os.path.join(root_path, f"lap{lap}/pred_to_train/patch/patch-r13.npz"))
    ps.append(os.path.join(root_path, f"lap{lap}/pred_to_train/patch/label.txt"))

    ps.append(os.path.join(root_path, f"lap{lap}/pred_to_train/PU-learning/check{PU_num}/check"))

    return ps

def make_savedirs(root):
    savepath = []
    for i in range(10):
        savepath.append(os.path.join(root, f"{i:02}-{i+1:02}"))
        os.makedirs(savepath[i], exist_ok=True)
        pass
    return savepath

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.tif")))
    
    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

def select_color(input, cm):
    assert all((0 <= input) & (input <= 1)), "input in (0, 1)"

    return list(map(lambda x: cm(int(x*255)), input))

def make_patch_bound(peak, img, r):
    xmin, xmax, ymin, ymax = peak[0]-r, peak[0]+r+1, peak[1]-r, peak[1]+r+1
    if peak[0] < r:
        xmin = 0
        xmax = 2*r+1
    elif peak[0] > img.shape[0] - 1 - r:
        xmax = img.shape[0]
        xmin = img.shape[0] - (2*r+1)
    if peak[1] < r:
        ymin = 0
        ymax = 2*r+1
    elif peak[1] > img.shape[1] - 1 - r:
        ymax = img.shape[1]
        ymin = img.shape[0] - (2*r+1)
    return int(xmin), int(xmax), int(ymin), int(ymax)

def plot_patch(frame, peak, image, value, label, patch, r, color):
    left, right, top, bottom = make_patch_bound(peak[[1, 0]], image, r)
    # PU = "P" if label == 1 else "U"
    if label == 1: PUN = "PP"
    elif label == 0: PUN = "UP"
    elif label == -1: PUN = "UN"
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    plt.gray()
    plt.axis("off")
    rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    if image.shape[0]/2 < peak[1]:
        tx, ha = right, "right"
    else:
        tx, ha = left, "left"
    if image.shape[1]/2 < peak[0]:
        ty, va = top, "bottom"
    else:
        ty, va = bottom, "top"

    bbox_props = dict(boxstyle="square,pad=0", linewidth=1, facecolor=color, edgecolor=color)
    text = f"frame{frame:02}:{value:.6f}, " + PUN
    ax.text(tx, ty, text, ha=ha, va=va, rotation=0, size=10, bbox=bbox_props)

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(patch)
    plt.gray()
    plt.axis("off")

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    
    # plt.show()

    value = int(value*10)
    save_path_vector = os.path.join(savepaths[value], f"check-id{id:04}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()

def plot_image(frame, peak, image, value, label, color):
    peak_PP = np.array([p for i, p in enumerate(peak) if label[i] == 1])
    peak_UP = np.array([p for i, p in enumerate(peak) if label[i] == 0])
    peak_UN = np.array([p for i, p in enumerate(peak) if label[i] == -1])
    value_PP = [v for i, v in enumerate(value) if label[i] == 1]
    value_UP = [v for i, v in enumerate(value) if label[i] == 0]
    value_UN = [v for i, v in enumerate(value) if label[i] == -1]
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(image)
    plt.gray()
    plt.axis("off")

    ax.scatter(peak_PP[:, 1], peak_PP[:, 0], marker="*", c=value_PP, cmap=cm, vmin=0, vmax=1, label="PP")
    ax.scatter(peak_UP[:, 1], peak_UP[:, 0], marker=".", c=value_UP, cmap=cm, vmin=0, vmax=1, label="UP")
    sc = ax.scatter(peak_UN[:, 1], peak_UN[:, 0], marker="x", c=value_UN, cmap=cm, vmin=0, vmax=1, label="UN")
    fig.colorbar(sc)

    # plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.)
    
    # plt.show()

    save_path_vector = os.path.join(savepath_image, f"check-f{frame:02}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()

def plot_hist(value, label, prior, savepath):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)

    PP_value = value[label == 1]
    UP_value = value[label == 0]
    UN_value = value[label == -1]


    plt.hist([PP_value, UP_value, UN_value], color=["red", "purple", "blue"], label=["PP", "UP", "UN"],
            bins=20, log=True, rwidth=0.9, align="mid", stacked=True)
    # plt.hist(value, bins=10, range=(0, 1), log=True, rwidth=0.9, align="mid")
    # plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.legend(loc='upper right')
    plt.title(f"range(0 - 1) prior = {prior}")
    
    # plt.show()

    save_path_vector = os.path.join(savepath, f"check_hist.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()

def plot_hist_highlow(value, label, prior, savepath):

    # PP_value = value_tmp[label_tmp == 1]
    UP_value = value[label == 0]
    UN_value = value[label == -1]
    U_value = value[label != 1]
    U_label = label[label != 1]

    U_min = U_value.min()
    U_min100 = U_value[U_value.argsort()[100]]

    plt.hist([UP_value, UN_value], color=["purple", "blue"], label=["UP", "UN"],
            bins=20, range=(U_min, U_min100), rwidth=0.9, align="mid", stacked=True, log=True)
    plt.legend(loc='upper left')
    plt.title(f"range({U_min} - {U_min100}) prior = {prior}")

    save_path_vector = os.path.join(savepath, f"check_hist_low.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()


    U_max = U_value.max()
    U_max100 = U_value[U_value.argsort()[-100]]

    plt.hist([UP_value, UN_value], color=["purple", "blue"], label=["UP", "UN"],
            bins=20, range=(U_max100, U_max), rwidth=0.9, align="mid", stacked=True, log=True)
    plt.legend(loc='upper left')
    plt.title(f"range({U_max100} - {U_max}) prior = {prior}")

    save_path_vector = os.path.join(savepath, f"check_hist_high.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()

def selectable_range(value, label, prior, savepath):
    U_value = value[label != 1]
    U_label = label[label != 1]

    # 出力が小さい順に見て、どこまでUNデータが取れるのか
    min_r = np.where(U_label[np.argsort(U_value)] != -1)[0][0]
    # 出力が大きい順に見て、どこまでUPデータが取れるのか
    max_r = np.where(U_label[np.argsort(U_value)] == -1)[0][-1]

    UPUN_size = np.array([value[label == 0].shape[0], value[label == -1].shape[0]])  # UPの数、UNの数
    UPUN_sele = np.array([U_label.shape[0] - max_r - 1, min_r])  # bottom Positive, top Negative in Unlabeled
    
    save_path_vector = os.path.join(savepath, f"selectable_range.txt")
    np.savetxt(save_path_vector, np.array([UPUN_size, UPUN_sele]), fmt="%d", header="UP, UN")



def main(root_path, lap, PU_num=1, prior=0.5):
    paths = make_paths(root_path, lap, PU_num)
    ## path list ##
    # 0: cellimage
    # 1: PU_pred
    # 2: patch
    # 3: labels
    # -1: savepath
    files = load_files(paths)
    ## file list ##
    # 0: cellimage_files
    # 1: predictimage_files
    os.makedirs(paths[-1], exist_ok=True)

    predict = np.loadtxt(paths[1])
    patch = io.imread(paths[2])
    label = np.loadtxt(paths[3])[:, 3]

    # peak_files, image_files, predict, patch = load_files()

    value = predict

    # cm = generate_cmap(["blue", "red"])
    # color = select_color(sigmoid(predict), cm)

    selectable_range(value, label, prior, paths[-1])
    plot_hist(value, label, prior, paths[-1])
    plot_hist_highlow(value, label, prior, paths[-1])

    # PU_learningの結果の上下から100ずつindexを取得
    sr = 100
    high_index = np.argpartition(predict, -sr)[-sr:]
    low_index = np.argpartition(predict, sr)[:sr]

    # r = 13
    # id = 0
    # with tqdm(total=len(peak_files), leave=False, position=0) as pbar0:
    #     for frame, (fn) in enumerate(zip(peak_files, image_files)):
    #         pbar0.set_description(f"frame:{frame:02}")
    #         pbar0.update(1)

    #         peaks = np.loadtxt(str(fn[0]))
    #         image = io.imread(str(fn[1]))

    #         val = value[id:id+peaks.shape[0]]
    #         lab = label[id:id+peaks.shape[0]]
    #         col = color[id:id+peaks.shape[0]]
    #         plot_image(frame, peaks, image, val, lab, col)

    #         with tqdm(total=peaks.shape[0], leave=False, position=1) as pbar1:
    #             for cell, peak in enumerate(peaks):
    #                 pbar1.set_description(f"cell:{cell:03}")
    #                 pbar1.update(1)

    #                 if id in low_index:
    #                     plot_patch(frame, peak, image, value[id], label[id], patch[id], r, color[id])

    #                 id += 1
    #                 pass
    #         pass

if __name__ == "__main__":
    main(root_path, lap)
    print("finished")