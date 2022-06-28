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

def make_savedirs(root):
    savepath = []
    for i in range(10):
        savepath.append(os.path.join(root, f"{i:02}-{i+1:02}"))
        os.makedirs(savepath[i], exist_ok=True)
        pass
    return savepath

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
    assert all((0 < input) & (input < 1)), "input in (0, 1)"

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

def plot_hist(value, label):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)

    PP_value = value[label == 1]
    UP_value = value[label == 0]
    UN_value = value[label == -1]


    plt.hist([PP_value, UP_value, UN_value], color=["red", "purple", "blue"], label=["PP", "UP", "UN"],
            bins=20, range=(0, 1), log=True, rwidth=0.9, align="mid", stacked=True)
    # plt.hist(value, bins=10, range=(0, 1), log=True, rwidth=0.9, align="mid")
    # plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.legend(loc='upper right')
    plt.title(f"range(0 - 1) prior = {prior}")
    
    # plt.show()

    save_path_vector = os.path.join(savepath, f"check_hist.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()

def plot_hist_highlow(value, label):

    # PP_value = value_tmp[label_tmp == 1]
    UP_value = value[label == 0]
    UN_value = value[label == -1]

    U_min = int(value[label != 1].min() * 100) / 100
    U_min100 = int(value[label != 1][value[label != 1].argsort()[100]] * 100) / 100 + 0.01

    plt.hist([UP_value, UN_value], color=["red", "blue"], label=["UP", "UN"],
            bins=20, range=(U_min, U_min100), rwidth=0.9, align="mid", stacked=True)
    plt.legend(loc='upper left')
    plt.title(f"range({U_min} - {U_min100}) prior = {prior}")

    save_path_vector = os.path.join(savepath, f"check_hist_low.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()


    U_max = int(value[label != 1].max() * 100) / 100 + 0.01
    U_max100 = int(value[label != 1][value[label != 1].argsort()[-100]] * 100) / 100

    plt.hist([UP_value, UN_value], color=["red", "blue"], label=["UP", "UN"],
            bins=20, range=(U_max100, U_max), rwidth=0.9, align="mid", stacked=True)
    plt.legend(loc='upper left')
    plt.title(f"range({U_max100} - {U_max}) prior = {prior}")

    save_path_vector = os.path.join(savepath, f"check_hist_high.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()

peak_path = "/home/fujii/hdd/BF-C2DL-HSC/02/train_data/precheck1/peak"
image_path = "/home/fujii/hdd/BF-C2DL-HSC/02/train_data/cellimage"
label_PU_path = "/home/fujii/hdd/BF-C2DL-HSC/02/train_data/precheck1/patch/label.txt"
label_PN_path = "/home/fujii/hdd/BF-C2DL-HSC/02/train_data/precheck1/patch/label-allGT.txt"
patch_path = "/home/fujii/hdd/BF-C2DL-HSC/02/train_data/precheck1/patch/patch-r13.npz"
check_num = 20
prior = 0.975

predict_path = f"/home/fujii/hdd/BF-C2DL-HSC/02/train_data/precheck1/PU-learning/checks/check{check_num}/predict.txt"
savepath = f"/home/fujii/hdd/BF-C2DL-HSC/02/train_data/precheck1/PU-learning/checks/check{check_num}"
savepath_image = os.path.join(savepath, "image_plot")
savepath_patch = os.path.join(savepath, "patch_plot")
os.makedirs(savepath, exist_ok=True)
os.makedirs(savepath_image, exist_ok=True)
os.makedirs(savepath_patch, exist_ok=True)
savepaths = make_savedirs(savepath_patch)

peak_files = sorted(Path(peak_path).glob("*.txt"))
image_files = sorted(Path(image_path).glob("*.tif"))
predict = np.loadtxt(predict_path)
label_PU = np.loadtxt(label_PU_path)
label_PN = np.loadtxt(label_PN_path)
label = (label_PU + label_PN) / 2  #PP = 1, UP = 0, UN = -1
patch = io.imread(patch_path)

value = sigmoid(predict)

cm = generate_cmap(["blue", "red"])
color = select_color(value, cm)

plot_hist(value, label)
plot_hist_highlow(value, label)

r = 13
id = 0
with tqdm(total=len(peak_files), leave=False, position=0) as pbar0:
    for frame, (fn) in enumerate(zip(peak_files, image_files)):
        pbar0.set_description(f"frame:{frame:02}")
        pbar0.update(1)

        peaks = np.loadtxt(str(fn[0]))
        image = io.imread(str(fn[1]))

        val = value[id:id+peaks.shape[0]]
        lab = label[id:id+peaks.shape[0]]
        col = color[id:id+peaks.shape[0]]
        plot_image(frame, peaks, image, val, lab, col)

        with tqdm(total=peaks.shape[0], leave=False, position=1) as pbar1:
            for cell, peak in enumerate(peaks):
                pbar1.set_description(f"cell:{cell:03}")
                pbar1.update(1)
                plot_patch(frame, peak, image, value[id], label[id], patch[id], r, color[id])

                id += 1
                pass
        pass