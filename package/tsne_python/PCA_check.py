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

from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score


##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"

check_num = 5
lap = 0
PU_num = 1
prior = 0.5


##

def make_paths(root_path, check_num, lap, PU_num):
    ps = []
    ps.append(os.path.join(root_path, f"cellimage"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/PU-learning/check{PU_num}/feature.txt"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/patch/label.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/PU-learning/check{PU_num}/check"))

    return ps


def make_savedirs(root):
    savepath = []
    for i in range(10):
        savepath.append(os.path.join(root, f"{i:02}-{i + 1:02}"))
        os.makedirs(savepath[i], exist_ok=True)
        pass
    return savepath

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.tif")))

    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

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
    plt.savefig(save_path_vector, bbox_inches='tight', pad_inches=0, dpi=300)

    plt.close()

def main(root_path, check_num, lap, PU_num=1, prior=0.5):
    paths = make_paths(root_path, check_num, lap, PU_num)
    ## path list ##
    # 0: cellimage
    # 1: feature
    # 2: labels
    # -1: savepath
    files = load_files(paths)
    ## file list ##
    # 0: cellimage_files
    os.makedirs(paths[-1], exist_ok=True)

    feature = np.loadtxt(paths[1])
    label = np.loadtxt(paths[2])

    featmin = feature.min()
    featmax = feature.max()
    feature = (feature - featmin) / (featmax - featmin)

    decomp = PCA(n_components=2)
    decomp.fit(feature)

    featdec = decomp.fit_transform(feature)
    UN_range = np.where((-2.2 < featdec[:, 0]) & (featdec[:, 0] < -1.1) & (-0.5 < featdec[:, 1]) & (featdec[:, 1] < 0))
    UN_label = label[UN_range[0]]

    peak_PP = UN_label[UN_label[:, 3] == 1]
    peak_UP = UN_label[UN_label[:, 3] == 0]
    peak_UN = UN_label[UN_label[:, 3] == -1]

    for frame, (fn) in enumerate(files[0]):
        cellimage = io.imread(str(fn))

        p_PP = peak_PP[peak_PP[:, 0] == frame]
        p_UP = peak_UP[peak_UP[:, 0] == frame]
        p_UN = peak_UN[peak_UN[:, 0] == frame]

        fig = plt.figure(figsize=(15, 10))

        plt.imshow(cellimage)
        plt.gray()
        plt.axis("off")

        left, right = plt.xlim()
        up, low = plt.ylim()
        plt.scatter(p_PP[:, 2], p_PP[:, 1], marker="*", c="red", label="PP")
        plt.scatter(p_UP[:, 2], p_UP[:, 1], marker=".", c="purple", label="PP")
        plt.scatter(p_UN[:, 2], p_UN[:, 1], marker="x", c="blue", label="PP")
        plt.xlim(left, right)
        plt.ylim(up, low)

        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)

        save_path_vector = os.path.join(paths[-1], f"check_PCA_PU_cellimage-f{frame}.png")
        plt.savefig(save_path_vector, bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.show()

        plt.close()


if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")