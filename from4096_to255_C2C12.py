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
root_path = "/home/fujii/hdd/C2C12P7/sequence/0318/sequ17"
##

def make_paths(root_path):
    ps = []
    ps.append(os.path.join(root_path, f"ori"))

    ps.append(os.path.join(root_path, f"ori_max255"))

    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.tif")))
    
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
        mask = mask[:, -ymin:]
        ymin = 0
    elif ymax > result.shape[1]:
        mask = mask[:, :-(ymax - result.shape[1])]
        ymax = result.shape[1]
    return xmin, xmax, ymin, ymax, mask

def main(root_path):
    paths = make_paths(root_path)
    ## path list ##
    # 0: cellimage
    # -1: savepath
    files = load_files(paths)
    ## file list ##
    # 0: cellimage
    os.makedirs(paths[-1], exist_ok=True)

    random.seed(0)
    np.random.seed(0)
    for frame, fn in enumerate(files[0]):
        cellimage = io.imread(str(fn))

        cellimage = (cellimage / 4096) * 255

        save_path_vector = os.path.join(paths[-1], f"cellimage-f{frame:04}.npz")
        np.savez_compressed(save_path_vector, cellimage)
        pass

if __name__ == "__main__":
    main(root_path)
    print("finished")