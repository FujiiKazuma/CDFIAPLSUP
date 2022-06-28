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
# root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root3"
root_path = "/home/fujii/hdd/C2C12P7/sequence/0318/sequ17/root1"

check_num = 1
lap = 0
##
 
def make_paths(root_path, check_num, lap):
    ps = []
    # ps.append(os.path.join(root_path, f"check{check_num}/cellimage"))
    ps.append(os.path.join(root_path, f"cellimage"))

    # ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/traindata/likelihoodmap"))
    # ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/traindata/lossmask"))
    # ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/traindata/coordinate"))

    # ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/traindata/check"))
    ps.append(os.path.join(root_path, f"traindata_first/likelihoodmap"))
    ps.append(os.path.join(root_path, f"traindata_first/lossmask"))
    ps.append(os.path.join(root_path, f"traindata_first/coordinate"))

    ps.append(os.path.join(root_path, f"traindata_first/check"))

    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.npz")))
    fs.append(sorted(Path(paths[1]).glob("*.npz")))
    fs.append(sorted(Path(paths[2]).glob("*.npz")))
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

def check_training_data(check_num, lap, frame, cellimage, likelihoodmap, lossmask, coords, savepath):
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(cellimage)
    plt.gray()
    plt.axis("off")

    left, right = plt.xlim()
    up, low = plt.ylim()
    plt.scatter(coords[:, 1], coords[:, 0], marker=".", s=5, linewidths=2, c="red")
    plt.xlim(left, right)
    plt.ylim(up, low)

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
    save_path_vector = os.path.join(savepath, f"traindata_check-lap00-f{frame:04}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0)
    plt.close()

def main(root_path, check_num, lap):
    paths = make_paths(root_path, check_num, lap)
    ## path list ##
    # 0: cellimage
    # 1: likelihoodmap
    # 2: lossmask
    # 3: coordinate
    # -1: savepath
    files = load_files(paths)
    ## file list ##
    # 0: cellimage
    # 1: likelihoodmap
    # 2: lossmask
    # 3: coordinate
    os.makedirs(paths[-1], exist_ok=True)

    for frame, (fn) in enumerate(zip(files[0], files[1], files[2], files[3])):
        cellimage = io.imread(str(fn[0]))
        likelihoodmap = io.imread(str(fn[1]))
        lossmask = io.imread(str(fn[2]))
        # coords = np.loadtxt(str(fn[3]), comments="%", dtype="int32")
        coords = np.loadtxt(str(fn[3]), comments="%", dtype="int32")
        if len(coords.shape) == 2:
            # coords = coords[:, [1, 0]]
            pass
        else:
            # coords = coords[np.newaxis, [1, 0]]
            coords = coords[np.newaxis, :]
        
        cellimage = minmax(cellimage)
        likelihoodmap = minmax(likelihoodmap)
        lossmask = minmax(lossmask)

        check_training_data(check_num, lap, frame, cellimage, likelihoodmap, lossmask, coords, paths[-1])
        pass

if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")