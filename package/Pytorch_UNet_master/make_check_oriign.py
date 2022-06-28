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
# root_path = "/home/fujii/hdd/C2C12P7/sequence/0318/sequ17/root1"
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root4"
##

def make_paths(root_path):
    ps = []
    ps.append(os.path.join(root_path, f"traindata_first/coordinate"))
    ps.append(os.path.join(root_path, f"traindata_first/likelihoodmap"))
    ps.append(os.path.join(root_path, f"traindata_first/lossmask"))
    ps.append(os.path.join(root_path, f"traindata_first/check"))
    ps.append(os.path.join(root_path, f"allGT"))
    ps.append(os.path.join(root_path, f"cellimage"))


    ps.append(os.path.join(root_path, f"check_origin/cellimage"))
    ps.append(os.path.join(root_path, f"check_origin/testdata/cellimage"))
    ps.append(os.path.join(root_path, f"check_origin/testdata/GT"))
    ps.append(os.path.join(root_path, f"check_origin/lap0/traindata/coordinate"))
    ps.append(os.path.join(root_path, f"check_origin/lap0/traindata/likelihoodmap"))
    ps.append(os.path.join(root_path, f"check_origin/lap0/traindata/lossmask"))
    ps.append(os.path.join(root_path, f"check_origin/allGT"))
    ps.append(os.path.join(root_path, f"check_origin/lap0/traindata/check"))

    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.txt")))
    fs.append(sorted(Path(paths[1]).glob("*.npz")))
    fs.append(sorted(Path(paths[2]).glob("*.npz")))
    fs.append(sorted(Path(paths[3]).glob("*.png")))
    fs.append(sorted(Path(paths[4]).glob("*.txt")))
    fs.append(sorted(Path(paths[5]).glob("*.*")))
    
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
    # 0: traindata coordinate
    # 1: traindata likelihoodamp
    # 2: traindata lossmask
    # 3: traindata check
    # 4: allGT
    # 5: cellimage
    # -8: savepath(cellimage) 
    # -7: savepath(testdata/cellimage) 
    # -6: savepath(testdata/GT) 
    # -5: savepath(traindata/coordinate)
    # -4: savepath(traindata/likelihoodmap)
    # -3: savepath(traindata/lossmask)
    # -2: savepath(traindata/allGT)
    # -1: savepath(traindata/check)
    files = load_files(paths)
    ## file list ##
    # 0: coordinate
    # 1: likelihoodmap
    # 2: lossmask
    # 3: check
    # 4: allGT
    # 5: cellimage
    os.makedirs(paths[-8], exist_ok=True)
    os.makedirs(paths[-7], exist_ok=True)
    os.makedirs(paths[-6], exist_ok=True)
    os.makedirs(paths[-5], exist_ok=True)
    os.makedirs(paths[-4], exist_ok=True)
    os.makedirs(paths[-3], exist_ok=True)
    os.makedirs(paths[-2], exist_ok=True)
    os.makedirs(paths[-1], exist_ok=True)

    random.seed(0)
    np.random.seed(0)
    for frame, (fn) in enumerate(zip(files[0], files[1], files[2], files[3], files[4], files[5])):
        if frame % 20 == 0:  # traindata
            shutil.copy(str(fn[0]), paths[-5])
            shutil.copy(str(fn[1]), paths[-4])
            shutil.copy(str(fn[2]), paths[-3])
            shutil.copy(str(fn[3]), paths[-1])
            shutil.copy(str(fn[4]), paths[-2])
            shutil.copy(str(fn[5]), paths[-8])
            pass
        else:  # test data
            shutil.copy(str(fn[4]), paths[-6])
            shutil.copy(str(fn[5]), paths[-7])
            pass

if __name__ == "__main__":
    main(root_path)
    print("finished")