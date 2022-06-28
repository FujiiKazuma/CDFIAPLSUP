import os
import sys
import math
import numpy as np
import pdb
from skimage import io
from pathlib import Path
from tqdm import tqdm

from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from pathlib import Path
import glob
import numpy as np
from tqdm import tqdm
import os
from matplotlib import rcParams


##
root_path = "/home/fujii/hdd/C2C12P7/sequence/0318/sequ17"
##

def make_paths(root_path):
    ps = []
    ps.append(os.path.join(root_path, f"9"))
    ps.append(os.path.join(root_path, f"ori"))


    ps.append(os.path.join(root_path, f"GT/check"))
    ps.append(os.path.join(root_path, f"GT"))

    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.tif")))
    fs.append(sorted(Path(paths[1]).glob("*.tif")))


    
    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def plot(img, img2, cellp, frame, savepath):
    fig = plt.figure(figsize=(15.0, 12.0))

    plt.subplot(1, 2, 1)
    plt.imshow(img2)
    plt.axis("off")
    plt.gray()

    left, right = plt.xlim()
    up, low = plt.ylim()
    plt.scatter(cellp[:, 1], cellp[:, 0], c="red", marker=".", s=20, linewidths=0)
    plt.xlim(left, right)
    plt.ylim(up, low)

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.axis("off")
    plt.gray()
    left, right = plt.xlim()
    up, low = plt.ylim()
    plt.scatter(cellp[:, 1], cellp[:, 0], c="red", marker=".", s=20, linewidths=0)
    plt.xlim(left, right)
    plt.ylim(up, low)

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    save_path_vector = os.path.join(savepath, f"GT_check-{frame:04}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0)

    plt.close()      

def main(root_path):
    paths = make_paths(root_path)
    ## path list ##
    # 0: likelihood map
    # 1: cell image
    # -2: savepath(check)
    # -1: savepath(GT_lsit)
    files = load_files(paths)
    ## file list ##
    # 0: likelihood map
    # 1: cell image
    os.makedirs(paths[-2], exist_ok=True)
    os.makedirs(paths[-1], exist_ok=True)

    peak_list = np.empty((0, 2))
    with tqdm(total=len(files[0]), leave=False, position=0) as pbar0:
        for frame, (fn) in enumerate(zip(files[0], files[1])):
            pbar0.set_description(f"frame{frame:02}")
            pbar0.update(1)

            likelihoodmap = io.imread(str(fn[0]))
            cellimage = io.imread(str(fn[1]))

            peak = peak_local_max(likelihoodmap, min_distance=3, threshold_abs=228, exclude_border=False, indices=True)

            save_path_vector = os.path.join(paths[-1], f"GT-f{frame:04}.txt")
            np.savetxt(save_path_vector, peak, fmt="%d")   

            # plot(likelihoodmap, cellimage, peak, frame, paths[-2])
            pass



if __name__ == "__main__":
    main(root_path)
    print("finished")