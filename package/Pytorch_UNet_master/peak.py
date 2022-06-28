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

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"

check_num = 2
lap = 0
##

def make_paths(root_path, check_num, lap, seed):
    ps = []
    ps.append(os.path.join(root_path, f"check{check_num}/cellimage"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/predicted_image/seed{seed:02}"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/peak/seed{seed:02}"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/peak/seed{seed:02}/check/check_all"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/peak/seed{seed:02}/check/check_pred"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/peak/seed{seed:02}/check/check_plot"))

    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.*")))
    fs.append(sorted(Path(paths[1]).glob("*.npz")))
    
    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def minmax(n):
    n_min = n.min()
    n_max = n.max()
    n = (n - n_min) / (n_max - n_min) if n_max != n_min else n - n_min
    n = n * 255
    return n

def plot(img, img2, cellp, check_num, lap, seed, frame, savepath):
    fig = plt.figure(figsize=(15.0, 12.0))

    plt.subplot(1, 3, 1)
    plt.imshow(img2)
    plt.axis("off")
    plt.gray()

    left, right = plt.xlim()
    up, low = plt.ylim()
    plt.scatter(cellp[:, 1], cellp[:, 0], c="red", marker=".", s=20, linewidths=0)
    plt.xlim(left, right)
    plt.ylim(up, low)

    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.axis("off")
    plt.gray()

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.axis("off")
    plt.gray()
    left, right = plt.xlim()
    up, low = plt.ylim()
    plt.scatter(cellp[:, 1], cellp[:, 0], c="red", marker=".", s=20, linewidths=0)
    plt.xlim(left, right)
    plt.ylim(up, low)

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    save_path_vector = os.path.join(savepath[0], f"peakcheck-check{check_num:02}-lap{lap:02}-seed{seed:02}-{frame:04}-all.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()      

    ## plot only predicted image

    plt.imshow(img)
    plt.axis("off")
    plt.gray()
    save_path_vector = os.path.join(savepath[1], f"peakcheck-check{check_num:02}-lap{lap:02}-seed{seed:02}-{frame:04}-pred.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()      

    ## plot predicted image with plot

    plt.imshow(img)
    plt.axis("off")
    plt.gray()
    left, right = plt.xlim()
    up, low = plt.ylim()
    plt.scatter(cellp[:, 1], cellp[:, 0], c="red", marker=".", s=20, linewidths=0)
    plt.xlim(left, right)
    plt.ylim(up, low)
    save_path_vector = os.path.join(savepath[2], f"peakcheck-check{check_num:02}-lap{lap:02}-seed{seed:02}-{frame:04}-plot.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0)

    plt.close()       

def main(root_path, check_num, lap, seed):
    paths = make_paths(root_path, check_num, lap, seed)
    ## path list ##
    # 0: cellimage
    # 1: predict_image
    # -4: savepath(peak)
    # -3: savepath(peak_check_all)
    # -2: savepath(peak_check_pred)
    # -1: savepath(peak_check_plot)
    files = load_files(paths)
    ## file list ##
    # 0: cellimage
    # 1: predict_image
    os.makedirs(paths[-4], exist_ok=True)
    os.makedirs(paths[-3], exist_ok=True)
    os.makedirs(paths[-2], exist_ok=True)
    os.makedirs(paths[-1], exist_ok=True)

    peak_list = np.empty((0, 3))
    with tqdm(total=len(files[0]), leave=False, position=0) as pbar0:
        for frame, (fn) in enumerate(zip(files[0], files[1])):
            pbar0.set_description(f"file{frame:02}")
            pbar0.update(1)

            cellimage = io.imread(str(fn[0]))
            predictmask = io.imread(str(fn[1]))
            cellimage = minmax(cellimage)
            predictmask = minmax(predictmask)

            peaks = peak_local_max(predictmask, min_distance=6, threshold_abs=128, exclude_border=False, indices=True)
            peaks = np.insert(peaks, 0, frame, axis=1)
            peak_list = np.concatenate([peak_list, peaks])

            plot(predictmask, cellimage, peaks[:, 1:], check_num, lap, seed, frame, paths[-3:])
            pass

    save_path_vector = os.path.join(paths[-4], f"peak.txt")
    np.savetxt(save_path_vector, peak_list, fmt="%d")

if __name__ == "__main__":
    main(root_path, check_num, lap, seed)
    print("finished")