import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
from skimage import io
from pathlib import Path
from tqdm import tqdm

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

from .EvaluationMetric import evaluate_detection

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"

check_num = 1
lap = 1
##

def make_paths(root_path, check_num, now_lap):
    ps = []

    now_peak_num = np.loadtxt(os.path.join(root_path, f"check{check_num}/lap{now_lap}/pred_to_train/peak/peak_num.txt"))
    now_seed = now_peak_num.argmax()

    ps.append(os.path.join(root_path, f"check{check_num}/cellimage"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{now_lap}/pred_to_train/predicted_image/seed{now_seed:02}"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{now_lap}/pred_to_train/peak/seed{now_seed:02}/peak.txt"))
    
    pre_lap = now_lap - 1
    assert pre_lap >= 0, "carry over lap error"
    if pre_lap == 0:
        pre_peak_num = np.loadtxt(os.path.join(root_path, f"check{check_num}/lap{pre_lap}/pred_to_train/peak/peak_num.txt"))
        pre_seed = pre_peak_num.argmax()
        ps.append(os.path.join(root_path, f"check{check_num}/lap{pre_lap}/pred_to_train/peak/seed{pre_seed:02}/peak.txt"))
    else:
        ps.append(os.path.join(root_path, f"check{check_num}/lap{pre_lap}/pred_to_train/carry_over_peak/peak_carryed.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{now_lap}/pred_to_train/carry_over_peak/check"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{now_lap}/pred_to_train/carry_over_peak"))
    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.*")))
    fs.append(sorted(Path(paths[1]).glob("*.npz")))
    
    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def plot(img, img2, now_peaks, pre_peaks, check_num, lap, seed, frame, savepath):
    fig = plt.figure(figsize=(15.0, 12.0))

    plt.subplot(1, 3, 1)  # .set_title("入力画像", fontsize=20)
    plt.imshow(img2)
    plt.axis("off")
    plt.gray()

    # left, right = plt.xlim()
    # up, low = plt.ylim()
    # plt.scatter(cellp[:, 1], cellp[:, 0], c="red", marker=".", s=20, linewidths=0)
    # plt.xlim(left, right)
    # plt.ylim(up, low)

    plt.subplot(1, 3, 2)  # .set_title("推定画像", fontsize=20)
    plt.imshow(img)
    plt.axis("off")
    plt.gray()

    plt.subplot(1, 3, 3)  # .set_title("推定画像(点あり)", fontsize=20)
    plt.imshow(img)
    plt.axis("off")
    plt.gray()

    left, right = plt.xlim()
    up, low = plt.ylim()
    plt.scatter(now_peaks[:, 1], now_peaks[:, 0], marker=".", s=20, linewidths=1, c="red", label="new peak")
    plt.scatter(pre_peaks[:, 1], pre_peaks[:, 0], marker=".", s=20, linewidths=1, c="yellow", label="peak carry overed")
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0, fontsize=15)
    plt.xlim(left, right)
    plt.ylim(up, low)


    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    save_path_vector = os.path.join(savepath, f"carrycheck-ch{check_num:02}-lap{lap:02}-seed{seed:02}-{frame:04}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0)

    # plt.show()
    plt.close()       

def main(root_path, check_num, lap):
    paths = make_paths(root_path, check_num, lap)
    ## path list ##
    # 0: cellimage
    # 1: predimage
    # 2: pre_peaks
    # 3: now_peaks
    # -1: savepath
    files = load_files(paths)
    ## file list ##
    # 0: cellimage_files
    # 1: predictimage_files
    os.makedirs(paths[-1], exist_ok=True)

    now_peaks = np.loadtxt(paths[2])
    pre_peaks = np.loadtxt(paths[3])
    
    seed = np.loadtxt(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/peak/peak_num.txt")).argmax()
    
    new_peaks = np.empty((0, 3))
    for frame, (fn) in enumerate(zip(files[0], files[1])):
        cellimage = io.imread(str(fn[0]))
        predimage = io.imread(str(fn[1]))
        
        now_p = now_peaks[now_peaks[:, 0] == frame]
        pre_p = pre_peaks[pre_peaks[:, 0] == frame]

        _1, _2, p_result, _3 = evaluate_detection(pre_p[:, 1:], now_p[:, 1:])

        ## 0: FP, 1: TP, 2: FN
        new_p = np.concatenate([now_p, pre_p[p_result[:, 2] == 0]])
        new_peaks = np.concatenate([new_peaks, new_p])

        plot(predimage, cellimage, now_p[:, 1:], pre_p[p_result[:, 2] == 0][:, 1:], check_num, lap, seed, frame, paths[-1])
        pass

    save_path_vector = os.path.join(paths[-1], f"peak_carryed.txt")
    np.savetxt(save_path_vector, new_peaks, fmt="%d")

if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")