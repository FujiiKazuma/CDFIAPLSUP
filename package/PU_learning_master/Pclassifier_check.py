import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from skimage import io
import glob
from pathlib import Path


##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"

lap = 1
PU_num = 1
PC_num = 1
push_direction = "N"
##

def make_paths(root_path, lap, PC_num, push_direction):
    ps = []
    ps.append(os.path.join(root_path, f"lap{lap}/pred_to_train/patch/label.txt"))
    ps.append(os.path.join(root_path, f"lap{lap}/pred_to_train/Pclassifier/check{PC_num}/{push_direction}/predict.txt"))

    ps.append(os.path.join(root_path, f"lap{lap}/pred_to_train/Pclassifier/check{PC_num}/{push_direction}/check"))
    return ps

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_hists(value, label, savepath):
    # U_value = value[label != 1]

    # U_min = U_value.min()
    # U_min100 = U_value[U_value.argsort()[100]]
    # U_max = U_value.max()
    # U_max100 = U_value[U_value.argsort()[-100]]
    min100 = value[value.argsort()[50]]
    max100 = value[value.argsort()[-50]]

    plot_hist_sub(value, label, value.min(), value.max(), "all", savepath)
    plot_hist_sub(value, label, value.min(), min100, "low", savepath)
    plot_hist_sub(value, label, max100, value.max(), "high", savepath)

def plot_hist_sub(value, label, lower, upper, tag, savepath):
    PP_value = value[label == 1]
    UP_value = value[label == 0]
    UN_value = value[label == -1]
    upper += 1e-1

    fig = plt.figure(figsize=(15, 10))

    plt.hist([PP_value, UP_value, UN_value], color=["red", "purple", "blue"], label=["PP", "UP", "UN"],
            bins=20, rwidth=0.9, align="mid", stacked=True, log=True, range=(lower, upper))
    plt.legend(loc='upper left')

    save_path_vector = os.path.join(savepath, f"check_hist_{tag}.png")
    plt.savefig(save_path_vector, bbox_inches = 'tight', pad_inches = 0, dpi=300)

    plt.close()

def selectable_range(value, label, savepath):
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

def main(root_path, lap):
    paths = make_paths(root_path, lap, PC_num, push_direction)
    ## path list ##
    # 0: label
    # 1: predict
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    label = np.loadtxt(paths[0])[:, 3]  #PP = 1, UP = 0, UN = -1
    # label = np.loadtxt(paths[0])

    predict = np.loadtxt(paths[1])

    value = predict

    plot_hists(value, label, paths[-1])
    selectable_range(value, label, paths[-1])

if __name__ == "__main__":
    main(root_path, lap)
    print("finished")