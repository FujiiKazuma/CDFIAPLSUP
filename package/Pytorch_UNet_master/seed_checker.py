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
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root4"

check_num = 1
lap = 0
##

def make_paths(root_path, check_num, lap):
    ps = []
    for seed in range(5):
        ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/peak/seed{seed:02}/peak.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_train/peak"))

    return ps

def main(root_path, check_num, lap):
    paths = make_paths(root_path, check_num, lap)
    ## path list ##
    # 0: peak(seed:00)
    # 1: peak(seed:01)
    # 2: peak(seed:02)
    # 3: peak(seed:03)
    # 4: peak(seed:04)
    # -1: savepath
    os.makedirs(paths[-1], exist_ok=True)

    peak_num = []
    for seed in range(5):
        peak = np.loadtxt(paths[seed])
        peak_num.append(peak.shape[0])
        pass

    peak_num = np.array(peak_num)
    save_path_vector = os.path.join(paths[-1], f"peak_num.txt")
    np.savetxt(save_path_vector, peak_num, fmt="%d")

if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")