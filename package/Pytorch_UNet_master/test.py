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


# files = sorted(Path("/home/fujii/hdd/BF-C2DL-HSC/02/root2/traindata_first/coordinate").glob("*.txt"))
# files = sorted(Path("/home/fujii/hdd/BF-C2DL-HSC/02/root2/allGT").glob("*.txt"))

# coords = np.empty((0, 3))
# for frame, fn in enumerate(files):
#     coo = np.loadtxt(str(fn))
#     if len(coo.shape) == 1:
#         coo = coo[np.newaxis, :]
#     coo = np.insert(coo, 0, frame, axis=1)
#     coords = np.concatenate([coords, coo], axis=0)

# for i in range(10):
#     print(test2.add(i, i+1))


savepath = "/home/fujii/hdd/BF-C2DL-HSC/02/root4/allGT"
os.makedirs(savepath, exist_ok=True)

allGT = np.loadtxt("/home/fujii/hdd/BF-C2DL-HSC/02/gt_plots_02.txt", delimiter=",", dtype="int32")
frame_list = np.unique(allGT[:, 0])

for frame in frame_list:
    # GT = allGT[allGT[:, 0] == frame][:, [3, 2]]

    # save_path_vector = os.path.join(savepath, f"GT-f{frame:04}.txt")
    # np.savetxt(save_path_vector, GT, fmt="%d")
    print(frame, ": ", np.sum(allGT[:, 0] == frame))



print("finished")