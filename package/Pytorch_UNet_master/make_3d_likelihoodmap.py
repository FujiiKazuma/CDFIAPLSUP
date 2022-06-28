import cv2
import numpy as np
import argparse
import os
import math
import matplotlib.pyplot as plt
import gc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image as pilimage
from skimage import io
from skimage.transform import rescale
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

def test2():  # plot tif data 2d
    for frame in range(60):
        path_vector = os.path.join("data/3d_data/TRA", f"man_track{frame:03}.tif")
        data = io.imread(path_vector)
        plt.imshow(data.max(axis=0))
        save_vector = os.path.join("data/3d_data/plt", f"man_track{frame:03}_plt.tif")
        plt.savefig(save_vector)
        
def test3():  # print cell area
    max = np.array([0, 0, 0])
    min = np.array([2000, 2000, 2000])
    for frame in range(60):
        path_vector = os.path.join("data/3d_data/TRA", f"man_track{frame:03}.tif")
        data = io.imread(path_vector)
        cell_area = np.where(data > 0)
        for i in range(3):
            max[i] = np.maximum(max[i], cell_area[i].max())
            min[i] = np.minimum(min[i], cell_area[i].min())
        print(max, min)
    return max, min

def test4():
    bar = tqdm(total=60, position=0)
    for frame in range(60):
        bar.set_description(f"frame{frame:03}")
        bar.update(1)
        oripath = os.path.join("data/3d_data/likelihoodmap2", f"man_track{frame:03}_lm.npz")
        hoge = io.imread(oripath)
        hoge = hoge.astype(np.float16)
        save_path = ("data/3d_data/likelihoodmap4")
        os.makedirs(save_path, exist_ok=True)
        save_path_vector = os.path.join(save_path, f"man_track{frame:03}_lm")
        np.savez_compressed(save_path_vector, hoge)
        

if __name__ == "__main__":

    #[ 893 1123  475] [310 120 104]
    
    img_path = ("data/3d_data/TRA")
    save_path = ("data/3d_data/likelihoodmap4")
    os.makedirs(save_path, exist_ok=True)

    blur_range = 24
    xmax, ymax, zmax, xmin, ymin, zmin = 893, 1123, 475, 310, 120, 104
    cell_range = (xmax - xmin + blur_range*2 + 1,
                  ymax - ymin + blur_range*2 + 1,
                  zmax - zmin + blur_range*2 + 1)
    image_size = (991, 1871, 965)

    lmrange = 24
    lm_parts = np.zeros((2*lmrange+1, 2*lmrange+1, 2*lmrange+1))  # 2*lmrange+1 = 49
    lm_parts[lmrange, lmrange, lmrange] = 1
    lm_parts = gaussian_filter(lm_parts, sigma=6, mode="constant")

    
    bar = tqdm(total=60, position=0)
    for frame in range(60):
        bar.set_description(f"frame{frame:03}")
        bar.update(1)
        img_path_vector = os.path.join(img_path, f"man_track{frame:03}.tif")
        track_data = io.imread(img_path_vector)[xmin - lmrange : xmax + lmrange + 1,
                                                ymin - lmrange : ymax + lmrange + 1,
                                                zmin - lmrange : zmax + lmrange + 1]
        id_list = np.unique(track_data)
        result = np.zeros(cell_range)

        bar2 = tqdm(total=id_list.size-1, position = 1)
        for ids in id_list[1:]:
            bar2.set_description(f"id{ids:03}")
            bar2.update(1)
            # result_tmp = np.zeros(cell_range)
            x, y, z = np.where(track_data == ids)
            x = round(x.mean()).astype("int16")
            y = round(y.mean()).astype("int16")
            z = round(z.mean()).astype("int16")
            # result_tmp[x + 24, y + 24, z + 24] = 1
            # result_tmp = gaussian_filter(result_tmp, sigma=6, mode="constant")
            # result = np.maximum(result, result_tmp)
            result[x-lmrange:x+lmrange+1, y-lmrange:y+lmrange+1, z-lmrange:z+lmrange+1] = np.maximum(result[x-lmrange:x+lmrange+1, y-lmrange:y+lmrange+1, z-lmrange:z+lmrange+1], lm_parts)

        result *= (255/result.max())
        zeros = np.zeros(image_size)
        zeros[xmin - blur_range : xmax + blur_range + 1,
              ymin - blur_range : ymax + blur_range + 1, 
              zmin - blur_range : zmax + blur_range + 1] = result

        del result
        gc.collect()

        zeros = zeros.astype(np.uint16)
        save_path_vector = os.path.join(save_path, f"man_track{frame:03}_lm")
        np.savez_compressed(save_path_vector, zeros)
        # io.imsave(save_path_vector, zeros)
        del zeros
        gc.collect()

    print("finished")