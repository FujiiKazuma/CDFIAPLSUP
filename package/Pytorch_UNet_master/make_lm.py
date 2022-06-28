import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('tracklet')  # data/090303_exp1_F0009_GT_full.txt
    parser.add_argument('imagepath')  # data/imgs/cell
    parser.add_argument('savepath')  # data/masks/cell_lm
    parser.add_argument('--sigma', default=6)

    args = parser.parse_args()

    tracklet = args.tracklet
    savepath = args.savepath
    imagepath = args.imagepath
    sigma = args.sigma

    track_data = np.loadtxt(tracklet, dtype='int16')
    image_path_vector = os.path.join(imagepath, "0000.png")
    cell_image = cv2.imread(image_path_vector)
    image_size = cv2.imread(image_path_vector, -1).shape
    os.makedirs(savepath, exist_ok=True)

    frame_list = np.unique(track_data[:, 0])
    zeros = np.zeros(image_size)
    bar = tqdm(total=len(frame_list), position=0)
    for frame in frame_list:
        bar.set_description(f'frame{frame}')
        bar.update(1)
        result = zeros.copy()
        for p in track_data[track_data[:, 0] == frame][:, 2:4]:
            result_tmp = zeros.copy()
            result_tmp[p[1], p[0]] = 1
            result_tmp = gaussian_filter(result_tmp, sigma=sigma, mode="constant")
            result = np.maximum(result, result_tmp)
        """
        image_path_vector = os.path.join(imagepath, f"{frame:04}.png")
        cell_image = cv2.imread(image_path_vector)
        cell_image = (cell_image * 0.7).astype(np.uint8)
        result *= (125/result.max())
        result = result + cell_image[:, :, 1]
        result = np.where(result > 255, 255, result)
        cell_image[:, :, 1] = result
        """
        result *= (255/result.max())

        save_path_vector = os.path.join(savepath, f"{frame:04}_lm.png")
        cv2.imwrite(save_path_vector, result)

    print('finished')