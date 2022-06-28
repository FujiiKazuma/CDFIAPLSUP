import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, random_split
import cv2
#
import argparse
import logging
import os
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from skimage import io
from PIL import Image
from torchvision import transforms

from .unet import UNet
from .utils.data_vis import plot_img_and_mask
from .utils.dataset import BasicDataset


def main():
    for lap in range(10):

    pass

if __name__ == '__main__':
    main()
    print("finished")