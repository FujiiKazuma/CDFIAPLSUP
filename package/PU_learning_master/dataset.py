import torch
from torch.utils import data
from torchvision.datasets import MNIST
from torch.utils.data import Dataset

from pathlib import Path
from glob import glob
from skimage import io
import numpy as np


class PU_Dataset(Dataset):
	def __init__(self, images_path, labels_path, unlabeled_rate=0.5):		
		self.images = io.imread(images_path)
		self.labels = np.loadtxt(labels_path)[:, 3]
		self.unlabeled_rate = unlabeled_rate
		
		self.image_max = self.images.max()

	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self, i):
		img = self.images[i]
		label = self.labels[i]

		img = img / self.image_max if self.image_max != 0 else img
		img = img * 255
		
		img = img.astype("float64")
		img = np.stack([img, img, img], 0)
		img = torch.from_numpy(img).type(torch.FloatTensor)

		label = torch.tensor(1) if label == 1 else torch.tensor(-1)

		return img, label

	def get_prior(self):
		unlabeled_rate = self.unlabeled_rate
		prior = (np.sum(self.labels == 1) + np.sum(self.labels != 1) * unlabeled_rate) / self.labels.shape[0]
		prior = torch.tensor(prior).type(torch.float32)

		return prior

class PC_Dataset(Dataset):
	def __init__(self, features_path, labels_path, push_direction):
		self.features = np.loadtxt(features_path)
		self.features = self.features / self.features.max()
		self.labels = np.loadtxt(labels_path)
		self.labels = self.labels[:, 3]
		self.p_dir = 1 if push_direction == "P" else -1
		# self.features = self.features * self.p_dir
		
	def __len__(self):
		return self.labels.shape[0]

	def __getitem__(self, i):
		feature = self.features[i]
		label = self.labels[i]
		
		feature = np.append(feature, 1)

		feature = feature.astype("float32")
		feature = torch.from_numpy(feature).type(torch.FloatTensor)
		label = torch.tensor(self.p_dir) if label == 1 else torch.tensor(-self.p_dir)
		return feature, label

class PN_MNIST(Dataset):
	def __getitem__(self, i):
		image_file = self.images_dir[i]
		label_file = self.labels_dir[i]


		input, target = super(PN_MNIST, self).__getitem__(i)
		if target % 2 == 0:
			target = torch.tensor(1)
		else:
			target = torch.tensor(-1)
	      
		return input, target