import torch
from torch import nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append("/home/fujii/ドキュメント/cell-detection")

from loss import PULoss

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'



class PUModel(nn.Module):
   """
   Basic Multi-layer perceptron as described in "Positive-Unlabeled Learning with Non-Negative Risk Estimator"
   """
   def __init__(self):
        super(PUModel, self).__init__()
        self.cv1 = nn.Conv2d(1, 16, 3)  # 27x27x1 -> 25x25x16
        self.pl1 = nn.MaxPool2d(2, 2)  # 25x25x16 -> 12x12x16
        self.cv2 = nn.Conv2d(16, 32, 3)  # 12x12x16 -> 10x10x32
        self.pl2 = nn.MaxPool2d(2, 2)  # 10x10x32 -> 5x5x32
        self.fc3 = nn.Linear(5*5*32,300, bias=False)
        self.bn3 = nn.BatchNorm1d(300)
        self.fc4 = nn.Linear(300,300, bias=False)
        self.bn4 = nn.BatchNorm1d(300)
        self.fc5 = nn.Linear(300,1)

   def forward(self, x):
        x = self.cv1(x)
        x = F.relu(x)
        x = self.pl1(x)
        x = self.cv2(x)
        x = F.relu(x)
        x = self.pl2(x)
        x2 = x.view(x.size()[0], -1)
        x = self.fc3(x2)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x, x2


# Alexnet
class AlexNet(nn.Module):
 
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 3 * 3)
        x = self.classifier(x)
        return x


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        # self.fc1 = nn.Linear(in_features=3, out_features=1, bias=False)
        self.fc1 = nn.Linear(in_features=2305, out_features=1, bias=False)
 
    def forward(self, x):
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    x = torch.randn(3,2)
    model = PNet()
    y = model(x)