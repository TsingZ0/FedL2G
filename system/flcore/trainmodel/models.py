import math
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from flcore.trainmodel.resnet import *
from flcore.trainmodel.mobilenet_v2 import *


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, args, cid):
        super().__init__()

        self.base = eval(args.models[cid % len(args.models)])
        head = None # you may need more code for pre-existing heterogeneous heads
        if hasattr(self.base, 'heads'):
            head = self.base.heads
            self.base.heads = nn.AdaptiveAvgPool1d(args.feature_dim)
        elif hasattr(self.base, 'head'):
            head = self.base.head
            self.base.head = nn.AdaptiveAvgPool1d(args.feature_dim)
        elif hasattr(self.base, 'fc'):
            head = self.base.fc
            self.base.fc = nn.AdaptiveAvgPool1d(args.feature_dim)
        elif hasattr(self.base, 'classifier'):
            head = self.base.classifier
            self.base.classifier = nn.AdaptiveAvgPool1d(args.feature_dim)
        else:
            raise('The base model does not have a classification head.')

        if hasattr(args, 'heads'):
            self.head = eval(args.heads[cid % len(args.heads)])
        elif 'vit' in args.models[cid % len(args.models)]:
            self.head = nn.Sequential(
                nn.Linear(args.feature_dim, 768), 
                nn.Tanh(),
                nn.Linear(768, args.num_classes)
            )
        else:
            self.head = nn.Linear(args.feature_dim, args.num_classes)
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out

###########################################################

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out
