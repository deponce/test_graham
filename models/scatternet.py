import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D
import kymatio.datasets as scattering_datasets
import argparse
from torchsummary import summary

class scatternet(nn.Module):
    def __init__(self):
        super(scatternet, self).__init__()
        #self.scattering = Scattering2D(J=4, shape=(224, 224))
        #if torch.cuda.is_available():
        #    self.scattering = self.scattering.cuda()
        self.F1 = nn.Conv2d(in_channels=1251,
                            out_channels=256,
                            kernel_size=2,
                            stride=2,
                            padding=0)
        self.LN1 = nn.LayerNorm([256, 7, 7])
        self.F2 = nn.Conv2d(in_channels=256,
                            out_channels=384,
                            kernel_size=3,
                            stride=2,
                            padding=0)
        self.LN2 = nn.LayerNorm([384, 3, 3])
        self.F3 = nn.Conv2d(in_channels=384,
                            out_channels=512,
                            kernel_size=3,
                            stride=2,
                            padding=0)
        self.LN3 = nn.LayerNorm([512, 1, 1])
        self.F4 = nn.Linear(in_features=512,
                            out_features=1000)
        
        #self.BN2D = nn.BatchNorm2d(num_features=1024)
        #self.BN1D = nn.BatchNorm1d(num_features=1524)

    def forward(self, x):
        #x = self.scattering(x)
        x = x.view(-1, 3 * 417, 14, 14)
        x = self.F1(x)
        x = self.LN1(x)
        x = nn.ReLU()(x)
        x = self.F2(x)
        x = self.LN2(x)
        x = nn.ReLU()(x)
        x = self.F3(x)
        x = self.LN3(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)
        x = self.F4(x)
        return x
