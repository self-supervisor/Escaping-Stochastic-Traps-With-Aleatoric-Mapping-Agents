import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# unet code from https://github.com/milesial/Pytorch-UNet
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ForwardModel(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(ForwardModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 64 - 16)
        factor = 2 if bilinear else 1
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.up4_sigma = Up(128, 64, bilinear)
        self.outc_sigma = OutConv(64, n_classes)

    def forward(self, x, action_vector):
        x = x.permute(0, 3, 1, 2)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        action_vector = torch.stack([action_vector] * 3, dim=2)
        action_vector = torch.stack([action_vector] * 3, dim=2)
        x2 = torch.cat((x2, action_vector), dim=1)
        x = self.up4(x2, x1)
        logits = self.outc(x).view(-1, 7, 7, 3)
        x_sigma = self.up4_sigma(x2, x1)
        logits_sigma = self.outc_sigma(x_sigma).view(-1, 7, 7, 3)
        return logits, logits_sigma


# class ForwardModel(nn.Module):
#    def __init__(self):
#        super(ForwardModel, self).__init__()
#        self.encoder = nn.Sequential(
#            nn.Conv2d(4, 16, 2, stride=1, padding=1),  # b, 16, 10, 10
#            nn.ReLU(True),
#            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#            nn.Conv2d(16, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
#            nn.ReLU(True),
#            nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2
#        )
#        self.decoder = nn.Sequential(
#            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
#            nn.ReLU(True),
#            nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1),  # b, 8, 15, 15
#            nn.ReLU(True),
#            nn.ConvTranspose2d(8, 3, 3, stride=1, padding=1),  # b, 1, 28, 28
#        )
#        self.decoder_sigma = nn.Sequential(
#            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
#            nn.ReLU(True),
#            nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1),  # b, 8, 15, 15
#            nn.ReLU(True),
#            nn.ConvTranspose2d(8, 3, 3, stride=1, padding=1),  # b, 1, 28, 28
#        )
#
#    def forward(self, x, action):
#        action_channel = (
#            action.type(torch.FloatTensor) * torch.ones([1, 7, 7, 16])
#        ).permute(3, 0, 1, 2)
#        x = x.permute(0, 3, 1, 2)
#        x = torch.cat([x, action_channel], dim=1)
#        x = self.encoder(x)
#        x_mean = self.decoder(x)
#        x_sigma = self.decoder_sigma(x)
#        x_mean = x_mean.permute(0, 2, 3, 1)
#        x_sigma = x_sigma.permute(0, 2, 3, 1)
##        return x_mean, x_sigma
