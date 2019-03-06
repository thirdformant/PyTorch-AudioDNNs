"""
Defines the CNN architectures in Huzaifah (2017)
"""
import torch.nn as nn
import torch.nn.functional as F
import torch


class Conv3(nn.Module):
    def __init__(self, n_classes=None, n_in=21060):
        super(Conv3, self).__init__()
        self.n_in = n_in
        self.n_classes = n_classes

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 180, kernel_size=3, stride=1, padding=1),
            nn.ReLu(),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=1),
            nn.Dropout(0.5)
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(self.n_in, 800),
            nn.ReLu(),
            nn.Dropout(0.5),
            nn.Linear(800, self.n_classes)
        )


    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

class Conv5(nn.Module):
    def __init__(self, ft, n_classes:int=None):
        super(Conv5, self).__init__()
        self.n_classes = n_classes
        self.n_in = 15360 if ft == 'wideband_linear' or ft == 'wideband_mel' else 13440
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.5),
            nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(self.n_in, 800),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(800, self.n_classes)
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
