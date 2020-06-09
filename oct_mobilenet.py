"""
Creates a MobileNet Model as defined in:
Andrew G. H., Menglong Z., Bo C., Dmitry K., Weijun W., Tobias W., Marco A., Hartwig A. (2017). 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
arXiv preprint arXiv:1704.04861.
import from https://github.com/marvis/pytorch-mobilenet
"""

import torch.nn as nn
from octconv import *

__all__ = ['oct_mobilenet']


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride, alpha_in=0.5, alpha_out=0.5):
    return nn.Sequential(
        Conv_BN_ACT(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False, \
            alpha_in=alpha_in, alpha_out=alpha_in if alpha_out != alpha_in else alpha_out),
        Conv_BN_ACT(inp, oup, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out)
    )


class OctMobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(OctMobileNet, self).__init__()

        self.features = nn.Sequential(
            conv_bn(  3,  32, 2),
            conv_dw( 32,  64, 1, 0, 0.5),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2), 
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1, 0.5, 0),
            conv_dw(512, 1024, 2, 0, 0),
            conv_dw(1024, 1024, 1, 0, 0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x_h, x_l = self.features(x)
        x = self.avgpool(x_h)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def oct_mobilenet(**kwargs):
    """
    Constructs a Octave MobileNet V1 model
    """
    return OctMobileNet(**kwargs)

