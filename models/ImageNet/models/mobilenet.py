import torch
import torch.nn as nn
from ImageNet.models.vgg import load_model_pytorch


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride, simplified=False):
    """
    Simplified depthwise block does not have BN and ReLU for depthwise Conv
    """
    if not simplified:
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),

            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )


class MobileNet(nn.Module):
    def __init__(self, simplified = False):
        super(MobileNet, self).__init__()

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1, simplified),
            conv_dw(64, 128, 2, simplified),
            conv_dw(128, 128, 1, simplified),
            conv_dw(128, 256, 2, simplified),
            conv_dw(256, 256, 1, simplified),
            conv_dw(256, 512, 2, simplified),
            conv_dw(512, 512, 1, simplified),
            conv_dw(512, 512, 1, simplified),
            conv_dw(512, 512, 1, simplified),
            conv_dw(512, 512, 1, simplified),
            conv_dw(512, 512, 1, simplified),
            conv_dw(512, 1024, 2, simplified),
            conv_dw(1024, 1024, 1, simplified),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def mobilenetv1(pretrained=False):
    model = MobileNet(simplified=False)
    if pretrained:
        load_model_pytorch(model, "checkpoint/mobilenetv1_imagenet.pth.tar")
    return model
