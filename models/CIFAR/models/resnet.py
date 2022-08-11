'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''
import torch
import torch.nn as nn
import math
from CIFAR.models.utils import StraightThrough
from CIFAR.models.spiking_layer import SpikeModel, SpikeModule, Union


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu1 = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10, use_bn=True, freeze_avg=True):
        super(ResNet_Cifar, self).__init__()

        global BN
        BN = nn.BatchNorm2d if use_bn else StraightThrough
        global ReLU
        ReLU = nn.ReLU
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BN(64)
        self.relu = ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BN(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class SpikeResModule(SpikeModule):
    """
    Spike-based Module that can handle spatial-temporal information.
    threshold :param that decides the maximum value
    conv :param is the original normal conv2d module
    """

    def __init__(self, sim_length: int, conv: Union[nn.Conv2d, nn.Linear], enable_shift: bool = True):
        super(SpikeResModule, self).__init__(sim_length, conv, enable_shift)

    def forward(self, input: torch.Tensor, residual: torch.Tensor):
        if self.use_spike:
            x = (self.fwd_func(input, self.weight, self.bias, **self.fwd_kwargs)) + residual
            if self.enable_shift is True and self.threshold is not None:
                x = x + self.threshold * 0.5 / self.sim_length
            self.mem_pot = self.mem_pot + x
            spike = (self.mem_pot >= self.threshold).float() * self.threshold
            self.mem_pot -= spike
            return spike
        else:
            return self.relu(self.fwd_func(input, self.org_weight, self.org_bias, **self.fwd_kwargs) + residual)


class SpikeBasicBlock(nn.Module):
    def __init__(self, basic_block: BasicBlock, **spike_params):
        super().__init__()
        self.conv1 = SpikeModule(conv=basic_block.conv1, **spike_params)
        self.conv1.add_module('relu', basic_block.relu1)
        self.conv2 = SpikeResModule(conv=basic_block.conv2, **spike_params)
        self.conv2.add_module('relu', basic_block.relu2)
        self.downsample = basic_block.downsample
        self.stride = basic_block.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv2(out, residual)
        return out


def resnet20(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


res_specials = {BasicBlock: SpikeBasicBlock}


if __name__ == '__main__':
    net = resnet20()
    net.eval()
    x = torch.randn(1, 3, 32, 32)
    net(x)
