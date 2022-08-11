import torch
import torch.nn as nn

from models.spiking_layer import SpikeModule, Union
from models.utils import AvgPoolConv, StraightThrough

from .vgg import load_model_pytorch

__all__ = ['ResNet', 'resnet34_snn']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=ConvBiasTrue, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=ConvBiasTrue)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu1 = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=True)
        self.relu3 = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 deep_stem=False,
                 avg_down=False,
                 freeze_layer=False,
                 adarelu=False,
                 use_bn=True):

        super(ResNet, self).__init__()

        global BN
        global ReLU

        ReLU = nn.ReLU if not adarelu else AdaptiveReLU
        BN = nn.BatchNorm2d if use_bn else StraightThrough

        global ConvBiasTrue
        if use_bn is False:
            norm_layer = StraightThrough
            ConvBiasTrue = True
        else:
            norm_layer = BN
            ConvBiasTrue = False

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.num_classes = num_classes
        self.freeze_layer = freeze_layer

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2,
                          padding=1, bias=ConvBiasTrue),
                norm_layer(32),
                ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1,
                          padding=1, bias=ConvBiasTrue),
                norm_layer(32),
                ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1,
                          padding=1, bias=ConvBiasTrue),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                                   stride=2, padding=3, bias=ConvBiasTrue)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not isinstance(m, AvgPoolConv):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride,
                                 ceil_mode=True, count_include_pad=False),
                    conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def freeze_conv_layer(self):
        layers = [
            nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool),
            self.layer1, self.layer2, self.layer3, self.layer4
        ]
        for layer in layers:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.freeze_layer:
            self.freeze_conv_layer()
        return self

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


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


class SpikeBottleneck(nn.Module):
    def __init__(self, bottleneck: Bottleneck, **spike_params):
        super().__init__()
        self.conv1 = SpikeModule(conv=bottleneck.conv1, **spike_params)
        self.conv1.add_module('relu', bottleneck.relu1)
        self.conv2 = SpikeModule(conv=bottleneck.conv2, **spike_params)
        self.conv2.add_module('relu', bottleneck.relu2)
        self.conv3 = SpikeResModule(conv=bottleneck.conv3, **spike_params)
        self.conv3.add_module('relu', bottleneck.relu3)
        self.downsample = bottleneck.downsample
        self.stride = bottleneck.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.conv3(out, residual)
        return out


def resnet34_snn(pretrained=True, use_bn=True, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], deep_stem=True, use_bn=use_bn, **kwargs)
    if pretrained:
        if use_bn:
            load_model_pytorch(model, 'checkpoints/res34_snn.pth.tar')
        else:
            load_model_pytorch(model, 'checkpoints/res34_snn_wobn.pth.tar')

    return model


res_spcials = {BasicBlock: SpikeBasicBlock,
               Bottleneck: SpikeBottleneck}
