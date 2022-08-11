from collections import OrderedDict
from typing import Any, Dict, List, Union, cast

import torch
import torch.nn as nn

from models.spiking_layer import SpikeModule

from ...utils import AvgPoolConv

__all__ = [
    'VGG', 'vgg16', 'vgg16_bn'
]


model_urls = {
    'vgg16': 'checkpoints/vgg16_snn.pth.tar',
    'vgg16_bn': 'checkpoints/vgg16_snn.pth.tar',
}


def load_model_pytorch(model, load_model, gpu_n=0):

    checkpoint = torch.load(load_model, map_location='cpu')

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    elif 'model' in checkpoint.keys():
        load_from = checkpoint['model']
    else:
        load_from = checkpoint

    # remove "module." in case the model is saved with Dist Mode
    if 'module.' in list(load_from.keys())[0]:
        load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    model.load_state_dict(load_from, strict=True)


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        drop_rate: float = 0.5
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        # This avgpool does not need to be converted
        # because the input shape is already (7, 7)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = VGGClassifier(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not isinstance(m, AvgPoolConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGClassifier(nn.Sequential):
    """
    A wrapper module for Special Spiking Conversion
    """

    def __init__(self, *module_list):
        super().__init__(*module_list)


class SpikeVGGClassifier(nn.Module):
    def __init__(self, classifier: VGGClassifier, **spike_params):
        super().__init__()
        self.fc1 = SpikeModule(conv=classifier[0], **spike_params)
        self.fc1.add_module('relu', classifier[1])
        self.dropout1 = classifier[2]
        self.fc2 = SpikeModule(conv=classifier[3], **spike_params)
        self.fc2.add_module('relu', classifier[4])
        self.dropout2 = classifier[5]
        # we do not convert the last layer to SpikeLinear
        self.fc3 = classifier[6]

    def forward(self, x):
        for c in self.children():
            x = c(x)
        return x


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        load_model_pytorch(model, model_urls[arch])
    return model


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


vgg_specials = {VGGClassifier: SpikeVGGClassifier}
