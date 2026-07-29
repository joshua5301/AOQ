import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt

__all__ = ["resnet20", "resnet32", "resnet44", "resnet110"]

import os

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Option A (paper-original) checkpoints from akamaster/pytorch_resnet_cifar10.
model_path = {
    "resnet20": os.path.join(_MODEL_DIR, "resnet20-12fca82f.th"),
    "resnet32": os.path.join(_MODEL_DIR, "resnet32-d509ac18.th"),
    "resnet44": os.path.join(_MODEL_DIR, "resnet44-014dd654.th"),
    "resnet110": os.path.join(_MODEL_DIR, "resnet110-1d1ed7c2.th"),
}


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            # Option A (He et al.): parameter-free shortcut, stride-2 subsampling
            # with zero padding. This is the standard backbone for CIFAR-10 QAT
            # experiments and matches the released pretrained checkpoints.
            # Note: it carries no weights, so no shortcut is quantized here.
            self.shortcut = LambdaLayer(
                lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4),
                    "constant",
                    0,
                )
            )
            # Option B (1x1 conv + BN) shortcut, kept for reference. Switching to
            # it makes the downsample layers quantizable, but the released
            # checkpoints above do not contain these weights.
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(
            #         inplanes,
            #         self.expansion * planes,
            #         kernel_size=1,
            #         stride=stride,
            #         bias=False,
            #     ),
            #     nn.BatchNorm2d(self.expansion * planes),
            # )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int = 10,
    ) -> None:
        super(ResNet, self).__init__()
        self.inplanes = 16

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(
        self,
        block,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # out = self.avgpool(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str, block, layers: List[int], pretrained: bool, progress: bool, **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)

    if pretrained:
        path = model_path[arch]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"pretrained checkpoint for {arch} not found: {path}\n"
                f"download it with: bash download_models.sh"
            )
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            state_dict[k[7:] if k.startswith("module.") else k] = v
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Guard against silently loading an incompatible checkpoint (e.g. an
        # option A checkpoint into an option B model): a single unloaded
        # shortcut collapses accuracy to chance level.
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint {os.path.basename(path)} does not match {arch}\n"
                f"  missing keys:    {missing}\n"
                f"  unexpected keys: {unexpected}"
            )
    return model


def resnet20(pretrained: bool = False, progress: bool = True) -> ResNet:
    return _resnet("resnet20", BasicBlock, [3, 3, 3], pretrained, progress)


def resnet32(pretrained: bool = False, progress: bool = True) -> ResNet:
    return _resnet("resnet32", BasicBlock, [5, 5, 5], pretrained, progress)


def resnet44(pretrained: bool = False, progress: bool = True) -> ResNet:
    return _resnet("resnet44", BasicBlock, [7, 7, 7], pretrained, progress)


def resnet110(pretrained: bool = False, progress: bool = True) -> ResNet:
    return _resnet("resnet110", BasicBlock, [18, 18, 18], pretrained, progress)
