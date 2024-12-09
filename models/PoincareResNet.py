import torch.nn as nn
from typing import List
from geoopt import ManifoldTensor, PoincareBall
import hyptorch.nn as hnn
from PoincareResidualBlock import PoincareResidualBlock

### visit for implementaiton of the layers https://github.com/maxvanspengler/hyperbolic_learning_library.git 


class PoincareResNet(nn.Module):
    """
    A ResNet model adapted for the Poincaré ball model of hyperbolic space.
    """

    def __init__(
        self,
        channel_sizes: List[int],
        group_depths: List[int],
        manifold: PoincareBall,
    ):
        super(PoincareResNet, self).__init__()
        self.manifold = manifold

        self.conv = hnn.HConvolution2d(
            in_channels=3,
            out_channels=channel_sizes[0],
            kernel_size=3,
            padding=1,
            manifold=manifold,
        )
        self.bn = hnn.HBatchNorm2d(channel_sizes[0], manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)

        self.group1 = self._make_group(
            in_channels=channel_sizes[0],
            out_channels=channel_sizes[0],
            depth=group_depths[0],
            manifold=manifold,
        )
        self.group2 = self._make_group(
            in_channels=channel_sizes[0],
            out_channels=channel_sizes[1],
            depth=group_depths[1],
            stride=2,
            manifold=manifold,
        )
        self.group3 = self._make_group(
            in_channels=channel_sizes[1],
            out_channels=channel_sizes[2],
            depth=group_depths[2],
            stride=2,
            manifold=manifold,
        )

        self.avg_pool = hnn.HAvgPool2d(kernel_size=8, manifold=manifold)
        self.fc = hnn.HLinear(
            in_features=channel_sizes[2],
            out_features=10,
            manifold=manifold,
        )

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)

        out = self.avg_pool(out)
        out = self.fc(out)
        return out

    def _make_group(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 1,
        manifold: PoincareBall = None,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = hnn.HConvolution2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                manifold=manifold,
            )

        layers = [
            PoincareResidualBlock(
                in_channels,
                out_channels,
                manifold=manifold,
                stride=stride,
                downsample=downsample,
            )
        ]

        for _ in range(1, depth):
            layers.append(
                PoincareResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    manifold=manifold,
                )
            )

        return nn.Sequential(*layers)
