import torch.nn as nn
from typing import Optional
import geoopt
from geoopt import ManifoldTensor, PoincareBall
import hyptorch.nn as hnn

### visit for implementaiton of the layers https://github.com/maxvanspengler/hyperbolic_learning_library.git 


class PoincareResidualBlock(nn.Module):
    """
    A Residual Block for Poincaré ResNet using hyperbolic layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        manifold: PoincareBall,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super(PoincareResidualBlock, self).__init__()
        self.manifold = manifold
        self.downsample = downsample

        self.conv1 = hnn.HConvolution2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            manifold=manifold,
        )
        self.bn1 = hnn.HBatchNorm2d(out_channels, manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)
        self.conv2 = hnn.HConvolution2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            manifold=manifold,
        )
        self.bn2 = hnn.HBatchNorm2d(out_channels, manifold=manifold)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.manifold.mobius_add(out, residual)
        out = self.relu(out)

        return out