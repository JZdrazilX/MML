import torch.nn as nn
from geoopt import PoincareBall
from PoincareResNet import PoincareResNet
from projectorSimCLR import Projector


class SimCLR(nn.Module):
    """
    SimCLR model adapted for hyperbolic space using a Poincaré ResNet backbone.
    """

    def __init__(
        self,
        backbone: PoincareResNet,
        manifold: PoincareBall,
        projection_size: int = 2,
    ):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.manifold = manifold

        # Replace the final fully connected layer with an identity mapping
        self.backbone.fc = nn.Identity()
        feature_size = backbone.channel_sizes[-1]

        self.projector = Projector(
            feature_size=feature_size,
            projection_size=projection_size,
            manifold=manifold,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.projector(x)
        return x
