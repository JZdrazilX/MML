import torch.nn as nn
from geoopt import PoincareBall
import hyptorch.nn as hnn

### visit for implementaiton of the layers https://github.com/maxvanspengler/hyperbolic_learning_library.git 


class Projector(nn.Module):
    """
    Projection head for SimCLR using hyperbolic layers.
    """

    def __init__(self, feature_size: int, projection_size: int, manifold: PoincareBall):
        super(Projector, self).__init__()
        self.manifold = manifold

        self.flatten = hnn.HFlatten()
        self.fc1 = hnn.HLinear(feature_size, feature_size, manifold=manifold)
        self.bn = hnn.HBatchNorm(feature_size, manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)
        self.fc2 = hnn.HLinear(feature_size, projection_size, manifold=manifold)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x