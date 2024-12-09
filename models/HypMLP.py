import torch.nn as nn
from geoopt import PoincareBall
import hyptorch.nn as hnn

### visit for implementaiton of the layers https://github.com/maxvanspengler/hyperbolic_learning_library.git 


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with hyperbolic layers.
    """

    def __init__(
        self,
        input_dim: int,
        manifold: PoincareBall,
        embedding_size: int = 256,
        hidden_size: int = 2048,
    ):
        """
        Initializes the MLP.

        Args:
            input_dim (int): Dimension of the input features.
            manifold (PoincareBall): The manifold representing hyperbolic space.
            embedding_size (int, optional): Size of the output embedding. Defaults to 256.
            hidden_size (int, optional): Size of the hidden layer. Defaults to 2048.
        """
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            hnn.HLinear(input_dim, hidden_size, manifold=manifold),
            hnn.HBatchNorm(hidden_size, manifold=manifold),
            hnn.HReLU(manifold=manifold),
            hnn.HLinear(hidden_size, embedding_size, manifold=manifold),
        )

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the MLP.
        """
        return self.net(x)
    
    
class AddProjHead(nn.Module):
    """
    Adds a projection head to a backbone model.
    """

    def __init__(
        self,
        model: nn.Module,
        in_features: int,
        layer_name: str,
        manifold: PoincareBall,
        hidden_size: int = 4096,
        embedding_size: int = 256,
    ):
        """
        Initializes the module by adding a projection head.

        Args:
            model (nn.Module): The backbone neural network model.
            in_features (int): Number of input features.
            layer_name (str): Name of the layer to replace with Identity.
            manifold (PoincareBall): The manifold representing hyperbolic space.
            hidden_size (int, optional): Size of the hidden layer. Defaults to 4096.
            embedding_size (int, optional): Size of the output embedding. Defaults to 256.
        """
        super(AddProjHead, self).__init__()
        self.backbone = model
        self.manifold = manifold

        # Replace the specified layer with Identity
        setattr(self.backbone, layer_name, nn.Identity())
        self.flatten = hnn.HFlatten()
        self.projection = MLP(
            input_dim=in_features,
            manifold=manifold,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
        )

    def forward(self, x, return_embedding: bool = False):
        """
        Forward pass of the model.

        Args:
            x: Input tensor.
            return_embedding (bool, optional): If True, returns the embedding. Defaults to False.

        Returns:
            Output tensor after passing through the projection head.
        """
        embedding = self.backbone(x)
        flattened_embedding = self.flatten(embedding)
        projection = self.projection(flattened_embedding)
        if return_embedding:
            return projection
        return projection
