from geoopt import PoincareBall
from geoopt import ManifoldTensor
import torch
import torch.nn.functional as F


class HyperbolicEMA:
    """
    Exponential Moving Average (EMA) updater for hyperbolic space.
    """

    def __init__(self, alpha: float, manifold: PoincareBall):
        """
        Initializes the EMA updater.

        Args:
            alpha (float): Decay rate for the moving average.
            manifold (PoincareBall): The manifold representing hyperbolic space.
        """
        self.alpha = alpha
        self.manifold = manifold

    def update_average(self, old: ManifoldTensor, new: ManifoldTensor) -> ManifoldTensor:
        """
        Updates the moving average in hyperbolic space.

        Args:
            old (ManifoldTensor): The old parameter tensor.
            new (ManifoldTensor): The new parameter tensor.

        Returns:
            ManifoldTensor: The updated parameter tensor.
        """
        if old is None:
            return new
        old_scaled = self.manifold.expmap0(self.manifold.logmap0(old) * self.alpha)
        new_scaled = self.manifold.expmap0(self.manifold.logmap0(new) * (1 - self.alpha))
        updated = self.manifold.mobius_add(old_scaled, new_scaled)
        return updated
    
def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the contrastive loss between two tensors.

    Args:
        x (torch.Tensor): Normalized projections from the student network.
        y (torch.Tensor): Normalized projections from the teacher network.

    Returns:
        torch.Tensor: The computed loss.
    """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)
