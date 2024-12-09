import torch
import torch.nn as nn
from geoopt import PoincareBall
from LossEmmaByol import HyperbolicEMA, loss_fn
from HypMLP import MLP, AddProjHead

### visit for implementaiton of the layers https://github.com/maxvanspengler/hyperbolic_learning_library.git 


class BYOL(nn.Module):
    """
    Bootstrap Your Own Latent (BYOL) model adapted for hyperbolic space.
    """

    def __init__(
        self,
        backbone: nn.Module,
        manifold: PoincareBall,
        in_features: int = 512,
        layer_name: str = 'fc',
        projection_size: int = 2,
        projection_hidden_size: int = 2048,
        moving_average_decay: float = 0.5,
        use_momentum: bool = True,
    ):
        """
        Initializes the BYOL model.

        Args:
            backbone (nn.Module): The backbone neural network model.
            manifold (PoincareBall): The manifold representing hyperbolic space.
            in_features (int, optional): Number of input features. Defaults to 512.
            layer_name (str, optional): Name of the layer to replace with Identity. Defaults to 'fc'.
            projection_size (int, optional): Size of the projection. Defaults to 2.
            projection_hidden_size (int, optional): Size of the hidden layer in the projector. Defaults to 2048.
            moving_average_decay (float, optional): Decay rate for the moving average. Defaults to 0.5.
            use_momentum (bool, optional): Whether to use momentum in the target encoder. Defaults to True.
        """
        super(BYOL, self).__init__()
        self.manifold = manifold
        self.use_momentum = use_momentum

        # Initialize student and teacher networks
        self.student_model = AddProjHead(
            model=backbone,
            in_features=in_features,
            layer_name=layer_name,
            manifold=manifold,
            embedding_size=projection_size,
            hidden_size=projection_hidden_size,
        )
        self.teacher_model = self._get_teacher()
        self.target_ema_updater = HyperbolicEMA(
            alpha=moving_average_decay, manifold=manifold
        )
        self.student_predictor = MLP(
            input_dim=projection_size,
            manifold=manifold,
            embedding_size=projection_size,
            hidden_size=projection_hidden_size,
        )

    @torch.no_grad()
    def _get_teacher(self):
        """
        Initializes the teacher model as a copy of the student model.

        Returns:
            AddProjHead: The teacher model.
        """
        teacher_model = AddProjHead(
            model=self.student_model.backbone,
            in_features=self.student_model.projection.net[0].in_features,
            layer_name=self.student_model.backbone.__class__.__name__,
            manifold=self.manifold,
            embedding_size=self.student_model.projection.net[-1].out_features,
            hidden_size=self.student_model.projection.net[0].out_features,
        )
        teacher_model.load_state_dict(self.student_model.state_dict())
        return teacher_model

    @torch.no_grad()
    def update_moving_average(self):
        """
        Updates the teacher model's parameters using an exponential moving average.
        """
        if not self.use_momentum:
            return

        for student_params, teacher_params in zip(
            self.student_model.parameters(), self.teacher_model.parameters()
        ):
            old_weight = teacher_params.data
            new_weight = student_params.data
            teacher_params.data = self.target_ema_updater.update_average(
                old_weight, new_weight
            )

    def forward(self, image_one, image_two=None, return_embedding: bool = False):
        """
        Forward pass of the BYOL model.

        Args:
            image_one: First set of images.
            image_two: Second set of images.
            return_embedding (bool, optional): If True, returns the embedding. Defaults to False.

        Returns:
            torch.Tensor: The computed loss or embeddings.
        """
        if return_embedding or (image_two is None):
            return self.student_model(image_one, return_embedding=True)

        # Student projections and predictions
        student_proj_one = self.student_model(image_one)
        student_proj_two = self.student_model(image_two)
        student_pred_one = self.student_predictor(student_proj_one)
        student_pred_two = self.student_predictor(student_proj_two)

        with torch.no_grad():
            # Teacher projections
            teacher_proj_one = self.teacher_model(image_one).detach()
            teacher_proj_two = self.teacher_model(image_two).detach()

        # Compute loss
        loss_one = loss_fn(student_pred_one, teacher_proj_one)
        loss_two = loss_fn(student_pred_two, teacher_proj_two)

        return (loss_one + loss_two).mean()
