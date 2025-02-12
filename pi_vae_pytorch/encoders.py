import torch
from torch import nn
from pi_vae_pytorch.layers import MLP


class MLPEncoder(nn.Module):
    """
    Defines mean and log of variance of q(z|x).

    Parameters
    ----------
    - x_dim (int) - observed x dimension
    - z_dim (int) - latent z dimension
    - n_hidden_layers (int, default=2) - number of MLP hidden layers
    - hidden_layer_dim (int, default=128) - dimension of each MLP hidden layer
    - activation (nn.Module, default=nn.Tanh) - activation function applied to each MLP hidden layer
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        n_hidden_layers: int = 2,
        hidden_layer_dim: int = 128,
        activation: nn.Module = nn.Tanh
        ) -> None:
        super().__init__()

        self.net = MLP(
            in_features=x_dim,
            out_features=z_dim*2,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_features=hidden_layer_dim,
            activation=activation
        )
    
    def forward(
        self,
        x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maps observed x to mean and log of variance of q(z|x).
        """

        q_z = self.net(x)
        # phi_mean, phi_log_variance
        return torch.chunk(input=q_z, chunks=2, dim=-1)
