import torch
from torch import nn
from pi_vae_pytorch.layers import MLP


class LabelPriorContinuous(nn.Module):
    """
    Compute the mean and log of variance of label prior p(z|u) for continuous label u.

    Parameters
    ----------
    - u_dim (int) - label u dimension
    - z_dim (int) - latent z dimension
    - n_hidden_layers (int) - number of MLP hidden layers. Default: 2
    - hidden_layer_dim (int) - dimension of MLP hidden layers. Default: 32
    - hidden_layer_activation (nn.Module) - activation function applied to MLP hidden layers. Default: nn.Tanh
    """

    def __init__(
        self,
        u_dim: int,
        z_dim: int,
        n_hidden_layers: int = 2,
        hidden_layer_dim: int = 32,
        hidden_layer_activation: nn.Module = nn.Tanh
        ) -> None:
        super().__init__()

        self.net = MLP(
            in_features=u_dim,
            out_features=z_dim*2,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_features=hidden_layer_dim,
            activation=hidden_layer_activation
        )
    
    def forward(
        self,
        u: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maps u to mean and log of variance of p(z|u).
        """
        
        z_prior = self.net(u)
        # lambda_mean, lambda_log_variance
        return torch.chunk(input=z_prior, chunks=2, dim=-1)
    

class LabelPriorDiscrete(nn.Module):
    """
    Compute the mean and log of variance of label prior p(z|u) for discrete label u.

    Parameters
    ----------
    - u_dim (int) - label u dimension
    - z_dim (int) - latent z dimension
    """

    def __init__(
        self,
        u_dim: int,
        z_dim: int
        ) -> None:
        super().__init__()

        self.embedded_mean = nn.Embedding(
            num_embeddings=u_dim,
            embedding_dim=z_dim
        )

        self.embedded_log_variance = nn.Embedding(
            num_embeddings=u_dim,
            embedding_dim=z_dim
        )

    def forward(
        self,
        u: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maps u to mean and log of variance of p(z|u).
        """

        # lambda_mean, lambda_log_variance
        return self.embedded_mean(u), self.embedded_log_variance(u)
