from typing import Optional
import torch
from torch import nn
from pi_vae_pytorch.layers import GINBlock, NFlowLayer, PermutationLayer


class GINFlowDecoder(nn.Module):
    """
    Define mean(p(x|z)) using GIN volume preserving flow.

    Parameters
    ----------
    - x_dim (int) - observed x dimension (should be much larger than z_dim)
    - z_dim (int) - latent z dimension
    - n_gin_blocks (int, default=2) - number of GIN blocks
    - gin_block_depth (int, default=2) - number of AffineCouplingLayers per GINBlock
    - affine_input_layer_slice_dim (int, default=None) - index at which to split an n-dimensional sample x input to each AffineCouplingLayer
    - affine_n_hidden_layers (int, default=2) - number of each AffineCouplingLayer's MLP hidden layers
    - affine_hidden_layer_dim (int, default=None) - dimension of each AffineCouplingLayer's MLP hidden layers
    - affine_hidden_layer_activation (nn.Module, default=nn.ReLU) - activation function applied to each AffineCouplingLayer's MLP hidden layers
    - nflow_n_hidden_layers (int, default=2) - number of the NFlowLayer's MLP hidden layers
    - nflow_hidden_layer_dim (int, default=None) - dimension of the NFlowLayer's MLP hidden layers
    - nflow_hidden_layer_activation (nn.Module, default=nn.ReLU) - activation function applied to the NFlowLayer's MLP hidden layers
    - observation_model (str, default=poisson) - poisson or gaussian. Default: poisson

    Notes
    -----
    - affine_input_layer_slice_dim (int) - when None, x_dim // 2 is assigned. Otherwise assigns the specified value, assuming 0 < affine_input_layer_slice_dim < x_dim.
    - affine_hidden_layer_dim (int) - when None, x_dim // 4 is assigned. Otherwise max(affine_hidden_layer_dim, x_dim // 4).
    - nflow_hidden_layer_dim (int) - when None, x_dim // 4 is assigned. Otherwise max(nflow_hidden_layer_dim, x_dim // 4).
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        n_gin_blocks: int = 2,
        gin_block_depth: int = 2,
        affine_input_layer_slice_dim: Optional[int] = None,
        affine_n_hidden_layers: int = 2,
        affine_hidden_layer_dim: Optional[int] = None,
        affine_hidden_layer_activation: nn.Module = nn.ReLU,
        nflow_n_hidden_layers: int = 2,
        nflow_hidden_layer_dim: Optional[int] = None,
        nflow_hidden_layer_activation: nn.Module = nn.ReLU,
        observation_model: str = 'poisson'
        ) -> None:
        super().__init__()

        self.output_activation = nn.Softplus() if observation_model == 'poisson' else nn.Identity()

        self.n_flow = NFlowLayer(
            x_dim=x_dim,
            z_dim=z_dim,
            n_hidden_layers=nflow_n_hidden_layers,
            hidden_layer_dim=nflow_hidden_layer_dim,
            hidden_layer_activation=nflow_hidden_layer_activation
        )

        layers = []
        for _ in range(n_gin_blocks):
            layers.append(PermutationLayer(x_dim))
            layers.append(
                GINBlock(
                    x_dim=x_dim,
                    n_affine_layers=gin_block_depth,
                    affine_input_layer_slice_dim=affine_input_layer_slice_dim,
                    affine_n_hidden_layers=affine_n_hidden_layers,
                    affine_hidden_layer_dim=affine_hidden_layer_dim,
                    affine_hidden_layer_activation=affine_hidden_layer_activation
                )
            )

        self.gin_blocks = nn.Sequential(*layers)
    
    def forward(
        self,
        z: torch.Tensor
        ) -> torch.Tensor:
        """
        Projects latent z into observation dimension (x_dim).
        """
       
        output = self.n_flow(z)
        output = self.gin_blocks(output)
        # Softplus or Identity
        return self.output_activation(output)
