from typing import Optional
import torch
from torch import nn


class MLP(nn.Module):
    """
    A basic Multilayer Perceptron (MLP) module.

    Parameters
    ----------
    - in_features (int) - number of input features 
    - out_features (int) - number of output features
    - n_hidden_layers (int) - number of hidden layers
    - hidden_layer_features (int) - number of features per hidden layer
    - activation (nn.Module) - activation function applied to hidden layers
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_hidden_layers: int,
        hidden_layer_features: int,
        activation: nn.Module
        ) -> None:
        super().__init__()

        # Construct layer dims
        dims = [in_features]
        for _ in range(n_hidden_layers * 2):
            dims.append(hidden_layer_features)
        dims.append(out_features)

        # Build layers
        layers = []
        act = activation

        for i in range(n_hidden_layers+1):
            # Output layer
            if i == n_hidden_layers:
                act = nn.Identity
            
            layers.append(nn.Linear(dims[2*i], dims[2*i+1]))
            layers.append(act())

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        input: torch.Tensor
        ) -> torch.Tensor:

        return self.net(input)


class PermutationLayer(nn.Module):
    """
    Randomly permutes n channels. 

    Parameters
    ----------
    - n_channels (int) - number of channels to permute
    """

    def __init__(
        self,
        n_channels: int
        ) -> None:
        super().__init__()

        self.register_buffer('perm', torch.randperm(n_channels))
        self.register_buffer('invperm', torch.argsort(self.perm))

    def forward(
        self,
        x: torch.Tensor
        ) -> torch.Tensor:

        return x[:, self.perm]

    def backward(
        self,
        x: torch.Tensor
        ) -> torch.Tensor:

        return x[:, self.invperm]


class NFlowLayer(nn.Module):
    """
    Define the first layer in GIN flow, which maps z to the cancatenation of z and t(z). t is parameterized by self.net. 
    This is equivalent to GIN model with input as z1:dim_z padding dim_x - dim_z zeros.

    Parameters
    ----------
    -x_dim (int) - observed x dimension
    - z_dim (int) - latent z dimension
    - n_hidden_layers (int, default=2) - number of MLP hidden layers
    - hidden_layer_dim (int, default=None) - dimension of MLP hidden layers
    - hidden_layer_activation (nn.Module, default=nn.ReLU) - activation function applied to MLP hidden layers

    Notes
    -----
    - hidden_layer_dim (int) - when None, x_dim // 4 is assigned. Otherwise max(hidden_layer_dim, x_dim // 4).
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        n_hidden_layers: int = 2,
        hidden_layer_dim: Optional[int] = None,
        hidden_layer_activation: nn.Module = nn.ReLU
        ) -> None:
        super().__init__()

        # Compute hidden layer dimension
        if hidden_layer_dim is None:
            hidden_dim = x_dim // 4
        else:
            hidden_dim = max(hidden_layer_dim, x_dim // 4)

        self.net = MLP(
            in_features=z_dim,
            out_features=x_dim - z_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_features=hidden_dim,
            activation=hidden_layer_activation
        )

    def forward(
        self,
        z: torch.Tensor
        ) -> torch.Tensor:
        """
        Maps z to the cancatenation of z and t(z). 
        """

        t_z = self.net(z)
        return torch.cat((z, t_z), dim=-1)


class AffineCouplingLayer(nn.Module):
    """
    Performs the affine coupling transform while preserving volume by mapping input x to
    [x_{1:x_slice_dim}, x_{x_slice_dim+1:n} * exp(s(x_{1:x_slice_dim})) + t(x_{1:x_slice_dim})].

    Parameters
    ----------
    - x_dim (int) - observed x dimension
    - x_slice_dim (int, default=None) - index at which to split an n-dimensional sample x
    - n_hidden_layers (int, default=2) - number of MLP hidden layers
    - hidden_layer_dim (int, default=None) - dimension of MLP hidden layers
    - hidden_layer_activation (nn.Module, default=nn.ReLU) - activation function applied to MLP hidden layers

    Notes
    -----
    - x_slice_dim (int) - when None, x_dim // 2 is assigned. Otherwise assigns the specified value, assuming 0 < x_slice_dim < x_dim. 
    - hidden_layer_dim (int) - when None, x_dim // 4 is assigned. Otherwise max(hidden_layer_dim, x_dim // 4).
    """

    def __init__(
        self,
        x_dim: int,
        x_slice_dim: Optional[int] = None,
        n_hidden_layers: int = 2,
        hidden_layer_dim: Optional[int] = None,
        hidden_layer_activation: nn.Module = nn.ReLU
        ) -> None:
        super().__init__()

        self.x_dim = x_dim

        # Compute slice dimension
        if x_slice_dim is None:
            self.slice_dim = x_dim // 2
        else:
            if x_slice_dim >= x_dim:
                raise ValueError('x_slice_dim must be less than x_dim.')
            self.slice_dim = x_slice_dim

        # Compute hidden layer dimension
        if hidden_layer_dim is not None:
            hidden_dim = max(hidden_layer_dim, x_dim // 4)
        else:
            hidden_dim = x_dim // 4

        self.net = MLP(
            in_features=self.slice_dim,
            out_features=2 * (x_dim - self.slice_dim) - 1,
            n_hidden_layers=n_hidden_layers,
            hidden_layer_features=hidden_dim,
            activation=hidden_layer_activation
        )

    def forward(
        self,
        x: torch.Tensor
        ) -> torch.Tensor:
        """
        Perform affine coupling transform while preserving volume.
        """

        # Split input
        x_1 = torch.narrow(x, 1, 0, self.slice_dim)
        x_2 = torch.narrow(x, 1, self.slice_dim, self.x_dim - self.slice_dim)

        # Compute scale and translation
        s_t = self.net(x_1)
        s_out = torch.narrow(s_t, 1, 0, self.x_dim - self.slice_dim - 1)
        t_out = torch.narrow(s_t, 1, self.x_dim - self.slice_dim - 1, self.x_dim - self.slice_dim)

        # clamp to ensure output of s is small
        s_out = 0.1 * torch.tanh(s_out)
        # preserve volume by ensuring the last layer has sum 0
        s_preserved = torch.cat((s_out, torch.sum(torch.neg(s_out), dim=-1, keepdim=True)), dim=-1)

        # perform transformation
        transform_x = x_2 * torch.exp(s_preserved) + t_out

        return torch.cat((transform_x, x_1), dim=-1)


class GINBlock(nn.Module):
    """
    General Incompressible-flow Network (GIN). Performs a series of affine coupling transformations 
    with inputs randomly permuted before applying each affine coupling function.

    Parameters
    ----------
    - x_dim (int) - observed x dimension
    - n_affine_layers (int, default=2) - number of AffineCouplingLayers in the block
    - affine_input_layer_slice_dim (int, default=None) - index at which to split an n-dimensional sample x input to each AffineCouplingLayer
    - affine_n_hidden_layers (int, default=2) - number of MLP hidden layers within each AffineCouplingLayer
    - affine_hidden_layer_dim (int, default=None) - dimension of each MLP hidden layer within each AffineCouplingLayer
    - affine_hidden_layer_activation (nn.Module, default=nn.ReLU) - activation function applied to MLP hidden layers within each AffineCouplingLayer

    Notes
    -----
    - affine_input_layer_slice_dim (int) - when None, x_dim // 2 is assigned. Otherwise assigns the specified value, assuming 0 < affine_input_layer_slice_dim < x_dim. 
    - affine_hidden_layer_dim (int) - when None, x_dim // 4 is assigned. Otherwise max(affine_hidden_layer_dim, x_dim // 4).
    """

    def __init__(
        self,
        x_dim: int,
        n_affine_layers: int = 2,
        affine_input_layer_slice_dim: Optional[int] = None,
        affine_n_hidden_layers: int = 2,
        affine_hidden_layer_dim: Optional[int] = None,
        affine_hidden_layer_activation: nn.Module = nn.ReLU
        ) -> None:
        super().__init__()

        layers = [
            AffineCouplingLayer(
                x_dim=x_dim,
                x_slice_dim=affine_input_layer_slice_dim,
                n_hidden_layers=affine_n_hidden_layers,
                hidden_layer_dim=affine_hidden_layer_dim,
                hidden_layer_activation=affine_hidden_layer_activation
            )
        ] * n_affine_layers

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor
        ) -> torch.Tensor:

        return self.net(x)
