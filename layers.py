import numpy as np
import torch
from torch import nn, Tensor
from typing import List, Tuple, Dict


class Permutation(nn.Module):
    """ A permutation layer, which permutes the channels of its input tensor.

    Parameters
    ----------
    in_channels: Number of channels to be permuted.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer('p', torch.randperm(in_channels))
        self.register_buffer('invp', torch.argsort(self.p))

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.in_channels
        return x[:, self.p]

    def backward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.in_channels
        return x[:, self.invp]


class MLP(nn.Module):
    """ Multilayer Perceptron (MLP) with n layers and fixed activation after each hidden
    layer, with linear output layer.

    Parameters
    ----------
    in_channels: Number of channels in input.
    out_channels: Number of output channels.
    n_layers: Number of layers including output layer.
    hidden_size: Size of each hidden layer.
    activation: Activation after each hidden layer.
    """

    def __init__(self,
        in_channels: int,
        out_channels: int,
        n_layers: int,
        hidden_size: int = 32,
        activation: nn.Module = nn.ReLU,
    ):
        assert n_layers >= 3
        super().__init__()

        layers = [nn.Linear(in_channels, hidden_size), activation()]  # first layer
        for _ in range(n_layers-2):  # intermediate layers
            layers.extend( [nn.Linear(hidden_size, hidden_size), activation()])
        layers.extend([nn.Linear(hidden_size, out_channels)])  # last layer

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class RealNVPLayer(nn.Module):
    """ Single layer of Real non-volume preserving (NVP) transform.
    Uses a multilayer perceptron (MLP) to transform half input channels into scaling and
    translation vectors to be used on remaining half.

    Parameters
    ----------
    in_channels: Number of channels of input.
    n_layers: Number of layers in MLP.
    activation: Activation of first (n_layers-1) layers (default ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        n_layers: int = 3,
        activation: nn.Module = nn.ReLU,
        ):
        super().__init__()
        assert in_channels%2 == 0, "Should have even dims"

        self.mlp = MLP(
            in_channels=in_channels//2,
            out_channels=in_channels,
            n_layers=n_layers,
            hidden_size=in_channels//2,
            activation=activation,
            )

    def forward(self, x: Tensor) -> Tensor:
        """Splits x into two halves (x0, x1), maps x0 through MLP to form s and t,
        then returns (s*x1+t, x0).
        """
        # split
        x0, x1 = torch.chunk(x, chunks=2, dim=-1)

        st = self.mlp(x0)

        # scale and translate
        s, t = torch.chunk(st, chunks=2, dim=-1)
        s =  .1 * torch.tanh(s)  # squash s
        transformed = x1 * torch.exp(s) + t
        y = torch.cat([transformed, x0], axis=-1)
        return y


class RealNVPBlock(nn.Module):
    """Real Non-volume preserving (NVP) block, consisting of n_layers layers.

    Parameters
    ----------
    in_channels: Number of channels of input.
    n_layers: Number of layers in each block.
    """

    def __init__(self, in_channels: int, n_layers: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.n_layers = n_layers
        self.nvp = nn.Sequential(
            *[RealNVPLayer(in_channels) for _ in range(n_layers)]
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.nvp(x)


class RealNVP(nn.Module):
    """ Real (non-volume preserving) NVP model (forward only).

    Parameters
    ----------
    in_channels: Size of NVP input. Half will remain untouched.
    n_blocks: Number of NVP blocks.
    n_layers: Number of layers within each NVP block.
    """

    def __init__(self, in_channels: int, n_blocks: int, n_layers: int):
        super().__init__()
        blocks = [RealNVPBlock(in_channels, n_layers)]

        for _ in range(n_blocks-1):
            blocks.extend([
                           Permutation(in_channels),
                           RealNVPBlock(in_channels, n_layers)
                           ]
            )

        self.nvp = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Run n_blocks of NVP"""
        return self.nvp(x)


class GINBlock(nn.Module):
    """ General incompressible-flow network block.

    Parameters
    ----------
    in_channels: Dimensionality of input.
    layers_per_block: Number of affine coupling GIN layers per block.
    layers_per_gin: Number of MLP layers in affine coupling transform.
    hidden_size: Number of hidden units in each MLP layer of GIN affine coupling layer.
    activation: Activation function in each GIN MLP layer.
    """

    def __init__(
        self,
        in_channels: int,
        layers_per_block: int = 2,
        layers_per_gin: int = 3,
        hidden_size: int = 32,
        activation: nn.Module = nn.ReLU,
        ):
        super().__init__()
        assert in_channels % 2 == 0
        layers = [
                  GINLayer(
                      in_channels=in_channels,
                      n_layers=layers_per_gin,
                      hidden_size=hidden_size,
                      activation=activation,
                    )
                  for _ in range(layers_per_block)
                  ]
        self.gin = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        y = self.gin(x)
        return y


class GINLayer(nn.Module):
    """ Affine coupling General Incompessible-flow Network (GIN) layer.
    Same as RealNVP transform except determinant of Jacobian is set to 1. This is done
    by ensuring the last entry of s is the neg sum of all other entries of s.

    Parameters
    ----------
    in_channels: Number of input channels.
    n_layers: Number of layers of MLP.
    hidden_size: Number of units in each hidden layer of MLP.
    activation: Activation function for each hideen layer of MLP.
    """

    def __init__(
        self,
        in_channels: int,
        n_layers: int = 3,
        hidden_size: int = 32,
        activation: nn.Module = nn.ReLU,
        ):
        super().__init__()
        assert n_layers >= 3
        assert in_channels % 2 == 0
        split_size = in_channels // 2

        self.split_size = split_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.mlp = MLP(
            in_channels=split_size,
            out_channels=(2*split_size)-1,
            n_layers=n_layers,
            hidden_size=hidden_size,
            activation=activation
            )

    def forward(self, x: Tensor) -> Tensor:
        """Affine coupling transform while preserving volume."""
        x0, x1 = torch.chunk(x, 2, -1)
        st = self.mlp(x0) # scaling and translation
        n = st.shape[-1]

        s, t = st[..., :n//2], st[..., n//2:]
        s = .1*torch.tanh(s)  # squash
        s = torch.cat([s, -s.sum(axis=-1, keepdim=True)], -1)  # preserve volume

        transformed = (x1 * torch.exp(s)) + t
        y = torch.cat([transformed, x0], -1) # untransformed x0 comes later
        return y


class GINBlock(nn.Module):
    """ General incompressible-flow network block.

    Parameters
    ----------
    in_channels: Dimensionality of input.
    layers_per_block: Number of affine coupling GIN layers per block.
    layers_per_gin: Number of MLP layers in affine coupling transform.
    hidden_size: Number of hidden units in each MLP layer of GIN affine coupling layer.
    activation: Activation function in each GIN MLP layer.
    """

    def __init__(
        self,
        in_channels: int,
        layers_per_block: int = 2,
        layers_per_gin: int = 3,
        hidden_size: int = 32,
        activation: nn.Module = nn.ReLU,
        ):
        super().__init__()
        assert in_channels % 2 == 0
        layers = [
                  GINLayer(
                      in_channels=in_channels,
                      n_layers=layers_per_gin,
                      hidden_size=hidden_size,
                      activation=activation,
                    )
                  for _ in range(layers_per_block)
                  ]
        self.gin = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        y = self.gin(x)
        return y


class ZPriorDiscrete(nn.Module):
    """Learnable Lambda mean and var params for condition prior.
    Uses embedding matrices  (look-up tables). Assumes Normal-distributed latent z.

    Parameters
    ----------
    z_dim:  Dimensionality of latent variable.
    u_dim:  Number of unique classes/condition labels.
    """

    def __init__(self, u_dim: int, z_dim: int):
        super().__init__()
        self.z_dim = z_dim
        self.u_dim = u_dim

        self.embed_mean = nn.Embedding(u_dim, z_dim)
        self.embed_log_var = nn.Embedding(u_dim, z_dim)

    def forward(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns mean and logvar of each sample. """
        lam_mean = self.embed_mean(u)
        lam_log_var = self.embed_log_var(u)
        return lam_mean, lam_log_var


class ZPriorContinuous(nn.Module):
    """ Map continuous u using MLP to a mean and logvar.
    Uses a single MLP to output both the mean and logvar of z label prior.

    Parameters
    ----------
    u_dim: Dimensionality of condition/label u.
    z_dim: Dimensionality of latent z.
    n_layers: Number of layers in MLP.
    hidden_size: Number of units in each hidden layer.
    activation: Activation of each hiden layer defaults to nn.Tanh().
    """

    def __init__(
        self,
        u_dim: int,
        z_dim: int,
        n_layers: int = 3,
        hidden_size: int = 32,
        activation: nn.Module = nn.Tanh
        ):
        super().__init__()
        self.mlp = MLP(u_dim, 2*z_dim, n_layers, hidden_size, activation)

    def forward(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.mlp(u)
        lam_mean, lam_log_var = torch.chunk(z, 2, -1)
        return lam_mean, lam_log_var

class Encoder(nn.Module):
    """ Recognition model determining mean and log_var of q(z|x)
    Goes from high-dim x down to mean and logvar of low-dim latent z.
    # This is done using a single MLP with both mean and logvar as outputs.
    This is done using two MLPs with mean and logvar as outputs, respectively.

    Parameters
    ----------
    x_dim: Dimensionality of observed x.
    z_dim: Dimensionality of latent z.
    n_layers: Number of layers in MLP.
    hidden_size: Number of units per hidden layer.
    activation: Activation function for each hidden layer of MLP.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        n_layers: int = 4,
        hidden_size: int = 64,
        activation: nn.Module = nn.Tanh
        ):
        super().__init__()
        # self.mlp = MLP(x_dim, 2*z_dim, n_layers, hidden_size, activation)
        self.mlp_mu = MLP(x_dim, z_dim, n_layers, hidden_size, activation)
        self.mlp_sig = MLP(x_dim, z_dim, n_layers, hidden_size, activation)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Go from x down to mean and logvar of z"""
        # y = self.mlp(x)
        # mean, log_var = torch.chunk(y, 2, -1)
        mean, log_var = self.mlp_mu(x), self.mlp_sig(x)
        return mean, log_var


class Decoder(nn.Module):
    """ Maps low-dim z to high-dim x by pushing through MLP then series of GIN blocks.

    p_f(x|z) = p_eps(x-f(z)) => x = f(z) + eps where eps is indep. noise
    f(z) nonlinearly lifts from z_dim to x_dim, then uses GIN blocks to map to mean
    firing rate of neurons.

    Parameters
    ----------
    z_dim: Dimensionality of latent variable z.
    x_dim: Dimensionality of observed data (should be much larger than that of z).
    n_gin_blocks: Number of GIN blocks.
    layers_per_gin: Number of MLP layers per GIN layer.
    layers_per_block: Number of GIN layers per block.
    hidden_size: Num units in each hidden layer in both MLP and GIN.
    lift_mlp_layers: Number of layers in MLP before GIN.
    obs_model: Observation model for x, either "poisson" or "gaussian".
    activation: Activation used in both MLP and GIN blocks.
    """

    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        n_gin_blocks: int,
        layers_per_gin: int,
        layers_per_block: int,
        hidden_size: int,
        lift_mlp_layers: int = 3,
        obs_model: str = "poisson",
        activation: nn.Module = nn.ReLU,
        ):
        super().__init__()
        assert x_dim >= z_dim
        assert obs_model in ["poisson", "gaussian"]
        self.obs_model = obs_model

        hidden_size = max(hidden_size, x_dim//4)
        self.mlp = MLP(
            in_channels=z_dim,
            out_channels=x_dim-z_dim,  # fill ambient space w/ nonlinear transform of z
            n_layers=lift_mlp_layers,
            hidden_size=hidden_size,
            activation=activation,
            )

        gin_blocks = []
        for _ in range(n_gin_blocks):
            gin_blocks.extend([Permutation(x_dim),
              GINBlock(
                  in_channels=x_dim,
                  layers_per_block=layers_per_block,
                  layers_per_gin=layers_per_gin,
                  hidden_size=hidden_size,
                  activation=activation,
                  )])

        self.gin = nn.Sequential(*gin_blocks)

    def forward(self, z: Tensor) -> Tensor:
        """Lifts z up to the dimensionality of x via MLP, then pushes through a GIN.
        Passes through softplus if observation model is Poisson.
        """
        y = self.mlp(z)  # lift from z_dim to x_dim
        x = torch.cat([z, y], -1)
        x = self.gin(x)
        return nn.functional.softplus(x) if self.obs_model == "poisson" else x