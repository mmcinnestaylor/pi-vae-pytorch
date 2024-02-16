import numpy as np
import torch
from torch import nn, Tensor
from typing import List, Tuple, Dict
from layers import *


def lift(
    x: Tensor,
    out_channels: int,
    nvp_blocks: int,
    nvp_layers: int
    ) -> Tensor:
    """ Nonlinearly transform x from lo -> hi dim w/ random-initialized MLP and realNVP.
    MLP lifts from in_channels to out_channels, then realNVP further transforms the
    data.

    Parameters
    ----------
    x: Input Tensor with Size([n, in_channels])
    out_channels: Dimensionality of output.
    nvp_blocks: Number of NVP blocks in NVP model.
    nvp_layers: Number of layers within each NVP block.

    Returns
    -------
    y: Output Tensor with size Size([n, out_channels]).
    """

    in_channels = x.shape[-1]
    mlp = MLP(in_channels, out_channels-in_channels, n_layers=3, activation=nn.ReLU).requires_grad_(False)
    realnvp_model = RealNVP(out_channels, nvp_blocks, nvp_layers).requires_grad_(False)

    # fill ambient space
    x_append = mlp(x)
    y = torch.cat([x, x_append], -1)
    y = realnvp_model(y)
    return y


def simulate_data_discrete(
    n_samples: int,
    n_cls: int,
    n_dim: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """ Create n-dimensional data with 2D intrinsic dimensionality.

    Parameters
    ----------
    n_samples: Number of samples.
    n_cls: Number of discrete classes.
    n_dim: Dimensionality of data.

    Returns
    -------
    x_true: Data to be used, Size([n_samples, n_dim]).
    z_true: True 2D data, Size([n_samples, 2]).
    u_true: Condition labels with n_cls unique classes, Size([n_samples, ]).
    lam_true: Poisson rate parameter of each sample, Size([n_samples, n_dim]).
    """

    mu_true = torch.empty((2, n_cls)).uniform_(-5, 5)
    var_true = torch.empty((2, n_cls)).uniform_(.5, 3)
    u_true = torch.tile(torch.arange(n_cls), (n_samples//n_cls, ))

    z0 = torch.normal(mu_true[0][u_true], np.sqrt(var_true[0][u_true]))
    z1 = torch.normal(mu_true[1][u_true], np.sqrt(var_true[1][u_true]))
    z_true = torch.stack([z0, z1], -1)

    ## Nonlinearly lift from 2D up to n_dims using RealNVP
    mean_true = lift(z_true, n_dim, nvp_blocks=4, nvp_layers=2)
    lam_true = torch.exp(2*torch.tanh(mean_true))  # Poisson rate param
    x_true = torch.poisson(lam_true)
    return x_true, z_true, u_true, lam_true


def simulate_data_continuous(n_samples: int, n_dim: int) -> torch.Tensor:
    """TODO: docstring. Haven't validated yet."""
    ## true 2D latent

    u_true = torch.empty((n_samples, )).uniform_(0, 2*np.pi)
    mu_true = torch.stack([u_true, 2*torch.sin(u_true)], -1)
    var_true = .15 * torch.abs(mu_true)
    var_true[:,0] = .6 - var_true[:,1]

    z_true = torch.randn((n_samples, 2)) * torch.sqrt(var_true) + mu_true

    mean_true = lift(z_true, n_dim, nvp_blocks=4, nvp_layers=2)
    lam_true = torch.exp(2.2 * torch.tanh(mean_true))
    x_true = torch.poisson(lam_true)

    return x_true, z_true, u_true, lam_true