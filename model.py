# Special thanks to the implementation provided by
# Lydon Duong @ https://www.lyndonduong.com/pivae/

import numpy as np
import torch
from torch import nn, Tensor
from typing import List, Tuple, Dict
from layers import *


class PIVAE(nn.Module):
    """ Poisson Identifiable Variational Auto-Encoder

    Uses label prior to inform and facilitate identifiability in the latent space of a
    VAE. More details can be found in original paper [0].

    Parameters
    ----------
    x_dim:
        Dimensionality of observed data.
    z_dim:
        Dimensionality of latent.
    u_dim:
        Dimensionality of condition/label (if discrete, then # unique conditions).
    encoder_n_layers:
        Number of MLP layers in Encoder mapping x to z.
    encoder_hidden_size:
        Number of units in hidden layer of Encoder.
    decoder_n_blocks:
        Number of GIN blocks in decoder.
    decoder_layers_per_gin_block:
        Number of affine coupling layers in each GIN block.
    decoder_layers_per_gin:
        Number of layers in each affine coupling layer.
    obs_model:
        Observation model ("poisson", "gaussian").
    discrete_prior:
        Whether or not label prior should be discrete (Default True).

    References
    ---------
    [0] Zhou, D., Wei, X. Learning identifiable and interpretable latent models of
        high-dimensional neural activity using pi-VAE. NeurIPS 2020.
        https://arxiv.org/abs/2011.04798
    """

    def __init__( self,
        x_dim: int,
        z_dim: int,
        u_dim: int,
        encoder_n_layers: int = 4,
        encoder_hidden_size: int = 64,
        decoder_n_blocks: int = 2,
        decoder_layers_per_gin_block: int = 3,
        decoder_layers_per_gin: int = 3,
        decoder_hidden_size: int = 32,
        obs_model: str = "poisson",
        discrete_prior: bool = True,
    ):
        super().__init__()

        self.obs_model = obs_model
        self.obs_noise_model  = nn.Linear(1, x_dim, bias=False)  # only for Gaussian

        # q(z|u)
        if discrete_prior:
            self.prior = ZPriorDiscrete(u_dim, z_dim)
        else:
            self.prior = ZPriorContinuous(u_dim, z_dim)

        # q(z|x)
        self.encoder = Encoder(
                x_dim,
                z_dim,
                encoder_n_layers,
                encoder_hidden_size,
                nn.Tanh
            )

        # p(x|z)
        self.decoder = Decoder(
                z_dim=z_dim,
                x_dim=x_dim,
                n_gin_blocks=decoder_n_blocks,
                layers_per_gin=decoder_layers_per_gin,
                layers_per_block=decoder_layers_per_gin_block,
                hidden_size=decoder_hidden_size,
                obs_model=obs_model
            )

    @staticmethod
    def sample(mean: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick, sample from factorial multivariate Gaussian
        q(z|x) = N(z; mu, diag(sqrt(var)))

        Parameters
        -----------
        mean: Means of each z sample, Size([n_samples, z_dim]).
        log_var: log vars of each z sample, Size([n_samples, z_dim]).

        Return
        ------
        s: Latent sample of z, Size([n_samples, z_dim]).
        """

        s = mean + torch.exp(.5 * log_var) * torch.randn_like(mean)
        return s

    @staticmethod
    def compute_posterior(
        z_mean: Tensor,
        z_log_var: Tensor,
        lam_mean: Tensor,
        lam_log_var: Tensor,
        ) -> Tensor:
        """ Computes approx posterior q(z|x,u) ~ q_{phi}(z|x) * p_{T, lambda}(z|u), as a product of Gaussians.
        p_{T, lambda}(z|u) is implemented as Gaussians instead of exponential family distribution.

        Parameters
        ----------
        z_mean: Means of encoded distribution q_{phi}(z|x), Size([n_samples, z_dim]).
        z_log_var: Log vars of the encoded distribution q_{phi}(z|x), Size([n_samples, z_dim]).
        lam_mean: Means of p_{T, lambda}(z|u), Size([n_samples, z_dim]).
        lam_log_var: Log vars of p_{T, lambda}(z|u), Size([n_samples, z_dim]).

        Returns
        -------
        post_mean: Approx posterior means, Size([n_samples, z_dim]).
        post_log_var: Approx posterior log vars, Size([n_samples, z_dim]).

        Notes
        -----
        Given identity covariance matrix:
            q(z) = q_{phi}(z|x) * p_{T, lambda}(z|u)
                = N(m1, var1) * N(mu2, var2)
                = N((mu1 * var2 + mu2 * var1)/(var1 + var2), var1 * var2 / (var1 + var2))
        """
        diff_log_var = z_log_var - lam_log_var
        post_mean = (z_mean/(1+torch.exp(diff_log_var))) + (lam_mean/(1+torch.exp(-diff_log_var)))
        post_log_var = z_log_var + lam_log_var - torch.log(torch.exp(z_log_var) + torch.exp(lam_log_var))

        return post_mean, post_log_var

    def loss(
        self,
        fr: Tensor,
        x: Tensor,
        post_mean: Tensor,
        post_log_var: Tensor,
        lam_mean: Tensor,
        lam_log_var: Tensor,
        ) -> Tensor:
        """ PI-VAE Loss function

        Minimize negloglik of data and KL between Gaussian latent posterior and
        Gaussian conditional label prior. LogLik will depend on observation noise model
        ("poisson" or "gaussian").

        Parameters
        ----------
        fr: Generated firing rates, Size([n_samples, x_dim]).
        x: Observed data, Size([n_samples, x_dim]).
        post_mean: Means from posterior q(z|x), Size([n_samples. z_dim]).
        post_log_var: Logvars from posterior q(z|x), Size([n_samples, z_dim]).
        lam_mean: Means from label prior q(z|u), Size([n_samples, z_dim]).
        lam_log_var: Logvars from label prior q(z|u), Size([n_samples, z_dim]).

        Notes
        -----
        Evidence lower bound:
            min -log p(x|z) + E_q log(q(z))-log(p(z|u))
            cross entropy
            q (mean1, var1) p (mean2, var2)
            E_q log(q(z))-log(p(z|u)) = -0.5*(1-log(var2/var1) - (var1+(mean2-mean1)^2)/var2)
            E_q(z|x,u) log(q(z|x,u))-log(p(z|u)) = -0.5*(log(2*pi*var2) + (var1+(mean2-mean1)^2)/var2)
            p(z) = q(z|x) = N(f(x), g(x)) parametrized by nn
        """

        if self.obs_model == "poisson":
            fr = torch.clamp(fr, min=1E-7, max=1E7)
            obs_log_lik = torch.sum(fr - x*torch.log(fr), -1)

        elif self.obs_model == "gaussian":
            # model noise
            obs_log_var = self.obs_noise_model(torch.ones((1, 1), device=device))
            E = torch.square(fr - x)/(2*torch.exp(obs_log_var))
            obs_log_lik = torch.sum(E + (obs_log_var/2), -1)

        diff_means2 = (post_mean-lam_mean)**2
        post_var, lam_var = torch.exp(post_log_var), torch.exp(lam_log_var)
        kl_loss = 1 + post_log_var - lam_log_var - ((diff_means2 + post_var) / lam_var)
        kl_loss = .5 * kl_loss.sum(-1)
        # kl_loss = torch.maximum(kl_loss, torch.ones(1)*.2)  # free bits

        total_loss = torch.mean(obs_log_lik - kl_loss)

        return total_loss

    def forward(
        self,
        x: Tensor,
        u: Tensor,
        valid: bool = False,
        ) -> Tuple[Tensor, Dict]:
        """ Run VAE encoder/decoder using condition label prior and compute loss.
        Parameters
        ----------
        x:
            Observed data, Size([n_samples, x_dim]).
        u:
            Condition label for each sample, Size([n_samples, ])
        valid:
            Validation mode to be used once training is complete. Outputs all
            distribution stats, z_sample, and generated fr in Dict. When False (default)
            Dict will be empty.

        Returns
        -------
        total_loss:
            Scalar loss negloglik+KL.
        out_dict:
            Dict output of posterior, label prior, and encoder means + logvars.
            Also includes z_sample and generated firing rate for each input sample.
        """

        # 1. Get means and logvars for each sample using label prior, p(z|u)
        lam_mean, lam_log_var = self.prior(u)

        # 2. Encode each data sample x to latent z w/ q(z|x), approximating distribution p(z|x)
        z_mean, z_log_var = self.encoder(x)

        # 3. For each sample, get approx posterior mean and logvar of q(z|x,u) ~ q(z|x)p(z|u)
        post_mean, post_log_var = self.compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)

        # 4. Use posterior stats to sample z using reparameterization trick
        z_sample = self.sample(post_mean, post_log_var)

        # 5. Decode (generate) firing rate using sampled z: p(x|z)
        fr = self.decoder(z_sample)

        # 6. Compute loss: negloglik(x, fr) + KL(posterior, label prior)
        total_loss = self.loss(fr, x, post_mean, post_log_var, lam_mean, lam_log_var)

        out_dict = {}
        if valid:
            out_dict = {
                "post_mean": post_mean.data,
                "post_log_var": post_log_var.data,
                "z_sample": z_sample.data,
                "firing_rate": fr.data,
                "lam_mean": lam_mean.data,
                "lam_log_var": lam_log_var.data,
                "z_mean": z_mean.data,
                "z_log_var": z_log_var.data,
            }

        return total_loss, out_dict