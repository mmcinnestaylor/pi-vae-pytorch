from typing import Optional
import torch
from torch import nn


class ELBOLoss(nn.Module):
    """ 
    pi-VAE Loss function which maximizes the evidence lower bound (ELBO) of p(x | u)

    Parameters
    ----------
    - version (int, default=2) - version of the loss function
    - alpha (float, default=0.5) - Weights the contribution of the encoder KL loss and posterior KL loss to the total KL loss
    - observation_model (str, default=poisson) - poisson or gaussian
    - device (torch.device, default=None) - object representing the device on which a Tensor will be allocated 
    
    Inputs
    ------
    - x (Tensor) - observed x. Size([n_samples, x_dim])
    - posterior_firing_rate (Tensor) - decoded firing rates. Size([n_samples, x_dim])
    - posterior_mean (Tensor) - means from posterior q(z|x,u)~q(z|x)p(z|u). Size([n_samples. z_dim])
    - posterior_log_variance (Tensor) - log of variances from posterior q(z|x,u)~q(z|x)p(z|u). Size([n_samples, z_dim])
    - label_mean (Tensor) - means from label prior p(z|u). Size([n_samples, z_dim])
    - label_log_variance (Tensor) - log of variances from label prior p(z|u). Size([n_samples, z_dim])
    - encoder_mean (Tensor, default=None) - means from encoder p(z|x). Size([n_samples, z_dim])
    - encoder_log_variance (Tensor, default=None) - means from encoder p(z|x). Size([n_samples, z_dim])
    - observation_noise_model (nn.Module, default=None) - if gaussian observation model, set the observation noise level as different real numbers
    - kl_weight (float, default=1.0) - KL annealing weight term used to scale the KL divergence component of the loss 

    Returns
    -------
    - (Tensor) - The total loss. Size([])

    Notes
    -----
    observation_noise_model (nn.Module) - PiVAE uses an nn.Linear(1, x_dim, bias=False) Module

    min -log p(x|z) + E_q log(q(z))-log(p(z|u))
    cross entropy
    q (mean1, var1) p (mean2, var2)
    E_q log(q(z))-log(p(z|u)) = -0.5*(1-log(var2/var1) - (var1+(mean2-mean1)^2)/var2)
    E_q(z|x,u) log(q(z|x,u))-log(p(z|u)) = -0.5*(log(2*pi*var2) + (var1+(mean2-mean1)^2)/var2)
    p(z) = q(z|x) = N(f(x), g(x)) parametrized by nn
    """

    def __init__(
        self,
        version: int = 2,
        alpha: float = 0.5,
        observation_model: str = 'poisson',
        device: Optional[torch.device] = None
        ) -> None:
        super().__init__()

        if version == 1:
            self.version = version
        elif version == 2:
            self.version = version

            if alpha >= 0 and alpha <= 1:
                self.alpha = alpha
            else:
                raise ValueError(f"Alpha outside of allowed interval [0, 1].")
        else:
            raise ValueError(f"Invalid loss version.")
        
        if observation_model == 'gaussian':
           self.ones = torch.ones(size=(1, 1), device=device)
        elif observation_model != 'poisson' and observation_model != 'gaussian':
            raise ValueError(f"Invalid observation_model: {observation_model}")
        
        self.observation_model = observation_model

    def forward(
        self,
        x: torch.Tensor,
        posterior_firing_rate: torch.Tensor,
        posterior_mean: torch.Tensor, 
        posterior_log_variance: torch.Tensor,
        label_mean: torch.Tensor, 
        label_log_variance: torch.Tensor,
        encoder_mean: Optional[torch.Tensor] = None,
        encoder_log_variance: Optional[torch.Tensor] = None,
        observation_noise_model: Optional[nn.Module] = None,
        kl_weight: float = 1.0
        ) -> torch.Tensor:

        if self.observation_model == 'poisson':
            observation_log_liklihood = torch.sum(posterior_firing_rate - x * torch.log(posterior_firing_rate), dim=-1)
        else:
            observation_log_variance = observation_noise_model(self.ones)
            observation_log_liklihood = torch.sum(torch.square(posterior_firing_rate - x) / (2 * torch.exp(observation_log_variance)) + (observation_log_variance / 2), dim=-1)

        if self.version == 1:
            kl_loss = 0.5 * self.compute_kl_loss(posterior_mean, posterior_log_variance, label_mean, label_log_variance)
        else:
            encoder_kl_loss = self.compute_kl_loss(encoder_mean, encoder_log_variance, label_mean, label_log_variance)
            posterior_kl_loss = self.compute_kl_loss(posterior_mean, posterior_log_variance, label_mean, label_log_variance)
            kl_loss = self.alpha * encoder_kl_loss + (1 - self.alpha) * posterior_kl_loss
        
        return torch.mean(observation_log_liklihood - kl_weight * kl_loss)
    
    @staticmethod
    def compute_kl_loss(
        mean_0: torch.Tensor,
        log_variance_0: torch.Tensor,
        mean_1: torch.Tensor,
        log_variance_1: torch.Tensor
        ) -> torch.Tensor:
        """
        Computes the Kullback–Leibler divergence between two distributions.

        Parameters
        ----------
        - mean_0 (Tensor) - predicted mean(s) of a distribution. Size([n_samples, sample_dim])
        - log_variance_0 (Tensor) - predicted log of variance(s) of a distribution. Size([n_samples, sample_dim])
        - mean_1 (Tensor) - predicted mean(s) of a distribution. Size([n_samples, sample_dim])
        - log_variance_1 (Tensor) - predicted log of variance(s) of a distribution. Size([n_samples, sample_dim])

        Returns
        -------
        - (Tensor) - the KL divergence loss. Size([])
        """

        kl_loss = 1 + log_variance_0 - log_variance_1 - ((torch.square(mean_0 - mean_1) + torch.exp(log_variance_0)) / torch.exp(log_variance_1))
        return 0.5 * torch.sum(kl_loss, dim=-1)
