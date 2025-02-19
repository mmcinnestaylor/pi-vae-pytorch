from typing import Union, Optional
import torch
from torch import nn
from pi_vae_pytorch.decoders import GINFlowDecoder
from pi_vae_pytorch.encoders import MLPEncoder
from pi_vae_pytorch.label_prior import LabelPriorContinuous, LabelPriorDiscrete


class PiVAE(nn.Module):
    """
    The pi-VAE model which constructs a latent variable model of neural activity while 
    simultaneously modeling the relation between the latent and task variables.

    Parameters
    ----------
    - x_dim (int) - observed x dimension
    - u_dim (int) -  label u dimension
    - z_dim (int) - latent z dimension
    - discrete_labels (bool, default=True) - label u format discrete(True) or continuous(False)
    - encoder_n_hidden_layers (int, default=2) - number of the MLPEncoder's MLP hidden layers
    - encoder_hidden_layer_dim (int, default=128) - dimension of the MLPEncoder's MLP hidden layers
    - encoder_hidden_layer_activation (nn.Module, default=nn.Tanh) - activation function applied to the MLPEncoder's MLP hidden layers
    - decoder_n_gin_blocks (int, default=2) - number of GINBlocks in the GINFlowDecoder
    - decoder_gin_block_depth (int, default=2) - depth of each GINBlock in the GINFlowDecoder
    - decoder_affine_input_layer_slice_dim (int, default=None) - index at which to split an n-dimensional sample x input to each AffineCouplingLayer
    - decoder_affine_n_hidden_layers (int, default=2) - number of each AffineCouplingLayer's MLP hidden layers
    - decoder_affine_hidden_layer_dim (int, default=None) - dimension of each AffineCouplingLayer's MLP hidden layers
    - decoder_affine_hidden_layer_activation (nn.Module, default=nn.ReLU) - activation function applied to each AffineCouplingLayer's MLP hidden layers
    - decoder_nflow_n_hidden_layers (int, default=2) - number of the NFlowLayer's MLP hidden layers
    - decoder_nflow_hidden_layer_dim (int, default=None) - dimension of the NFlowLayer's MLP hidden layers
    - decoder_nflow_hidden_layer_activation (nn.Module, default=nn.ReLU) - activation function applied to the NFlowLayer's MLP hidden layers
    - decoder_observation_model (str, default=poisson) - GINFlowDecoder's observation model poisson/gaussian
    - decoder_fr_clamp_min (float, default=1E-7) - min value used when clamping decoded firing rates
    - decoder_fr_clamp_max (float, default=1E7) - max value used when clamping decoded firing rates
    - label_prior_n_hidden_layers (int, default=2) - number of the LabelPriorContinuous's MLP hidden layers
    - label_prior_hidden_layer_dim (int, default=32) - dimension of the LabelPriorContinuous's MLP hidden layers
    - label_prior_hidden_layer_activation (nn.Module, default=nn.Tanh) - activation function applied to the LabelPriorContinuous's MLP hidden layers

    Notes
    -----
    - decoder_affine_input_layer_slice_dim (int) - when None, x_dim // 2 is assigned. Otherwise assigns the specified value, assuming 0 < decoder_affine_input_layer_slice_dim < x_dim.
    - decoder_affine_hidden_layer_dim (int) - when None, x_dim // 4 is assigned. Otherwise max(decoder_affine_hidden_layer_dim, x_dim // 4).
    - decoder_nflow_hidden_layer_dim (int) - when None, x_dim // 4 is assigned. Otherwise max(decoder_nflow_hidden_layer_dim, x_dim // 4).
    """
    
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        z_dim: int,
        discrete_labels: bool = True,
        encoder_n_hidden_layers: int = 2,
        encoder_hidden_layer_dim: int = 128,
        encoder_hidden_layer_activation: nn.Module = nn.Tanh,
        decoder_n_gin_blocks: int = 2,
        decoder_gin_block_depth: int = 2,
        decoder_affine_input_layer_slice_dim: Optional[int] = None,
        decoder_affine_n_hidden_layers: int = 2,
        decoder_affine_hidden_layer_dim: Optional[int] = None,
        decoder_affine_hidden_layer_activation: nn.Module = nn.ReLU,
        decoder_nflow_n_hidden_layers: int = 2,
        decoder_nflow_hidden_layer_dim: Optional[int] = None,
        decoder_nflow_hidden_layer_activation: nn.Module = nn.ReLU,
        decoder_observation_model: str = 'poisson',
        decoder_fr_clamp_min: float = 1E-7,
        decoder_fr_clamp_max: float = 1E7,
        label_prior_n_hidden_layers: int = 2,
        label_prior_hidden_layer_dim: int = 32,
        label_prior_hidden_layer_activation: nn.Module = nn.Tanh
        ) -> None:
        super().__init__()

        if decoder_observation_model == 'gaussian':
            self.observation_noise_model = nn.Linear(
                in_features=1,
                out_features=x_dim, 
                bias=False
            )
        elif decoder_observation_model == 'poisson':
            self.observation_noise_model = None
        else:
            raise ValueError(f"Invalid observation model: {decoder_observation_model}")
        
        self.decoder_observation_model = decoder_observation_model
        
        self.decoder = GINFlowDecoder(
            x_dim=x_dim,
            z_dim=z_dim,
            n_gin_blocks=decoder_n_gin_blocks,
            gin_block_depth=decoder_gin_block_depth,
            affine_input_layer_slice_dim=decoder_affine_input_layer_slice_dim,
            affine_n_hidden_layers=decoder_affine_n_hidden_layers,
            affine_hidden_layer_dim=decoder_affine_hidden_layer_dim,
            affine_hidden_layer_activation=decoder_affine_hidden_layer_activation,
            nflow_n_hidden_layers=decoder_nflow_n_hidden_layers,
            nflow_hidden_layer_dim=decoder_nflow_hidden_layer_dim,
            nflow_hidden_layer_activation=decoder_nflow_hidden_layer_activation,
            observation_model=decoder_observation_model
        )

        if decoder_fr_clamp_min < decoder_fr_clamp_max:
            self.decoder_fr_clamp_min = decoder_fr_clamp_min
            self.decoder_fr_clamp_max = decoder_fr_clamp_max
        else:
            raise ValueError(f"decoder_fr_clamp_min: {decoder_fr_clamp_min} must be less than decoder_fr_clamp_max: {decoder_fr_clamp_max}")
        
        
        self.encoder = MLPEncoder(
            x_dim=x_dim,
            z_dim=z_dim,
            n_hidden_layers=encoder_n_hidden_layers,
            hidden_layer_dim=encoder_hidden_layer_dim,
            activation=encoder_hidden_layer_activation
        )
        
        if discrete_labels:
            self.label_prior = LabelPriorDiscrete(
                u_dim=u_dim,
                z_dim=z_dim
            )
        else:
            self.label_prior = LabelPriorContinuous(
                u_dim=u_dim,
                z_dim=z_dim,
                n_hidden_layers=label_prior_n_hidden_layers,
                hidden_layer_dim=label_prior_hidden_layer_dim,
                hidden_layer_activation=label_prior_hidden_layer_activation
            )

        self.inference = False

    def forward(
        self,
        x: torch.Tensor,
        u: Optional[torch.Tensor] = None
        ) -> dict[str, torch.Tensor]:

        # Encode each sample observation x to latent z approximating q(z|x)
        encoder_mean, encoder_log_variance = self.encoder(x)

        with torch.no_grad():
            # Sample latent z using reparameterization trick
            encoder_z_sample = self.reparameterization_trick(mean=encoder_mean, log_variance=encoder_log_variance)

            # Generate firing rate using sampled latent z
            encoder_firing_rate = self.decoder(encoder_z_sample)

            if self.decoder_observation_model == 'poisson':
                encoder_firing_rate = torch.clamp(encoder_firing_rate, min=self.decoder_fr_clamp_min, max=self.decoder_fr_clamp_max)
        
        if self.inference:
            return {
                'encoder_firing_rate': encoder_firing_rate,
                'encoder_z_sample': encoder_z_sample,
                'encoder_mean': encoder_mean,
                'encoder_log_variance': encoder_log_variance                
            }
        else:
            # Mean and log of variance for each sample using label prior p(z|u)
            label_mean, label_log_variance = self.label_prior(u)

            # Compute the full posterior of q(z|x,u)~q(z|x)p(z|u) as a product of Gaussians
            posterior_mean, posterior_log_variance = self.compute_posterior(encoder_mean, encoder_log_variance, label_mean, label_log_variance)  

            # Sample latent z using reparameterization trick
            posterior_z_sample = self.reparameterization_trick(mean=posterior_mean, log_variance=posterior_log_variance)

            # Generate firing rate using sampled latent z
            posterior_firing_rate = self.decoder(posterior_z_sample)
            if self.decoder_observation_model == 'poisson':
                posterior_firing_rate = torch.clamp(posterior_firing_rate, min=self.decoder_fr_clamp_min, max=self.decoder_fr_clamp_max)

            return {
                'encoder_firing_rate': encoder_firing_rate,
                'encoder_z_sample': encoder_z_sample,
                'encoder_mean':   encoder_mean,
                'encoder_log_variance': encoder_log_variance,
                'label_mean': label_mean,
                'label_log_variance': label_log_variance,
                'posterior_firing_rate': posterior_firing_rate,
                'posterior_z_sample': posterior_z_sample,
                'posterior_mean': posterior_mean,
                'posterior_log_variance': posterior_log_variance
            }

    def decode(
        self,
        z: torch.Tensor
        ) -> torch.Tensor:
        """
        Projects samples in the model's latent space (`z_dim`) into the model's observation space (`x_dim`) by passing them through the model's decoder module.

        Parameters
        ----------
        - `z` (Tensor) - sample(s) the model's latent space (`z_dim`) to be decoded. `Size([n_samples, z_dim])`

        Returns
        -------
        - `decoded` (Tensor) - sample(s) in the model's observation space (`x_dim`). `Size([n_samples, x_dim])` 
        """

        with torch.no_grad():
            decoded = self.decoder(z)

            if self.decoder_observation_model == 'poisson':
                decoded = torch.clamp(decoded, min=self.decoder_fr_clamp_min, max=self.decoder_fr_clamp_max)

        return decoded

    def encode(
        self,
        x: torch.Tensor,
        return_stats: bool = False
        ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Projects samples in the model's observation space (`x_dim`) into the model's latent space (`z_dim`) by passing them through the model's encoder module.

        Parameters
        ----------
        - `x` (Tensor) - the sample(s) in the model's observation space (`x_dim`) to be encoded. `Size([n_samples, x_dim])`
        - `return_stats` (bool, default=False) - if `True`, the mean and log of variance associated with the encoded sample are returned; otherwise only the encoded sample is returned.

        Returns
        -------
        - `encoded` (Tensor) - sample(s) in the model's latent space(`z_dim`). `Size([n_samples, z_dim])`
        - `encoded_mean` (Tensor) [optional] - mean(s) of the encoded sample(s). `Size([n_samples, z_dim])`
        - `encoded_log_variance` (Tensor) [optional] - log of variance(s) of the encoded sample(s). `Size([n_samples, z_dim])`
        """

        with torch.no_grad():
            encoded_mean, encoded_log_variance = self.encoder(x)
            # Sample latent z using reparameterization trick
            encoded = self.reparameterization_trick(mean=encoded_mean, log_variance=encoded_log_variance)

        if return_stats:
            return encoded, encoded_mean, encoded_log_variance
        else:
            return encoded

    def get_label_statistics(
        self,
        u: Union[float, int, list, tuple, torch.Tensor],
        device: Optional[torch.device] = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and log of the variance associated with label `u`.

        Parameters
        ----------
        - `u` (int, float, list, tuple, or Tensor) - the label of the generated samples
        - `device` (torch.device, default=None) - the torch device on which the model resides

        Returns
        -------
        - `label_mean` (Tensor) - mean associated with the specified label. `Size([1, z_dim])`
        - `label_log_variance` (Tensor) - log of the variance associated with  the specified label. `Size([1, z_dim])`
        """

        with torch.no_grad():
            if isinstance(u, int): # discrete label
                u = torch.as_tensor([u], device=device)
            elif isinstance(u, float): # continuous label
                u = torch.as_tensor([u], device=device).unsqueeze(dim=0)
            elif isinstance(u, list) or isinstance(u, tuple): # continuous label
                u = torch.as_tensor(u, dtype=torch.float, device=device).unsqueeze(dim=0)

            # Mean and log of variance of label u using label prior estimator of p(z|u)
            label_mean, label_log_variance = self.label_prior(u) 
        
        return label_mean, label_log_variance
    
    def predict_labels(
        self,
        x: torch.Tensor,
        n_samples: int = 1000,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Returns the label probabilities associated with observation(s) `x`.

        Parameters
        ----------
        - `x` (Tensor) - data in the model's observation space (`x_dim`). `Size([n_observations, x_dim])`
        - `n_samples` (int, default=1000) - number of samples drawn per label during Monte Carlo estimation
        - `device` (torch.device, default=None) - the torch device on which the model resides

        Returns
        -------
        - `predictions` (Tensor) - the label probabilities of the observation(s). `Size([n_observations, u_dim])`
        """

        with torch.no_grad():
            n_labels = self.label_prior.embedded_mean.weight.shape[0]
            label_means, label_stds = None, None

            z = self.encode(x)
            _, latent_dim = z.shape

            for u in range(n_labels):
                mean, log_variance = self.get_label_statistics(u=u, device=device)

                if label_means is None:
                    label_means, label_stds = mean, torch.exp(0.5 * log_variance)
                else:
                    label_means = torch.cat((label_means, mean), dim=0)
                    label_stds = torch.cat((label_stds, torch.exp(0.5 * log_variance)), dim=0)

            # Sample from p(z|u) for each class
            z_samples = label_means.unsqueeze(dim=0) + label_stds.unsqueeze(dim=0) * torch.randn(n_samples, n_labels, latent_dim, device=device)

            # Compute distances between sampled z and actual z
            z = z.unsqueeze(1).unsqueeze(dim=0)  # Size([1, batch_size, 1, latent_dim])
            distances = torch.square(z - z_samples.unsqueeze(1)).sum(dim=-1)  # Size([n_samples, batch_size, n_labels])

            # Estimate p(z|u) using Monte Carlo (negative distances as proxy for likelihood)
            proxy_log_liklihood = torch.neg(distances.mean(dim=0))  # Average over samples, Size([batch_size, n_labels])
            
            # Normalize with softmax to approximate p(u|z)
            return torch.nn.functional.softmax(proxy_log_liklihood, dim=1)

    def sample(
        self,
        u: Union[float, int, list, tuple, torch.Tensor],
        n_samples: int = 1,
        return_z: bool = False,
        device: Optional[torch.device] = None
        ) -> torch.Tensor:
        """
        Generates samples in the model's x dimension using a specified label u.

        Parameters
        ----------
        - u (int, float, list, tuple, or Tensor) - the label of the generated samples
        - n_samples (int, default=1) - the number of samples to generate
        - return_z (bool, default=False) - if True the latent space samples are also returned
        - device (torch.device, default=None) - the torch device on which the model resides

        Returns
        -------
        - samples (Tensor) - the projected latent space samples corresponding to the specified label. Size([n_samples, x_dim])
        - z_samples (Tensor) [optional] - the generated latent space samples corresponding to the specified label. Size([n_samples, z_dim]) 
        """
        with torch.no_grad():
            if isinstance(u, int): # discrete label
                u = torch.as_tensor([u], device=device)
            elif isinstance(u, float): # continuous label
                u = torch.as_tensor([u], device=device).unsqueeze(dim=0)
            elif isinstance(u, list) or isinstance(u, tuple): # continuous label
                u = torch.as_tensor(u, dtype=torch.float, device=device).unsqueeze(dim=0)
            
            # Mean and log of variance of label u
            mean, log_variance = self.label_prior(u)

            # Create covariance matrix
            variance = torch.exp(log_variance).squeeze(dim=0)
            covariance = torch.diag_embed(variance)

            # Create distribution used for sampling
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)
            
            # Generate z samples
            z_samples = distribution.sample((n_samples,))
            z_samples = z_samples.squeeze(dim=1)

            # Lift samples to x dimension
            samples = self.decoder(z_samples)

        if return_z:
            return samples, z_samples
        else:
            return samples

    def sample_z(
        self,
        u: Union[float, int, list, tuple, torch.Tensor],
        n_samples: int = 1,
        device: Optional[torch.device] = None
        ) -> torch.Tensor:
        """
        Generates samples in the model's z dimension using a specified label u.

        Parameters
        ----------
        - u (int, float, list, tuple, or Tensor) - the label of the generated samples
        - n_samples (int, default=1) - the number of samples to generate
        - device (torch.device, default=None) - the torch device on which the model resides

        Returns
        -------
        - samples (Tensor) - the generated samples corresponding to the specified label. Size([n_samples, z_dim]) 
        """

        with torch.no_grad():
            if isinstance(u, int): # discrete label
                u = torch.as_tensor([u], device=device)
            elif isinstance(u, float): # continuous label
                u = torch.as_tensor([u], device=device).unsqueeze(dim=0)
            elif isinstance(u, list) or isinstance(u, tuple): # continuous label
                u = torch.as_tensor(u, dtype=torch.float, device=device).unsqueeze(dim=0)

            # Mean and log of variance of label u
            mean, log_variance = self.label_prior(u)

            # Create covariance matrix
            variance = torch.exp(log_variance).squeeze(dim=0)
            covariance = torch.diag_embed(variance)

            # Create distribution used for sampling
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)
                
            # Generate z samples
            samples = distribution.sample((n_samples,))
            samples = samples.squeeze(dim=1)                          

        return samples
    
    def set_inference_mode(
        self,
        state: bool
        ) -> None:
        """
        Toggles the model's inference state flag. When `True`, the model's `forward` method does not utilize the `u` parameter. When `False`,  the `u` parameter is utilized.

        Parameters
        ----------
        - `state` (bool) - the desired inference state

        Returns
        -------
        - `None`
        """

        self.inference = state

    @staticmethod
    def compute_posterior(
    mean_0: torch.Tensor, 
    log_variance_0: torch.Tensor, 
    mean_1: torch.Tensor, 
    log_variance_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the posterior of two distributions as a product of Gaussians.

        Parameters
        ----------
        - mean_0 (Tensor) - mean of a distribution. Size([n_samples, sample_dim])
        - log_variance_0 (Tensor) - log of variance of a distribution. Size([n_samples, sample_dim])
        - mean_1 (Tensor) - mean of a distribution. Size([n_samples, sample_dim])
        - log_variance_1 (Tensor) - log of variance of a distribution. Size([n_samples, sample_dim])

        Returns
        -------
        - posterior_mean: mean of the posterior. Size([n_samples, sample_dim])
        - posterior_log_variance: log of variances of the posterior. Size([n_samples, sample_dim])

        Notes
        -----
        - The sample_dim should be equivalent amongst all four parameters. 
        """

        variance_difference = log_variance_0 - log_variance_1
        posterior_mean = (mean_0 / (1 + torch.exp(variance_difference))) + (mean_1 / (1 + torch.exp(torch.neg(variance_difference))))
        posterior_log_variance = log_variance_0 + log_variance_1 - torch.log(torch.exp(log_variance_0) + torch.exp(log_variance_1))

        return posterior_mean, posterior_log_variance
    
    @staticmethod
    def reparameterization_trick(
    mean: torch.Tensor, 
    log_variance: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples from an isotropic unit Gaussian distribution using the reparameterization trick.

        Parameters
        ----------
        - mean (Tensor) - mean of a distribution. Size([n_samples, sample_dim])
        - log_variance (Tensor) - log of variance of a distribution. Size([n_samples, sample_dim])

        Returns
        -------
        - samples: a sample from the distribution. Size([n_samples, sample_dim])
        """

        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
