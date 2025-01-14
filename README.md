# Poisson Identifiable VAE (pi-VAE)

This is a Pytorch implementation of [Poisson Identifiable VAE (pi-VAE)](https://arxiv.org/abs/2011.04798), used to construct latent variable models of neural activity while simultaneously modeling the relation between the latent and task variables (non-neural variables, e.g. sensory, motor, and other externally observable states).

The original implementation by [Ding Zhou](https://zhd96.github.io/) and [Xue-Xin Wei](https://sites.google.com/view/xxweineuraltheory/) in Tensorflow 1.13 is available [here](https://github.com/zhd96/pi-vae).

Another Pytorch implementation by [Lyndon Duong](http://lyndonduong.com/) is available [here](https://github.com/lyndond/lyndond.github.io/blob/0865902edb4648a8690ed8d449573d9236a72406/code/2021-11-25-pivae.ipynb).  

A special thank you to [Zhongxuan Wu](https://github.com/ZhongxuanWu) who helped in the design and testing of this implementation. 

## Install

```
pip install pi-vae-pytorch
```

### From Source

For those interested in modifying and testing the codebase, using an editable `pip` installation is recommended:

```
# pi-vae-pytorch/

pip install -e .
```

## Model Architecture

pi-VAE is comprised of three main components: the encoder, the label prior estimator, and the decoder. 

### MLP Structure

The Multi Layer Perceptron (MLP) is the primary building block of the aforementioned components. Each MLP used in this implementation is configurable by specifying the appropriate parameters when `PiVAE` is initialized:  
- number of hidden layers  
- hidden layer dimension  
    - applied to all hidden layers within a given MLP
- hidden layer activation function  
    - applied to all non-output layer activations within a given MLP

### Encoder

The model's encoder is comprised of a single MLP, which learns the distribution q(z \| x). 

### Label Prior Estimator

The model's label prior estimator learns the distribution p(z \| u). In the discrete label regime this module is comprised of a single `nn.Embedding` layer, while in the continuous label regime the module is comprised of a MLP.

### Decoder

The model's decoder learns to map a latent sample `z` to the predicted firing rate of the latent sample. Inputs to the decoder are passed through the following submodules. 

#### NFlowLayer

This module is comprised of a MLP which maps `z` to the concatenation of `z` and `t(z)`.

#### GINBlock

Outputs from the `NFlowLayer` are passed to a series of `GINBlock` modules. Each `GINBlock` is comprised of a specified number of `AffineCouplingLayer` modules. Each `AffineCouplingLayer` is comprised of a MLP and performs an affine coupling transformation.

## Initialization

- `x_dim`: int  
    Dimension of observation `x`
- `u_dim`: int  
    Dimension of label `u`
- `z_dim`: int  
    Dimension of latent `z`
- `discrete_labels`: bool  
    - Default: `True`  

    Flag denoting `u`'s label type - `True`: discrete or `False`: continuous.
- `encoder_n_hidden_layers`: int  
    - Default: `2`  

    Number of hidden layers in the MLP of the model's encoder. 
- `encoder_hidden_layer_dim`: int  
    - Default: `120`  

    Dimensionality of each hidden layer in the MLP of the model's encoder. 
- `encoder_hidden_layer_activation`: nn.Module    
    - Default: `nn.Tanh`  

    Activation function applied to the outputs of each hidden layer in the MLP of the model's encoder. 
- `decoder_n_gin_blocks`: int  
    - Default: `2`  

    Number of GIN blocks used within the model's decoder. 
- `decoder_gin_block_depth`: int   
    - Default: `2`  

    Number of AffineCouplingLayers which comprise each GIN block.
- `decoder_affine_input_layer_slice_dim`: int  
    - Default None (corresponds to `x_dim // 2`)  

    Index at which to split an n-dimensional input x. 
- `decoder_affine_n_hidden_layers`: int  
    - Default: `2`  

    Number of hidden layers in the MLP of the model's encoder. 
- `decoder_affine_hidden_layer_dim`: int  
    - Default: `None` (corresponds to `x_dim // 4`)  

    Dimensionality of each hidden layer in the MLP of each AffineCouplingLayer. 
- `decoder_affine_hidden_layer_activation`: nn.Module  
    - Default: `nn.ReLU`  

    Activation function applied to the outputs of each hidden layer in the MLP of each AffineCouplingLayer. 
- `decoder_nflow_n_hidden_layers`: int  
    - Default: `2`  

    Number of hidden layers in the MLP of the decoder's NFlowLayer. 
- `decoder_nflow_hidden_layer_dim`: int  
    - Default: `None` (corresponds to `x_dim // 4`)  

    Dimensionality of each hidden layer in the MLP of the decoder's NFlowLayer. 
- `decoder_nflow_hidden_layer_activation`: nn.Module   
    - Default: `nn.ReLU`  

    Activation function applied to the outputs of each hidden layer in the MLP of the decoder's NFlowLayer. 
- `decoder_observation_model`: str  
    - Default: `poisson`  
    - One of `gaussian` or `poisson`

    Observation model used by the model's decoder. 
- `decoder_fr_clamp_min`: float  
    - Default: `1E-7`  
    - Only applied when `decoder_observation_model="poisson"`

    Mininimum threshold used when clamping decoded firing rates.
- `decoder_fr_clamp_max`: float  
    - Default: `1E7` 
    - Only applied when `decoder_observation_model="poisson"`

    Maximum threshold used when clamping decoded firing rates.
- `z_prior_n_hidden_layers`: int  
    - Default: `2`  
    - Only applied when `discrete_labels=False`  

    Number of hidden layers in the MLP of the label prior estimator module. 
- `z_prior_hidden_layer_dim`: int  
    - Default: `20`  
    - Only applied when `discrete_labels=False`

    Dimensionality of each hidden layer in the MLP of the label prior estimator module. 
- `z_prior_hidden_layer_activation`: nn.Module  
    - Default: `nn.Tanh`  
    - Only applied when `discrete_labels=False`

    Activation function applied to the outputs of each hidden layer in the MLP of the label prior estimator module. 

## Basic operation

```
import torch
from pi_vae_pytorch import PiVAE

model = PiVAE(
    x_dim = 100,
    u_dim = 3,
    z_dim = 2,
    discrete_labels=False
)

x = torch.randn(1, 100) # Size([n_samples, x_dim])

u = torch.randn(1, 3) # Size([n_samples, u_dim])

outputs = model(x, u) # Dict
```

### Output

A `Dict` with the following items. 

- `firing_rate`: Tensor   
    - Size([n_samples, x_dim])  

    Predicted firing rates of `z_sample`. 
- `lambda_mean`: Tensor  
    - Size([n_samples, z_dim])  

    Mean for each sample using label prior p(z \| u). 
- `lambda_log_variance`: Tensor  
    - Size([n_samples, z_dim])  
    
    Log of variance for each sample using label prior p(z \| u). 
- `posterior_mean`: Tensor  
    - Size([n_samples, z_dim])  

    Mean for each sample using full posterior of q(z \| x,u) ~ q(z \| x) &times; p(z \| u). 
- `posterior_log_variance`: Tensor  
    - Size([n_samples, z_dim])  

    Log of variance for each sample using full posterior of q(z \| x,u) ~ q(z \| x) &times; p(z \| u). 
- `z_mean`: Tensor  
    - Size([n_samples, z_dim])  

    Mean for each sample using approximation of q(z \| x). 
- `z_log_variance`: Tensor  
    - Size([n_samples, z_dim])  

    Log of variance for each sample using approximation of q(z \| x). 
- `z_sample`: Tensor  
    - Size([n_samples, z_dim])  
    
    Generated latents `z`. 

## Loss Function - `ELBOLoss`

pi-VAE learns the deep generative model and the approximate posterior q(z \| x, u) of the true posterior p(z \| x, u) by maximizing the evidence lower bound (ELBO) of p(x \| u). This loss function is implemented in the included `ELBOLoss` class.

### Parameters

- `observation_model`: str  
    - Default: `poisson`
    - One of `poisson` or `gaussian`  
    - Should use the same value passed to `decoder_observation_model` when initializing `PiVAE`.  

    The observation model used by pi-VAE's decoder.

- `device`: torch.device  
    - Default: None (uses the current device for the default tensor type)  
    - Only applied when `observation_model="gaussian"`  

    An object representing the device on which a `torch.Tensor` will be allocated. Should match the `device` on which the model resides.  

### Inputs

The loss computation requires the following inputs.

- `x`: Tensor  
    - Size([n_samples, x_dim])  

    Observations `x`.  
- `firing_rate`: Tensor 
    - Size([n_samples, x_dim])  

    Predicted firing rate of generated latent `z`. 
- `lambda_mean`: Tensor 
    - Size([n_samples, z_dim])  
    
    Means from label prior p(z \| u). 
- `lambda_log_variance`: Tensor 
    - Size([n_samples, z_dim])  
    
    Log of variances from label prior p(z \| u). 
- `posterior_mean`: Tensor 
    - Size([n_samples, z_dim])  
    
    Means from full posterior of q(z \| x,u) ~ q(z \| x) &times; p(z \| u). 
- `posterior_log_variance`: Tensor 
    - Size([n_samples, z_dim])  
    
    Log of variances from full posterior of q(z \| x,u) ~ q(z \| x) &times; p(z \| u).
- `observation_noise_model`: nn.Module 
    - Default: None  
    - Only required when `observation_model="gaussian"`  
    
    The noise model used when pi-VAE's decoder utilizes a Gaussian observation model. When `PiVAE` is initialized with `decoder_observation_model="gaussian"`, the model's `observation_noise_model` attribute should be used.

### Examples

#### Poisson observation model

```
from pi_vae_pytorch import ELBOLoss

loss_fn = ELBOLoss()

outputs = model(x, u) # Initialized with decoder_observation_model="poisson"

loss = loss_fn(
    x=x,
    firing_rate=outputs["firing_rate"],
    lambda_mean=outputs["lambda_mean"],
    lambda_log_variance=outputs["lambda_log_variance"],
    posterior_mean=outputs["posterior_mean"],
    posterior_log_variance=outputs["posterior_log_variance"]
)

loss.backward()
```

#### Gaussian observation model

```
from pi_vae_pytorch import ELBOLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device) # Initialized with decoder_observation_model="gaussian"

loss_fn = ELBOLoss(observation_model='gaussian', device=device)

outputs = model(x, u) 

loss = loss_fn(
    x=x,
    firing_rate=outputs["firing_rate"],
    lambda_mean=outputs["lambda_mean"],
    lambda_log_variance=outputs["lambda_log_variance"],
    posterior_mean=outputs["posterior_mean"],
    posterior_log_variance=outputs["posterior_log_variance"],
    observation_noise_model=model.observation_noise_model
)

loss.backward()
```

## Class methods

- `decode(X)`  
    Projects samples in the model's latent space (`z_dim`) into the model's observation space (`x_dim`) by passing them through the model's decoder module.  
    **Parameters**  
    - `X`: *Tensor of shape(n_samples, z_dim)*  
        Samples to be projected into the model's observation space.  
    
    **Returns**  
    - `decoded`: *Tensor of shape(n_samples, x_dim)*  
        Samples projected into the model's observation space.  
- `encode(X, return_stats=False)`  
    Projects samples in the model's observation space (`x_dim`) into the model's latent space (`z_dim`) by passing them through the model's encoder module.  
    **Parameters**  
    - `X`: *Tensor of shape(n_samples, x_dim)*  
        Samples to be projected into the model's observation space.  
    - `return_stats`: *bool, default=False*  
        If `True`, the mean and log of the variance associated with the encoded sample are returned; otherwise only the encoded sample is returned.  
    
    **Returns**  
    - `encoded`: *Tensor of shape(n_samples, z_dim)*  
        Samples projected into the model's latent space.  
    - `encoded_mean`: *Tensor of shape(n_samples, z_dim), optional*  
        Mean associated with a projected sample.  
    - `encoded_log_variance`: *Tensor of shape(n_samples, z_dim), optional*  
        Log of the variances associated with a projected sample.  
- `get_label_statistics(u, device=None)`  
    Returns the mean and log of the variance associated with a label `u` using the label prior estimator of p(z \| u).  
    **Parameters**  
    - `u`: *int, float, list, tuple, or Tensor of shape(1, u_dim)*  
        Label whose statictics will be returned. An integer is expected in the discrete label regime, while a float, list, tuple or Pytorch Tensor is expected in the continuous label regime.    
    - `device`: *torch.device, default=None*  
        Pytorch [device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) on which the model currently resides. A value of `None` may be used when utilizing the default device.  
    
    **Returns**  
    - `label_mean`: *Tensor of shape(1, z_dim)*  
        Mean of label `u`.  
    - `label_log_variance`: *Tensor of shape(1, z_dim)*  
        Log of the variance of label `u`.  
- `sample(u, n_samples=1, device=None)`  
    Generates random samples in the model's observation dimension (`x_dim`). Samples are initially drawn from a Gaussian distribution in the model's latent dimension (`z_dim`) corresponding to specified label `u`. Samples are subsequently lifted to the model's observation dimension (`x_dim`) by passing them through the model's decoder.  
    **Parameters**  
    - `u`: *int, float, list, tuple, or Tensor of shape(1, u_dim)*  
        Label of the samples to generate. An integer is expected in the discrete label regime, while a float, list, tuple or Pytorch Tensor is expected in the continuous label regime.  
    - `n_samples`: *int, default=1*  
        Number of samples to generate.  
    - `device`: *torch.device, default=None*  
        Pytorch [device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) on which the model currently resides. A value of `None` may be used when utilizing the default device.   
    
    **Returns**  
    - `samples`: *Tensor of shape(n_samples, x_dim)*  
        Randomly generated sample(s).  
- `sample_z(u, n_samples=1, device=None)`  
    Generates random samples in the model's latent dimension (`z_dim`). Samples are drawn from a Gaussian distribution corresponding to specified label `u`.  
    **Parameters**  
    - `u`: *int, float, list, tuple, or Tensor of shape(1, u_dim)*  
        Label of the samples to generate. An integer is expected in the discrete label regime, while a float, list, tuple or Pytorch Tensor is expected in the continuous label regime.  
    - `n_samples`: *int, default=1*  
        Number of samples to generate.  
    - `device`: *torch.device, default=None*  
        Pytorch [device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) on which the model currently resides. A value of `None` may be used when utilizing the default device.  
    
    **Returns**  
    - `samples`: *Tensor of shape(n_samples, z_dim)*  
        Randomly generated sample(s).  

## Citation

```
@misc{zhou2020learning,
    title={Learning identifiable and interpretable latent models of high-dimensional neural activity using pi-VAE}, 
    author={Ding Zhou and Xue-Xin Wei},
    year={2020},
    eprint={2011.04798},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```
