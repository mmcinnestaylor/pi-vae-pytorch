# Poisson Identifiable VAE (pi-VAE) 2.0
This is a Pytorch implementation of [Poisson Identifiable Variational Autoencoder (pi-VAE)](https://arxiv.org/abs/2011.04798), used to construct latent variable models of neural activity while simultaneously modeling the relation between the latent and task variables (non-neural variables, e.g. sensory, motor, and other externally observable states).  

A special thank you to [Zhongxuan Wu](https://github.com/ZhongxuanWu) who helped in the design and testing of this implementation.  

#### Code status

![GitHub Actions Test Status](https://img.shields.io/github/actions/workflow/status/mmcinnestaylor/pi-vae-pytorch/test-package.yml?logo=github&label=Tests)
![GitHub Actions Publish Status](https://img.shields.io/github/actions/workflow/status/mmcinnestaylor/pi-vae-pytorch/publish-to-pypi.yml?logo=github&logoColor=white&label=Publish%20to%20PyPI)
![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/pypi/l/pi-vae-pytorch)  

#### PyPI status

![PyPI - Version](https://img.shields.io/pypi/v/pi-vae-pytorch?label=package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pi-vae-pytorch?label=downloads)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/pi-vae-pytorch)

#### conda-forge status

![Conda - Version](https://img.shields.io/conda/vn/conda-forge/pi-vae-pytorch?label=package)
![Conda - Downloads](https://img.shields.io/conda/d/conda-forge/pi-vae-pytorch?label=downloads)
![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/pi-vae-pytorch)

### Model Versions

pi-VAE 1.0 and 2.0 differ solely in their loss function, specifically how the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) component of the loss is computed. Additional information is available in the *Loss Function - ELBOLoss* section of this documentation.

#### Version 2.0

- This codebase is the only known publically available implementation, and also includes an implementation of the Version 1.0 loss function.  

#### Version 1.0

- The original implementation by [Ding Zhou](https://zhd96.github.io/) and [Xue-Xin Wei](https://sites.google.com/view/xxweineuraltheory/) in Tensorflow 1.13 is available [here](https://github.com/zhd96/pi-vae).

- Another Pytorch implementation by [Lyndon Duong](http://lyndonduong.com/) is available [here](https://github.com/lyndond/lyndond.github.io/blob/0865902edb4648a8690ed8d449573d9236a72406/code/2021-11-25-pivae.ipynb).  



## Installation

It is possible to install this project using `pip`:
```
pip install pi-vae-pytorch
```

or `conda`, using the `conda-forge` channel:
```
conda install -c conda-forge pi-vae-pytorch
```

It is also possible to clone this repo and install it using `pip`: 
```
git clone https://github.com/mmcinnestaylor/pi-vae-pytorch.git
cd pi-vae-pytorch
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

The model's encoder is comprised of a single MLP, which learns to approximate the distribution q(z \| x). 

### Label Prior Estimator

The model's label prior estimator learns to approximate the distribution p(z \| u). In the discrete label regime this module is comprised of two `nn.Embedding` submodules, while in the continuous label regime the module is comprised of a single MLP.

### Decoder

The model's decoder learns to map a latent sample `z` to its predicted firing rate in the model's observation space. Inputs to the decoder are passed through the following submodules:  

- **NFlowLayer**  
This module is comprised of a MLP which maps `z` to the concatenation of `z` and `t(z)`.  

- **GINBlock(s)**  
Outputs from the `NFlowLayer` are passed to a series of `GINBlock` modules. Each `GINBlock` is comprised of a `PermutationLayer` and a specified number of `AffineCouplingLayer` modules. Each `AffineCouplingLayer` is comprised of a MLP and performs an affine coupling transformation.

## Initialization

```
pi_vae_pytorch.PiVAE(
    x_dim,
    u_dim,
    z_dim,
    discrete_labels=True,
    encoder_n_hidden_layers=2,
    encoder_hidden_layer_dim=128,
    encoder_hidden_layer_activation=nn.Tanh,
    decoder_n_gin_blocks=2,
    decoder_gin_block_depth=2,
    decoder_affine_input_layer_slice_dim=None,
    decoder_affine_n_hidden_layers=2,
    decoder_affine_hidden_layer_dim=None,
    decoder_affine_hidden_layer_activation=nn.ReLU,
    decoder_nflow_n_hidden_layers=2,
    decoder_nflow_hidden_layer_dim=None,
    decoder_nflow_hidden_layer_activation=nn.ReLU,
    decoder_observation_model='poisson',
    decoder_fr_clamp_min=1E-7,
    decoder_fr_clamp_max=1E7,
    label_prior_n_hidden_layers=2,
    label_prior_hidden_layer_dim=32,
    label_prior_hidden_layer_activation=nn.Tanh)
```

- **x_dim:** *int*  
    Dimension of observation `x`  

- **u_dim:** *int*  
    Dimension of observation labels `u`. In the discrete regime, this corresponds to the number of unique classes/labels. In the continuous regime, this corresponds to the dimension of each label.  

- **z_dim:** *int*  
    Dimension of latent `z`  

- **discrete_labels:** *bool, default=*`True`  
    - `True`: discrete or `False`: continuous  

    Flag denoting the observation's label regime. 
- **encoder_n_hidden_layers:** *int, default=*`2`  
    Number of hidden layers in the MLP of the model's encoder.  

- **encoder_hidden_layer_dim:** *int, default=*`128`  
    Dimensionality of each hidden layer in the MLP of the model's encoder.  

- **encoder_hidden_layer_activation:** *nn.Module, default=*`nn.Tanh`  
    Activation function applied to the outputs of each hidden layer in the MLP of the model's encoder.  

- **decoder_n_gin_blocks:** *int, default=*`2`  
    Number of GIN blocks used within the model's decoder.  

- **decoder_gin_block_depth:** *int, default=*`2`  
    Number of AffineCouplingLayers which comprise each GIN block.  

- **decoder_affine_input_layer_slice_dim:** *int, default=*`None` *(equivalent to* `x_dim // 2`*)*  
    Index at which to split an n-dimensional input x.  

- **decoder_affine_n_hidden_layers:** *int, default=*`2`  
    Number of hidden layers in the MLP of each AffineCouplingLayer.  

- **decoder_affine_hidden_layer_dim:** *int, default=*`None` *(equivalent to* `x_dim // 4`*)*  
    Dimensionality of each hidden layer in the MLP of each AffineCouplingLayer.  

- **decoder_affine_hidden_layer_activation:** *nn.Module, default=*`nn.ReLU`  
    Activation function applied to the outputs of each hidden layer in the MLP of each AffineCouplingLayer.  

- **decoder_nflow_n_hidden_layers:** *int, default=*`2`  
    Number of hidden layers in the MLP of the decoder's NFlowLayer.  

- **decoder_nflow_hidden_layer_dim:** *int, default=*`None` *(equivalent to* `x_dim // 4`*)*  
    Dimensionality of each hidden layer in the MLP of the decoder's NFlowLayer.  

- **decoder_nflow_hidden_layer_activation:** *nn.Module, default=*`nn.ReLU`  
    Activation function applied to the outputs of each hidden layer in the MLP of the decoder's NFlowLayer.  

- **decoder_observation_model:** *str, default=*`'poisson'`  
    - Either `gaussian` or `poisson`

    Observation model used by the model's decoder.  

- **decoder_fr_clamp_min:** *float, default=*`1E-7`  
    - Only applied when `decoder_observation_model='poisson'`

    Mininimum threshold used when clamping decoded firing rates.  

- **decoder_fr_clamp_max:** *float, default=*`1E7`  
    - Only applied when `decoder_observation_model='poisson'`

    Maximum threshold used when clamping decoded firing rates.  

- **label_prior_n_hidden_layers:** *int, default=*`2`  
    - Only applied when `discrete_labels=False`  

    Number of hidden layers in the MLP of the label prior estimator module.  

- **label_prior_hidden_layer_dim:** *int, default=*`32`  
    - Only applied when `discrete_labels=False`

    Dimensionality of each hidden layer in the MLP of the label prior estimator module.  

- **label_prior_hidden_layer_activation:** *nn.Module, default=*`nn.Tanh`  
    - Only applied when `discrete_labels=False`

    Activation function applied to the outputs of each hidden layer in the MLP of the label prior estimator module.  

## Attributes

- **decoder:** *nn.Module*  
    The model's decoder module which projects a latent space sample into the model's observation space.  

- **decoder_observation_model:** *str*  
    - Either `poisson` or `gaussian`.  

    The distribution of the obervsation space samples.  

- **decoder_fr_clamp_min:** *float*  
    Mininimum threshold used when clamping decoded firing rates.  

- **decoder_fr_clamp_max:** *float*  
    Maximum threshold used when clamping decoded firing rates.  

- **encoder:** *nn.Module*  
    The model's encoder module which approximates q(z \| x).  

- **inference:** *bool, default=*`False`  
    Flag denoting the model inference mode. When `True` the model is in inference mode.  

- **observation_noise_model:** *nn.Module*  
    - Only used when `decoder_observation_model='gaussian'`  

    The noise model used when computing the pi-VAE's loss.  

- **label_prior:** *nn.Module*  
    The model's label prior module which approximates p(z \| u).  

## Basic operation

For every observation space sample `x` and associated label `u` provided to pi-VAE's `forward` method, the encoder and label statistics (mean & log of variance) are obtained from the encoder  and label prior modules. These values are used to obtain the same statistics from the posterior q(z \| x,u). 

The [reparameterization trick](https://en.wikipedia.org/wiki/Reparameterization_trick) is performed with the resulting mean & log of variance to obtain the sample's representation in the model's latent space. This latent representation is then passed through the model's decoder module, which generates the predicted firing rate in the model's observation space. 

### Inputs

- **x:** *Tensor of shape(n_samples, x_dim)*  
    Samples in the model's observation space.  
- **u:** *Tensor, default=*`None`  
    - *shape(n_samples)* when using discrete labels
    - *shape(n_samples, u_dim)* when using continuous labels  
    
    Label corresponding to each sample. This parameter is not used when the model is in inference mode.  

### Outputs

A `dict` with the following items: 

- **encoder_firing_rate:** *Tensor of shape(n_samples, x_dim)*  
    Predicted firing rate of `encoder_z_sample`.  

- **encoder_z_sample:** *Tensor of shape(n_samples, z_dim)*  
    Latent space representation of each input sample computed from the encoder module's approximation of q(z \| x).  

- **encoder_mean:** *Tensor of shape(n_samples, z_dim)*  
    Mean of each input sample using the encoder module's approximation of q(z \| x).  

- **encoder_log_variance:** *Tensor of shape(n_samples, z_dim)*  
    Log of variance of each input sample using the encoder module's approximation of q(z \| x).  

- **label_mean:** *Tensor of shape(n_samples, z_dim)*  
    Mean of each input sample using the label prior module's approximation of p(z \| u).  

- **label_log_variance:** *Tensor of shape(n_samples, z_dim)*  
    Log of variance of input each sample using the label prior module's approximation of p(z \| u).  

- **posterior_firing_rate:** *Tensor of shape(n_samples, x_dim)*  
    Predicted firing rate of `posterior_z_sample`.  

- **posterior_z_sample:** *Tensor of shape(n_samples, z_dim)*      
    Latent space representation of each input sample computed from the approximation of posterior q(z \| x,u) ~ q(z \| x) &times; p(z \| u).  

- **posterior_mean:** *Tensor of shape(n_samples, z_dim)*  
    Mean of each input sample using the approximation of posterior of q(z \| x,u) ~ q(z \| x) &times; p(z \| u).  

- **posterior_log_variance:** *Tensor of shape(n_samples, z_dim)*  
    Log of variance of each input sample using the approximation of posterior q(z \| x,u) ~ q(z \| x) &times; p(z \| u).  

#### Inference Mode

A `dict` with the following items:  

- **encoder_firing_rate:** *Tensor of shape(n_samples, x_dim)*  
- **encoder_z_sample:** *Tensor of shape(n_samples, z_dim)*  
- **encoder_mean:** *Tensor of shape(n_samples, z_dim)*  
- **encoder_log_variance**: *Tensor of shape(n_samples, z_dim)*  

### Examples

#### Continuous Labels

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

outputs = model(x, u) # dict
```

#### Discrete Labels

```
import torch
from pi_vae_pytorch import PiVAE

model = PiVAE(
    x_dim = 100,
    u_dim = 3,
    z_dim = 2,
    discrete_labels=True
)

x = torch.randn(1, 100) # Size([n_samples, x_dim])

u = torch.randint(u_dim, (1,)) # Size([n_samples])

outputs = model(x, u) # dict
```

## Class Methods

- **decode(*z*)**  
    Projects samples in the model's latent space (`z_dim`) into the model's observation space (`x_dim`) by passing them through the model's decoder module.  

    > **Parameters:**  

    - **z**: *Tensor of shape(n_samples, z_dim)*  
        Samples to be projected into the model's observation space.  
    
    > **Returns:**  

    - **decoded**: *Tensor of shape(n_samples, x_dim)*  
        Samples projected into the model's observation space.  

    **Example:**  
    ```
    mdl = PiVAE(x_dim=100, u_dim=3, z_dim=2)
    z_samples = torch.randn(10, 2) # Size([n_samples, z_dim])

    decoded = mdl.decode(z_samples) # Size([n_samples, x_dim])
    ```  

- **encode(*x, return_stats=False*)**  
    Projects samples in the model's observation space (`x_dim`) into the model's latent space (`z_dim`) by passing them through the model's encoder module.  
    
    > **Parameters:**  

    - **x**: *Tensor of shape(n_samples, x_dim)*  
        Samples to be projected into the model's latent space.  

    - **return_stats**: *bool, default=False*  
        If `True`, the mean and log of the variance associated with the encoded sample are returned; otherwise only the encoded sample is returned.  
    
    > **Returns:**  

    When `return_stats=True` a tuple of tensors, otherwise a single tensor.  

    - **encoded**: *Tensor of shape(n_samples, z_dim)*  
        Samples projected into the model's latent space.  
        
    - **encoded_mean**: *Tensor of shape(n_samples, z_dim), optional*  
        Mean associated with a projected sample.  

    - **encoded_log_variance**: *Tensor of shape(n_samples, z_dim), optional*  
        Log of the variance associated with a projected sample.  

   **Example:**   
    ```
    mdl = PiVAE(x_dim=100, u_dim=3, z_dim=2)
    x_samples = torch.randn(10, 100) # Size([n_samples, x_dim])

    encoded = mdl.encode(x_samples) # Size([n_samples, z_dim])
    encoded, encoded_mean, encoded_log_variance = mdl.encode(x_samples, return_stats=True) # each of Size([n_samples, z_dim])
    ```  

- **get_label_statistics(*u, device=None*)**  
    Returns the mean and log of the variance associated with a label `u` using the label prior estimator of p(z \| u).  

    > **Parameters:**  

    - **u**: *int, float, list, tuple, or Tensor of shape(1, u_dim)*  
        Label whose statictics will be returned. An integer is expected in the discrete label regime, while a float, list, tuple or Pytorch Tensor is expected in the continuous label regime.    
    - **device**: *torch.device, default=*`None` *(uses the CPU*)  
        A [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) object representing the device on which operations will be performed. Should match the `torch.device` on which the model resides.  
    
    > **Returns:**  

    A tuple of tensors.  

    - **label_mean**: *Tensor of shape(1, z_dim)*  
        Mean of label `u`.  

    - **label_log_variance**: *Tensor of shape(1, z_dim)*  
        Log of the variance of label `u`.  

    **Examples:**   
    ```
    ## Discrete labels ##

    from random import randrange

    mdl = PiVAE(x_dim=100, u_dim=3, z_dim=2)
    label = randrange(3)

    mean, log_variance = mdl.get_label_statistics(label) # each of Size([1, z_dim])
    ```
    ```
    ## Continuous labels ##

    # 1-D label #
    mdl = PiVAE(x_dim=100, u_dim=1, z_dim=2, discrete_labels=False)
    label = 0.37

    mean, log_variance = mdl.get_label_statistics(label) # each of Size([1, z_dim])


    # n-D label #
    mdl = PiVAE(x_dim=100, u_dim=3, z_dim=2, discrete_labels=False)
    
    # ex tuple: label = (1.33, .82, .4)
    # ex list: label = [1.33, .82, .4]
    label = torch.randn(3) # Size([1, u_dim])

    mean, log_variance = mdl.get_label_statistics(label) # each of Size([1, z_dim])
    ```  

- **predict_labels(*x, n_samples=1000, device=None*)**  
    Returns the label probabilities associated with observation(s) `x`. An observation is first projected into the model's latent space by passing it through the encoder module. For each label in `u_dim`, an estimation of p(z \| u) is generated using Monte Carlo estimation with the specified number of samples. Next, the average distance between the projected observation and generated samples is computed, which serves as a proxy for the log likelihood. Softmax is then applied to these logits to produce the label probabilities.  
    
    *NOTE: Label prediction is currently supported only in the discrete label regime.*   

    > **Parameters:**  

    - **x**: *Tensor of shape(n_observations, x_dim)*  
        Data in the model's observation space.  

    - **n_samples**: *int, default=*`1000`  
        Number of samples drawn per label during Monte Carlo estimation.  

    - **device**: *torch.device, default=*`None` *(uses the CPU)*  
        A [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) object representing the device on which operations will be performed. Should match the `torch.device` on which the model resides.  
    
    > **Returns:**  
    
    - **predictions**: *Tensor of shape(n_observations, u_dim)*  
        Label probabilities of the observation(s).  

    **Example:**   
    ```
    mdl = PiVAE(x_dim=100, u_dim=3, z_dim=2) # discrete labels
    x = torch.randn(10, 100) # Size([n_observations, x_dim])

    predictions = mdl.predict_labels(x) # Size([n_observations, u_dim])
    ```  

- **sample(*u, n_samples=1, return_z=False, device=None*)**  
    Generates random samples in the model's observation space (`x_dim`). Samples are initially drawn from a Gaussian distribution in the model's latent space (`z_dim`) corresponding to specified label `u`. Samples are subsequently projected into the model's observation space (`x_dim`) by passing them through the model's decoder.  

    > **Parameters:**  

    - **u**: *int, float, list, tuple, or Tensor of shape(1, u_dim)*  
        Label of the samples to generate. An integer is expected in the discrete label regime, while a float, list, tuple or Pytorch Tensor is expected in the continuous label regime.  

    - **n_samples**: *int, default=*`1`  
        Number of samples to generate.  

    - **return_z**: *bool, default=*`False`  
        If `True` the latent space samples are returned along with the observation space samples. Otheriwse only the observation space samples are returned.  

    - **device**: *torch.device, default=*`None` *(uses the CPU)*  
        A [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) object representing the device on which operations will be performed. Should match the `torch.device` on which the model resides.  
    
    > **Returns:**  

    When `return_z=True` a tuple of tensors, otherwise a single tensor.  

    - **samples**: *Tensor of shape(n_samples, x_dim)*  
        Randomly generated sample(s) projected into the model's observation space.  

    - **z_samples**: *Tensor of shape(n_samples, z_dim), optional*  
        Randomly generated sample(s) in the model's latent space.  

    **Examples:**   
    ```
    ## Discrete labels ##

    from random import randrange

    mdl = PiVAE(x_dim=100, u_dim=3, z_dim=2)
    label = randrange(3)

    samples = mdl.sample(label, n_samples=10) # Size([n_samples, x_dim])
    samples, z_samples = mdl.sample(label, n_samples=10, return_z=True) # Size([n_samples, x_dim]). Size([n_samples, z_dim])
    ```  
    ```
    ## Continuous labels ##

    # 1-D label #
    mdl = PiVAE(x_dim=100, u_dim=1, z_dim=2, discrete_labels=False)
    label = 0.37

    samples = mdl.sample(label, n_samples=10) # Size([n_samples, x_dim])
    samples, z_samples = mdl.sample(label, n_samples=10, return_z=True) # Size([n_samples, x_dim]), Size([n_samples, z_dim])


    # n-D label #
    mdl = PiVAE(x_dim=100, u_dim=3, z_dim=2, discrete_labels=False)
    
    # ex tuple: label = (1.33, .82, .4)
    # ex list: label = [1.33, .82, .4]
    label = torch.randn(3) # Size([1, u_dim])

    samples = mdl.sample(label, n_samples=10) # Size([n_samples, x_dim])
    samples, z_samples = mdl.sample(label, n_samples=10, return_z=True) # Size([n_samples, x_dim]), Size([n_samples, z_dim])
    ```  

- **sample_z(*u, n_samples=1, device=None*)**  
    Generates random samples in the model's latent space (`z_dim`). Samples are drawn from a Gaussian distribution corresponding to specified label `u`.  

    > **Parameters:**  

    - **u**: *int, float, list, tuple, or Tensor of shape(1, u_dim)*  
        Label of the samples to generate. An integer is expected in the discrete label regime, while a float, list, tuple or Pytorch Tensor is expected in the continuous label regime.  

    - **n_samples**: *int, default=*`1`  
        Number of samples to generate.  

    - **device**: *torch.device, default=*`None` *(uses the CPU)*  
        A [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) object representing the device on which operations will be performed. Should match the `torch.device` on which the model resides.  
    
    > **Returns:**  

    - **samples**: *Tensor of shape(n_samples, z_dim)*  
        Randomly generated sample(s).  

    **Examples:**   
    ```
    ## Discrete labels ##

    from random import randrange

    mdl = PiVAE(x_dim=100, u_dim=3, z_dim=2)
    label = randrange(3)

    samples = mdl.sample(label, n_samples=10) # Size([n_samples, z_dim])
    ```  
    ```
    ## Continuous labels ##

    # 1-D label #
    mdl = PiVAE(x_dim=100, u_dim=1, z_dim=2, discrete_labels=False)
    label = 0.37

    samples = mdl.sample(label, n_samples=10) # Size([n_samples, z_dim])


    # n-D label #
    mdl = PiVAE(x_dim=100, u_dim=3, z_dim=2, discrete_labels=False)
    
    # ex tuple: label = (1.33, .82, .4)
    # ex list: label = [1.33, .82, .4]
    label = torch.randn(3) # Size([1, u_dim])

    samples = mdl.sample(label, n_samples=10) # Size([n_samples, z_dim])
    ```  

- **set_inference_mode(*state*)**  
    Toggles the model's inference state flag. When `True`, the model's `forward` method does not utilize the `u` parameter. When `False`,  the `u` parameter is utilized. Useful for working with unlabeled data. *NOTE: Inference mode must be disabled during model training.*  

    > **Parameters:**  

    - **state**: *bool*  
        The desired inference state.  
    
    > **Returns:**  

    - None  

    **Example:**   
    ```
    mdl = PiVAE(x_dim=100, u_dim=3, z_dim=2) # Inference Mode disabled by default
    x_samples = torch.randn(10, 100) # Size([n_samples, x_dim])

    mdl.set_inference_mode(True) # Inference Mode enabled
    outputs = mdl(x_samples) # dict
    ```  

## Static Methods

- **compute_posterior(*mean_0, log_variance_0, mean_1, log_variance_1*)**  
    Computes the posterior of two distributions as a product of Gaussians.  

    > **Parameters:**  

    - **mean_0:** *Tensor of shape(n_samples, sample_dim)*  
        Mean of a distribution.  

    - **log_variance_0:** *Tensor of shape(n_samples, sample_dim)*  
        Log of variance of a distribution.  

    - **mean_1:** *Tensor of shape(n_samples, sample_dim)*  
        Mean of a distribution.  

    - **log_variance_1:** *Tensor of shape(n_samples, sample_dim)*  
        Log of variance of a distribution.  
    
    > **Returns:**  

    - **posterior_mean:** *Tensor of shape(n_samples, sample_dim)*  
        Mean of the posterior distribution.  

    - **posterior_log_variance:** *Tensor of shape(n_samples, sample_dim)*  
        Log of variance of the posterior distribution.  

- **reparameterization_trick(*mean, log_variance*)**  
    Samples from an isotropic unit Gaussian distribution using the reparameterization trick.  

    > **Parameters:**  

    - **mean:** *Tensor of shape(n_samples, sample_dim)*  
        Mean of a distribution.  

    - **log_variance:** *Tensor of shape(n_samples, sample_dim)*  
        Log of variance of a distribution.  
    
    > **Returns:**  

    - **samples:** *Tensor of shape(n_samples, sample_dim)*  
        A sample from the distribution.  

## Loss Function - ELBOLoss

pi-VAE learns the deep generative model and the approximate posterior q(z \| x, u) of the true posterior p(z \| x, u) by maximizing the evidence lower bound (ELBO) of p(x \| u). This loss function is implemented in the included `ELBOLoss` class.  

### Initialization

```
pi_vae_pytorch.ELBOLoss(
    version=2,
    alpha=0.5,
    observation_model='poisson',
    device=None)
```  

- **version:** *int, default=*`2`  
    - Either `1` or `2`  

    The version of the loss function.  
    - **Version 1:** Computes the KL divergence between the posterior and the label prior.  
    - **Version 2:** Computes the KL divergence between the posterior and the label prior as well as between the encoder and label prior. These two values are then weighted by the `alpha` parameter.  

- **alpha:** *float, default=*`0.5`  
    - Only applied when `version=2`  
    - Must reside within [0, 1]  

    Weights the contribution of the encoder KL loss and posterior KL loss to the total KL loss. 

    ```
    kl_loss = (alpha * encoder_kl_loss) + ((1 - alpha) * posterior_kl_loss)
    ```   

- **observation_model:** *str, default=*`'poisson'`  
    - Either `poisson` or `gaussian`  
    - Should use the same value passed to `decoder_observation_model` when initializing pi-VAE.  

    The observation model used by pi-VAE's decoder.  

- **device:** *torch.device, default=*`None` *(uses the CPU)*  
    - Only applied when `observation_model='gaussian'`  

    A [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) object representing the device on which operations will be performed. Should match the `torch.device` on which the model resides.  

### Inputs

- **x:** *Tensor of shape(n_samples, x_dim)*  
    Sample(s) in the model's observation space.  

- **posterior_firing_rate:** *Tensor of shape(n_samples, x_dim)*  
    Predicted firing rate of latent(s) generated from posterior q(z \| x,u).  

- **posterior_mean:** *Tensor of shape(n_samples, z_dim)*  
    Mean from posterior q(z \| x,u) ~ q(z \| x) &times; p(z \| u).  

- **posterior_log_variance:** *Tensor of shape(n_samples, z_dim)*  
    Log of variance from posterior q(z \| x,u) ~ q(z \| x) &times; p(z \| u).  

- **label_mean:** *Tensor of shape(n_samples, z_dim)*  
    Mean from the label prior estimator which approximates p(z \| u).  

- **label_log_variance:** *Tensor of shape(n_samples, z_dim)*  
    Log of variance from the label prior estimator which approximates p(z \| u).  

- **encoder_mean:** *Tensor of shape(n_samples, z_dim), default=*`None`  
    - Only used when `version=2`  
    
    Mean from the encoder which approximates p(z \| x).  

- **encoder_log_variance:** *Tensor of shape(n_samples, z_dim), default=*`None`  
    - Only used when `version=2`  

    Log of variance from the encoder which approximates p(z \| x).  

- **observation_noise_model:** *nn.Module, default=*`None`  
    - Only used when `observation_model='gaussian'`  
    
    The noise model used when pi-VAE's decoder utilizes a Gaussian observation model. When pi-VAE is initialized with `decoder_observation_model='gaussian'`, the model's `observation_noise_model` attribute should be used.  

- **kl_weight:** *float, default=*`1.0`  
    Weight term used to scale the KL divergence component of the loss. Enables the use of KL annealing.  

### Outputs

- **loss:** *Tensor of shape(1)*  
    The total loss of the samples.

### Static Methods

- **compute_kl_loss(*mean_0, log_variance_0, mean_1, log_variance_1*)**  
    Computes the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between two distributions.  

    > **Parameters:**  

    - **mean_0:** *Tensor of shape(n_samples, sample_dim)*  
        Mean of a distribution.  

    - **log_variance_0:** *Tensor of shape(n_samples, sample_dim)*  
        Log of variance of a distribution.  

    - **mean_1:** *Tensor of shape(n_samples, sample_dim)*  
        Mean of a distribution.  

    - **log_variance_1:** *Tensor of shape(n_samples, sample_dim)*  
        Log of variance of a distribution.  
    
    > **Returns:**  

    - **kl_loss**: *Tensor of shape(1)*  
        The Kullback-Leibler divergence loss.  

### Examples

#### Poisson observation model

```
from pi_vae_pytorch import ELBOLoss

loss_fn = ELBOLoss()

outputs = model(x, u) # Initialized with decoder_observation_model='poisson'

loss = loss_fn(
    x=x,
    posterior_firing_rate=outputs['posterior_firing_rate'],
    posterior_mean=outputs['posterior_mean'],
    posterior_log_variance=outputs['posterior_log_variance'],
    label_mean=outputs['label_mean'],
    label_log_variance=outputs['label_log_variance'],
    encoder_mean=outputs['encoder_mean'],
    encoder_log_variance=outputs['encoder_log_variance']
)

loss.backward()
```

#### Gaussian observation model

```
import torch
from pi_vae_pytorch import ELBOLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device) # Initialized with decoder_observation_model='gaussian'

loss_fn = ELBOLoss(observation_model='gaussian', device=device)

outputs = model(x, u) 

loss = loss_fn(
    x=x,
    posterior_firing_rate=outputs['posterior_firing_rate'],
    posterior_mean=outputs['posterior_mean'],
    posterior_log_variance=outputs['posterior_log_variance'],
    label_mean=outputs['label_mean'],
    label_log_variance=outputs['label_log_variance'],
    encoder_mean=outputs['encoder_mean'],
    encoder_log_variance=outputs['encoder_log_variance']
    observation_noise_model=model.observation_noise_model
)

loss.backward()
```

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
