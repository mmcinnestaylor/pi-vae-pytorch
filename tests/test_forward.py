import torch
from pi_vae_pytorch import PiVAE


n_samples = 10
x_dim = 100
u_dim = 3
z_dim = 2
x = torch.randn(n_samples, x_dim) 

def test_continuous():
    model = PiVAE(
        x_dim = x_dim,
        u_dim = u_dim,
        z_dim = z_dim,
        discrete_labels=False
    )

    u = torch.randn(n_samples, u_dim)
    outputs = model(x, u) 

    for key in outputs.keys():
        if 'firing_rate' in key:
            dim2 = x_dim
        else:
            dim2 = z_dim
        
        assert outputs[key].shape == (n_samples, dim2), f"Incorrect {key} shape outputted (continuous)"

def test_discrete():
    model = PiVAE(
        x_dim = x_dim,
        u_dim = u_dim,
        z_dim = z_dim,
        discrete_labels=True
    )

    u = torch.randint(0, u_dim, (n_samples,)) 
    outputs = model(x, u) 

    for key in outputs.keys():
        if 'firing_rate' in key:
            dim2 = x_dim
        else:
            dim2 = z_dim

        assert outputs[key].shape == (n_samples, dim2), f"Incorrect {key} shape outputted (discrete)"
