import torch

from pi_vae_pytorch import PiVAE


n_samples = 1
x_dim = 100
u_dim = 3
z_dim = 2
discrete_labels = False

model = PiVAE(
    x_dim = x_dim,
    u_dim = u_dim,
    z_dim = z_dim,
    discrete_labels=discrete_labels
)

x = torch.randn(n_samples, x_dim) # Size([n_samples, x_dim])
u = torch.randn(n_samples, u_dim) # Size([n_samples, u_dim])

outputs = model(x, u) # dict

for key in outputs.keys():
    if key == "firing_rate":
        dim2 = x_dim
    else:
        dim2 = z_dim
    
    assert outputs[key].shape == (n_samples, dim2), f"correct {key} shape outputted"

'''
assert outputs["firing_rate"].shape == (n_samples, x_dim), "correct firing_rate shape outputted"
assert outputs["lambda_mean"].shape == (n_samples, z_dim), "correct lambda_mean shape outputted"
assert outputs["lambda_log_variance"].shape == (n_samples, z_dim), "correct lambda_log_variance shape outputted"
assert outputs["posterior_mean"].shape == (n_samples, z_dim), "correct posterior_mean shape outputted"
assert outputs["posterior_log_variance"].shape == (n_samples, z_dim), "correct posterior_log_variance shape outputted"
assert outputs["z_mean"].shape == (n_samples, z_dim), "correct z_mean shape outputted"
assert outputs["z_log_variance"].shape == (n_samples, z_dim), "correct z_log_variance shape outputted"
assert outputs["z_sample"].shape == (n_samples, z_dim), "correct z_sample shape outputted"
'''
