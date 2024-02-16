import numpy as np
import torch
from torch import nn, Tensor
from pytorch_model_summary import summary
from typing import List, Tuple, Dict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
seed = 69420
torch.manual_seed(seed)

n_samples = 10000
z_dim = 2  # ground truth here
x_dim = 100

discrete_prior = True
if discrete_prior:
    n_classes = 5
    x_true, z_true, u_true, lam_true = simulate_data_discrete(n_samples, n_classes, x_dim)
else:
    x_true, z_true, u_true, lam_true = simulate_data_continuous(n_samples, x_dim)

x_true = x_true.to(device)
u_true = u_true.to(device)

x_all = x_true.reshape(50, -1, x_dim)
u_all = u_true.reshape(50, -1)
u_all = u_all if discrete_prior else u_all.unsqueeze(-1)

x_train = x_all[:40]
u_train = u_all[:40]

x_valid = x_all[40:45]
u_valid = u_all[40:45]

x_test = x_all[45:]
u_test = u_all[45:]


vae = PIVAE(x_dim=x_dim,
            z_dim=2,
            u_dim=n_classes if discrete_prior else 1,
            encoder_n_layers=3,
            encoder_hidden_size=60,
            decoder_n_blocks=2,
            decoder_layers_per_gin_block=2,
            decoder_layers_per_gin=3,
            decoder_hidden_size=30,
            obs_model='poisson',
            discrete_prior=discrete_prior,
        ).to(device)

if discrete_prior:
    print(summary(vae, torch.rand(200,x_dim,device=device), torch.randint(0, n_classes, (200,), device=device)))
else:
    print(summary(vae, x_all[0], u_all[0]))

loss_train = []
loss_valid = []

optimizer = torch.optim.Adam(params=vae.parameters(), lr=5E-4)

pbar = tqdm(range(1000))
valid_every = 10

n_valid = len(x_valid)
n_train = len(x_train)
n_samples = x_train[0].shape[0]
for epoch in pbar:
    train_loss = 0.
    for batch in range(n_train):
        optimizer.zero_grad()
        x, u = x_train[batch], u_train[batch]
        loss, _ = vae(x, u)
        loss = loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / n_train
    loss_train.append(train_loss)

    if epoch % valid_every == 0:
        with torch.no_grad():
            valid_loss = 0
            for i in range(n_valid):
                x, u = x_valid[i], u_valid[i]
                valid_loss += vae(x, u)[0].item() / n_valid
            if np.isnan(loss):
                print("Loss is nan")
                break

            pbar.set_postfix({"valid-loss": f"{valid_loss:.04E}"})
            loss_valid.append(valid_loss)

with sns.plotting_context("talk"):
    fig, ax = plt.subplots(1, 1)
    ax.plot(loss_train, ".-", label="train")
    ax.plot(np.arange(1, len(loss_valid)+1) * valid_every, loss_valid, ".-", label="valid")
    ax.set(xlabel="epoch", ylabel="neg ELBO")
    ax.legend()
    sns.despine()


with torch.no_grad():
    _, out_dict = vae(x_true, u_true if discrete_prior else u_true.unsqueeze(-1), valid=True)
    post_means = out_dict["post_mean"]
    post_log_vars = out_dict["post_log_var"]
    post_lam_means = out_dict["lam_mean"]
    post_lam_log_vars = out_dict["lam_log_var"]
    z_means = out_dict["z_mean"]
    z_log_vars = out_dict["z_log_var"]

post_means = post_means.cpu()
z_means = z_means.cpu()
z_true = z_true.cpu()
u_true = u_true.cpu()
ll = n_samples

if discrete_prior:
    c_vec = np.array(['crimson','orange','dodgerblue','limegreen','indigo'])
    idx = u_true
else:
    length = 30
    c_vec = plt.cm.viridis(np.linspace(0,1, length))
    bins = np.linspace(0, 2*np.pi, length)
    centers = (bins[1:]+bins[:-1])/2
    idx = np.digitize(u_true.squeeze(), centers)

with sns.plotting_context("talk", font_scale=1):

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
    ax[0].scatter(z_true[:,0], z_true[:,1], c=c_vec[idx], s=1, alpha=0.5, rasterized=True)
    ax[1].scatter(post_means[:,1], post_means[:,0], c=c_vec[idx], s=1, alpha=0.5, rasterized=True)
    ax[2].scatter(z_means[:,1], z_means[:,0], s=1, c=c_vec[idx], alpha=0.5, rasterized=True)

    ax[0].set(xlabel="latent 1", ylabel="latent 2", title="ground truth")
    ax[1].set(xlabel="latent 1", ylabel="latent 2", title=r"posterior means $q(z|x,u)\propto q(z|x)p(z|u)$")
    ax[2].set(xlabel="latent 1", ylabel="latent 2", title=r"encoder mean $q(z|x)$")
    fig.tight_layout()
    sns.despine()
