from joblib import dump, load
import numpy as np
import torch
import matplotlib.pyplot as plt
from data import simulate_data_discrete, simulate_data_continuous

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
seed = 2024
torch.manual_seed(seed)

n_samples = 10000
z_dim = 2  # ground truth here
x_dim = 100

discrete_prior = False # True
if discrete_prior:
    n_classes = 5
    x_true, z_true, u_true, lam_true = simulate_data_discrete(n_samples, n_classes, x_dim)
else:
    x_true, z_true, u_true, lam_true = simulate_data_continuous(n_samples, x_dim)

x_true = x_true.detach().cpu().numpy()
z_true = z_true.detach().cpu().numpy()
u_true = u_true.detach().cpu().numpy()
lam_true = lam_true.detach().cpu().numpy()

# Visualize dataset
if discrete_prior:
    c_vec = np.array(['crimson', 'orange', 'dodgerblue', 'limegreen', 'indigo'])
    idx = u_true
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(z_true[:, 0], z_true[:, 1], c=c_vec[idx], s=1, alpha=0.5, rasterized=True)
    plt.savefig('dataset.png')
    plt.close()
else:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    scatter = ax.scatter(
        z_true[:, 0], z_true[:, 1],
        c=u_true, cmap='viridis',
        s=1, alpha=0.5, rasterized=True)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Auxillary Label')
    plt.savefig('dataset.png', dpi=300)
    # plt.savefig('dataset.pdf')
    plt.close()

# Save dataset
# data = {
#     'x': x_true,
#     'z': z_true,
#     'u': u_true,
#     'lam': lam_true}
# dump(data, 'data.jl')