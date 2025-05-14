import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -- Load data for the chosen plot --
plot_nm = '46'
magnitudes    = np.load(f'temp_results/Flow_map_nD/magnitudes_{plot_nm}.npy')
a_b_gradients = np.load(f'temp_results/Flow_map_nD/gradients_{plot_nm}.npy')
loss_grid     = np.load(f'temp_results/Flow_map_nD/loss_grid_{plot_nm}.npy')
mu_sigmamu    = np.load(f'temp_results/Flow_map_nD/mu_{plot_nm}.npy')
Neff          = np.load(f'temp_results/Flow_map_nD/Neff_{plot_nm}.npy')
mu, sigmamu  = mu_sigmamu[:, 0], mu_sigmamu[:, 1]
a_b_c         = np.load(f'temp_results/Flow_map_nD/ad_bd_sig_{plot_nm}.npy')


# Configuration per plot_nm: scale and base/target points
plot3d_configs = {
    '19': {'scale': 1e3, 'base': (0.68, 0.98, 0.335), 'target': (0.78, 0.88, 0.335)},
    '20': {'scale': 1e3, 'base': (0.68, 0.98, 0.335), 'target': (0.78, 0.88, 0.335)},
    '21': {'scale': 1e4, 'base': (0.68, 0.98, 0.335), 'target': (0.78, 0.88, 0.335)},
    '22': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.78, 0.88, 0.335)},
    '23': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.58, 0.88, 0.31)},
    '24': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.78, 0.88, 0.335)},
    '25': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.78, 0.88, 0.335)},
    '26': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.78, 0.88, 0.325)},
    '27': {'scale': 5e2, 'base': (0.68, 0.98, 0.335), 'target': (0.58, 0.88, 0.31)},
    '28': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.58, 0.88, 0.31)},
    '29': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.58, 0.88, 0.31)},
    '30': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.58, 0.88, 0.31)},
    '31': {'scale': 5e2, 'base': (0.68, 0.98, 0.335), 'target': (0.58, 0.88, 0.31)},
    '32': {'scale': 5e2, 'base': (0.68, 0.98, 0.335), 'target': (0.58, 0.88, 0.31)},
    '33': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.58, 0.88, 0.31)},
    '34': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.78, 0.88, 0.33)},
    '35': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.78, 0.88, 0.33)},
    '36': {'scale': 2e4, 'base': (0.68, 0.98, 0.335), 'target': (0.76, 0.98, 0.33)},
    '37': {'scale': 2e4, 'base': (0.68, 0.98, 0.335), 'target': (0.76, 0.98, 0.33)},
    '38': {'scale': 2e4, 'base': (0.68, 0.98, 0.335), 'target': (1.48, 0.98, 0.33)},
    '39': {'scale': 2e4, 'base': (0.68, 0.98, 0.335), 'target': (1.48, 0.98, 0.33)},
    '40': {'scale': 8e2, 'base': (0.68, 0.98, 0.335), 'target': (0.74, 0.88, 0.33)},
    '41': {'scale': 2e4, 'base': (0.68, 0.98, 0.335), 'target': (0.74, 0.88, 0.33)},
    '42': {'scale': 8e2, 'base': (0.68, 0.98, 0.335), 'target': (0.74, 0.88, 0.33)},
    '43': {'scale': 8e3, 'base': (0.68, 0.98, 0.335), 'target': (0.74, 0.88, 0.33)},
    '44': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.74, 0.88, 0.33)},
    '45': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.74, 0.88, 0.33)},
    '46': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.74, 0.88, 0.33)},
    '47': {'scale': 5e3, 'base': (0.68, 0.98, 0.335), 'target': (0.74, 0.88, 0.33)},
}
config = plot3d_configs.get(plot_nm)
if config is None:
    raise ValueError(f"Unsupported plot_nm: {plot_nm}")

# Unpack coordinates and gradients
ad, bd, sig = a_b_c[:, 0], a_b_c[:, 1], a_b_c[:, 2]
g1, g2, g3  = -a_b_gradients[:, 0], -a_b_gradients[:, 1], -a_b_gradients[:, 2]

# Unpack tuner values
# points = np.load('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Tuner/jupyter/ADAgrad_N50k_5/RSA_tuning_params.npy')
points = np.load('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Tuner/jupyter/RSA_tuning_params_SGD_lr_0.001.npy')

# Unpack the three columns
x, y, z = points[:, 0], points[:, 1], points[:, 2]




# Create 3D figure
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')



# Plot base and target
ax.scatter(*config['base'],  marker='o', color='black', s=50, label='base')
ax.scatter(*config['target'], marker='o', color='red',   s=50, label='target')
ax.plot(x, y, z, marker='o', c='blue')  # 3D scatter plot

# Quiver for gradient field
ax.quiver(ad, bd, sig,
          g1 / config['scale'],
          g2 / config['scale'],
          g3 / config['scale'],
          length=1.0, normalize=False)

# Scatter colored by loss value
sc = ax.scatter(ad, bd, sig,
                c=np.log10(loss_grid), cmap='plasma',
                s=20, alpha=0.8)
cbar = fig.colorbar(sc, pad=0.1)
cbar.set_label('log10(loss)')

# Labels and legend
ax.set_xlabel('a_d')
ax.set_ylabel('b_d')
ax.set_zlabel('sigma')
ax.set_title(f'3D Loss Landscape and Gradients (plot {plot_nm})')
ax.legend()
plt.tight_layout()
plt.show()