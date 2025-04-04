import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

nevents = 10
Neffs_file_name = f'Neffs_3d1_N{nevents}'
Sigma_file_name = 'Sigmas_3d1_N{nevents}'
A_file_name = 'Ad_3d1_N{nevents}'
B_file_name = 'B_3d1_N{nevents}'
C_file_name = 'Au_3d1_N{nevents}'


# np.save('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Metrics/' + Neffs_file_name, Neffs)
# np.save('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Metrics/' + Sigma_file_name, Sigmas)
# np.save('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Metrics/' + A_file_name, Ad)
# np.save('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Metrics/' + B_file_name, B)
# np.save('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Metrics/' + C_file_name, Au)

# Loading the saved files into variables with '_loaded' suffix
Neffs_loaded = np.load('temp_results/Metrics/' + Neffs_file_name + '.npy')
Sigmas_loaded = np.load('temp_results/Metrics/' + Sigma_file_name + '.npy')
Ad_loaded = np.load('temp_results/Metrics/' + A_file_name + '.npy')
B_loaded = np.load('temp_results/Metrics/' + B_file_name + '.npy')
Au_loaded = np.load('temp_results/Metrics/' + C_file_name + '.npy')


# Flatten the grids and function values for plotting
x_vals = Ad_loaded.flatten()
y_vals = Au_loaded.flatten()
z_vals = B_loaded.flatten()
N_eff = Neffs_loaded.flatten()

# Create the 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with a color map
scat = ax.scatter(x_vals, y_vals, z_vals, c=N_eff, cmap='viridis', s=60, alpha=0.8)
ax.scatter(0.68,0.68,0.98, s=100, c='red')

# Labels and title with larger font size and bold styling
ax.set_xlabel('a_d', fontsize=14, fontweight='bold')
ax.set_ylabel('a_u', fontsize=14, fontweight='bold')
ax.set_zlabel('b', fontsize=14, fontweight='bold')
ax.set_title('N_eff', fontsize=16, fontweight='bold')

# Add a colorbar with improved style
colorbar = fig.colorbar(scat)
colorbar.set_label('N_eff', fontsize=12, fontweight='bold')
colorbar.ax.tick_params(labelsize=10)

# Improve layout and tight layout
plt.tight_layout()

# Set background color for a softer look
fig.patch.set_facecolor('whitesmoke')
ax.set_facecolor('white')

# Adding grid for better visibility
ax.grid(True, linestyle='--', color='gray', alpha=0.6)

plt.show()


# Assuming the variables Ad_loaded, Au_loaded, B_loaded, and Neffs_loaded are already defined and loaded with data
# Flatten the grids and function values for plotting
x_vals = Ad_loaded.flatten()
y_vals = Au_loaded.flatten()
z_vals = B_loaded.flatten()
Sigmas = Sigmas_loaded.flatten()

# Create the 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with a color map
scat = ax.scatter(x_vals, y_vals, z_vals, c=Sigmas, cmap='viridis', s=60, alpha=0.8)
ax.scatter(0.68,0.68,0.98, s=100, c='red')

# Labels and title with larger font size and bold styling
ax.set_xlabel('a_d', fontsize=14, fontweight='bold')
ax.set_ylabel('a_u', fontsize=14, fontweight='bold')
ax.set_zlabel('b', fontsize=14, fontweight='bold')
ax.set_title('Sigma', fontsize=16, fontweight='bold')

# Add a colorbar with improved style
colorbar = fig.colorbar(scat)
colorbar.set_label('Sigma', fontsize=12, fontweight='bold')
colorbar.ax.tick_params(labelsize=10)

# Improve layout and tight layout
plt.tight_layout()

# Set background color for a softer look
fig.patch.set_facecolor('whitesmoke')
ax.set_facecolor('white')

# Adding grid for better visibility
ax.grid(True, linestyle='--', color='gray', alpha=0.6)
plt.show()