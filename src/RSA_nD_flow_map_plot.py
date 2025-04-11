# Clone a_b_init into numpy array
# a_b_c = ad_bd_au_init.detach().numpy()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# print(a_b_c)

magnitudes = np.load('temp_results/Flow_map_nD/magnitudes_4.npy')
a_b_gradients = np.load('temp_results/Flow_map_nD/gradients_4.npy')
a_b_c = np.load('temp_results/Flow_map_nD/ad_bd_au_4.npy')
loss_grid = np.load('temp_results/Flow_map_nD/loss_grid_4.npy')
mu_sigmamu = np.load('temp_results/Flow_map_nD/mu_4.npy')
Neff = np.load('temp_results/Flow_map_nD/Neff_4.npy')
mu = mu_sigmamu[:,0]
sigmamu = mu_sigmamu[:,1]


# Calculate the magnitude of each vector in a_b_gradients
magnitudes = np.linalg.norm(a_b_gradients, axis=1)
print(magnitudes.shape)

#to save:
# mu = metrics[0]
# Neff = metrics[1]
# np.save('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map_nD/mu_5', mu)
# np.save('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map_nD/Neff_5', Neff)
# np.save('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map_nD/magnitudes_5',magnitudes)
# np.save('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map_nD/gradients_5',a_b_gradients)
# np.save('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map_nD/ad_bd_au_5',a_b_c)
# np.save('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map_nD/loss_grid_5',loss_grid)

# 1 -> ad = 0.78, au =0.58, bd = 0.88
# 2 -> ad = 0.72, au =0.64, bd = 0.88
# np.save('/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map_nD/metrics_1',metrics)



fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ad = a_b_c[:,0]
bd = a_b_c[:,1]
au = a_b_c[:,2]

g1,g2,g3 =  -a_b_gradients[:,0], -a_b_gradients[:,1], -a_b_gradients[:,2],
scale = 5e1
ax.quiver(ad,bd,au,g1/scale,g2/scale,g3/scale)
# sc = ax.scatter(ad, bd, au, c=magnitudes, cmap='plasma', s=20)  # 's' is size, 'c' is color
sc = ax.scatter(ad, bd, au, c=np.log10((loss_grid)), cmap='plasma', s=50)  # 's' is size, 'c' is color
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)

ax.scatter(0.68,0.98,0.68, label = 'base', c= 'black', s=50)
ax.scatter(0.78,0.88,0.58, label = 'target', c='red', s=50)
plt.legend()
plt.title('Loss')
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ad = a_b_c[:,0]
bd = a_b_c[:,1]
au = a_b_c[:,2]


g1,g2,g3 =  -a_b_gradients[:,0], -a_b_gradients[:,1], -a_b_gradients[:,2],
scale = 5e1
ax.quiver(ad,bd,au,g1/scale,g2/scale,g3/scale)
# sc = ax.scatter(ad, bd, au, c=magnitudes, cmap='plasma', s=20)  # 's' is size, 'c' is color
sc = ax.scatter(ad, bd, au, c=Neff, cmap='plasma', s=50)  # 's' is size, 'c' is color
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)


ax.scatter(0.68,0.98,0.68, label = 'base', c= 'black', s=50)
ax.scatter(0.78,0.88,0.58, label = 'target', c='red', s=50)
plt.legend()
plt.title('N_eff')
plt.show()



fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ad = a_b_c[:,0]
bd = a_b_c[:,1]
au = a_b_c[:,2]


g1,g2,g3 =  -a_b_gradients[:,0], -a_b_gradients[:,1], -a_b_gradients[:,2],
scale = 5e1
ax.quiver(ad,bd,au,g1/scale,g2/scale,g3/scale)
# sc = ax.scatter(ad, bd, au, c=magnitudes, cmap='plasma', s=20)  # 's' is size, 'c' is color
sc = ax.scatter(ad, bd, au, c=mu, cmap='plasma', s=50)  # 's' is size, 'c' is color
cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)


ax.scatter(0.68,0.98,0.68, label = 'base', c= 'black', s=50)
ax.scatter(0.78,0.88,0.58, label = 'target', c='red', s=50)
plt.legend()
plt.title('Mu')
plt.show()
