"""
# Flowmap for reweighting of parameters including sigma using Joker observable histograms
# Run file from src directory to avoid path errors
# Define base parameters in params_base, define parameters to be reweighted in params_learn and parameters for the grid computation in ad_au_bd_init
# !NOTE: parameters need to be in the same order in grid, params_base and params_learn
"""

#Change for 2d: parameter input, grid, loss_func to Joker_nosigma if no sigma is involved

import importlib
from RSA_ND_tuner_Joker import *
import RSA_ND_tuner_Joker
importlib.reload(RSA_ND_tuner_Joker)
from RSA_ND_tuner_Joker import *
 
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ObservableDataset(Dataset):
	"""
	Converts observable dataset into PyTorch syntax.
	"""
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		sample = self.data[idx]
		return sample
	
class ObservableDatasetJoker(Dataset):
    """
    Converts observable dataset into PyTorch syntax.
    """
    def __init__(self, data):
        self.mult = data[0]
        self.pT = data[1]
        self.z_accept = data[2]

    def __len__(self):
        return self.mult.shape[0]

    def __getitem__(self, idx):
        return (self.mult[idx], self.pT[idx], self.z_accept[idx])
	
def a_b_c_grid(x_range, y_range, z_range, n_points):
    """
    Creates a grid of values within a three-dimensional range and returns it in a flattened tensor.

    Parameters:
    x_range (tuple): A tuple of (min, max) for the x-axis range.
    y_range (tuple): A tuple of (min, max) for the y-axis range.
    z_range (tuple): A tuple of (min, max) for the z-axis range.
    steps (int): --- The number of steps/points in each dimension.

    Returns:
    torch.Tensor: A flattened tensor containing all the grid points.
    """
    # Create linearly spaced points for each range
    x_points = torch.linspace(x_range[0], x_range[1], n_points)
    y_points = torch.linspace(y_range[0], y_range[1], n_points)
    z_points = torch.linspace(z_range[0], z_range[1], n_points)

    # Create a meshgrid from the x and y points
    x_grid, y_grid, z_grid = torch.meshgrid(x_points, y_points, z_points, indexing='ij')

    # Flatten the grid and stack the coordinates
    grid_flattened = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)

    return grid_flattened

def a_b_c_grid_custom(x_points, y_points, z_points):
    """
    Creates a grid of values within a three-dimensional range and returns it in a flattened tensor.

    Parameters:
    x_points (tuple): An array of points for evaluation for the x-axis range.
    y_points (tuple): An array of points for evaluation for the y-axis range.
    z_points (tuple): An array of points for evaluation for the z-axis range.

    Returns:
    torch.Tensor: A flattened tensor containing all the grid points.
    """

    # Create a meshgrid from the x and y points
    x_grid, y_grid, z_grid = torch.meshgrid(x_points, y_points, z_points, indexing='ij')

    # Flatten the grid and stack the coordinates
    grid_flattened = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)

    return grid_flattened

def grid_from_dict(params_grid: dict, params_learn: dict) -> torch.Tensor:
# def grid_from_dict(params_grid, params_learn):
    """
    Creates a grid of all parameter combinations based on values in `params_grid`,
    ordered by the keys in `params_learn`.

    Args:
        params_grid (dict): Dictionary of parameter_name: tensor_of_grid_points.
        params_learn (dict): Dictionary that defines the key order.

    Returns:
        torch.Tensor: A tensor of shape (num_points, num_parameters),
                      with rows representing parameter combinations.
    """
    # 1. Get ordered keys from params_learn
    ordered_keys = list(params_learn.keys())

    # 2. Collect the grid vectors in the desired order
    grid_axes = [params_grid[k] for k in ordered_keys]

    # 3. Build meshgrid in correct order
    mesh = torch.meshgrid(*grid_axes, indexing='ij')

    # 4. Flatten and stack each mesh into (N_points, N_dims)
    grid = torch.stack([m.flatten() for m in mesh], dim=-1)

    return grid, ordered_keys

def charged_mask_from_pdg(pdg_ids, charged=1):
    """
    Minimal PDG charge mask for common stable hadrons.
    Extend with a full PDG charge table if needed.
    """
    charged_set = {
        -3334,  # anti-Omega+
        -3324,  # anti-Xi*+
        -3224,  # anti-Sigma*-
        -3114,  # anti-Sigma*+
        -2214,  # anti-Delta-
        -1114,  # anti-Delta+
        -323,   # anti-K*-
        -321,   # K-
        -213,   # rho-
        -211,   # pi-
        211,   # pi+
        213,   # rho+
        321,   # K+
        323,   # K*+
        1114,  # Delta-
        2212,  # proton
        2214,  # Delta+
        2224,  # Delta++
        3112,  # Sigma-
        3114,  # Sigma*-
        3222,  # Sigma+
        3224,  # Sigma*+
        3312,  # Xi-
        3314,  # Xi*-
        3334,  # Omega-
    }
    
    return np.isin(pdg_ids, list(charged_set))

# def momentum_fraction_raw_moments(p_frac, charged_mask): #take for raw moments -> try later
#     """
#     Compute first 3 raw moments of momentum fractions
#     for the charged subset of particles in each event.

#     Parameters
#     ----------
#     p_frac : array, shape (n_events, n_particles)
#         Momentum fractions per event (e.g., |p_i| / sum|p|).
#     charged_mask : bool array of same shape
#         True where particle is charged, False otherwise.

#     Returns
#     -------
#     m1 : (n_events,) array
#         First raw moment <z>.
#     m2 : (n_events,) array
#         Second raw moment <z^2>.
#     m3 : (n_events,) array
#         Third raw moment <z^3>.
#     """
#     mask = charged_mask.astype(bool)
#     counts = np.sum(mask, axis=1, keepdims=True)
#     safe = counts > 0

#     m1 = np.zeros_like(counts, dtype=float)
#     m2 = np.zeros_like(counts, dtype=float)
#     m3 = np.zeros_like(counts, dtype=float)

#     # compute only for events with charged particles
#     m1[safe] = np.sum((p_frac)    * mask, axis=1, keepdims=True)[safe] / counts[safe]
#     m2[safe] = np.sum((p_frac**2) * mask, axis=1, keepdims=True)[safe] / counts[safe]
#     m3[safe] = np.sum((p_frac**3) * mask, axis=1, keepdims=True)[safe] / counts[safe]

#     return m1.ravel(), m2.ravel(), m3.ravel()

def momentum_fraction_moments(p_frac, charged_mask):
    """
    Compute first 3 moments (mean, variance, skewness) of momentum fractions
    for the charged subset of particles in each event.

    Parameters
    ----------
    p_frac : array, shape (n_events, n_particles)
        Momentum fractions per event (e.g., |p_i| / sum|p|).
    charged_mask : bool array of same shape
        True where particle is charged, False otherwise.

    Returns
    -------
    mean : (n_events,) array
        First moment (mean) over charged subset.
    m2_raw : (n_events,) array
        Second raw moment <z^2>.
    var : (n_events,) array
        Variance <(z - mean)^2>.
    skew : (n_events,) array
        Skewness (third central moment normalized by variance^(3/2)).
    """
    mask = charged_mask.astype(bool)
    counts = np.sum(mask, axis=1, keepdims=True)
    safe = counts > 0
    eps = 1e-12

    # First moment <z>
    m1 = np.zeros_like(counts, dtype=float)
    m1[safe] = np.sum(p_frac * mask, axis=1, keepdims=True)[safe] / counts[safe]

    # Second raw moment <z^2>
    m2_raw = np.zeros_like(counts, dtype=float)
    m2_raw[safe] = np.sum((p_frac**2) * mask, axis=1, keepdims=True)[safe] / counts[safe]

    # Variance <(z - m1)^2>
    var = np.zeros_like(counts, dtype=float)
    var[safe] = np.sum(((p_frac - m1)**2) * mask, axis=1, keepdims=True)[safe] / counts[safe]

    # Skewness
    skew = np.zeros_like(counts, dtype=float)
    skew[safe] = np.sum((((p_frac - m1) / (np.sqrt(var) + eps))**3) * mask,
                         axis=1, keepdims=True)[safe] / counts[safe]

    return m1.ravel(), m2_raw.ravel(), var.ravel(), skew.ravel()


#2d datasets: not okay
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD-0.05_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.0e+05_id_mT2_accept_reject_z.npy
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD-0.05_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.0e+05_hadrons.npy
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD-0.05_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.0e+05_fPrel.npy
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.0e+05_id_mT2_accept_reject_z.npy
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.0e+05_hadrons.npy
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.0e+05_fPrel.npy

# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.8_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_2.0e+04_id_mT2_accept_reject_z.npy
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.8_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_2.0e+04_hadrons.npy
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.8_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_2.0e+04_fPrel.npy
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.06_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_2.5e+04_id_mT2_accept_reject_z.npy
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.06_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_2.5e+04_hadrons.npy
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.06_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_2.5e+04_fPrel.npy

# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_5.0e+05_fPrel.npy
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_5.0e+05_hadrons.npy
# /pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_5.0e+05_id_mT2_accept_reject_z.npy


# Paths to the datasets
# exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhad_a0.68_b0.98_aD0.1_aU-0.1_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.0e+05_hadrons.npy'
# exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.3_N_1.0e+05_hadrons.npy'
# exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.0e+04_hadrons.npy'
# exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_1.5e+05_hadrons.npy'
# exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.06_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_2.0e+04_hadrons.npy'
# exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.06_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_1.5e+05_hadrons.npy'
# exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.17_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.25_N_1.0e+06_hadrons.npy'
# exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.085_aU0_aS0_aC0_aB0_aH0.97_bD0.93_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.2925_N_1.0e+05_hadrons.npy'
exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_5.0e+05_hadrons.npy'
# exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.085_aU0_aS0_aC0_aB0_aH0.97_bD0.93_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.2925_N_1.0e+05_hadrons.npy'


# sim_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.0e+05_hadrons.npy'
# sim_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.5e+05_hadrons.npy'
sim_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1275_aU0_aS0_aC0_aB0_aH0.97_bD0.905_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_1.0e+05_hadrons.npy'
# sim_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.8_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_2.0e+04_hadrons.npy'

# sim_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.085_aU0_aS0_aC0_aB0_aH0.97_bD0.93_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.2925_N_1.0e+05_hadrons.npy'
# sim_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1275_aU0_aS0_aC0_aB0_aH0.97_bD0.905_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_1.0e+05_hadrons.npy'
# sim_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.17_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.25_N_1.0e+05_hadrons.npy'
# sim_accept_reject_PATH = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.0e+05_id_mT2_accept_reject_z.npy'
sim_accept_reject_PATH = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1275_aU0_aS0_aC0_aB0_aH0.97_bD0.905_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_1.0e+05_id_mT2_accept_reject_z.npy'
# sim_accept_reject_PATH = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.8_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_2.0e+04_id_mT2_accept_reject_z.npy'

# sim_accept_reject_PATH = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.085_aU0_aS0_aC0_aB0_aH0.97_bD0.93_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.2925_N_1.0e+05_id_mT2_accept_reject_z.npy'
# sim_accept_reject_PATH = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1275_aU0_aS0_aC0_aB0_aH0.97_bD0.905_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_1.0e+05_id_mT2_accept_reject_z.npy'
# sim_accept_reject_PATH = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.17_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.25_N_1.0e+05_id_mT2_accept_reject_z.npy'
# sim_fPrel_PATH         = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.0e+05_fPrel.npy'
sim_fPrel_PATH         = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1275_aU0_aS0_aC0_aB0_aH0.97_bD0.905_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_1.0e+05_fPrel.npy'
# sim_fPrel_PATH         = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.8_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_2.0e+04_fPrel.npy'

# sim_fPrel_PATH         = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.085_aU0_aS0_aC0_aB0_aH0.97_bD0.93_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.2925_N_1.0e+05_fPrel.npy'
# sim_fPrel_PATH         = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1275_aU0_aS0_aC0_aB0_aH0.97_bD0.905_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_1.0e+05_fPrel.npy'
# sim_fPrel_PATH         = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.17_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.25_N_1.0e+05_fPrel.npy'

# Extra: exp_accept_reject values for optimal ("Joker") fit
# exp_accept_reject_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.17_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.25_N_1.0e+06_id_mT2_accept_reject_z.npy'
# exp_accept_reject_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.085_aU0_aS0_aC0_aB0_aH0.97_bD0.93_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.2925_N_1.0e+05_id_mT2_accept_reject_z.npy'
exp_accept_reject_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_5.0e+05_id_mT2_accept_reject_z.npy'


# Load the arrays
exp_hadrons       = np.load(exp_hadrons_PATH, mmap_mode="r")
sim_hadrons       = np.load(sim_hadrons_PATH, mmap_mode="r")
sim_accept_reject = np.load(sim_accept_reject_PATH, mmap_mode = "r")
sim_fPrel         = np.load(sim_fPrel_PATH, mmap_mode = "r")
# Extra: ("Joker" fit)
exp_accept_reject = np.load(exp_accept_reject_PATH, mmap_mode = "r")

# Print dataset shapes
print('Experimental observable shape:', exp_hadrons.shape)
# print('Experimental observable:', exp_hadrons[0,:]) #the first 10 hadrons and their 5 properties
print('Simulated observable shape:', sim_hadrons.shape)
print('Simulated z shape:', sim_accept_reject.shape)
print('Simulated fPrel shape:', sim_fPrel.shape)

# Restrict to a subset of the full dataset (for memory)
# N_events = int(50000)
# N_target = min(int(1000000), len(exp_hadrons)) # the number of events in the experimental dataset
N_events = int(30000)
N_target = min(int(30000), len(exp_hadrons)) # the number of events in the experimental dataset
print('N_target: ', N_target)

# Observables:
exp_hadrons_temp = exp_hadrons[:N_target]
sim_hadrons_temp = sim_hadrons[:N_events]   

px_exp, py_exp, pz_exp = exp_hadrons_temp[..., 0], exp_hadrons_temp[..., 1], exp_hadrons_temp[..., 2]
E_exp, m_exp, pid_exp = exp_hadrons_temp[..., 3], exp_hadrons_temp[..., 4], exp_hadrons_temp[..., 5]

px_sim, py_sim, pz_sim = sim_hadrons_temp[..., 0], sim_hadrons_temp[..., 1], sim_hadrons_temp[..., 2]
E_sim, m_sim, pid_sim = sim_hadrons_temp[..., 3], sim_hadrons_temp[..., 4], sim_hadrons_temp[..., 5]

p_mag_exp = np.sqrt(px_exp**2 + py_exp**2 + pz_exp**2)
p_mag_sim = np.sqrt(px_sim**2 + py_sim**2 + pz_sim**2)

# Charged masks
charged_mask_exp = charged_mask_from_pdg(pid_exp) & (pid_exp != 0)
charged_mask_sim = charged_mask_from_pdg(pid_sim) & (pid_sim != 0)
uncharged_mask_exp = ~charged_mask_exp & (pid_exp != 0)
uncharged_mask_sim = ~charged_mask_sim & (pid_sim != 0)
pion_mask_exp = (pid_exp == 211) | (pid_exp == -211) | (pid_exp == 111)
pion_mask_sim = (pid_sim == 211) | (pid_sim == -211) | (pid_sim == 111)

# --- Charged-only views (same shape as originals; others set to 0.0) ---
px_exp_charged = np.where(charged_mask_exp, px_exp, 0.0)
py_exp_charged = np.where(charged_mask_exp, py_exp, 0.0)
pz_exp_charged = np.where(charged_mask_exp, pz_exp, 0.0)
E_exp_charged  = np.where(charged_mask_exp,  E_exp, 0.0)
m_exp_charged  = np.where(charged_mask_exp,  m_exp, 0.0)

px_sim_charged = np.where(charged_mask_sim, px_sim, 0.0)
py_sim_charged = np.where(charged_mask_sim, py_sim, 0.0)
pz_sim_charged = np.where(charged_mask_sim, pz_sim, 0.0)
E_sim_charged  = np.where(charged_mask_sim,  E_sim, 0.0)
m_sim_charged  = np.where(charged_mask_sim,  m_sim, 0.0)

# --- Uncharged-only views (same shape; charged entries set to 0.0) ---
px_exp_uncharged = np.where(uncharged_mask_exp, px_exp, 0.0)
py_exp_uncharged = np.where(uncharged_mask_exp, py_exp, 0.0)
pz_exp_uncharged = np.where(uncharged_mask_exp, pz_exp, 0.0)
E_exp_uncharged  = np.where(uncharged_mask_exp,  E_exp, 0.0)
m_exp_uncharged  = np.where(uncharged_mask_exp,  m_exp, 0.0)

px_sim_uncharged = np.where(uncharged_mask_sim, px_sim, 0.0)
py_sim_uncharged = np.where(uncharged_mask_sim, py_sim, 0.0)
pz_sim_uncharged = np.where(uncharged_mask_sim, pz_sim, 0.0)
E_sim_uncharged  = np.where(uncharged_mask_sim,  E_sim, 0.0)
m_sim_uncharged  = np.where(uncharged_mask_sim,  m_sim, 0.0)

charged_mult_exp = np.sum(charged_mask_exp, axis=1)
charged_mult_sim = np.sum(charged_mask_sim, axis=1)

pion_mult_exp = np.sum(pion_mask_exp, axis=1)
pion_mult_sim = np.sum(pion_mask_sim, axis=1)

uncharged_mult_exp = np.sum(uncharged_mask_exp, axis=1)
uncharged_mult_sim = np.sum(uncharged_mask_sim, axis=1)

mask = np.abs(exp_hadrons_temp[:, :, 0]) > 0.0
exp_mult = np.sum(mask, axis=1)
mask = np.abs(sim_hadrons_temp[:, :, 0]) > 0.0
sim_mult = np.sum(mask, axis=1)

p_mag_exp = np.sqrt(px_exp**2 + py_exp**2 + pz_exp**2)
p_mag_sim = np.sqrt(px_sim**2 + py_sim**2 + pz_sim**2)

p_frac_exp = p_mag_exp / np.sum(p_mag_exp, axis=1, keepdims=True)
p_frac_sim = p_mag_sim / np.sum(p_mag_sim, axis=1, keepdims=True)


# --- Experimental data ---
p_charged_1mom_exp, p_charged_2raw_exp, p_charged_var_exp, p_charged_skew_exp = \
    momentum_fraction_moments(p_frac_exp, charged_mask_exp)

# --- Simulated data ---
p_charged_1mom_sim, p_charged_2raw_sim, p_charged_var_sim, p_charged_skew_sim = \
    momentum_fraction_moments(p_frac_sim, charged_mask_sim)

# Mask for all real particles (exclude padding)
all_mask_exp = pid_exp != 0
all_mask_sim = pid_sim != 0

p_all_1mom_exp, p_all_2raw_exp, p_all_var_exp, p_all_skew_exp = \
    momentum_fraction_moments(p_frac_exp, all_mask_exp)

p_all_1mom_sim, p_all_2raw_sim, p_all_var_sim, p_all_skew_sim = \
    momentum_fraction_moments(p_frac_sim, all_mask_sim)


exp_accept_reject = torch.Tensor(exp_accept_reject[0:N_target].copy())
sim_accept_reject = torch.Tensor(sim_accept_reject[0:N_events].copy())


# Check the accepted z-values, if z == 1 reduce it by epsilon (a very nasty bug to find).
# The a-coefficient when computing the likelihood has a term proportional to log(1-z). If 
# z = 1, this term diverges to -inf and completely destroys the backward pass.
epsilon = 1e-5
sim_accept_reject[:,:,2:][sim_accept_reject[:,:,2:] == 1] = 1 - epsilon # Do not change pid values!
exp_accept_reject[:,:,2:][exp_accept_reject[:,:,2:] == 1] = 1 - epsilon # Do not change pid values!


mask_base = sim_accept_reject[:, :, 5] > 0.0
mask_sim = exp_accept_reject[:, :, 5] > 0.0
px_sim = torch.tensor(sim_accept_reject[:, :, 3])
py_sim = torch.tensor(sim_accept_reject[:, :, 4])
pT_sim = torch.pow(torch.pow(px_sim,2) + torch.pow(py_sim,2), 1/2)
px_exp = torch.tensor(exp_accept_reject[:, :, 3])
py_exp = torch.tensor(exp_accept_reject[:, :, 4])
pT_exp = torch.pow(torch.pow(px_exp,2) + torch.pow(py_exp,2), 1/2)

#z_accept_obs
z_accept_sim = sim_accept_reject[:,:,5]
z_accept_exp = exp_accept_reject[:,:,5]

# Define observables:
# observable = 'mult,charged_mult,p_all_1mom,p_charged_1mom,p_all_var,p_charged_var,p_all_skew,p_charged_skew'
# sim_observables = [sim_mult, charged_mult_sim, p_all_1mom_sim, p_charged_1mom_sim, p_all_var_sim, p_charged_var_sim, p_all_skew_sim, p_charged_skew_sim]
# exp_observables = [exp_mult, charged_mult_exp, p_all_1mom_exp, p_charged_1mom_exp, p_all_var_exp, p_charged_var_exp, p_all_skew_exp, p_charged_skew_exp]
observable = 'Joker'
exp_observable = [pion_mult_exp, pT_exp, z_accept_exp]
sim_observable = [pion_mult_sim, pT_sim, z_accept_sim]

# sim_observable = np.stack(sim_observables, axis=0)
# exp_observable = np.stack(exp_observables, axis=0)

# Convert into torch objects
# sim_obs          = torch.Tensor(sim_observable[0:N_events].copy())
sim_fPrel         = torch.Tensor(sim_fPrel[0:N_events].copy())
# exp_obs          = torch.Tensor(exp_observable[0:N_target].copy())
sim_obs = sim_observable
exp_obs = exp_observable

# Check the accepted z-values, if z == 1 reduce it by epsilon (a very nasty bug to find).
# The a-coefficient when computing the likelihood has a term proportional to log(1-z). If 
# z = 1, this term diverges to -inf and completely destroys the backward pass.
epsilon = 1e-5
sim_accept_reject[:,:,2:][sim_accept_reject[:,:,2:] == 1] = 1 - epsilon # Do not change pid values!
exp_accept_reject[:,:,2:][exp_accept_reject[:,:,2:] == 1] = 1 - epsilon # Do not change pid values!


# Print dataset shapes
print('Experimental multiplicity shape:', exp_mult.shape)
print('Simulated multiplicity shape:', sim_mult.shape)
print('Simulated z shape:', sim_accept_reject.shape) # only has the z values, accepted and rejected
print('Simulated fPrel shape:', sim_fPrel.shape)

# Prepare data for DataLoader
sim_obs          = ObservableDatasetJoker(sim_obs)
sim_accept_reject = ObservableDataset(sim_accept_reject)
sim_fPrel            = ObservableDataset(sim_fPrel) #!NOTE: changed sim_mt to sim_fPrel
exp_obs          = ObservableDatasetJoker(exp_obs)

# Set batch size -- set it eqaul to the number of events, we only want one 'batch'
batch_size = N_events

# Initialize data-loaders
sim_observable_dataloader    = DataLoader(sim_obs,          batch_size = batch_size, shuffle = False, pin_memory=True)
sim_accept_reject_dataloader = DataLoader(sim_accept_reject, batch_size = batch_size, shuffle = False, pin_memory=True)
sim_fPrel_dataloader         = DataLoader(sim_fPrel,         batch_size = batch_size, shuffle = False, pin_memory=True)
exp_observable_dataloader    = DataLoader(exp_obs,          batch_size = N_target, shuffle = False, pin_memory=True)

print('Size of sim_observable_dataloader:', len(sim_observable_dataloader.dataset))
print('Size of sim_accept_reject_dataloader:', len(sim_accept_reject_dataloader.dataset))
print('Size of sim_fPrel_dataloader:', len(sim_fPrel_dataloader.dataset))
print('Size of exp_observable_dataloader:', len(exp_observable_dataloader.dataset))
# print('Shape of sim_observable_dataloader:', sim_observable_dataloader.dataset.data.shape)
# print('Shape of exp_observable_dataloader:', exp_observable_dataloader.dataset.data.shape)



# Training hyperparameters
over_sample_factor = 10.0
# The flow map will be dependent on the learning rate (size of the gradients)
learning_rate = 0.01
fixed_binning = True
# Length of event buffer
dim_multiplicity  = sim_accept_reject_dataloader.dataset.data.shape[1]
dim_accept_reject = sim_accept_reject_dataloader.dataset.data.shape[2]

print('Each event has been zero-padded to a length of', dim_multiplicity)
print('Each emission has been zero-padded to a length of', dim_accept_reject)

# Define base parameters of simulated data (a, b)
# params_base = torch.tensor([0.72, 0.88])

aExtraDQuark = 0.1275
# aExtraDQuark = 0.17
aExtraUQuark = 0
aExtraSQuark = 0
aExtraCquark = 0
aExtraBquark = 0
aExtraDiquark = 0.97

bNonstandardD = 0.905
# bNonstandardD = 0.93
bNonstandardU = 0.98
bNonstandardS = 0.98
bNonstandardC = 0.98
bNonstandardB = 0.98
bNonstandardH = 0.98

aLund = 0.68
bLund = 0.98
sigma_base = 0.27125

aLundD = aLund + aExtraDQuark
bLundD = bNonstandardD
# bLundD = bLund
aLundU = aLund + aExtraUQuark
bLundU = bNonstandardU
# bLundU = bLund
aLundS = aLund + aExtraSQuark
bLundS = bNonstandardS
# bLundS = bLund
aLundDiquark = aLund + aExtraDiquark
bLundDiquark = bNonstandardH
# bLundDiquark = bLund


# params_base = {'a0': torch.tensor(aLund), 'b0': torch.tensor(bLund),
params_base = {'a0': torch.tensor(0.0), 'b0': torch.tensor(0.0),
            'a1': torch.tensor(aLundD), 'b1': torch.tensor(bLundD), 'a2': torch.tensor(aLundU), 'b2': torch.tensor(bLundU),
            'a3': torch.tensor(aLundS), 'b3': torch.tensor(bLundS), 'a1103': torch.tensor(aLundDiquark), 'b1103': torch.tensor(bLundDiquark),
            'a2101': torch.tensor(aLundDiquark), 'b2101': torch.tensor(bLundDiquark), 'a2103': torch.tensor(aLundDiquark), 'b2103': torch.tensor(bLundDiquark),
            'a2203': torch.tensor(aLundDiquark), 'b2203': torch.tensor(bLundDiquark), 'a3101': torch.tensor(aLundDiquark), 'b3101': torch.tensor(bLundDiquark),
            'a3103': torch.tensor(aLundDiquark), 'b3103': torch.tensor(bLundDiquark), 'a3201': torch.tensor(aLundDiquark), 'b3201': torch.tensor(bLundDiquark),
            'a3203': torch.tensor(aLundDiquark), 'b3203': torch.tensor(bLundDiquark), 'a3303': torch.tensor(aLundDiquark), 'b3303': torch.tensor(bLundDiquark),
            'sigma': torch.tensor(sigma_base)}

# params_learn = {'a1': torch.tensor(aLundD), 'b1': torch.tensor(bLundD), 'a2': torch.tensor(aLundU)}
# params_learn = {'a1': torch.tensor(aLundD), 'b1': torch.tensor(bLundD),'a2': torch.tensor(aLundU),'sigma': torch.tensor(sigma_base)}
# params_learn = {'a1': torch.tensor(aLundD), 'b1': torch.tensor(bLundD),'sigma': torch.tensor(sigma_base)}

#temp 2d
params_learn = {'a1': torch.tensor(aLundD), 'b1': torch.tensor(bLundD)}

# Define a grid of initial parameters
# a1_points  = torch.linspace(0.7,1.0,10)
# a1_points  = torch.linspace(0.68,1.0,20)
a1_points  = torch.linspace(0.66,0.71,10)
# a2_points  = torch.linspace(0.68,0.68,10)
# a2_points  = torch.tensor([0.67,0.68,0.69])
# b1_points  = torch.linspace(0.88, 0.98, 2)
b1_points  = torch.linspace(0.95, 1.0, 10)
# bd_points  = torch.tensor([0.88])
# sigma_points = torch.arange(0.328, 0.337, 0.001)
# sigma_points = torch.linspace(0.200, 0.400, 20)
# sigma_points = torch.linspace(0.220, 0.310, 10)
sigma_points = torch.linspace(0.27125, 0.2925, 2)

# params_grid_dict = {'a1':a1_points, 'b1': b1_points,'a2': a2_points, 'sigma': sigma_points}
# params_grid_dict = {'a1':a1_points, 'b1': b1_points, 'sigma': sigma_points}
#temp 2d
params_grid_dict = {'a1':a1_points, 'b1': b1_points}

params_grid, ordered_keys = grid_from_dict(params_grid=params_grid_dict, params_learn=params_learn)


print('Initial ad_au_bd grid shape:', params_grid.shape)
print('Parameter order: ', ordered_keys)


# Irrelevant parameters for the flow plot that must be initialized for the RSA class
epochs = 1

# Create an RSA instance
RSA = RSA_nD_tuner(epochs = epochs, dim_multiplicity = dim_multiplicity, dim_accept_reject = dim_accept_reject, over_sample_factor = over_sample_factor,
				params_base = params_base, sim_observable_dataloader = sim_observable_dataloader, sim_z_dataloader = sim_accept_reject_dataloader, 
				sim_fPrel_dataloader = sim_fPrel_dataloader, exp_observable_dataloader = exp_observable_dataloader, print_details = False, 
				results_dir = "/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flowmap", params_init = params_learn, fixed_binning = True, loss_type='Joker_nosigma')

# for k,v in RSA.weight_nexus.params.items():
#     print(v)
	
# print(a_b_init)
# Set the optimizer
optimizer = torch.optim.Adam(RSA.weight_nexus.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(macroscopic_trainer.weight_nexus.parameters(), lr=learning_rate)
# Generate gradients
gradients, loss_grid, metrics = RSA.RSA_flow(optimizer, params_grid)
a_b_c = params_grid.detach().numpy()

#to save:
# Calculate the magnitude of each vector in a_b_gradients
magnitudes = np.linalg.norm(gradients, axis=1)
print(magnitudes.shape)

mu = metrics[0]
Neff = metrics[1]
plt_nm = 4
dim = len(params_learn)

print('Observable: ', observable)
print('Plot number: ', plt_nm)

# Construct full path
folder_path = f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map/{dim}D/{observable}/'
file_path = os.path.join(folder_path, f'ordered_params_{plt_nm}.npy')

# Create directory if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map/{dim}D/{observable}/mu_{plt_nm}', mu)
np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map/{dim}D/{observable}/Neff_{plt_nm}', Neff)
np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map/{dim}D/{observable}/magnitudes_{plt_nm}',magnitudes)
np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map/{dim}D/{observable}/gradients_{plt_nm}',gradients)
np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map/{dim}D/{observable}/params_grid_{plt_nm}',params_grid)
np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map/{dim}D/{observable}/loss_grid_{plt_nm}',loss_grid)
np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Flow_map/{dim}D/{observable}/ordered_params_{plt_nm}', ordered_keys)

