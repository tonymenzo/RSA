"""
Need to set: 
N_events
batch_size
learning_rate
epochs
"""

import importlib
from RSA_ND_tuner_Joker import *
import RSA_ND_tuner_Joker
importlib.reload(RSA_ND_tuner_Joker)
from RSA_ND_tuner_Joker import *
 
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


import torch_optimizer as optim
import os


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

# 2D dataset
# sim_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1275_aU0_aS0_aC0_aB0_aH0.97_bD0.905_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_1.0e+05_hadrons.npy'
# sim_accept_reject_PATH = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1275_aU0_aS0_aC0_aB0_aH0.97_bD0.905_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_1.0e+05_id_mT2_accept_reject_z.npy'
# sim_fPrel_PATH         = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1275_aU0_aS0_aC0_aB0_aH0.97_bD0.905_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_1.0e+05_fPrel.npy'
# exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_5.0e+05_hadrons.npy'
# # Extra: exp_accept_reject values for optimal ("Joker") fit
# exp_accept_reject_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_5.0e+05_id_mT2_accept_reject_z.npy'

# 3D dataset
sim_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.085_aU0_aS0_aC0_aB0_aH0.97_bD0.93_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.2925_N_1.0e+05_hadrons.npy'
sim_accept_reject_PATH = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.085_aU0_aS0_aC0_aB0_aH0.97_bD0.93_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.2925_N_1.0e+05_id_mT2_accept_reject_z.npy'
sim_fPrel_PATH         = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.085_aU0_aS0_aC0_aB0_aH0.97_bD0.93_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.2925_N_1.0e+05_fPrel.npy'
exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.27125_N_5.0e+05_hadrons.npy'
# Extra: exp_accept_reject values for optimal ("Joker") fit
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


# # --- Experimental data ---
# p_charged_1mom_exp, p_charged_2raw_exp, p_charged_var_exp, p_charged_skew_exp = \
#     momentum_fraction_moments(p_frac_exp, charged_mask_exp)

# # --- Simulated data ---
# p_charged_1mom_sim, p_charged_2raw_sim, p_charged_var_sim, p_charged_skew_sim = \
#     momentum_fraction_moments(p_frac_sim, charged_mask_sim)

# # Mask for all real particles (exclude padding)
# all_mask_exp = pid_exp != 0
# all_mask_sim = pid_sim != 0

# p_all_1mom_exp, p_all_2raw_exp, p_all_var_exp, p_all_skew_exp = \
#     momentum_fraction_moments(p_frac_exp, all_mask_exp)

# p_all_1mom_sim, p_all_2raw_sim, p_all_var_sim, p_all_skew_sim = \
#     momentum_fraction_moments(p_frac_sim, all_mask_sim)


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
sigma_base = 0.2925

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
epsilon = 1e-5
params_learn = {'a1': torch.tensor(aLundD+epsilon), 'b1': torch.tensor(bLundD+epsilon),'sigma': torch.tensor(sigma_base+epsilon)}

#temp 2d
# params_learn = {'a1': torch.tensor(aLundD), 'b1': torch.tensor(bLundD)}

# Irrelevant parameters for the flow plot that must be initialized for the RSA class
epochs = 75

# Create an RSA instance
RSA = RSA_nD_tuner(epochs = epochs, dim_multiplicity = dim_multiplicity, dim_accept_reject = dim_accept_reject, over_sample_factor = over_sample_factor,
				params_base = params_base, sim_observable_dataloader = sim_observable_dataloader, sim_z_dataloader = sim_accept_reject_dataloader, 
				sim_fPrel_dataloader = sim_fPrel_dataloader, exp_observable_dataloader = exp_observable_dataloader, print_details = False, 
				results_dir = "/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Tuner_nD", params_init = params_learn, fixed_binning = True, loss_type='Joker')



# optimizer = optim.Adahessian(RSA.weight_nexus.parameters())
optimizer = torch.optim.Adam(RSA.weight_nexus.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(macroscopic_trainer.weight_nexus.parameters(), lr=learning_rate)

# Generate gradients
params_final, all_params, loss_values = RSA.RSA_tune(optimizer)


save_nm = 3
dim = len(params_learn)

# Construct full path
folder_path = f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Tune_map/Joker/{dim}D/'

# Create directory if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Tune_map/Joker/{dim}D/all_params_{save_nm}', all_params)
np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Tune_map/Joker/{dim}D/params_final_{save_nm}', params_final)
np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Tune_map/Joker/{dim}D/loss_values_{save_nm}', loss_values)
