'''
# Gradient stability calculation on a line_grid connecting target and base parametrization points
# Define base parameters in params_base, target parameters in r_t and define parameters to be reweighted in params_learn
# !NOTE: parameters need to be in the same order in r_t, params_base and params_learn
'''

print('ONE')

import multiprocessing as mp
import os
import psutil
# mp.set_start_method('spawn')
nthreads = psutil.cpu_count(logical=True)
ncores = psutil.cpu_count(logical=False)
nthreads_per_core = nthreads // ncores
nthreads_available = len(os.sched_getaffinity(0))
ncores_available = nthreads_available // nthreads_per_core

assert nthreads == os.cpu_count()
assert nthreads == mp.cpu_count()

print(f'{nthreads=}')
print(f'{ncores=}')
print(f'{nthreads_per_core=}')
print(f'{nthreads_available=}')
print(f'{ncores_available=}')

import torch
from multiprocessing import Pool, cpu_count
from RSA_nD_tuner import RSA_nD_tuner, Dataset
from torch.utils.data import DataLoader
import multiprocessing as mp

import importlib
from RSA_nD_tuner import *
import RSA_nD_tuner
importlib.reload(RSA_nD_tuner)
from RSA_nD_tuner import *
import numpy as np

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
	

# Paths to the datasets

# exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_1.5e+05_hadrons.npy'
# exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.06_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.33_N_1.5e+05_hadrons.npy'
exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.1_aU0_aS0_aC0_aB0_aH0.97_bD0.88_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.3_N_1.5e+05_hadrons.npy'

sim_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.5e+05_hadrons.npy'
sim_accept_reject_PATH = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.5e+05_id_mT2_accept_reject_z.npy'
sim_fPrel_PATH         = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.5e+05_fPrel.npy'


# Load the arrays
exp_hadrons       = np.load(exp_hadrons_PATH, mmap_mode="r")
sim_hadrons       = np.load(sim_hadrons_PATH, mmap_mode="r")
sim_accept_reject = np.load(sim_accept_reject_PATH, mmap_mode = "r")
sim_fPrel         = np.load(sim_fPrel_PATH, mmap_mode = "r")

# Print dataset shapes
print('Experimental observable shape:', exp_hadrons.shape)
print('Simulated observable shape:', sim_hadrons.shape)
print('Simulated z shape:', sim_accept_reject.shape)
print('Simulated fPrel shape:', sim_fPrel.shape)


# Training hyperparameters
over_sample_factor = 10.0
# The flow map will be dependent on the learning rate (size of the gradients)
learning_rate = 0.01
fixed_binning = True

# Define base parameters of simulated data (a, b)
aExtraDQuark = 0
aExtraUQuark = 0
aExtraSQuark = 0
aExtraCquark = 0
aExtraBquark = 0
aExtraDiquark = 0.97

bNonstandardD = 0.88
bNonstandardU = 0.88
bNonstandardS = 0.88
bNonstandardC = 0.88
bNonstandardB = 0.88
bNonstandardH = 0.88

aLund = 0.68
bLund = 0.98
sigma_base = 0.335

aLundD = aLund + aExtraDQuark
# bLundD = bNonstandardD
bLundD = bLund
aLundU = aLund + aExtraUQuark
# bLundU = bNonstandardU
bLundU = bLund
aLundS = aLund + aExtraSQuark
# bLundS = bNonstandardS
bLundS = bLund
aLundDiquark = aLund + aExtraDiquark
# bLundDiquark = bNonstandardH
bLundDiquark = bLund


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
params_learn = {'a1': torch.tensor(aLundD), 'b1': torch.tensor(bLundD),'sigma': torch.tensor(sigma_base)}


r_b = torch.tensor([aLundD,bLundD,sigma_base])
r_t = torch.tensor([0.78,0.88,0.3])
s = torch.linspace(0,1,4)
# delta_s = s[1]-s[0]
s_eps = 1e-5
s[0] = s[0] + s_eps # for stability
s = torch.cat([s, torch.tensor([1.1])])
s = s[:, torch.newaxis]

line_grid = r_b + s*(r_t-r_b)

line_grid = torch.tensor(line_grid)

all_gradients = []
all_loss_grid = []
all_metrics_Neff = []
all_metrics_mu = []


Ns = [100,500,1000,2500,5000,7500,10000,25000,50000]
# Ns = [100,500,1000,2500]
# Ns = [1000,5000]
Ns = [int(N) for N in Ns]
repeat = {100:50, 500: 50, 1000:20, 2500:10, 5000:10, 7500:10, 10000:10, 25000:5, 50000:3}
# repeat = {100:100, 500: 50, 1000:20, 2500:10}
# repeat = {1000: 5, 5000:4, }


def run_one_repeat(args):
        N_events, r = args
        # 1) slice out your r-th chunk:
        exp_mult = np.array([len(exp_hadrons[i,:][np.abs(exp_hadrons[i,:,0])>0.0]) 
                                for i in range(len(exp_hadrons))])
        sim_mult = np.array([len(sim_hadrons[i,:][np.abs(sim_hadrons[i,:,0])>0.0]) 
                                for i in range(len(exp_hadrons))])

        sim_mult_t          = torch.Tensor(sim_mult[r*N_events:(r+1)*N_events].copy())
        sim_accept_reject_t = torch.Tensor(sim_accept_reject[r*N_events:(r+1)*N_events].copy())
        sim_fPrel_t         = torch.Tensor(sim_fPrel[r*N_events:(r+1)*N_events].copy())
        exp_mult_t          = torch.Tensor(exp_mult[r*N_events:(r+1)*N_events].copy())

        # clamp z:
        epsilon = 1e-5
        sim_accept_reject_t[sim_accept_reject_t == 1] = 1 - epsilon

        # wrap into Datasets + DataLoaders
        sim_obs_dl = DataLoader(ObservableDataset(sim_mult_t), batch_size=N_events, shuffle=False)
        sim_z_dl   = DataLoader(ObservableDataset(sim_accept_reject_t), batch_size=N_events, shuffle=False)
        sim_f_dl   = DataLoader(ObservableDataset(sim_fPrel_t), batch_size=N_events, shuffle=False)
        exp_dl     = DataLoader(ObservableDataset(exp_mult_t), batch_size=N_events, shuffle=False)

        # infer dims (your fixed logic, or adapt as needed):
        dim_mult = sim_z_dl.dataset.data.shape[1]
        dim_z = sim_z_dl.dataset.data.shape[2]

        # build RSA object
        rsa = RSA_nD_tuner(
                epochs=1,
                dim_multiplicity=dim_mult,
                dim_accept_reject=dim_z,
                over_sample_factor=over_sample_factor,
                params_base=params_base,
                sim_observable_dataloader=sim_obs_dl,
                sim_z_dataloader=sim_z_dl,
                sim_fPrel_dataloader=sim_f_dl,
                exp_observable_dataloader=exp_dl,
                print_details=False,
                results_dir=None,
                params_init=params_learn,
                fixed_binning=True,
        )

        optimizer = torch.optim.Adam(rsa.weight_nexus.parameters(), lr=learning_rate)
        grads, loss_grid, (mu, neff) = rsa.RSA_flow(optimizer, line_grid)

        return grads, loss_grid, mu, neff

if __name__ == "__main__":
        all_gradients   = []
        all_loss_grids  = []
        all_mus         = []
        all_neffs       = []

        plt_nm = 4

        a_b_c = line_grid.detach().numpy()
        np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/grad_stability_errors/all_Ns{plt_nm}',
                np.array(Ns, dtype=object))
        np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/grad_stability_errors/all_line_grid{plt_nm}',
                np.array(a_b_c, dtype=object))


        for N_events in Ns:
                print(f'Running {N_events} events...')
                print(f'Number of repeats: {repeat[N_events]}')
                reps = repeat[N_events]
                # prepare (N_events, r) pairs
                tasks = [(N_events, r) for r in range(reps)]

                # with Pool(processes=min(reps, cpu_count())) as pool:
                #         out = pool.map(run_one_repeat, tasks)
				
                out = []
                for task in tasks:
                        out.append(run_one_repeat(task))


                # unzip results
                grads_list, loss_list, mu_list, neff_list = zip(*out)

                all_gradients.append(grads_list)
                all_loss_grids.append(loss_list)
                all_metrics_mu.append(mu_list)
                all_metrics_Neff.append(neff_list)


                np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/grad_stability_errors/all_gradients{plt_nm}',
                        np.array(all_gradients, dtype=object))
                np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/grad_stability_errors/all_loss_grid{plt_nm}',
                        np.array(all_loss_grids, dtype=object))
                np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/grad_stability_errors/all_mu_metrics{plt_nm}',
                        np.array(all_metrics_mu, dtype=object))
                np.save(f'/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/grad_stability_errors/all_Neff_metrics{plt_nm}',
                        np.array(all_metrics_Neff, dtype=object))


