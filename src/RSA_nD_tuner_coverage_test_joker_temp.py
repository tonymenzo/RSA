# Coverage test for Joker observable RSA_nD_tuner class (locked parameters)
# Nested bootstrap:
#   Outer: bootstrap target (NT)
#   Inner: bootstrap base   (NB)
# Produces empirical coverage for CI built from base-bootstrapped fits.

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import importlib
import RSA_ND_tuner_Joker_locked
importlib.reload(RSA_ND_tuner_Joker_locked)
from RSA_ND_tuner_Joker_locked import RSA_nD_tuner


# -----------------------------
# Datasets
# -----------------------------
class ObservableDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        return self.data[idx]


class ObservableDatasetJoker(Dataset):
    """
    Returns (mult, pT, z_accept) per event.
    """
    def __init__(self, mult: torch.Tensor, pT: torch.Tensor, z_accept: torch.Tensor):
        self.mult = mult
        self.pT = pT
        self.z_accept = z_accept

    def __len__(self):
        return int(self.mult.shape[0])

    def __getitem__(self, idx):
        return (self.mult[idx], self.pT[idx], self.z_accept[idx])


# -----------------------------
# Helpers
# -----------------------------
def fix_z_equal_one(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Do not touch pid columns; you’re already applying on [:,:,2:] so OK.
    x = x.clone()
    x[:, :, 2:][x[:, :, 2:] == 1.0] = 1.0 - eps
    return x


def to_device_float(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(device=device, dtype=torch.float32)


def params_to_vector(params_final, group_order):
    """
    Convert params_final (whatever RSA returns) into a vector in a stable order.
    params_final contains all parameters, but we only care about the ones in group_order (e.g. 'a', 'b').
    Supports:
      - dict-like: {'a':..., 'b':...} or {'a1':..., ...}
      - list/tuple/np.ndarray/torch.Tensor
    """
    if isinstance(params_final, dict):
        # Expect grouped keys like 'a', 'b', 'sigma' OR raw keys.
        vec = []
        for k in group_order:
            if k in params_final:
                v = params_final[k]
            else:
                # allow mapping from group label to some representative raw key
                # (if needed you can customize this)
                raise KeyError(f"params_final missing key '{k}'. Keys: {list(params_final.keys())}")
            if torch.is_tensor(v):
                vec.append(float(v.detach().cpu().item()))
            else:
                vec.append(float(v))
        return np.array(vec, dtype=np.float64)

    if torch.is_tensor(params_final):
        return params_final.detach().cpu().numpy().astype(np.float64)

    if isinstance(params_final, (list, tuple, np.ndarray)):
        return np.array(params_final, dtype=np.float64)

    raise TypeError(f"Unsupported params_final type: {type(params_final)}")


# -----------------------------
# Main
# -----------------------------
def main():
    # -----------------------------
    # Paths (your 2D dataset)
    # -----------------------------
    sim_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.04_aU0.04_aS0.04_aC0.04_aB0.04_aH0.97_bD0.88_bU0.88_bS0.88_bC0.88_bB0.88_bH0.98_sigma_0.335_N_5.0e+05_hadrons.npy'
    sim_accept_reject_PATH = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.04_aU0.04_aS0.04_aC0.04_aB0.04_aH0.97_bD0.88_bU0.88_bS0.88_bC0.88_bB0.88_bH0.98_sigma_0.335_N_5.0e+05_id_mT2_accept_reject_z.npy'
    sim_fPrel_PATH         = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.04_aU0.04_aS0.04_aC0.04_aB0.04_aH0.97_bD0.88_bU0.88_bS0.88_bC0.88_bB0.88_bH0.98_sigma_0.335_N_5.0e+05_fPrel.npy'

    exp_hadrons_PATH       = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.5e+05_hadrons.npy'
    exp_accept_reject_PATH = '/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.5e+05_id_mT2_accept_reject_z.npy'

    # -----------------------------
    # Coverage-test knobs
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NT = 25          # bootstrap targets
    NT = 25          # bootstrap targets
    NB = 25          # bootstrap bases
    k_sigma = 1.0    # 1σ interval; set 2.0 for 2σ
    eps_z = 1e-5

    # Choose bootstrap sample sizes (usually = original sample sizes)
    N_target_draw = 30_000   # number of target events per pseudo-experiment
    N_base_draw   = 30_000   # number of base events per base-bootstrap fit
    # N_target_draw = 30   # number of target events per pseudo-experiment
    # N_base_draw   = 30   # number of base events per base-bootstrap fit

    # RSA hyperparams
    epochs = 75
    batch_size = 30_000
    # batch_size = 30
    over_sample_factor = 10.0
    learning_rate = 0.1
    fixed_binning = True
    loss_type = "Joker_nosigma"

    # Random seed for reproducibility
    rng = np.random.default_rng(43)

    # -----------------------------
    # Load arrays
    # -----------------------------
    exp_hadrons_np       = np.load(exp_hadrons_PATH, mmap_mode="r")
    sim_hadrons_np       = np.load(sim_hadrons_PATH, mmap_mode="r")
    sim_accept_reject_np = np.load(sim_accept_reject_PATH, mmap_mode="r")
    sim_fPrel_np         = np.load(sim_fPrel_PATH, mmap_mode="r")
    exp_accept_reject_np = np.load(exp_accept_reject_PATH, mmap_mode="r")

    # #temp: access only small slices to check loading (remove later):
    # N_temp = 10_000
    # exp_hadrons_np = exp_hadrons_np[:N_temp]
    # sim_hadrons_np = sim_hadrons_np[:N_temp]
    # sim_accept_reject_np = sim_accept_reject_np[:N_temp]
    # sim_fPrel_np = sim_fPrel_np[:N_temp]
    # exp_accept_reject_np = exp_accept_reject_np[:N_temp]

    N_target_avail = exp_hadrons_np.shape[0]
    N_base_avail   = sim_hadrons_np.shape[0]

    if N_target_draw > N_target_avail:
        raise ValueError(f"N_target_draw={N_target_draw} > available target events {N_target_avail}")
    if N_base_draw > N_base_avail:
        raise ValueError(f"N_base_draw={N_base_draw} > available base events {N_base_avail}")

    # -----------------------------
    # Precompute EXP and SIM observables (full arrays),
    # then we just index them in the bootstraps.
    # -----------------------------
    def compute_pion_mult(hadrons_np):
        pid = hadrons_np[..., 5]
        pion_mask = (pid == 211) | (pid == -211) | (pid == 111)
        return pion_mask.sum(axis=1).astype(np.float32)  # (N,)

    pion_mult_exp_full = compute_pion_mult(exp_hadrons_np)
    pion_mult_sim_full = compute_pion_mult(sim_hadrons_np)

    # Convert accept/reject to torch (full), fix z==1 once globally
    exp_accept_reject_full = torch.from_numpy(exp_accept_reject_np.copy()).float()
    sim_accept_reject_full = torch.from_numpy(sim_accept_reject_np.copy()).float()

    exp_accept_reject_full = fix_z_equal_one(exp_accept_reject_full, eps=eps_z)
    sim_accept_reject_full = fix_z_equal_one(sim_accept_reject_full, eps=eps_z)

    # pT and accepted z from accept/reject full tensors
    def compute_pT_and_z(accrej: torch.Tensor):
        px = accrej[:, :, 3]
        py = accrej[:, :, 4]
        pT = torch.sqrt(px * px + py * py)            # (N, T)
        z_acc = accrej[:, :, 5]                       # (N, T)
        return pT, z_acc

    pT_exp_full, z_acc_exp_full = compute_pT_and_z(exp_accept_reject_full)
    pT_sim_full, z_acc_sim_full = compute_pT_and_z(sim_accept_reject_full)

    # Convert multiplicities to torch
    pion_mult_exp_full_t = torch.from_numpy(pion_mult_exp_full)  # (N,)
    pion_mult_sim_full_t = torch.from_numpy(pion_mult_sim_full)

    # -----------------------------
    # Physics parameters (base + learn), grouped into {a,b} for 2D locked case
    # -----------------------------
    aExtraDQuark = 0.04
    aExtraUQuark = 0.04
    aExtraSQuark = 0.04
    aExtraDiquark = 0.97

    bNonstandardD = 0.88
    bNonstandardU = 0.88
    bNonstandardS = 0.88
    bNonstandardH = 0.98

    aLund = 0.68
    bLund = 0.98
    sigma_base = 0.335

    aLundD = aLund + aExtraDQuark
    aLundU = aLund + aExtraUQuark
    aLundS = aLund + aExtraSQuark
    aLundDiquark = aLund + aExtraDiquark

    bLundD = bNonstandardD
    bLundU = bNonstandardU
    bLundS = bNonstandardS
    bLundDiquark = bNonstandardH

    params_base = {
        "a0": torch.tensor(0.0), "b0": torch.tensor(0.0),
        "a1": torch.tensor(aLundD), "b1": torch.tensor(bLundD),
        "a2": torch.tensor(aLundU), "b2": torch.tensor(bLundU),
        "a3": torch.tensor(aLundS), "b3": torch.tensor(bLundS),
        "a1103": torch.tensor(aLundDiquark), "b1103": torch.tensor(bLundDiquark),
        "a2101": torch.tensor(aLundDiquark), "b2101": torch.tensor(bLundDiquark),
        "a2103": torch.tensor(aLundDiquark), "b2103": torch.tensor(bLundDiquark),
        "a2203": torch.tensor(aLundDiquark), "b2203": torch.tensor(bLundDiquark),
        "a3101": torch.tensor(aLundDiquark), "b3101": torch.tensor(bLundDiquark),
        "a3103": torch.tensor(aLundDiquark), "b3103": torch.tensor(bLundDiquark),
        "a3201": torch.tensor(aLundDiquark), "b3201": torch.tensor(bLundDiquark),
        "a3203": torch.tensor(aLundDiquark), "b3203": torch.tensor(bLundDiquark),
        "a3303": torch.tensor(aLundDiquark), "b3303": torch.tensor(bLundDiquark),
        "sigma": torch.tensor(sigma_base),
    }

    eps_params = 1e-5
    params_learn = {
        "a1": torch.tensor(aLundD + eps_params),
        "a2": torch.tensor(aLundU + eps_params),
        "a3": torch.tensor(aLundS + eps_params),
        "b1": torch.tensor(bLundD + eps_params),
        "b2": torch.tensor(bLundU + eps_params),
        "b3": torch.tensor(bLundS + eps_params),
    }
    global_param_groups = {
        "a1": "a", "a2": "a", "a3": "a",
        "b1": "b", "b2": "b", "b3": "b",
    }
    group_order = ["a", "b"]  # stable order for vectors

    # -----------------------------
    # Define "truth" for coverage
    # -----------------------------
    #   aD=aU=aS=0 and bD=bU=bS=0.98
    #   a_true = aLund + 0.00 = 0.68
    #   b_true = 0.98
    theta_true = np.array([0.68, 0.98], dtype=np.float64)

    # -----------------------------
    # Output containers
    # -----------------------------
    # Store all fitted params: (NT, NB, D)
    D = len(group_order)
    theta_hat = np.zeros((NT, NB, D), dtype=np.float64)
    mu_t = np.zeros((NT, D), dtype=np.float64)
    sig_t = np.zeros((NT, D), dtype=np.float64)
    cov_t = np.zeros((NT, D), dtype=np.int32)

    # Optional: keep loss curves if you want (can be huge); store only final loss here
    final_loss = np.zeros((NT, NB), dtype=np.float64)

    # -----------------------------
    # Nested bootstrap
    # -----------------------------
    for it in range(NT):
        # --- bootstrap target (with replacement)
        idx_t = rng.choice(N_target_avail, size=N_target_draw, replace=False)

        # Build target observables dataloader ONCE per it
        exp_mult_t   = pion_mult_exp_full_t[idx_t]
        exp_pT_t     = pT_exp_full[idx_t]
        exp_zacc_t   = z_acc_exp_full[idx_t]
        exp_ds = ObservableDatasetJoker(
            to_device_float(exp_mult_t, device),
            to_device_float(exp_pT_t, device),
            to_device_float(exp_zacc_t, device),
        )
        exp_loader = DataLoader(exp_ds, batch_size=batch_size, shuffle=False)

        # Inner loop: bootstrap base NB times
        for ib in range(NB):
            # --- bootstrap base (with replacement)
            idx_b = rng.choice(N_base_avail, size=N_base_draw, replace=False)

            # Base accept/reject + fPrel
            sim_acc_t = sim_accept_reject_full[idx_b]
            sim_acc_t = fix_z_equal_one(sim_acc_t, eps=eps_z)  # safe, though already fixed globally
            sim_fPrel_t = torch.from_numpy(sim_fPrel_np[idx_b].copy()).float()

            # Base Joker observables
            sim_mult_b = pion_mult_sim_full_t[idx_b]
            sim_pT_b   = pT_sim_full[idx_b]
            sim_zacc_b = z_acc_sim_full[idx_b]

            # Move to device
            sim_ds_obs = ObservableDatasetJoker(
                to_device_float(sim_mult_b, device),
                to_device_float(sim_pT_b, device),
                to_device_float(sim_zacc_b, device),
            )
            sim_loader_obs = DataLoader(sim_ds_obs, batch_size=batch_size, shuffle=False)

            sim_ds_acc = ObservableDataset(to_device_float(sim_acc_t, device))
            sim_loader_acc = DataLoader(sim_ds_acc, batch_size=batch_size, shuffle=False)

            sim_ds_fprel = ObservableDataset(to_device_float(sim_fPrel_t, device))
            sim_loader_fprel = DataLoader(sim_ds_fprel, batch_size=batch_size, shuffle=False)

            # Dimensions inferred from accept/reject tensor
            dim_multiplicity  = sim_loader_acc.dataset.data.shape[1]
            dim_accept_reject = sim_loader_acc.dataset.data.shape[2]

            # RSA instance
            RSA = RSA_nD_tuner(
                epochs=epochs,
                dim_multiplicity=dim_multiplicity,
                dim_accept_reject=dim_accept_reject,
                over_sample_factor=over_sample_factor,
                params_base=params_base,
                sim_observable_dataloader=sim_loader_obs,
                sim_z_dataloader=sim_loader_acc,
                sim_fPrel_dataloader=sim_loader_fprel,
                exp_observable_dataloader=exp_loader,
                print_details=False,
                results_dir="/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Tuner_nD",
                params_init=params_learn,
                fixed_binning=fixed_binning,
                loss_type=loss_type,
                params_groups=global_param_groups,
            )

            optimizer = torch.optim.Adam(RSA.weight_nexus.parameters(), lr=learning_rate)

            params_final, all_params, loss_values = RSA.RSA_tune(optimizer)
            #temp ugly solution: all a_x and b_x should be exactly the same within each group, so just take the first one as representative for the vector. 
            # This is because we defined group_order = ['a', 'b'] and global_param_groups maps all a_i to 'a' and all b_i to 'b'.
            params_final = params_final[[0,3]] # take a1 and b1 as representatives for 'a' and 'b' groups, respectively.
            # print(f"Finished fit for it={it+1}/{NT} ib={ib+1}/{NB}")
            # print(params_final.shape)
            # print(params_final)

            # Convert params_final to stable vector [a, b]
            theta_hat[it, ib, :] = params_to_vector(params_final, group_order=group_order)

            # Save final loss (optional)
            try:
                final_loss[it, ib] = float(loss_values[-1])
            except Exception:
                final_loss[it, ib] = np.nan

            print(f"[it={it+1}/{NT}] [ib={ib+1}/{NB}] theta_hat={theta_hat[it, ib, :]} final_loss={final_loss[it, ib]:.6g}")

        # After NB fits for this target bootstrap: compute μ_t and σ_t over ib
        mu_t[it, :] = theta_hat[it, :, :].mean(axis=0)
        sig_t[it, :] = theta_hat[it, :, :].std(axis=0, ddof=1)

        # CI and coverage (per dimension)
        lo = mu_t[it, :] - k_sigma * sig_t[it, :]
        hi = mu_t[it, :] + k_sigma * sig_t[it, :]
        cov_t[it, :] = ((theta_true >= lo) & (theta_true <= hi)).astype(np.int32)

        print(f"--- Target bootstrap it={it+1}: mu={mu_t[it,:]}, sigma={sig_t[it,:]}, cov={cov_t[it,:]}")


        #temp save intermediate results after each target bootstrap (can be large, but useful for debugging and analysis if something crashes later; also you can remove the large arrays if you just want the final coverage rates and losses)
        # Empirical coverage (per dimension)
        coverage_rate = cov_t.mean(axis=0)
        print("\n============================================================")
        print(f"Empirical coverage for k={k_sigma}σ: {coverage_rate}  (per parameter)")
        print("============================================================\n")

        # -----------------------------
        # Save results
        # -----------------------------
        save_nm = 1
        out_dir = f"/pscratch/sd/l/ljpuslar/RSA/RSA/src/temp_results/Tuner_Coverage_ND/Joker/locked/{D}D"
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, f"theta_hat_NT{NT}_NB{NB}_k{k_sigma}_{save_nm}.npy"), theta_hat)
        np.save(os.path.join(out_dir, f"mu_t_NT{NT}_NB{NB}_k{k_sigma}_{save_nm}.npy"), mu_t)
        np.save(os.path.join(out_dir, f"sigma_t_NT{NT}_NB{NB}_k{k_sigma}_{save_nm}.npy"), sig_t)
        np.save(os.path.join(out_dir, f"cov_t_NT{NT}_NB{NB}_k{k_sigma}_{save_nm}.npy"), cov_t)
        np.save(os.path.join(out_dir, f"final_loss_NT{NT}_NB{NB}_k{k_sigma}_{save_nm}.npy"), final_loss)

    # Also save a tiny summary text
    with open(os.path.join(out_dir, f"summary_NT{NT}_NB{NB}_k{k_sigma}_{save_nm}.txt"), "w") as f:
        f.write(f"NT={NT}\nNB={NB}\nk_sigma={k_sigma}\n")
        f.write(f"theta_true={theta_true.tolist()}\n")
        f.write(f"coverage_rate={coverage_rate.tolist()}\n")
        f.write(f"number of target events drawn per bootstrap: {N_target_draw}\n")
        f.write(f"number of base events drawn per bootstrap: {N_base_draw}\n")
        f.write(f"parameter groups: {global_param_groups}\n")
        f.write(f"=============================================================\n")
        f.write(f"RSA hyperparameters:\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"dim_multiplicity={dim_multiplicity}\n")
        f.write(f"dim_accept_reject={dim_accept_reject}\n")
        f.write(f"over_sample_factor={over_sample_factor}\n")
        f.write(f"learning_rate={learning_rate}\n")
        f.write(f"loss_type={loss_type}\n")
        f.write(f"=============================================================\n")
        f.write('Parameter values at base:\n')
        for name, value in params_base.items():
            f.write(f"  {name}: {value.item()}\n")
        f.write('Parameter values at target:\n')
        for value in theta_true:
            f.write(f"  {value}\n")
        f.write(f"=============================================================\n")
        f.write(f"dataset_paths:\n")
        f.write(f"sim_hadrons_PATH={sim_hadrons_PATH}\n")
        f.write(f"sim_accept_reject_PATH={sim_accept_reject_PATH}\n")
        f.write(f"sim_fPrel_PATH={sim_fPrel_PATH}\n")
        f.write(f"exp_hadrons_PATH={exp_hadrons_PATH}\n")
        f.write(f"exp_accept_reject_PATH={exp_accept_reject_PATH}\n")
        f.write(f"=============================================================\n")
        


if __name__ == "__main__":
    main()
