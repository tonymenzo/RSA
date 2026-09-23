# Single tuning run for RSA_ND_tuner_Joker locked parameters
# Produces one fitted parameter pair (a,b).

import sys
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# Add RSA/RSA/src to PYTHONPATH
# Assumes this script is placed in RSA/RSA/src/Coverage_tests/
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
sys.path.insert(0, str(SRC_DIR))

import importlib
import RSA_ND_tuner_Joker
importlib.reload(RSA_ND_tuner_Joker)
from RSA_ND_tuner_Joker import RSA_nD_tuner


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
SEED = 12345

LOSS_TYPE = "emd"
# Other examples:
# LOSS_TYPE = "chi2"
# LOSS_TYPE = "Joker_nosigma"
# LOSS_TYPE = "Joker_nosigma_chi2"   # only if implemented in RSA_ND_tuner_Joker

MULTIPLICITY_TYPE = "pion"
# Options:
# "pion" : count pi+, pi-, pi0
# "all"  : count all nonzero PID entries

N_TARGET_DRAW = 100_000
N_BASE_DRAW = 30_000

EPOCHS = 75
OVER_SAMPLE_FACTOR = 10.0
LEARNING_RATE = 0.1
FIXED_BINNING = False

EPS_Z = 1e-5
EPS_PARAMS = 1e-5

RUN_NM = 1
OUT_DIR = Path(f"./single_tune_results_{RUN_NM}")


# ------------------------------------------------------------
# Dataset paths
# ------------------------------------------------------------
sim_hadrons_PATH = "/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.04_aU0.04_aS0.04_aC0.04_aB0.04_aH0.97_bD0.88_bU0.88_bS0.88_bC0.88_bB0.88_bH0.98_sigma_0.335_N_5.0e+05_hadrons.npy"

sim_accept_reject_PATH = "/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.04_aU0.04_aS0.04_aC0.04_aB0.04_aH0.97_bD0.88_bU0.88_bS0.88_bC0.88_bB0.88_bH0.98_sigma_0.335_N_5.0e+05_id_mT2_accept_reject_z.npy"

sim_fPrel_PATH = "/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0.04_aU0.04_aS0.04_aC0.04_aB0.04_aH0.97_bD0.88_bU0.88_bS0.88_bC0.88_bB0.88_bH0.98_sigma_0.335_N_5.0e+05_fPrel.npy"

exp_hadrons_PATH = "/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.5e+05_hadrons.npy"

exp_accept_reject_PATH = "/pscratch/sd/l/ljpuslar/RSA/RSA/data/structured_data/pgun_uubar_allhadsigma_a0.68_b0.98_aD0_aU0_aS0_aC0_aB0_aH0.97_bD0.98_bU0.98_bS0.98_bC0.98_bB0.98_bH0.98_sigma_0.335_N_1.5e+05_id_mT2_accept_reject_z.npy"


# ------------------------------------------------------------
# Dataset classes
# ------------------------------------------------------------
class ObservableDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        return self.data[idx]


class ObservableDatasetJoker(Dataset):
    """
    Returns (multiplicity, pT, z_accept) per event.
    """
    def __init__(self, mult: torch.Tensor, pT: torch.Tensor, z_accept: torch.Tensor):
        self.mult = mult
        self.pT = pT
        self.z_accept = z_accept

    def __len__(self):
        return int(self.mult.shape[0])

    def __getitem__(self, idx):
        return self.mult[idx], self.pT[idx], self.z_accept[idx]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def to_device_float(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(device=device, dtype=torch.float32)


def fix_z_equal_one(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Same convention as the coverage script:
    replace entries equal to 1.0 in columns 2 onward by 1-eps.
    """
    x = x.clone()
    x[:, :, 2:][x[:, :, 2:] == 1.0] = 1.0 - eps
    return x


def compute_multiplicity(hadrons_np, multiplicity_type="pion"):
    """
    Compute event multiplicity from padded hadron arrays.

    Assumes PID is stored in column 5.

    multiplicity_type:
        "pion" : count pi+, pi-, pi0
        "all"  : count all nonzero PID entries
    """
    pid = hadrons_np[..., 5]

    if multiplicity_type == "pion":
        mask = (pid == 211) | (pid == -211) | (pid == 111)

    elif multiplicity_type == "all":
        mask = pid != 0

    else:
        raise ValueError(
            f"Unknown multiplicity_type={multiplicity_type}. "
            "Use 'pion' or 'all'."
        )

    return mask.sum(axis=1).astype(np.float32)


def compute_pT_and_z(accrej: torch.Tensor):
    """
    Extract pT and accepted z from accept/reject tensor.

    Expected columns:
        3 -> px
        4 -> py
        5 -> accepted z
    """
    px = accrej[:, :, 3]
    py = accrej[:, :, 4]

    pT = torch.sqrt(px * px + py * py)
    z_acc = accrej[:, :, 5]

    return pT, z_acc


def params_to_vector(params_final):
    """
    Convert representative fitted params to numpy vector.
    In the locked setup we take:
        params_final[0] -> representative a
        params_final[3] -> representative b
    """
    if torch.is_tensor(params_final):
        return params_final.detach().cpu().numpy().astype(np.float64)

    return np.asarray(params_final, dtype=np.float64)


# ------------------------------------------------------------
# Main tuning script
# ------------------------------------------------------------
def main():
    print("============================================================")
    print("Single RSA tuning run")
    print("============================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"loss_type: {LOSS_TYPE}")
    print(f"multiplicity_type: {MULTIPLICITY_TYPE}")

    rng = np.random.default_rng(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Load arrays
    # ------------------------------------------------------------
    print("Loading arrays...")

    exp_hadrons_np = np.load(exp_hadrons_PATH, mmap_mode="r")
    sim_hadrons_np = np.load(sim_hadrons_PATH, mmap_mode="r")

    exp_accept_reject_np = np.load(exp_accept_reject_PATH, mmap_mode="r")
    sim_accept_reject_np = np.load(sim_accept_reject_PATH, mmap_mode="r")
    sim_fPrel_np = np.load(sim_fPrel_PATH, mmap_mode="r")

    N_target_avail = exp_hadrons_np.shape[0]
    N_base_avail = sim_hadrons_np.shape[0]

    if N_TARGET_DRAW > N_target_avail:
        raise ValueError(
            f"N_TARGET_DRAW={N_TARGET_DRAW} > available target events {N_target_avail}"
        )

    if N_BASE_DRAW > N_base_avail:
        raise ValueError(
            f"N_BASE_DRAW={N_BASE_DRAW} > available base events {N_base_avail}"
        )

    # ------------------------------------------------------------
    # Compute full observables once
    # ------------------------------------------------------------
    print("Computing observables...")

    mult_exp_full = compute_multiplicity(
        exp_hadrons_np,
        multiplicity_type=MULTIPLICITY_TYPE,
    )

    mult_sim_full = compute_multiplicity(
        sim_hadrons_np,
        multiplicity_type=MULTIPLICITY_TYPE,
    )

    mult_exp_full_t = torch.from_numpy(mult_exp_full)
    mult_sim_full_t = torch.from_numpy(mult_sim_full)

    exp_accept_reject_full = torch.from_numpy(exp_accept_reject_np.copy()).float()
    sim_accept_reject_full = torch.from_numpy(sim_accept_reject_np.copy()).float()

    exp_accept_reject_full = fix_z_equal_one(exp_accept_reject_full, eps=EPS_Z)
    sim_accept_reject_full = fix_z_equal_one(sim_accept_reject_full, eps=EPS_Z)

    pT_exp_full, z_acc_exp_full = compute_pT_and_z(exp_accept_reject_full)
    pT_sim_full, z_acc_sim_full = compute_pT_and_z(sim_accept_reject_full)

    # ------------------------------------------------------------
    # Draw one target sample and one base sample
    # ------------------------------------------------------------
    print("Drawing target and base samples...")

    idx_t = rng.choice(
        N_target_avail,
        size=N_TARGET_DRAW,
        replace=True,
    )

    idx_b = rng.choice(
        N_base_avail,
        size=N_BASE_DRAW,
        replace=True,
    )

    exp_mult_t = mult_exp_full_t[idx_t]
    exp_pT_t = pT_exp_full[idx_t]
    exp_zacc_t = z_acc_exp_full[idx_t]

    sim_mult_b = mult_sim_full_t[idx_b]
    sim_pT_b = pT_sim_full[idx_b]
    sim_zacc_b = z_acc_sim_full[idx_b]

    sim_acc_b = sim_accept_reject_full[idx_b]
    sim_acc_b = fix_z_equal_one(sim_acc_b, eps=EPS_Z)

    sim_fPrel_b = torch.from_numpy(sim_fPrel_np[idx_b].copy()).float()

    # ------------------------------------------------------------
    # Build dataloaders
    # ------------------------------------------------------------
    batch_size_target = N_TARGET_DRAW
    batch_size_base = N_BASE_DRAW

    exp_ds = ObservableDatasetJoker(
        to_device_float(exp_mult_t, device),
        to_device_float(exp_pT_t, device),
        to_device_float(exp_zacc_t, device),
    )

    sim_ds_obs = ObservableDatasetJoker(
        to_device_float(sim_mult_b, device),
        to_device_float(sim_pT_b, device),
        to_device_float(sim_zacc_b, device),
    )

    sim_ds_acc = ObservableDataset(
        to_device_float(sim_acc_b, device),
    )

    sim_ds_fprel = ObservableDataset(
        to_device_float(sim_fPrel_b, device),
    )

    exp_loader = DataLoader(
        exp_ds,
        batch_size=batch_size_target,
        shuffle=False,
    )

    sim_loader_obs = DataLoader(
        sim_ds_obs,
        batch_size=batch_size_base,
        shuffle=False,
    )

    sim_loader_acc = DataLoader(
        sim_ds_acc,
        batch_size=batch_size_base,
        shuffle=False,
    )

    sim_loader_fprel = DataLoader(
        sim_ds_fprel,
        batch_size=batch_size_base,
        shuffle=False,
    )

    dim_multiplicity = sim_loader_acc.dataset.data.shape[1]
    dim_accept_reject = sim_loader_acc.dataset.data.shape[2]

    print(f"dim_multiplicity: {dim_multiplicity}")
    print(f"dim_accept_reject: {dim_accept_reject}")

    # ------------------------------------------------------------
    # Physics parameters
    # ------------------------------------------------------------
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
        "a0": torch.tensor(0.0),
        "b0": torch.tensor(0.0),

        "a1": torch.tensor(aLundD),
        "b1": torch.tensor(bLundD),

        "a2": torch.tensor(aLundU),
        "b2": torch.tensor(bLundU),

        "a3": torch.tensor(aLundS),
        "b3": torch.tensor(bLundS),

        "a1103": torch.tensor(aLundDiquark),
        "b1103": torch.tensor(bLundDiquark),

        "a2101": torch.tensor(aLundDiquark),
        "b2101": torch.tensor(bLundDiquark),

        "a2103": torch.tensor(aLundDiquark),
        "b2103": torch.tensor(bLundDiquark),

        "a2203": torch.tensor(aLundDiquark),
        "b2203": torch.tensor(bLundDiquark),

        "a3101": torch.tensor(aLundDiquark),
        "b3101": torch.tensor(bLundDiquark),

        "a3103": torch.tensor(aLundDiquark),
        "b3103": torch.tensor(bLundDiquark),

        "a3201": torch.tensor(aLundDiquark),
        "b3201": torch.tensor(bLundDiquark),

        "a3203": torch.tensor(aLundDiquark),
        "b3203": torch.tensor(bLundDiquark),

        "a3303": torch.tensor(aLundDiquark),
        "b3303": torch.tensor(bLundDiquark),

        "sigma": torch.tensor(sigma_base),
    }

    params_learn = {
        "a1": torch.tensor(aLundD + EPS_PARAMS),
        "a2": torch.tensor(aLundU + EPS_PARAMS),
        "a3": torch.tensor(aLundS + EPS_PARAMS),

        "b1": torch.tensor(bLundD + EPS_PARAMS),
        "b2": torch.tensor(bLundU + EPS_PARAMS),
        "b3": torch.tensor(bLundS + EPS_PARAMS),
    }

    global_param_groups = {
        "a1": "a",
        "a2": "a",
        "a3": "a",

        "b1": "b",
        "b2": "b",
        "b3": "b",
    }

    theta_true = np.array([0.68, 0.98], dtype=np.float64)

    # ------------------------------------------------------------
    # Create RSA tuner
    # ------------------------------------------------------------
    print("Initializing RSA tuner...")

    RSA = RSA_nD_tuner(
        epochs=EPOCHS,
        dim_multiplicity=dim_multiplicity,
        dim_accept_reject=dim_accept_reject,
        over_sample_factor=OVER_SAMPLE_FACTOR,
        params_base=params_base,
        sim_observable_dataloader=sim_loader_obs,
        sim_z_dataloader=sim_loader_acc,
        sim_fPrel_dataloader=sim_loader_fprel,
        exp_observable_dataloader=exp_loader,
        params_init=params_learn,
        print_details=False,
        results_dir=str(OUT_DIR),
        fixed_binning=FIXED_BINNING,
        device=device,
        loss_type=LOSS_TYPE,
        params_groups=global_param_groups,
    )

    optimizer = torch.optim.Adam(
    RSA.weight_nexus.parameters(),
    lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",        # reduce LR when loss stops decreasing
        factor=0.5,        # multiply LR by 0.5
        patience=10,       # wait 10 scheduler steps before reducing
        threshold=1e-4,    # minimum relative improvement
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-5,
    )

    # ------------------------------------------------------------
    # Run tuning
    # ------------------------------------------------------------
    print("Starting tuning...")

    params_final, all_params, loss_values = RSA.RSA_tune(optimizer, scheduler=scheduler)

    # In locked setup:
    # params_final order follows params_learn:
    # [a1, a2, a3, b1, b2, b3]
    # Since a1=a2=a3 and b1=b2=b3 are locked, use a1 and b1.
    params_final_ab = params_final[[0, 3]]
    theta_fit = params_to_vector(params_final_ab)

    final_loss = float(loss_values[-1]) if len(loss_values) > 0 else np.nan

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------
    print("\n============================================================")
    print("Single tuning result")
    print("============================================================")
    print(f"loss_type: {LOSS_TYPE}")
    print(f"multiplicity_type: {MULTIPLICITY_TYPE}")
    print(f"N_TARGET_DRAW: {N_TARGET_DRAW}")
    print(f"N_BASE_DRAW: {N_BASE_DRAW}")
    print(f"epochs: {EPOCHS}")
    print("------------------------------------------------------------")
    print(f"theta_true      = {theta_true}")
    print(f"theta_fit [a,b] = {theta_fit}")
    print(f"final_loss      = {final_loss}")
    print("============================================================\n")

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    np.save(OUT_DIR / "theta_fit.npy", theta_fit)
    np.save(OUT_DIR / "params_final_full.npy", params_final.detach().cpu().numpy())
    np.save(OUT_DIR / "all_params.npy", np.array(all_params, dtype=object))
    np.save(OUT_DIR / "loss_values.npy", np.asarray(loss_values, dtype=np.float64))

    with open(OUT_DIR / "single_tune_summary.txt", "w") as f:
        f.write("Single RSA tuning result\n")
        f.write("============================================================\n")
        f.write(f"loss_type: {LOSS_TYPE}\n")
        f.write(f"multiplicity_type: {MULTIPLICITY_TYPE}\n")
        f.write(f"N_TARGET_DRAW: {N_TARGET_DRAW}\n")
        f.write(f"N_BASE_DRAW: {N_BASE_DRAW}\n")
        f.write(f"epochs: {EPOCHS}\n")
        f.write(f"over_sample_factor: {OVER_SAMPLE_FACTOR}\n")
        f.write(f"learning_rate: {LEARNING_RATE}\n")
        f.write(f"fixed_binning: {FIXED_BINNING}\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"theta_true: {theta_true.tolist()}\n")
        f.write(f"theta_fit: {theta_fit.tolist()}\n")
        f.write(f"final_loss: {final_loss}\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"sim_hadrons_PATH: {sim_hadrons_PATH}\n")
        f.write(f"sim_accept_reject_PATH: {sim_accept_reject_PATH}\n")
        f.write(f"sim_fPrel_PATH: {sim_fPrel_PATH}\n")
        f.write(f"exp_hadrons_PATH: {exp_hadrons_PATH}\n")
        f.write(f"exp_accept_reject_PATH: {exp_accept_reject_PATH}\n")

    print(f"Saved outputs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()