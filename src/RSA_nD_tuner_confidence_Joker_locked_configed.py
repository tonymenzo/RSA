# Fitting distributions for Joker observable RSA_nD_tuner class
# locked parameters
# use case: python RSA_nD_tuner_confidence_Joker_locked_configed.py --config /pscratch/sd/l/ljpuslar/RSA/RSA/src/configs/tuner_confidence_Joker_locked.yaml

import argparse
import importlib
import os
import sys

import numpy as np
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
import torch_optimizer as optim  # unused but kept as in original

import RSA_ND_tuner_Joker_locked
from RSA_ND_tuner_Joker_locked import RSA_nD_tuner


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


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main():
    # ----------------- Parse CLI -----------------
    parser = argparse.ArgumentParser(description="RSA nD Joker tuner with YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     required=True,
    #     help="Random seed value.",
    # )

    args = parser.parse_args()
    cfg = load_config(args.config)
    # seed = args.seed

    # Helper: require key in config (no silent defaults)
    def require(key: str):
        if key not in cfg:
            raise KeyError(f"Missing required config entry '{key}' in {args.config}")
        return cfg[key]

    # Reload RSA module (if edited interactively)
    importlib.reload(RSA_ND_tuner_Joker_locked)

    # ----------------- Basic config -----------------
    nD = int(require("nD"))
    # np.random.seed(int(require("random_seed")))

    # ----------------- Select dataset paths -----------------
    if nD == 2:
        sim_hadrons_PATH       = require("sim_hadrons_path_2D")
        sim_accept_reject_PATH = require("sim_accept_reject_path_2D")
        sim_fPrel_PATH         = require("sim_fPrel_path_2D")
        exp_hadrons_PATH       = require("exp_hadrons_path_2D")
        exp_accept_reject_PATH = require("exp_accept_reject_path_2D")
        sigma_base_cfg         = float(require("sigma_base_2D"))
        loss_type              = require("loss_type_2D")
    elif nD == 3:
        sim_hadrons_PATH       = require("sim_hadrons_path_3D")
        sim_accept_reject_PATH = require("sim_accept_reject_path_3D")
        sim_fPrel_PATH         = require("sim_fPrel_path_3D")
        exp_hadrons_PATH       = require("exp_hadrons_path_3D")
        exp_accept_reject_PATH = require("exp_accept_reject_path_3D")
        sigma_base_cfg         = float(require("sigma_base_3D"))
        loss_type              = require("loss_type_3D")
    else:
        raise ValueError(f"Unsupported nD={nD}, expected 2 or 3.")

    # ----------------- Load arrays -----------------
    exp_hadrons          = np.load(exp_hadrons_PATH, mmap_mode="r")
    sim_hadrons          = np.load(sim_hadrons_PATH, mmap_mode="r")
    sim_accept_reject_np = np.load(sim_accept_reject_PATH, mmap_mode="r")
    sim_fPrel            = np.load(sim_fPrel_PATH, mmap_mode="r")
    exp_accept_reject_np = np.load(exp_accept_reject_PATH, mmap_mode="r")

    N_events_max  = int(require("N_events_max"))
    N_target_max  = min(int(require("N_target_max")), len(exp_hadrons))

    # Harmonise number of events across sim/exp
    # N_total = min(
    #     N_events_max,
    #     N_target_max,
    #     exp_hadrons.shape[0],
    #     sim_hadrons.shape[0],
    #     sim_accept_reject_np.shape[0],
    #     sim_fPrel.shape[0],
    #     exp_accept_reject_np.shape[0],
    # )

    exp_hadrons_temp       = exp_hadrons[:N_target_max]
    sim_hadrons_temp       = sim_hadrons[:N_events_max]
    sim_accept_reject_np   = sim_accept_reject_np[:N_events_max]
    sim_fPrel              = sim_fPrel[:N_events_max]
    exp_accept_reject_np   = exp_accept_reject_np[:N_target_max]
    N_target = N_target_max
    print("N_target:", N_target)

    # ----------------- Observables -----------------
    px_exp, py_exp, pz_exp = exp_hadrons_temp[..., 0], exp_hadrons_temp[..., 1], exp_hadrons_temp[..., 2]
    E_exp, m_exp, pid_exp  = exp_hadrons_temp[..., 3], exp_hadrons_temp[..., 4], exp_hadrons_temp[..., 5]

    px_sim, py_sim, pz_sim = sim_hadrons_temp[..., 0], sim_hadrons_temp[..., 1], sim_hadrons_temp[..., 2]
    E_sim, m_sim, pid_sim  = sim_hadrons_temp[..., 3], sim_hadrons_temp[..., 4], sim_hadrons_temp[..., 5]

    p_mag_exp = np.sqrt(px_exp**2 + py_exp**2 + pz_exp**2)
    p_mag_sim = np.sqrt(px_sim**2 + py_sim**2 + pz_sim**2)

    pion_mask_exp = (pid_exp == 211) | (pid_exp == -211) | (pid_exp == 111)
    pion_mask_sim = (pid_sim == 211) | (pid_sim == -211) | (pid_sim == 111)

    pion_mult_exp = np.sum(pion_mask_exp, axis=1)
    pion_mult_sim = np.sum(pion_mask_sim, axis=1)

    mask = np.abs(exp_hadrons_temp[:, :, 0]) > 0.0
    exp_mult = np.sum(mask, axis=1)
    mask = np.abs(sim_hadrons_temp[:, :, 0]) > 0.0
    sim_mult = np.sum(mask, axis=1)

    p_frac_exp = p_mag_exp / np.sum(p_mag_exp, axis=1, keepdims=True)
    p_frac_sim = p_mag_sim / np.sum(p_mag_sim, axis=1, keepdims=True)

    # Convert accept/reject to torch
    exp_accept_reject = torch.tensor(exp_accept_reject_np.copy(), dtype=torch.float32)
    sim_accept_reject = torch.tensor(sim_accept_reject_np.copy(), dtype=torch.float32)

    # ----------------- z == 1 fix -----------------
    epsilon_z = float(require("epsilon_z"))
    sim_accept_reject[:, :, 2:][sim_accept_reject[:, :, 2:] == 1] = 1 - epsilon_z
    exp_accept_reject[:, :, 2:][exp_accept_reject[:, :, 2:] == 1] = 1 - epsilon_z

    # pT and z
    mask_base = sim_accept_reject[:, :, 5] > 0.0
    mask_sim  = exp_accept_reject[:, :, 5] > 0.0

    px_sim_t = sim_accept_reject[:, :, 3]
    py_sim_t = sim_accept_reject[:, :, 4]
    pT_sim   = torch.sqrt(px_sim_t**2 + py_sim_t**2)

    px_exp_t = exp_accept_reject[:, :, 3]
    py_exp_t = exp_accept_reject[:, :, 4]
    pT_exp   = torch.sqrt(px_exp_t**2 + py_exp_t**2)

    z_accept_sim = sim_accept_reject[:, :, 5]
    z_accept_exp = exp_accept_reject[:, :, 5]

    # Observables definition
    observable = require("observable")
    if observable != "Joker":
        raise ValueError(f"Only 'Joker' observable is implemented here, got {observable}.")

    exp_observable = [pion_mult_exp, pT_exp, z_accept_exp]
    sim_observable = [pion_mult_sim, pT_sim, z_accept_sim]

    N = exp_hadrons_temp.shape[0]
    print("Total N:", N)

    # ----------------- Training hyperparameters -----------------
    repeat             = int(require("repeat"))
    batch_size         = int(require("batch_size"))
    N_events_draw      = int(require("sample_events"))
    epochs             = int(require("epochs"))
    over_sample_factor = float(require("over_sample_factor"))
    learning_rate      = float(require("learning_rate"))
    fixed_binning      = bool(require("fixed_binning"))

    all_params_list  = []
    params_final_list = []
    all_loss_values  = []

    # ----------------- Main training loop -----------------
    for i in range(repeat):
        random_indices = np.random.choice(N, size=N_events_draw, replace=False)

        print(sim_accept_reject.shape)
        sim_accept_reject_t = torch.tensor(sim_accept_reject[random_indices])
        sim_fPrel_t         = torch.tensor(sim_fPrel[random_indices])

        sim_scores = [x[random_indices] for x in sim_observable]
        exp_scores = [x[random_indices] for x in exp_observable]

        # z == 1 protection again for the subset
        sim_accept_reject_t[:, :, 2:][sim_accept_reject_t[:, :, 2:] == 1] = 1 - epsilon_z

        print("Simulated z shape:", sim_accept_reject_t.shape)
        print("Simulated fPrel shape:", sim_fPrel_t.shape)

        sim_scores_ds          = ObservableDatasetJoker(sim_scores)
        sim_accept_reject_ds   = ObservableDataset(sim_accept_reject_t)
        sim_fPrel_ds           = ObservableDataset(sim_fPrel_t)
        exp_scores_ds          = ObservableDatasetJoker(exp_scores)

        sim_observable_dataloader    = DataLoader(sim_scores_ds,          batch_size=batch_size, shuffle=False)
        sim_accept_reject_dataloader = DataLoader(sim_accept_reject_ds,   batch_size=batch_size, shuffle=False)
        sim_fPrel_dataloader         = DataLoader(sim_fPrel_ds,           batch_size=batch_size, shuffle=False)
        exp_observable_dataloader    = DataLoader(exp_scores_ds,          batch_size=batch_size, shuffle=False)

        dim_multiplicity  = sim_accept_reject_dataloader.dataset.data.shape[1]
        dim_accept_reject = sim_accept_reject_dataloader.dataset.data.shape[2]

        print("Each event has been zero-padded to length:", dim_multiplicity)
        print("Each emission has been zero-padded to length:", dim_accept_reject)

        # ----------------- Physics parameters -----------------
        aExtraDQuark  = float(require("aExtraDQuark"))
        aExtraUQuark  = float(require("aExtraUQuark"))
        aExtraSQuark  = float(require("aExtraSQuark"))
        aExtraCquark  = float(require("aExtraCquark"))
        aExtraBquark  = float(require("aExtraBquark"))
        aExtraDiquark = float(require("aExtraDiquark"))

        bNonstandardD = float(require("bNonstandardD"))
        bNonstandardU = float(require("bNonstandardU"))
        bNonstandardS = float(require("bNonstandardS"))
        bNonstandardC = float(require("bNonstandardC"))
        bNonstandardB = float(require("bNonstandardB"))
        bNonstandardH = float(require("bNonstandardH"))

        aLund      = float(require("aLund"))
        bLund      = float(require("bLund"))
        sigma_base = sigma_base_cfg

        aLundD = aLund + aExtraDQuark
        bLundD = bNonstandardD

        aLundU = aLund + aExtraUQuark
        bLundU = bNonstandardU

        aLundS = aLund + aExtraSQuark
        bLundS = bNonstandardS

        aLundDiquark = aLund + aExtraDiquark
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

        epsilon_params = float(require("epsilon_params"))
        if nD == 2:
            params_learn = {
                "a1": torch.tensor(aLundD + epsilon_params),
                "b1": torch.tensor(bLundD + epsilon_params),
                "a2": torch.tensor(aLundU + epsilon_params),
                "b2": torch.tensor(bLundU + epsilon_params),
                "a3": torch.tensor(aLundS + epsilon_params),
                "b3": torch.tensor(bLundS + epsilon_params),
            }
            global_param_groups = {
                "a1": "a", "a2": "a", "a3": "a",
                "b1": "b", "b2": "b", "b3": "b"
            }
        elif nD == 3:
            params_learn = {
                "a1": torch.tensor(aLundD + epsilon_params),
                "b1": torch.tensor(bLundD + epsilon_params),
                "a2": torch.tensor(aLundU + epsilon_params),
                "b2": torch.tensor(bLundU + epsilon_params),
                "a3": torch.tensor(aLundS + epsilon_params),
                "b3": torch.tensor(bLundS + epsilon_params),
                "sigma": torch.tensor(sigma_base + epsilon_params),
            }
            global_param_groups = {
                "a1": "a", "a2": "a", "a3": "a",
                "b1": "b", "b2": "b", "b3": "b",
                "sigma": "sigma"
            }

        # ----------------- RSA instance -----------------



        print("\n" + "="*80)
        print("🔍  MEMORY / SHAPE DIAGNOSTICS BEFORE RSA INITIALIZATION")
        print("="*80)

        # -----------------------------
        # Dataset sizes
        # -----------------------------
        print(f"N_total (events base):       {N}")
        print(f"Batch size:                  {batch_size}")
        print(f"N_events_draw (per iter):    {N_events_draw if 'N_events_draw' in locals() else N_events}")
        print(f"Repeat:                      {repeat}")
        print()

        # -----------------------------
        # Shape of loaded arrays (numpy)
        # -----------------------------
        print("📦 Numpy array shapes:")
        print(f"exp_hadrons_temp:            {exp_hadrons_temp.shape}")
        print(f"sim_hadrons_temp:            {sim_hadrons_temp.shape}")
        print(f"sim_accept_reject_np:        {sim_accept_reject_np.shape if 'sim_accept_reject_np' in locals() else sim_accept_reject.shape}")
        print(f"sim_fPrel:                   {sim_fPrel.shape}")
        print(f"exp_accept_reject_np:        {exp_accept_reject_np.shape if 'exp_accept_reject_np' in locals() else exp_accept_reject.shape}")
        print()

        # -----------------------------
        # Torch tensors (actual memory)
        # -----------------------------
        def tensor_info(name, t):
            print(f"{name:<30} shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")

        print("🔥 Torch tensor diagnostic:")
        tensor_info("sim_accept_reject_t", sim_accept_reject_t)
        tensor_info("sim_fPrel_t",         sim_fPrel_t)

        # Scores
        if isinstance(sim_scores, list):
            for i, s in enumerate(sim_scores):
                tensor_info(f"sim_scores[{i}]", torch.tensor(s) if not torch.is_tensor(s) else s)
        else:
            tensor_info("sim_scores", sim_scores)

        if isinstance(exp_scores, list):
            for i, s in enumerate(exp_scores):
                tensor_info(f"exp_scores[{i}]", torch.tensor(s) if not torch.is_tensor(s) else s)
        else:
            tensor_info("exp_scores", exp_scores)

        print()

        # -----------------------------
        # DataLoader dataset shapes
        # -----------------------------
        print("📁 Dataloader dataset shapes:")
        print("sim_observable_dataloader.dataset:")
        try:
            print(f"  mult:    {sim_observable_dataloader.dataset.mult.shape}")
            print(f"  pT:      {sim_observable_dataloader.dataset.pT.shape}")
            print(f"  z_accept:{sim_observable_dataloader.dataset.z_accept.shape}")
        except:
            pass

        print(f"sim_accept_reject_dataloader.dataset.data shape: {sim_accept_reject_dataloader.dataset.data.shape}")
        print(f"sim_fPrel_dataloader.dataset.data shape:         {sim_fPrel_dataloader.dataset.data.shape}")
        print(f"exp_observable_dataloader.dataset.mult shape:    {exp_observable_dataloader.dataset.mult.shape}")
        print()

        # -----------------------------
        # Padding configuration
        # -----------------------------
        print("🧩 Padding dimensions:")
        print(f"dim_multiplicity (events padded length):  {dim_multiplicity}")
        print(f"dim_accept_reject (per-emission length):  {dim_accept_reject}")
        print()

        # -----------------------------
        # Parameters (base + learn)
        # -----------------------------
        print("⚙️ Parameter dictionaries:")
        print(f"params_base keys:   {list(params_base.keys())}")
        print(f"params_learn keys:  {list(params_learn.keys())}")
        print(f"params_groups:      {global_param_groups}")
        print()

        # -----------------------------
        # Optional GPU memory check
        # -----------------------------
        if torch.cuda.is_available():
            print("💾 CUDA memory usage:")
            print(f"Allocated:   {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"Reserved:    {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
            print(f"Max alloc:   {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
            print()

        print("="*80 + "\n")






        RSA = RSA_nD_tuner(
            epochs=epochs,
            dim_multiplicity=dim_multiplicity,
            dim_accept_reject=dim_accept_reject,
            over_sample_factor=over_sample_factor,
            params_base=params_base,
            sim_observable_dataloader=sim_observable_dataloader,
            sim_z_dataloader=sim_accept_reject_dataloader,
            sim_fPrel_dataloader=sim_fPrel_dataloader,
            exp_observable_dataloader=exp_observable_dataloader,
            print_details=False,
            results_dir=require("results_dir"),
            params_init=params_learn,
            fixed_binning=fixed_binning,
            loss_type=loss_type,
            params_groups=global_param_groups,
        )

        optimizer = torch.optim.Adam(
            RSA.weight_nexus.parameters(),
            lr=learning_rate
        )

        params_final, all_params, loss_values = RSA.RSA_tune(optimizer)

        all_params_list.append(all_params)
        params_final_list.append(params_final)
        all_loss_values.append(loss_values)
        print("Shape of all_params_list:", np.shape(all_params_list))

        # ----------------- Saving -----------------
        if i == 0:
            save_nm = int(require("save_nm"))
            dim = len(set(global_param_groups.values()))

            base_conf_dir = require("confidence_dir_base")
            folder_path = os.path.join(base_conf_dir, f"{dim}D")
            os.makedirs(folder_path, exist_ok=True)

        folder_path = os.path.join(require("confidence_dir_base"), f"{dim}D")
        print("Shape of all_params_list:", np.shape(all_params_list))
        np.save(os.path.join(folder_path, f"all_params_{save_nm}.npy"),
                all_params_list)
        np.save(os.path.join(folder_path, f"params_final_{save_nm}.npy"),
                params_final_list)
        np.save(os.path.join(folder_path, f"loss_values_{save_nm}.npy"),
                all_loss_values)


if __name__ == "__main__":
    main()
