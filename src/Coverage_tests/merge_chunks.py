import re
import numpy as np
from pathlib import Path

def extract_range(name: str):
    # expects ..._it<start>-<end>.npy
    m = re.search(r"_it(\d+)-(\d+)\.npy$", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def load_and_concat(folder: str, prefix: str):
    files = sorted(Path(folder).glob(f"{prefix}_*.npy"))
    ranges = []
    for f in files:
        r = extract_range(f.name)
        if r is not None:
            ranges.append((r[0], r[1], f))

    if not ranges:
        raise RuntimeError(f"No chunk files found for prefix={prefix} in {folder}")

    # sort by start
    ranges.sort(key=lambda x: x[0])

    arrays = []
    expected_start = ranges[0][0]
    for start, end, f in ranges:
        if start != expected_start:
            raise RuntimeError(f"Gap or overlap: expected start {expected_start}, got {start} ({f})")
        arr = np.load(f)
        arrays.append(arr)
        expected_start = end + 1

    return np.concatenate(arrays, axis=0)

def main():
    folder = "/pscratch/.../Tuner_Coverage_ND/Joker/locked/2D"  # set this
    out = Path(folder)

    theta_hat = load_and_concat(folder, "theta_hat")
    mu_t      = load_and_concat(folder, "mu_t")
    sigma_t   = load_and_concat(folder, "sigma_t")
    cov_t     = load_and_concat(folder, "cov_t")
    final_loss= load_and_concat(folder, "final_loss")

    coverage_rate = cov_t.mean(axis=0)

    np.save(out / "theta_hat_merged.npy", theta_hat)
    np.save(out / "mu_t_merged.npy", mu_t)
    np.save(out / "sigma_t_merged.npy", sigma_t)
    np.save(out / "cov_t_merged.npy", cov_t)
    np.save(out / "final_loss_merged.npy", final_loss)

    with open(out / "summary_merged.txt", "w") as f:
        f.write(f"coverage_rate={coverage_rate.tolist()}\n")
        f.write(f"NT={cov_t.shape[0]} NB={theta_hat.shape[1]} D={theta_hat.shape[2]}\n")

    print("Merged. Coverage:", coverage_rate)

if __name__ == "__main__":
    main()
