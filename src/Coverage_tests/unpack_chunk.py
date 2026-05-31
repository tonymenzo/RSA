#!/usr/bin/env python3
"""
inspect_chunk.py

Load and inspect saved coverage-test outputs from ONE chunk.

Usage examples:
  python inspect_chunk.py --dir /pscratch/.../Coverage_tests/results/run1 --chunk-id 0
  python inspect_chunk.py --dir ./results/run1 --chunk-id 2 --save-json summary_chunk2.json
  python inspect_chunk.py --dir ./results/run1 --chunk-id 0 --pattern "NT25_NB25"
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


PREFIXES = ["theta_hat", "mu_t", "sigma_t", "cov_t", "final_loss"]


def find_chunk_file(
    folder: Path,
    prefix: str,
    chunk_id: int,
    pattern: Optional[str] = None,
) -> Path:
    """
    Find a single .npy file matching:
      <prefix>_*chunk<chunk_id>* .npy
    Optionally also requiring `pattern` to appear in the filename.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    glob_pat = f"{prefix}_*chunk{chunk_id}_*.npy"
    candidates = sorted(folder.glob(glob_pat))

    if pattern is not None:
        candidates = [p for p in candidates if pattern in p.name]

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No files found for prefix='{prefix}', chunk_id={chunk_id} "
            f"in {folder} (glob: {glob_pat})"
        )

    if len(candidates) > 1:
        # Prefer the most specific / latest by sorting; but warn
        print(f"[warn] Multiple matches for {prefix}, chunk {chunk_id}:")
        for c in candidates:
            print(f"  - {c.name}")
        print("[warn] Using the last one (lexicographically).")

    return candidates[-1]


def extract_it_range(filename: str) -> Optional[Tuple[int, int]]:
    """
    Extract it<start>-<end> from filename.
    Example: theta_hat_NT25_NB25_chunk0_it0-4.npy -> (0,4)
    """
    m = re.search(r"_it(\d+)-(\d+)\.npy$", filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def describe_array(name: str, arr: np.ndarray) -> Dict:
    info = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    # Only compute stats for numeric arrays with at least 1 element
    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        finite = np.isfinite(arr)
        info["n_finite"] = int(finite.sum())
        if finite.any():
            vals = arr[finite]
            info["min"] = float(vals.min())
            info["max"] = float(vals.max())
            info["mean"] = float(vals.mean())
            info["std"] = float(vals.std(ddof=0))
        else:
            info["min"] = info["max"] = info["mean"] = info["std"] = None
    return info


def main():
    ap = argparse.ArgumentParser(description="Inspect one coverage-test chunk output.")
    ap.add_argument("--dir", required=True, help="Directory containing chunk .npy files")
    ap.add_argument("--chunk-id", type=int, required=True, help="Chunk id to inspect")
    ap.add_argument(
        "--pattern",
        default=None,
        help="Optional substring to further filter filenames (e.g. 'NT25_NB25')",
    )
    ap.add_argument(
        "--save-json",
        default=None,
        help="Optional path to write a JSON summary of the inspection",
    )
    ap.add_argument(
        "--print-samples",
        action="store_true",
        help="Print a few sample values (useful for quick sanity checks)",
    )
    args = ap.parse_args()

    folder = Path(args.dir).expanduser().resolve()
    chunk_id = args.chunk_id

    loaded: Dict[str, np.ndarray] = {}
    files_used: Dict[str, str] = {}
    it_range: Optional[Tuple[int, int]] = None

    print(f"\nInspecting chunk {chunk_id} in: {folder}\n")

    # Load expected arrays
    for prefix in PREFIXES:
        path = find_chunk_file(folder, prefix, chunk_id, pattern=args.pattern)
        files_used[prefix] = path.name
        arr = np.load(path, allow_pickle=False)
        loaded[prefix] = arr

        if it_range is None:
            it_range = extract_it_range(path.name)

    # Summaries
    summary = {
        "dir": str(folder),
        "chunk_id": chunk_id,
        "it_range": list(it_range) if it_range is not None else None,
        "files_used": files_used,
        "arrays": {},
    }

    for name, arr in loaded.items():
        info = describe_array(name, arr)
        summary["arrays"][name] = info

        print(f"{name}: shape={arr.shape} dtype={arr.dtype}")
        if "mean" in info:
            print(
                f"  finite={info.get('n_finite')}/{arr.size} "
                f"min={info.get('min')} max={info.get('max')} "
                f"mean={info.get('mean')} std={info.get('std')}"
            )

    # Some deeper sanity checks (typical expectations)
    if "theta_hat" in loaded:
        th = loaded["theta_hat"]
        # Expected: (NT_local, NB, D)
        print("\nSanity checks:")
        if th.ndim == 3:
            NT_local, NB, D = th.shape
            print(f"  theta_hat dims: NT_local={NT_local}, NB={NB}, D={D}")
        else:
            print(f"  [warn] theta_hat expected 3D, got {th.ndim}D")

        if "cov_t" in loaded:
            cov = loaded["cov_t"]
            # Expected: (NT_local, D)
            if cov.ndim == 2 and th.ndim == 3 and cov.shape[0] == th.shape[0]:
                cov_rate_local = cov.mean(axis=0)
                print(f"  local coverage rate per dim: {cov_rate_local}")
                summary["local_coverage_rate"] = cov_rate_local.tolist()
            else:
                print("  [warn] cov_t shape doesn't match theta_hat NT_local")

    # Print some samples
    if args.print_samples:
        print("\nSample values:")
        for k in ["mu_t", "sigma_t", "cov_t"]:
            if k in loaded:
                arr = loaded[k]
                print(f"{k}[0]: {arr[0]}")
        if "theta_hat" in loaded:
            th = loaded["theta_hat"]
            print(f"theta_hat[0,0,:]: {th[0,0,:]}")
        if "final_loss" in loaded:
            fl = loaded["final_loss"]
            print(f"final_loss[0,0]: {fl[0,0]}")

    # Save JSON summary if requested
    if args.save_json:
        outp = Path(args.save_json).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote JSON summary to: {outp}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
