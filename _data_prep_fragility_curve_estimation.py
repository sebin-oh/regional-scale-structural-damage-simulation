#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 07/20/2026
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

Generate baseline lognormal fragility parameters from IDA capacities,
then create sigma-perturbed variants by multiplying the (scale) parameter
by a lognormal noise factor.

"""


from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm


# =============================================================================
# Settings
# =============================================================================
TARGET_REGION = "Milpitas"

IDA_CSV = Path(f"../data/IDA_results/{TARGET_REGION}/IDA_results_sigma000.csv")
OUT_DIR = Path(f"../data/fragility_params/{TARGET_REGION}")

DTYPE_NP = np.float64
SEED = 42

SIGMAS = np.round(np.arange(0.0, 1.001, 0.01), 3)  # lognormal sigma values


# =============================================================================
# Helpers
# =============================================================================
def sigma_label(sigma: float) -> str:
    return f"{int(round(float(sigma) * 100)):03d}"


def load_capacities(csv_path: Path, dtype: np.dtype) -> np.ndarray:
    """
    Load IDA results CSV and return capacities as (n_bldgs, n_gms).
    Assumes first column is GM_ID (or similar) and remaining are capacities.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"IDA CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    capa = df.to_numpy()[:, 1:].astype(dtype, copy=False).T
    return capa


def fit_lognormal_params(capa: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Fit lognormal(shape, loc, scale) per building from capacity samples.

    Rule:
      - If a building has ANY non-finite value in its capacity row, return [NaN, NaN, NaN].
      - Else fit using positive samples; if too few positive samples, return NaNs.

    Returns array of shape (n_bldgs, 3) = [shape, loc, scale].
    """
    n_bldgs = capa.shape[0]
    out = np.full((n_bldgs, 3), np.nan, dtype=dtype)

    for i in range(n_bldgs):
        row = capa[i]

        # Keep building, but mark frag params as NaN if any capacity value is NaN/inf
        if not np.all(np.isfinite(row)):
            continue

        x = row[row > 0]
        if x.size < 2:
            continue

        shape, loc, scale = lognorm.fit(x, floc=0)
        out[i] = (dtype(shape), dtype(loc), dtype(scale))

    return out


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load capacities (do NOT drop buildings)
    capa = load_capacities(IDA_CSV, DTYPE_NP)
    n_bldgs, n_gms = capa.shape
    print(f"Loaded capacities: {n_bldgs:,} buildings x {n_gms:,} GMs")

    # 2) Fit baseline fragility params (sigma000), with NaN rows preserved
    frag0 = fit_lognormal_params(capa, DTYPE_NP)

    n_nan = int(np.sum(~np.isfinite(frag0[:, 2])))
    if n_nan > 0:
        print(f"Baseline frag params: {n_nan:,}/{n_bldgs:,} buildings set to NaN (due to NaN/inf or insufficient samples).")

    np.save(OUT_DIR / "frag_params_sigma000.npy", frag0)

    # 3) Generate sigma-perturbed variants by scaling the lognormal scale parameter
    for k, sigma in enumerate(SIGMAS, start=1):
        lab = sigma_label(sigma)
        print(f"{k}/{len(SIGMAS)}  sigma={sigma:.3f} -> {lab}")

        frag = frag0.copy()

        if sigma > 0:
            noise = rng.lognormal(mean=0.0, sigma=float(sigma), size=frag.shape[0]).astype(DTYPE_NP, copy=False)

            # Only apply noise where baseline scale is finite; NaNs stay NaN
            mask = np.isfinite(frag[:, 2])
            frag[mask, 2] *= noise[mask]

        np.save(OUT_DIR / f"frag_params_sigma{lab}.npy", frag)


if __name__ == "__main__":
    main()