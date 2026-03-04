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
SIGMA_LABEL_SCALE = 100  # sigma=0.25 -> "025"


# =============================================================================
# Helpers
# =============================================================================
def sigma_label(sigma: float) -> str:
    return f"{int(round(float(sigma) * SIGMA_LABEL_SCALE)):03d}"


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


def drop_nan_buildings(capa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop buildings (rows) with any NaN and return:
      - filtered capacities
      - keep mask (True for retained buildings)
    """
    has_nan = np.any(~np.isfinite(capa), axis=1)
    keep = ~has_nan
    return capa[keep, :], keep


def fit_lognormal_params(capa: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Fit lognormal(shape, loc, scale) per building from capacity samples.

    Returns array of shape (n_bldgs, 3) = [shape, loc, scale].
    """
    n_bldgs = capa.shape[0]
    out = np.empty((n_bldgs, 3), dtype=dtype)

    for i in range(n_bldgs):
        x = capa[i]
        x = x[np.isfinite(x) & (x > 0)]

        # Robust fallback if too few samples
        if x.size < 2:
            x = np.array([1e-6, 2e-6], dtype=dtype)

        shape, loc, scale = lognorm.fit(x, floc=0)
        out[i] = (dtype(shape), dtype(loc), dtype(scale))

    return out


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load capacities and drop NaN buildings
    capa = load_capacities(IDA_CSV, DTYPE_NP)
    capa, keep_mask = drop_nan_buildings(capa)

    print(f"Loaded capacities: {capa.shape[0]:,} buildings x {capa.shape[1]:,} GMs")
    if not np.all(keep_mask):
        print(f"Dropped {np.sum(~keep_mask):,} buildings due to NaNs")

    # 2) Fit baseline fragility params (sigma000)
    frag0 = fit_lognormal_params(capa, DTYPE_NP)

    # Optionally save baseline too (useful for consistency)
    np.save(OUT_DIR / "frag_params_sigma000.npy", frag0)

    # 3) Generate sigma-perturbed variants by scaling the lognormal scale parameter
    for k, sigma in enumerate(SIGMAS, start=1):
        lab = sigma_label(sigma)
        print(f"{k}/{len(SIGMAS)}  sigma={sigma:.3f} -> {lab}")

        frag = frag0.copy()

        if sigma > 0:
            noise = rng.lognormal(mean=0.0, sigma=float(sigma), size=frag.shape[0])
            frag[:, 2] *= noise  # perturb scale only

        np.save(OUT_DIR / f"frag_params_sigma{lab}.npy", frag)


if __name__ == "__main__":
    main()