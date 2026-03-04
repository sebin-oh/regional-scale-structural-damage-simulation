#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 07/20/2025
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

Regional damage-volume simulation on GPU.

Workflow (per Mw, per SIGMA):
  1) Compute GMPE ln(demand) mean/std and spatial correlation of intra-residuals.
  2) Use capacity correlation from IDA results to build ln(capacity) covariance.
  3) Sample X = ln(C) - ln(D) ~ MVN(X_mean, X_cov) on GPU.
  4) Damage indicator: X < 0. Save:
       - damage_fraction    = # damaged buildings / # total buildings
       - repair_cost        = sum(repair_cost of damaged buildings)

"""


from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import cupy as cp
import utm
from scipy.stats import lognorm

warnings.simplefilter(action="ignore", category=FutureWarning)

from fn_GMPE_parallel_gpu_vec import (
    gmm_CY14,
    gmm_ASK14,
    gmm_BSSA14,
    gmm_CB14,
    intra_residuals_corr_fn_cupy,
)


# =============================================================================
# Config
# =============================================================================
TARGET_REGION = "Milpitas"
TARGET_STRUCTURE = "MultiStory"  # "SingleStory", "TwoStory", "MultiStory", or "All"

SIGMAS = np.round(np.arange(0.0, 1.001, 0.01), 3)
MWS = np.round(np.arange(6.25, 8.51, 0.05), 2)

GMM_NAME = "CY14"  # "CY14", "ASK14", "BSSA14", "CB14"
NUM_SIM = 10_000
BATCH_SIZE = 1024  # GPU-friendly chunking for sampling

DTYPE_NP = np.float64
DTYPE_CP = cp.float64

SOIL_CASE = 1
VS30_MIN = 180.0

# Epicenter (lat, lon)
EPICENTER = (37.666, -122.076)
DEPTH_TO_TOP = DTYPE_NP(3.0)
STRIKE, RAKE, DIP = 325, 180, 90

# Paths
IDA_CSV = Path(f"../data/IDA_results/{TARGET_REGION}/IDA_results_sigma000.csv")
REGION_CSV = Path(f"../data/building_inventories/RegionalInventory_{TARGET_REGION}.csv")
FRAG_DIR = Path(f"../data/fragility_params/{TARGET_REGION}")
OUT_DIR = Path(f"../data/damage_simulation_results/{TARGET_REGION}")

# If True, capacity covariance is fixed from sigma000.
# If False, capacity covariance is rebuilt per-SIGMA using that SIGMA’s sigma (slower).
FIX_CAPACITY_COV_FROM_SIGMA000 = True

# Cholesky stability (only used if FIX_CAPACITY_COV_FROM_SIGMA000=True)
CHOLESKY_JITTER0 = DTYPE_CP(1e-10)
CHOLESKY_MAX_TRIES = 6

# =============================================================================
# GPU utilities
# =============================================================================
def free_gpu() -> None:
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# =============================================================================
# Helpers
# =============================================================================
def structure_mask(df: pd.DataFrame, structure: str) -> np.ndarray:
    ns = df["NumberOfStories"].to_numpy()
    if structure == "SingleStory":
        return ns == 1
    if structure == "TwoStory":
        return ns == 2
    if structure == "MultiStory":
        return ns > 1
    return np.ones(len(df), dtype=bool)


def gmm_dispatch(name: str) -> Callable:
    name = name.upper()
    if name == "CY14":
        return gmm_CY14
    if name == "ASK14":
        return gmm_ASK14
    if name == "BSSA14":
        return gmm_BSSA14
    if name == "CB14":
        return gmm_CB14
    raise ValueError(f"Invalid GMM_NAME: {name!r}")


def sigma_label(sigma: float) -> str:
    return f"{int(round(sigma * 100)):03d}"


def mw_label(mw: float) -> str:
    return f"{int(round(mw * 100)):03d}"


def robust_cholesky(A: cp.ndarray, jitter0: float = CHOLESKY_JITTER0, max_tries: int = CHOLESKY_MAX_TRIES) -> cp.ndarray:
    """Cholesky with diagonal jitter escalation."""
    n = A.shape[0]
    I = cp.eye(n, dtype=A.dtype)
    jitter = jitter0
    for k in range(max_tries):
        try:
            return cp.linalg.cholesky(A + jitter * I)
        except Exception:
            jitter *= 10.0
    raise RuntimeError("Cholesky failed even after jitter escalation. Covariance may be indefinite.")


# =============================================================================
# Load data
# =============================================================================
# IDA capacities: (num_gms, num_bldgs+1 columns) -> keep numeric capacities and transpose to (num_bldgs, num_gms)
bldg_capa = pd.read_csv(IDA_CSV)
bldg_capa = bldg_capa.to_numpy()[:, 1:].astype(DTYPE_NP, copy=False).T

# Drop buildings with any NaN capacity
keep_nonan = ~np.any(np.isnan(bldg_capa), axis=1)

# Region inventory (must align with IDA order)
region_info = pd.read_csv(REGION_CSV)

# Structure filter
keep_structure = structure_mask(region_info, TARGET_STRUCTURE)

# Final keep mask (aligned to original building order)
keep = keep_nonan & keep_structure

# Apply masks
region_info = region_info.loc[keep].reset_index(drop=True)
bldg_capa = bldg_capa[keep, :]

num_bldgs, num_gms = bldg_capa.shape
print(f"Buildings kept: {num_bldgs:,} | GMs: {num_gms:,}")

# =============================================================================
# Capacity correlation from IDA
# =============================================================================
# Avoid -inf if any zeros slip in
ln_capa = np.log(np.clip(bldg_capa, 1e-12, None)).astype(DTYPE_NP, copy=False)
ln_capa_corr_np = np.corrcoef(ln_capa, rowvar=True).astype(DTYPE_NP, copy=False)

# Baseline capacity mean/std (sigma000). Prefer loading a precomputed file;
# otherwise fit lognormal params from IDA capacities.
frag_path = FRAG_DIR / f"frag0_sigma{sigma_label(0.0)}.npy"

frag0 = None

if frag_path.exists():
    try:
        frag0_loaded = np.load(frag_path).astype(DTYPE_NP, copy=False)

        # Align if the file contains the full inventory and you have a keep-mask
        # (safe no-op if already aligned)
        if "keep" in globals():
            keep_mask = np.asarray(keep, dtype=bool)
            if frag0_loaded.shape[0] == keep_mask.size:
                frag0_loaded = frag0_loaded[keep_mask, :]

        frag0 = frag0_loaded

    except Exception as e:
        warnings.warn(
            f"Failed to load baseline frag file {frag_path} ({type(e).__name__}: {e}). "
            "Falling back to fitting from IDA capacities.",
            RuntimeWarning,
        )

if frag0 is None:
    # Fit from IDA capacities (assumes bldg_capa is already aligned to your current building set)
    num_bldgs = bldg_capa.shape[0]
    frag0 = np.empty((num_bldgs, 3), dtype=DTYPE_NP)

    for i in range(num_bldgs):
        temp = bldg_capa[i]
        temp = temp[np.isfinite(temp) & (temp > 0)]

        # Robust fallback if too few samples
        if temp.size < 2:
            temp = np.array([1e-6, 2e-6], dtype=DTYPE_NP)

        shape, loc, scale = lognorm.fit(temp, floc=0)
        frag0[i] = (shape, loc, scale)

    warnings.warn(
        f"{frag_path.name} not found (or could not be loaded).\n"
        "Using lognormal fits from IDA capacities for baseline capacity mean/std. ",
        RuntimeWarning,
    )

ln_capa_mean0_np = np.log(frag0[:, 2]).astype(DTYPE_NP, copy=False)  # mu = log(scale)
ln_capa_std0_np  = frag0[:, 0].astype(DTYPE_NP, copy=False)          # sigma (shape)
del frag0

# Move constant capacity correlation to GPU
ln_capa_corr_cp = cp.asarray(ln_capa_corr_np, dtype=DTYPE_CP)

# Capacity covariance used when FIX_CAPACITY_COV_FROM_SIGMA000=True
ln_capa_std0_cp = cp.asarray(ln_capa_std0_np, dtype=DTYPE_CP)
ln_capa_cov0_cp = cp.outer(ln_capa_std0_cp, ln_capa_std0_cp) * ln_capa_corr_cp

free_gpu()

# =============================================================================
# Site info (UTM + Vs30 + repair cost)
# =============================================================================
easting, northing, _, _ = utm.from_latlon(region_info["Latitude"].to_numpy(), region_info["Longitude"].to_numpy())
easting = easting.astype(DTYPE_NP, copy=False)
northing = northing.astype(DTYPE_NP, copy=False)

vs30 = region_info["Vs30"].to_numpy(dtype=DTYPE_NP, copy=False)
vs30 = np.maximum(vs30, VS30_MIN)

assets = pd.DataFrame({"x": easting, "y": northing, "Vs30": vs30})

bldg_repair_cost_cp = cp.asarray(region_info["RepairCost"].to_numpy(dtype=DTYPE_NP, copy=False), dtype=DTYPE_CP)

# =============================================================================
# Hazard info (epicenter UTM)
# =============================================================================
event_e, event_n, _, _ = utm.from_latlon(*EPICENTER)
event_xy = (DTYPE_NP(event_e), DTYPE_NP(event_n))

# =============================================================================
# Demand correlation (GPU)
# =============================================================================
ln_dmnd_corr_cp = intra_residuals_corr_fn_cupy(
    cp.asarray(assets["x"].to_numpy(), dtype=DTYPE_CP),
    cp.asarray(assets["y"].to_numpy(), dtype=DTYPE_CP),
    soil_case=SOIL_CASE,
).astype(DTYPE_CP)

# =============================================================================
# Output dirs
# =============================================================================
(out_damage_fraction := OUT_DIR / "damage_fraction").mkdir(parents=True, exist_ok=True)
(out_repair_cost := OUT_DIR / "repair_cost").mkdir(parents=True, exist_ok=True)

# =============================================================================
# Main loop
# =============================================================================
gmm_fn = gmm_dispatch(GMM_NAME)
rng = cp.random.default_rng(12345)

t0_all = time.time()

for mw in MWS:
    print(f"\n=== Mw = {mw:.2f} ===")
    t0_mw = time.time()

    eq_info = dict(
        depth_to_top=DEPTH_TO_TOP,
        strike=STRIKE,
        rake=RAKE,
        dip=DIP,
        mechanism="SS",
        region="california",
        on_hanging_wall=False,
        vs_source="inferred",
        Mw=float(mw),
    )

    # GMPE ln(demand) mean/std on CPU, then to GPU
    ln_dmnd_mean_np, ln_dmnd_std_np = gmm_fn(assets, event_xy, eq_info)
    ln_dmnd_mean_cp = cp.asarray(ln_dmnd_mean_np, dtype=DTYPE_CP)
    ln_dmnd_std_cp = cp.asarray(ln_dmnd_std_np, dtype=DTYPE_CP)

    # Demand covariance for this Mw
    ln_dmnd_cov_cp = cp.outer(ln_dmnd_std_cp, ln_dmnd_std_cp) * ln_dmnd_corr_cp

    # If we keep capacity covariance fixed, X_cov is fixed across SIGMAS -> precompute Cholesky once.
    if FIX_CAPACITY_COV_FROM_SIGMA000:
        X_cov_cp = ln_dmnd_cov_cp + ln_capa_cov0_cp
        L_cp = robust_cholesky(X_cov_cp)
        LT_cp = L_cp.T
    else:
        L_cp = None
        LT_cp = None

    damage_fraction_cp = cp.empty((NUM_SIM,), dtype=DTYPE_CP)
    repair_cost_cp = cp.empty((NUM_SIM,), dtype=DTYPE_CP)

    for k, sigma in enumerate(SIGMAS):
        t0_sigma = time.time()
        sigma_lab = sigma_label(float(sigma))
        mw_lab = mw_label(float(mw))

        frag = np.load(FRAG_DIR / f"frag0_sigma{sigma_lab}.npy").astype(DTYPE_NP, copy=False)
        frag = frag[keep, :]

        ln_capa_mean_cp = cp.asarray(np.log(frag[:, 2]), dtype=DTYPE_CP)

        # X_mean changes with SIGMA
        X_mean_cp = ln_capa_mean_cp - ln_dmnd_mean_cp

        # Optionally rebuild capacity covariance per SIGMA (slower)
        if not FIX_CAPACITY_COV_FROM_SIGMA000:
            ln_capa_std_cp = cp.asarray(frag[:, 0], dtype=DTYPE_CP)
            ln_capa_cov_cp = cp.outer(ln_capa_std_cp, ln_capa_std_cp) * ln_capa_corr_cp
            X_cov_cp = ln_dmnd_cov_cp + ln_capa_cov_cp
            L_cp = robust_cholesky(X_cov_cp)
            LT_cp = L_cp.T

        # --- Sample in batches, compute damage_fraction and repair_cost on GPU ---
        invalid_total = 0

        for s0 in range(0, NUM_SIM, BATCH_SIZE):
            bs = min(BATCH_SIZE, NUM_SIM - s0)

            Z = rng.standard_normal((bs, num_bldgs), dtype=DTYPE_CP)
            X = Z @ LT_cp + X_mean_cp[None, :]  # (bs, num_bldgs)

            valid = cp.isfinite(X)
            damaged = (X < 0) & valid

            valid_cnt = valid.sum(axis=1)
            damaged_cnt = damaged.sum(axis=1)

            # avoid divide-by-zero if a row is entirely invalid
            valid_cnt = cp.maximum(valid_cnt, 1)

            damage_fraction_cp[s0 : s0 + bs] = damaged_cnt / valid_cnt
            repair_cost_cp[s0 : s0 + bs] = (bldg_repair_cost_cp[None, :] * damaged).sum(axis=1)

            invalid_total += int((~valid).sum().get())

            # free temporary batch arrays
            del Z, X, valid, damaged, valid_cnt, damaged_cnt
            free_gpu()

        if invalid_total > 0.01 * (NUM_SIM * num_bldgs):
            print(f"Warning: many invalid samples ({invalid_total:,}) for Mw={mw:.2f}, sigma={sigma:.3f}")

        # Save
        damage_fraction_np = cp.asnumpy(damage_fraction_cp)
        repair_cost_np = cp.asnumpy(repair_cost_cp)

        np.save(out_damage_fraction / f"{TARGET_STRUCTURE}_damage_fraction_Mw{mw_lab}_sigma{sigma_lab}_{TARGET_REGION}_{GMM_NAME}.npy", damage_fraction_np)
        np.save(out_repair_cost / f"{TARGET_STRUCTURE}_repair_cost_Mw{mw_lab}_sigma{sigma_lab}_{TARGET_REGION}_{GMM_NAME}.npy", repair_cost_np)

        if k % 25 == 0:
            print(f"  sigma={sigma:.3f} done in {time.time() - t0_sigma:.2f}s")

        del frag, ln_capa_mean_cp, X_mean_cp, damage_fraction_np, repair_cost_np
        free_gpu()

    # cleanup per Mw
    del ln_dmnd_mean_cp, ln_dmnd_std_cp, ln_dmnd_cov_cp, X_cov_cp, L_cp, LT_cp
    free_gpu()

    print(f"=== Mw {mw:.2f} done in {time.time() - t0_mw:.2f}s ===")

print(f"\nAll done in {time.time() - t0_all:.2f}s")