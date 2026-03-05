#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 02/09/2026
@Author       :
@Contact      : sebin.oh@berkeley.edu
@Description  : 
Compute (per structural-diversity sigma):
  - Susceptibility chi from equilibrium-conditioned damage-state ensembles
  - Correlation length xi from the radial covariance curve C(r)/C(0)
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import cupy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utm
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist
from scipy.stats import linregress, lognorm

from fn_GMPE import gmm_CY14, intra_residuals_corr

warnings.simplefilter(action="ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Plot styling / palette
# -----------------------------------------------------------------------------
CP_HEX = [
    "#4056A1", "#F12815", "#D79922", "#14A098",
    "#CB2D6F", "#116466", "#501F3A", "#F76C6C",
    "#EFE2BA", "#C5CBE3", "#8C9AC7", "#0072B5",
]

StructureType = Literal["SingleStory", "TwoStory", "MultiStory", "All"]

LOAD_CHI_XI = True
SAVE_FIG = False

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def _default_sigmas() -> np.ndarray:
    return np.round(np.arange(0.0, 1.001, 0.01), 3)


@dataclass(frozen=True)
class Config:
    # Scenario
    target_region: str = "Milpitas"
    target_structure: StructureType = "MultiStory"

    # Paths
    data_dir: Path = Path("../data")

    # Simulation control
    Mw_val: float = 5.55
    sigmas: np.ndarray = field(default_factory=_default_sigmas)
    num_sim: int = 10_000
    num_disorder_realizations: int = 50
    base_seed: int = 12345

    # Hazard geometry (for GMPE)
    epicenter_latlon: Tuple[float, float] = (37.666, -122.076)
    depth_to_top_km: float = 3.0
    strike: float = 325.0
    rake: float = 180.0
    dip: float = 90.0
    mechanism: str = "SS"
    region: str = "california"
    on_hanging_wall: bool = False
    vs_source: str = "inferred"

    # Demand correlation model
    soil_case: int = 1

    # Numerical types
    dtype_np: type = np.float64
    dtype_cp: object = cp.float64

    # Sampling
    batch_sim: int = 1024  # number of simulations per GPU batch

    # Equilibrium selection
    eq_frac: float = 0.10
    value_range: Tuple[float, float] = (0.0, 1.0)

    # Radial covariance / xi fit
    bin_w_m: float = 20.0
    r_max_m: float = 2000.0
    var_eps: float = 1e-8

    # Output
    save_arrays: bool = True
    plot_results: bool = True


# -----------------------------------------------------------------------------
# GPU utilities
# -----------------------------------------------------------------------------
def free_gpu() -> None:
    """Release cached blocks so memory is reusable between iterations."""
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def robust_cholesky(A: cp.ndarray, *, jitter0: float = 1e-10, max_tries: int = 8) -> cp.ndarray:
    """Cholesky with diagonal jitter escalation."""
    n = int(A.shape[0])
    I = cp.eye(n, dtype=A.dtype)
    jitter = float(jitter0)
    for _ in range(max_tries):
        try:
            return cp.linalg.cholesky(A + jitter * I)
        except Exception:
            jitter *= 10.0
    raise RuntimeError("Cholesky failed after jitter escalation; covariance may be indefinite.")


# -----------------------------------------------------------------------------
# Data prep helpers
# -----------------------------------------------------------------------------
def structure_mask(df: pd.DataFrame, structure: StructureType) -> np.ndarray:
    ns = df["NumberOfStories"].to_numpy()
    if structure == "SingleStory":
        return ns == 1
    if structure == "TwoStory":
        return ns == 2
    if structure == "MultiStory":
        return ns > 1
    return np.ones(len(df), dtype=bool)


def load_ida_capacities(csv_path: Path, *, dtype: type) -> np.ndarray:
    """
    Load IDA CSV and return capacities as (B, num_gms).
    Assumes first column is an index and remaining columns are capacities.
    """
    df = pd.read_csv(csv_path)
    return df.to_numpy()[:, 1:].astype(dtype, copy=False).T


def fit_lognormal_params_per_building(capa: np.ndarray, *, dtype: type) -> np.ndarray:
    """
    Fit (shape, loc, scale) lognormal parameters for each building using positive, finite samples.
    """
    B = capa.shape[0]
    out = np.zeros((B, 3), dtype=dtype)
    for i in range(B):
        x = capa[i]
        x = x[np.isfinite(x) & (x > 0)]
        if x.size < 2:
            x = np.array([1e-6, 2e-6], dtype=dtype)
        shape, loc, scale = lognorm.fit(x, floc=0)
        out[i] = (dtype(shape), dtype(loc), dtype(scale))
    return out


def build_pairwise_dist_flat(
    df: pd.DataFrame,
    *,
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    utm_epsg: str = "EPSG:32610",
    dtype: type = np.float32,
) -> np.ndarray:
    """
    Pairwise Euclidean distances (meters), condensed vector of length N*(N-1)/2.
    Uses lon/lat -> UTM transform.
    """
    lon = df[lon_col].to_numpy(dtype=float)
    lat = df[lat_col].to_numpy(dtype=float)
    if np.isnan(lon).any() or np.isnan(lat).any():
        raise ValueError("NaNs found in Longitude/Latitude. Drop/fill before computing distances.")

    transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    x, y = transformer.transform(lon, lat)
    xy = np.column_stack([x, y]).astype(dtype, copy=False)

    return pdist(xy, metric="euclidean")  # meters

# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------
def building_cov(X: np.ndarray, *, dtype: type = np.float32) -> np.ndarray:
    """
    Covariance across buildings.
      X: (B, S) array (float or 0/1)
    Returns:
      cov: (B, B)
    """
    X = np.asarray(X, dtype=dtype)
    S = X.shape[1]
    mu = X.mean(axis=1, keepdims=True)
    EXX = (X @ X.T) / S
    return EXX - (mu @ mu.T)


def radial_cov_curve_from_ds(
    ds: np.ndarray,
    pairwise_dist_flat: np.ndarray,
    *,
    bin_w_m: float,
    rmax_m: float,
    var_eps: float = 1e-8,
    eps: float = 1e-12,
    residual_cov: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ds: (B, S_eq) boolean or 0/1
    pairwise_dist_flat: condensed distances (meters), length B*(B-1)/2

    Returns:
      r_km, C_norm, SE_norm  (all 1D arrays, filtered up to rmax_m)
    """
    ds = np.asarray(ds)
    B, S = ds.shape

    # C0 from average Bernoulli variance across buildings
    p_i = ds.mean(axis=1)
    var_i = p_i * (1.0 - p_i)
    mvar = var_i > var_eps
    if not np.any(mvar):
        raise ValueError("All buildings appear deterministic (var ~ 0); cannot define C0.")
    C0 = float(var_i[mvar].mean())

    # Residual covariance or raw covariance
    if residual_cov:
        delta = ds - p_i[:, None]
        cov = building_cov(delta, dtype=np.float32)
    else:
        cov = building_cov(ds, dtype=np.float32)

    y = cov[np.triu_indices(B, k=1)].astype(np.float64, copy=False)
    x = np.asarray(pairwise_dist_flat, dtype=np.float64)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"pairwise_dist_flat has {x.size} pairs but ds implies {y.size} pairs.")

    # Bin distances
    edges = np.arange(0.0, x.max() + bin_w_m, bin_w_m)
    nb = len(edges) - 1
    idx = np.searchsorted(edges, x, side="right") - 1

    m = (idx >= 0) & (idx < nb) & np.isfinite(x) & np.isfinite(y)
    idx = idx[m]
    y = y[m]

    cnt = np.bincount(idx, minlength=nb).astype(np.int64)
    sumy = np.bincount(idx, weights=y, minlength=nb)
    sumy2 = np.bincount(idx, weights=y * y, minlength=nb)

    mean = sumy / np.maximum(cnt, 1)
    var = np.maximum(sumy2 / np.maximum(cnt, 1) - mean**2, 0.0)
    std = np.sqrt(var)

    centers_m = edges[:-1] + 0.5 * bin_w_m
    valid = (cnt > 0) & (centers_m <= float(rmax_m))

    r_km = centers_m[valid] / 1e3
    C_norm = (mean / (C0 + eps))[valid]
    SE_norm = ((std / np.sqrt(np.maximum(cnt, 1))) / (C0 + eps))[valid]

    return r_km.astype(np.float64), C_norm.astype(np.float64), SE_norm.astype(np.float64)


def estimate_xi_from_slope(
    r_km: np.ndarray,
    y: np.ndarray,
    *,
    rmin_km: Optional[float] = None,
    rmax_km: Optional[float] = None,
    include_r0: bool = False,
) -> dict:
    """
    Semi-log regression: ln(y) = a + s r, xi = -1/s.
    """
    r = np.asarray(r_km, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(r) & np.isfinite(y) & (y > 0)
    if not include_r0:
        m &= (r > 0)
    if rmin_km is not None:
        m &= (r >= rmin_km)
    if rmax_km is not None:
        m &= (r <= rmax_km)

    r_fit = r[m]
    y_fit = y[m]
    if r_fit.size < 3:
        raise ValueError(f"Need >=3 points for regression; got {r_fit.size}.")

    ly = np.log(y_fit)
    res = linregress(r_fit, ly)

    if res.slope >= 0:
        raise ValueError(f"Slope is non-negative (s={res.slope:.3g}). Check fit window/data.")

    xi = -1.0 / res.slope
    xi_stderr = res.stderr / (res.slope * res.slope)

    rr = np.linspace(r_fit.min(), r_fit.max(), 200)
    y_line = np.exp(res.intercept + res.slope * rr)

    return {
        "xi_km": float(xi),
        "xi_stderr_km": float(xi_stderr),
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r2": float(res.rvalue**2),
        "r_fit": r_fit,
        "y_fit": y_fit,
        "rr": rr,
        "y_line": y_line,
    }


def equilibrium_range_from_min_width(
    ds_mean: np.ndarray,
    *,
    frac: float,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, Tuple[float, float], dict]:
    """
    Minimum-width interval containing `frac` of the valid values.
    Returns mask over the ORIGINAL ds_mean array (NaNs -> False).
    """
    x_full = np.asarray(ds_mean, dtype=float)
    finite = np.isfinite(x_full)

    xmin, xmax = value_range
    in_range = finite & (x_full >= xmin) & (x_full <= xmax)

    x = x_full[in_range]
    S = x.size
    if S == 0:
        mask = np.zeros_like(x_full, dtype=bool)
        return mask, (np.nan, np.nan), {"reason": "no valid data"}

    frac = float(frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0,1], got {frac}")

    xs = np.sort(x)
    k = int(np.ceil(frac * S))
    k = max(1, min(k, S))

    if k == 1:
        j = S // 2
        lo = hi = float(xs[j])
        width = 0.0
        best_i = j
    else:
        widths = xs[k - 1 :] - xs[: S - k + 1]
        best_i = int(np.argmin(widths))
        lo = float(xs[best_i])
        hi = float(xs[best_i + k - 1])
        width = float(hi - lo)

    mask_full = in_range & (x_full >= lo) & (x_full <= hi)
    info = {
        "method": "min_width_interval",
        "frac": frac,
        "k": k,
        "S_valid": int(S),
        "selected_count": int(mask_full.sum()),
        "m_lo": lo,
        "m_hi": hi,
        "width": width,
        "best_start_rank": int(best_i),
        "best_end_rank": int(best_i + k - 1),
        "value_range": tuple(value_range),
    }
    return mask_full, (lo, hi), info


def susceptibility_from_ds_eq(ds_eq: np.ndarray, *, scale_by_B: bool = True) -> float:
    """
    ds_eq: (B, S_eq) boolean or 0/1
    chi = B * Var(m) where m is damage fraction per realization.
    """
    ds_eq = np.asarray(ds_eq, dtype=np.float64)
    B, _ = ds_eq.shape
    m = ds_eq.mean(axis=0)
    var_m = m.var(ddof=0)
    return float(B * var_m) if scale_by_B else float(var_m)


# -----------------------------------------------------------------------------
# SimulationConfig / preparation
# -----------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    # filtered inventory
    region_info: pd.DataFrame
    num_bldgs: int
    assets: pd.DataFrame
    repair_cost_cp: cp.ndarray

    # capacity stats
    baseline_frag: np.ndarray  # (B,3)
    ln_capa_cov_cp: cp.ndarray

    # demand correlation
    ln_dmnd_corr_cp: cp.ndarray

    # geometry
    event_xy: Tuple[float, float]
    eq_info_static: dict

    # distances (pairs)
    pairwise_dist_flat: np.ndarray

    # dtypes
    dtype_np: type
    dtype_cp: object


def prepare_context(cfg: Config) -> SimulationConfig:
    # --- Load baseline IDA capacities (cov000) ---
    ida_csv = cfg.data_dir / "IDA_results" / f"{cfg.target_region}" / "IDA_results_sigma000.csv"
    bldg_capa = load_ida_capacities(ida_csv, dtype=cfg.dtype_np)

    # drop rows with any NaN for capacity-based stats
    keep_cap = ~np.any(np.isnan(bldg_capa), axis=1)
    if not np.all(keep_cap):
        bldg_capa = bldg_capa[keep_cap, :]

    # --- Load region inventory (must align with IDA ordering after NaN drop) ---
    inv_csv = cfg.data_dir / "building_inventories" / f"RegionalInventory_{cfg.target_region}.csv"
    region_info = pd.read_csv(inv_csv)

    if not np.all(keep_cap):
        region_info = region_info.loc[keep_cap].reset_index(drop=True)

    # structure filter
    keep_struct = structure_mask(region_info, cfg.target_structure)
    region_info = region_info.loc[keep_struct].reset_index(drop=True)
    bldg_capa = bldg_capa[keep_struct, :]

    B, _ = bldg_capa.shape
    print(f"[prep] Buildings kept: {B:,}")

    # --- Baseline frag params from capacities (used for multi-disorder runs) ---
    baseline_frag = fit_lognormal_params_per_building(bldg_capa, dtype=cfg.dtype_np)

    # --- Capacity covariance in log-space ---
    ln_std = baseline_frag[:, 0].astype(cfg.dtype_np, copy=False)
    ln_capa = np.log(np.clip(bldg_capa.astype(cfg.dtype_np, copy=False), 1e-12, None))
    ln_corr = np.corrcoef(ln_capa, rowvar=True).astype(cfg.dtype_np, copy=False)
    ln_cov = (np.outer(ln_std, ln_std) * ln_corr).astype(cfg.dtype_np, copy=False)

    ln_capa_cov_cp = cp.asarray(ln_cov, dtype=cfg.dtype_cp)

    # --- Site info (UTM + Vs30 + repair cost) ---
    e, n, _, _ = utm.from_latlon(region_info["Latitude"].to_numpy(), region_info["Longitude"].to_numpy())
    e = e.astype(cfg.dtype_np, copy=False)
    n = n.astype(cfg.dtype_np, copy=False)

    vs30 = region_info["Vs30"].to_numpy(dtype=cfg.dtype_np, copy=False)
    vs30 = np.maximum(vs30, 180.0)

    assets = pd.DataFrame({"x": e, "y": n, "Vs30": vs30})
    repair_cost_cp = cp.asarray(region_info["RepairCost"].to_numpy(dtype=cfg.dtype_np, copy=False), dtype=cfg.dtype_cp)

    # --- Event geometry in UTM ---
    ev_e, ev_n, _, _ = utm.from_latlon(*cfg.epicenter_latlon)
    event_xy = (cfg.dtype_np(ev_e), cfg.dtype_np(ev_n))

    # --- Demand correlation (GPU) ---
    ln_dmnd_corr_cp = intra_residuals_corr(
        cp.asarray(assets["x"].to_numpy(), dtype=cfg.dtype_cp),
        cp.asarray(assets["y"].to_numpy(), dtype=cfg.dtype_cp),
        soil_case=cfg.soil_case,
    ).astype(cfg.dtype_cp)

    # --- Pairwise distance ---
    pairwise_dist_flat = build_pairwise_dist_flat(region_info, dtype=cfg.dtype_np)

    eq_info_static = dict(
        depth_to_top=cfg.depth_to_top_km,
        strike=cfg.strike,
        rake=cfg.rake,
        dip=cfg.dip,
        mechanism=cfg.mechanism,
        region=cfg.region,
        on_hanging_wall=cfg.on_hanging_wall,
        vs_source=cfg.vs_source,
    )

    free_gpu()

    return SimulationConfig(
        region_info=region_info,
        num_bldgs=B,
        assets=assets,
        repair_cost_cp=repair_cost_cp,
        baseline_frag=baseline_frag,
        ln_capa_cov_cp=ln_capa_cov_cp,
        ln_dmnd_corr_cp=ln_dmnd_corr_cp,
        event_xy=event_xy,
        eq_info_static=eq_info_static,
        pairwise_dist_flat=pairwise_dist_flat,
        dtype_np=cfg.dtype_np,
        dtype_cp=cfg.dtype_cp,
    )


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def prepare_hazard_mvn(ctx: SimulationConfig, *, Mw: float) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    For a fixed Mw:
      - compute ln(demand) mean on GPU
      - build X_cov = Cov(lnD) + Cov(lnC)
      - return (ln_dmnd_mean_cp, chol_T) where chol_T is Cholesky factor transpose
    """
    eq_info = dict(ctx.eq_info_static)
    eq_info["Mw"] = float(Mw)

    ln_mean_np, ln_std_np = gmm_CY14(ctx.assets, ctx.event_xy, eq_info)
    ln_mean_cp = cp.asarray(ln_mean_np, dtype=ctx.dtype_cp)
    ln_std_cp = cp.asarray(ln_std_np, dtype=ctx.dtype_cp)

    ln_dmnd_cov_cp = cp.outer(ln_std_cp, ln_std_cp) * ctx.ln_dmnd_corr_cp
    X_cov_cp = ln_dmnd_cov_cp + ctx.ln_capa_cov_cp

    L = robust_cholesky(X_cov_cp)
    LT = L.T

    # free intermediates
    del ln_std_cp, ln_dmnd_cov_cp, X_cov_cp, L
    free_gpu()

    return ln_mean_cp, LT


def sample_ds_from_frag(
    frag_params: np.ndarray,
    *,
    ln_dmnd_mean_cp: cp.ndarray,
    chol_T: cp.ndarray,
    num_sim: int,
    cp_rng: cp.random.Generator,
    dtype_np: type,
    dtype_cp: object,
    batch_sim: int,
) -> np.ndarray:
    """
    Sample ds (B,S) on GPU and return as numpy bool array.
    """
    B = frag_params.shape[0]

    ln_capa_mean_cp = cp.asarray(np.log(frag_params[:, 2].astype(dtype_np, copy=False)), dtype=dtype_cp)
    X_mean_cp = ln_capa_mean_cp - ln_dmnd_mean_cp  # (B,)

    ds = np.empty((B, num_sim), dtype=bool)

    for s0 in range(0, num_sim, batch_sim):
        bs = min(batch_sim, num_sim - s0)

        Z = cp_rng.standard_normal((bs, B), dtype=dtype_cp)
        X = Z @ chol_T + X_mean_cp[None, :]

        valid = cp.isfinite(X)
        ds_batch = (X < 0) & valid  # (bs,B)

        ds[:, s0 : s0 + bs] = cp.asnumpy(ds_batch.T)

        del Z, X, valid, ds_batch
        free_gpu()

    return ds


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------
def run_xi_chi_sweep(cfg: Config, ctx: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      xi_km: (Nsig, R)
      chi  : (Nsig, R)
    """
    ln_dmnd_mean_cp, chol_T = prepare_hazard_mvn(ctx, Mw=cfg.Mw_val)

    Nsig = int(cfg.sigmas.size)
    R = int(cfg.num_disorder_realizations)

    xi_km = np.full((Nsig, R), np.nan, dtype=cfg.dtype_np)
    chi = np.full((Nsig, R), np.nan, dtype=cfg.dtype_np)

    r_max_km = cfg.r_max_m / 1e3

    for i, sigma in enumerate(cfg.sigmas):
        t_sigma = time.time()
        print(f"\n=== sigma={sigma:.2f} ({i+1}/{Nsig}) ===")

        for r in range(R):
            if r % 10 == 0:
                print(f"  realization {r+1}/{R}")

            seed = cfg.base_seed + i * 100_000 + r
            np_rng = np.random.default_rng(seed)
            cp_rng = cp.random.default_rng(seed)

            frag = ctx.baseline_frag.copy()
            if sigma > 0:
                noise = np_rng.lognormal(mean=0.0, sigma=float(sigma), size=ctx.num_bldgs).astype(cfg.dtype_np, copy=False)
                frag[:, 2] *= noise

            ds = sample_ds_from_frag(
                frag,
                ln_dmnd_mean_cp=ln_dmnd_mean_cp,
                chol_T=chol_T,
                num_sim=cfg.num_sim,
                cp_rng=cp_rng,
                dtype_np=cfg.dtype_np,
                dtype_cp=cfg.dtype_cp,
                batch_sim=cfg.batch_sim,
            )

            ds_mean = ds.mean(axis=0)  # (S,)
            mask_eq, _, info = equilibrium_range_from_min_width(ds_mean, frac=cfg.eq_frac, value_range=cfg.value_range)

            ds_eq = ds[:, mask_eq]  # (B, S_eq)
            if ds_eq.shape[1] < 10:
                print(f"    warning: S_eq={ds_eq.shape[1]} too small; skipping.")
                continue

            chi[i, r] = susceptibility_from_ds_eq(ds_eq, scale_by_B=True)

            try:
                r_km, C_r, _ = radial_cov_curve_from_ds(
                    ds_eq,
                    ctx.pairwise_dist_flat,
                    bin_w_m=cfg.bin_w_m,
                    rmax_m=cfg.r_max_m,
                    var_eps=cfg.var_eps,
                    residual_cov=True,
                )
                fit = estimate_xi_from_slope(r_km, C_r, rmin_km=0.0, rmax_km=r_max_km)
                xi_km[i, r] = fit["xi_km"]
            except Exception as e:
                print(f"    warning: xi fit failed (sigma={sigma:.2f}, r={r}): {e}")

            del ds, ds_eq
            free_gpu()

        print(f"=== sigma={sigma:.2f} done in {time.time() - t_sigma:.2f}s ===")

    del ln_dmnd_mean_cp, chol_T
    free_gpu()

    return xi_km, chi


# -----------------------------------------------------------------------------
# Plot: curtain heatmap over sigma
# -----------------------------------------------------------------------------
def curtain_heatmap_smooth(
    x_vals: np.ndarray,
    Y: np.ndarray,  # (Nsig, R)
    *,
    ylabel: str,
    yscale: Optional[Literal["log"]] = None,
    nbins_y: int = 120,
    quantile_clip: Tuple[float, float] = (1, 99),
    x_refine: Optional[int] = None,
    smooth_sigma_y: float = 2.0,
    smooth_sigma_x: float = 1.0,
    min_density_mode: Literal["auto", "fixed"] = "auto",
    auto_frac: float = 1e-3,
    min_density: float = 0.0,
    color: str = CP_HEX[0],
    savefig: bool = False,
    out_path: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    x_vals = np.asarray(x_vals, float)
    Y = np.asarray(Y, float)

    y_flat = Y[np.isfinite(Y)]
    if y_flat.size == 0:
        raise ValueError("No finite values in Y.")

    qlo, qhi = np.percentile(y_flat, quantile_clip)
    if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi <= qlo:
        qlo, qhi = float(np.nanmin(y_flat)), float(np.nanmax(y_flat))

    if yscale == "log":
        qlo = max(qlo, 1e-12)
        y_edges = np.logspace(np.log10(qlo), np.log10(qhi), nbins_y + 1)
    else:
        y_edges = np.linspace(qlo, qhi, nbins_y + 1)

    Nsig = x_vals.size
    D = np.zeros((nbins_y, Nsig), dtype=float)

    for i in range(Nsig):
        yi = Y[i, :]
        yi = yi[np.isfinite(yi)]
        if yi.size == 0:
            continue
        h, _ = np.histogram(yi, bins=y_edges, density=True)
        D[:, i] = h

    # refine x by interpolation (optional)
    if x_refine is not None and x_refine > 1 and Nsig >= 2:
        x_new = np.linspace(x_vals.min(), x_vals.max(), (Nsig - 1) * x_refine + 1)
        D_ref = np.empty((nbins_y, x_new.size), dtype=float)
        for k in range(nbins_y):
            D_ref[k, :] = np.interp(x_new, x_vals, D[k, :])
        x_plot, D_plot = x_new, D_ref
    else:
        x_plot, D_plot = x_vals, D

    # smooth 2D density
    if (smooth_sigma_y and smooth_sigma_y > 0) or (smooth_sigma_x and smooth_sigma_x > 0):
        D_plot = gaussian_filter(D_plot, sigma=(smooth_sigma_y, smooth_sigma_x), mode="nearest")

    # transparency threshold after smoothing
    if min_density_mode == "auto":
        nz = D_plot[D_plot > 0]
        thr = (auto_frac * nz.max()) if nz.size else 0.0
    else:
        thr = float(min_density)

    D_masked = np.ma.masked_less_equal(D_plot, thr)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "curtain",
        [(0.0, "white"), (0.5, color), (1.0, CP_HEX[2])],
        N=256,
    )
    cmap.set_bad((0, 0, 0, 0))

    # x edges for pcolormesh
    x = x_plot
    dx0 = x[1] - x[0]
    dx1 = x[-1] - x[-2]
    x_edges = np.concatenate(([x[0] - dx0 / 2], 0.5 * (x[1:] + x[:-1]), [x[-1] + dx1 / 2]))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(x_vals, Y, linestyle="None", marker=".", markersize=2, color=color, alpha=1.0, zorder=0)
    pcm = ax.pcolormesh(x_edges, y_edges, D_masked, shading="auto", cmap=cmap, zorder=1, alpha=0.6)

    cb = fig.colorbar(pcm, ax=ax, pad=0.05)
    cb.set_label("Occurrence frequency", fontsize=18)
    cb.set_ticks([])
    cb.ax.text(0.5, -0.01, "Low", ha="center", va="top", fontsize=14, transform=cb.ax.transAxes)
    cb.ax.text(0.5, 1.01, "High", ha="center", va="bottom", fontsize=14, transform=cb.ax.transAxes)

    ax.set_xlabel(r"Structural diversity $\sigma$", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    if yscale == "log":
        ax.set_yscale("log")

    fig.tight_layout()

    if savefig and out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved: {out_path}")

    plt.show()
    return fig, ax


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main() -> None:
    cfg = Config()

    outdir = cfg.data_dir / "chi_xi_calculation" / f"{cfg.target_region}"
    outdir.mkdir(parents=True, exist_ok=True)

    if not LOAD_CHI_XI:
        ctx = prepare_context(cfg)

        t0 = time.time()
        xi_km, chi = run_xi_chi_sweep(cfg, ctx)
        print(f"\n(Mw={cfg.Mw_val:.2f}) All done in {time.time() - t0:.2f}s")

        if cfg.save_arrays:
            np.save(outdir / f"xi_km_Mw{cfg.Mw_val:.2f}.npy", xi_km)
            np.save(outdir / f"chi_Mw{cfg.Mw_val:.2f}.npy", chi)
            np.save(outdir / "sigmas.npy", cfg.sigmas)
            print(f"Saved arrays to: {outdir}")
    else:
        xi_km = np.load(outdir / f"xi_km_Mw{cfg.Mw_val:.2f}.npy")
        chi = np.load(outdir / f"chi_Mw{cfg.Mw_val:.2f}.npy")
        print(f"Loaded arrays from: {outdir}")

    if cfg.plot_results:
        curtain_heatmap_smooth(
            cfg.sigmas,
            chi,
            ylabel=r"Susceptibility $\chi$",
            yscale=None,
            nbins_y=100,
            x_refine=None,
            smooth_sigma_y=2,
            smooth_sigma_x=1,
            min_density_mode="auto",
            auto_frac=1e-3,
            color=CP_HEX[0],
            savefig=SAVE_FIG,
            out_path=f"../results/chi_vs_sigma_Mw{cfg.Mw_val:.2f}.png",
        )

        curtain_heatmap_smooth(
            cfg.sigmas,
            xi_km,
            ylabel=r"Correlation length $\xi$ (km)",
            yscale=None,
            nbins_y=100,
            x_refine=None,
            smooth_sigma_y=2,
            smooth_sigma_x=1,
            min_density_mode="auto",
            auto_frac=1e-3,
            color=CP_HEX[5],
            savefig=SAVE_FIG,
            out_path=f"../results/xi_vs_sigma_Mw{cfg.Mw_val:.2f}.png",
        )


if __name__ == "__main__":
    main()