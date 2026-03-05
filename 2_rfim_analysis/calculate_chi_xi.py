#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 02/09/2026
@Author       :
@Contact      : sebin.oh@berkeley.edu
@Description  : 

"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import cupy as cp
import utm
import time, os, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from fn_GMPE_parallel_gpu_vec import (
    gmm_CY14,
    intra_residuals_corr_fn_cupy,
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "cm"

CP_HEX = [
    "#4056A1", "#F12815", "#D79922", "#14A098",
    "#CB2D6F", "#116466", "#501F3A", "#F76C6C",
    "#EFE2BA", "#C5CBE3", "#8C9AC7", "#0072B5",
]

# %% Helper functions
from pyproj import Transformer
from scipy.spatial.distance import pdist, squareform

def pairwise_building_distance(
    gdf,
    lon_col="Longitude",
    lat_col="Latitude",
    utm_epsg="EPSG:32610",   # UTM Zone 10N (WGS84). Good for Bay Area.
    return_square=False,     # False -> condensed vector (pdist); True -> NxN matrix
    dtype=np.float32,
):
    """
    Compute pairwise distances between points given lon/lat columns.

    Returns:
      - condensed distance vector (length N*(N-1)/2) if return_square=False
      - full NxN distance matrix if return_square=True
    """
    lon = gdf[lon_col].to_numpy(dtype=float)
    lat = gdf[lat_col].to_numpy(dtype=float)

    if np.isnan(lon).any() or np.isnan(lat).any():
        raise ValueError("NaNs found in lon/lat columns. Drop/fill them before computing distances.")

    # lon/lat (EPSG:4326) -> UTM meters
    transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
    x, y = transformer.transform(lon, lat)

    xy = np.column_stack([x, y]).astype(dtype, copy=False)

    d = pdist(xy, metric="euclidean")  # meters
    return squareform(d) if return_square else d

if load_pairwise_distance := False:
    target_region = "Milpitas"
    gdf_region = pd.read_csv(f"../data/building_inventories/RegionalInventory_{target_region}.csv")
    pairwise_dist_flat = pairwise_building_distance(gdf_region)         # shape: (N*(N-1)/2,)
    np.save(f"../data/pairwise_distances_{target_region}.npy", pairwise_dist_flat)  # save for future runs (skip distance recomputation)
else:
    pairwise_dist_flat = np.load(f"../data/pairwise_distances_{target_region}.npy")


def radial_cov_curve_from_ds(
    ds: np.ndarray,
    pairwise_dist_flat: np.ndarray,
    bin_w_m: float = 100.0,
    rmax_m: float = 6000.0,
    var_eps: float = 1e-8,
    eps: float = 1e-12,
    dtype=np.float32,
    return_all: bool = False,
    residual_cov: bool = False,
):
    ds = np.asarray(ds)
    if residual_cov:
        mean_i = ds.mean(axis=1)                         # (B,)
        delta = ds - mean_i[:, np.newaxis]                # (B, n_sims)
        ds_cov = building_cov(delta)
    else:
        ds_cov = building_cov(ds)
    ds_cov_flat = ds_cov[np.triu_indices(ds_cov.shape[0], k=1)]
    
    pairwise_dist_flat = np.asarray(pairwise_dist_flat)

    # --- C0: zero-lag variance scale (mean_i Var(D_i)) ---
    p_i = ds.mean(axis=1)                         # (B,)
    var_i = p_i * (1.0 - p_i)                     # (B,)
    mask_var = var_i > var_eps
    if not np.any(mask_var):
        raise ValueError("All buildings appear deterministic (var_i ~ 0); cannot define C0.")
    C0 = var_i[mask_var].mean()

    # --- covariance upper triangle (pairs) ---
    if ds_cov_flat is None:
        # NOTE: this computes full BxB covariance then extracts UT; heavy for B~5943
        X = ds.astype(dtype, copy=False)
        S = X.shape[1]
        p = X.mean(axis=1, keepdims=True)         # (B,1)
        EXX = (X @ X.T) / S
        cov = EXX - (p @ p.T)
        y = cov[np.triu_indices(cov.shape[0], k=1)]
    else:
        y = np.asarray(ds_cov_flat)

    x = pairwise_dist_flat

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Length mismatch: pairwise_dist_flat={x.shape[0]} vs ds_cov_flat={y.shape[0]}")

    # --- binning setup ---
    bin_w_m = float(bin_w_m)
    edges = np.arange(0.0, x.max() + bin_w_m, bin_w_m)
    nb = len(edges) - 1

    idx = np.searchsorted(edges, x, side="right") - 1
    m = (idx >= 0) & (idx < nb) & np.isfinite(y) & np.isfinite(x)
    idx = idx[m]
    y = y[m]

    # --- per-bin stats across pairs ---
    cnt  = np.bincount(idx, minlength=nb).astype(np.int64)
    sumy = np.bincount(idx, weights=y, minlength=nb)
    sumy2 = np.bincount(idx, weights=y*y, minlength=nb)

    mean = sumy / np.maximum(cnt, 1)
    var = sumy2 / np.maximum(cnt, 1) - mean**2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    xc_m = edges[:-1] + bin_w_m / 2.0

    valid = (cnt > 0) & (xc_m <= float(rmax_m))
    r_km = xc_m[valid] / 1e3

    C_norm = (mean / (C0 + eps))[valid]
    SE_norm = ((std / np.sqrt(np.maximum(cnt, 1))) / (C0 + eps))[valid]
    # C_norm = mean[valid]
    # SE_norm = std[valid] / np.sqrt(np.maximum(cnt[valid], 1))

    # # Insert r=0 point
    # r_km = np.insert(r_km, 0, 0.0)
    # C_norm = np.insert(C_norm, 0, 1.0)
    # SE_norm = np.insert(SE_norm, 0, 0.0)

    if return_all:
        out = dict(
            C0=C0,
            cnt=cnt,
            mean_raw=mean,
            std_raw=std,
            bin_centers_m=xc_m,
            valid_mask=valid,
            p_i=p_i,
            var_i=var_i,
        )
        return r_km, C_norm, SE_norm, out

    return r_km, C_norm, SE_norm

def building_cov(ds: np.ndarray, dtype=np.float32, eps=1e-12):
    """
    ds: (n_buildings, n_sims) with 0/1 entries
    returns: (n_buildings, n_buildings) covariance matrix
    """
    # Use float32 to cut memory; keep ds as 0/1
    X = ds.astype(dtype, copy=False)
    n_sims = X.shape[1]

    # p_i = E[X_i]
    p = X.mean(axis=1, keepdims=True)                  # (B,1)

    # E[X_i X_j] = (X X^T) / n_sims   (since entries are 0/1)
    EXX = (X @ X.T) / n_sims                          # (B,B)

    cov = EXX - (p @ p.T)

    return cov

from scipy.stats import linregress
def estimate_xi_from_slope(r_km, y, rmin_km=None, rmax_km=None, include_r0=False):
    """
    Estimate correlation length xi (km) from semi-log slope:
      ln y = a + s r,  xi = -1/s

    Returns dict with xi, slope, intercept, stderr, r2, and fit arrays.
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
        raise ValueError(f"Need >=3 valid points for regression; got {r_fit.size}.")

    ly = np.log(y_fit)
    res = linregress(r_fit, ly)  # slope, interc ept, stderr, rvalue, ...

    slope = res.slope
    intercept = res.intercept

    if slope >= 0:
        # This usually means you're not in an exponential-decay regime in the chosen window.
        raise ValueError(f"Slope is non-negative (s={slope:.3g}). Check fit window/data.")

    xi = -1.0 / slope

    # propagate slope stderr -> xi stderr (approx)
    xi_stderr = res.stderr / (slope * slope)

    # fitted line for plotting (on original y-scale)
    rr = np.linspace(r_fit.min(), r_fit.max(), 200)
    y_line = np.exp(intercept + slope * rr)

    return {
        "xi_km": xi,
        "xi_stderr_km": xi_stderr,
        "slope": slope,
        "intercept": intercept,
        "r2": res.rvalue**2,
        "r_fit": r_fit,
        "y_fit": y_fit,
        "rr": rr,
        "y_line": y_line,
    }

from scipy.signal import find_peaks
def phase_coexistence_detect(
    dv_temp,
    bins=200,
    win=9,
    prominence_frac=0.03,
    tol=0.10,
    value_range=(0.0, 1.0)
):
    """
    Returns 1 if histogram has peaks near both edges (0 and 1), else 0.
    """

    x = np.asarray(dv_temp, dtype=float)
    x = x[np.isfinite(x)]

    # Enforce range if desired
    xmin, xmax = value_range
    x = x[(x >= xmin) & (x <= xmax)]

    if len(x) == 0:
        return 0   # no data → no peaks

    # Histogram
    counts, edges = np.histogram(
        x, bins=bins, range=value_range, density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Smooth
    if win % 2 == 0:
        win += 1  # ensure odd window
    kernel = np.ones(win) / win
    smooth = np.convolve(counts, kernel, mode="same")

    # Peak detection
    peaks, _ = find_peaks(
        smooth,
        prominence=prominence_frac * smooth.max(),
        distance=bins // 10
    )

    if len(peaks) == 0:
        return 0

    peak_locs = centers[peaks]

    # Edge checks
    has_left  = np.any(peak_locs <= xmin + tol)
    has_right = np.any(peak_locs >= xmax - tol)

    return int(has_left and has_right)

def _single_ds_from_frag(
    *,
    bldg_frag_params: np.ndarray,
    num_bldgs: int,
    num_sim: int,
    X_cov,
    ln_dmnd_mean_cp,
    dtype_np,
    dtype_cp,
):
    # capacity mean on GPU
    ln_capa_mean_cp = cp.asarray(
        np.log(bldg_frag_params[:, 2].astype(dtype_np, copy=False)),
        dtype=dtype_cp
    )

    X_mean = ln_capa_mean_cp - ln_dmnd_mean_cp           # (B,)
    X_samples = cp.random.multivariate_normal(X_mean, X_cov, num_sim)  # (S,B)

    valid = ~cp.isnan(X_samples)                          # (S,B)

    ds = cp.zeros((num_bldgs, num_sim), dtype=cp.bool_)
    ds[:, :] = (X_samples.T < 0) & valid.T                # invalid -> False

    return cp.asnumpy(ds)

def simulate_ds(
    Mw: float,
    cov: float,
    *,
    target_region: str,
    idx_target: np.ndarray,
    num_sim: int,
    num_bldgs: int,
    dtype_np=np.float32,
    dtype_cp=cp.float32,
    frag_dir_suffix: str = "bldg_frag_params_for_[sim_total]",
    num_disorder_realizations: int = 1,
    seed: int | None = None,
):
    if seed is not None:
        np.random.seed(seed)
        cp.random.seed(seed)
    
    cov_label = f"{int(round(float(cov) * 1000)):04d}"

    # -----------------------------
    # GMPE mean/std for this Mw
    # -----------------------------
    eq_info = dict(
        depth_to_top     = depth_to_top,
        strike           = strike,
        rake             = rake,
        dip              = dip,
        mechanism        = 'SS',
        region           = 'california',
        on_hanging_wall  = False,
        vs_source        = 'inferred',
        Mw               = float(Mw),
    )
    ln_dmnd_mean_np, ln_dmnd_std_np = gmm_CY14(assets, event_xy, eq_info)

    ln_dmnd_mean_cp = cp.asarray(ln_dmnd_mean_np, dtype=dtype_cp)
    ln_dmnd_std_cp  = cp.asarray(ln_dmnd_std_np,  dtype=dtype_cp)

    ln_dmnd_cov_cp = cp.outer(ln_dmnd_std_cp, ln_dmnd_std_cp) * ln_dmnd_corr_cp
    X_cov = ln_dmnd_cov_cp + ln_capa_cov_cp

    # -----------------------------
    # One-shot simulation (single disorder realization)
    # -----------------------------
    if num_disorder_realizations == 1:
        frag_path = f"Savio/{target_region}/{frag_dir_suffix}/bldg_frag_params_cov{cov_label}.npy"
        bldg_frag_params = np.load(frag_path)
        bldg_frag_params = bldg_frag_params[idx_target, :]

        if bldg_frag_params.shape[0] != num_bldgs:
            raise ValueError(
                f"num_bldgs mismatch: bldg_frag_params has {bldg_frag_params.shape[0]} rows, "
                f"but num_bldgs={num_bldgs}. Check idx_target / filtering."
            )

        ds_np = _single_ds_from_frag(
            bldg_frag_params=bldg_frag_params,
            num_bldgs=num_bldgs,
            num_sim=num_sim,
            X_cov=X_cov,
            ln_dmnd_mean_cp=ln_dmnd_mean_cp,
            dtype_np=dtype_np,
            dtype_cp=dtype_cp,
        )

        return ds_np
    
    # -----------------------------
    # Multiple disorder realizations: build baseline (cov=0) frag params once
    # -----------------------------
    # NOTE: uses IDA_results_OpenSeesPy_cov000.csv as "base" to estimate shape/loc/scale
    bldg_info = pd.read_csv(f"Savio/{target_region}/IDA_results_OpenSeesPy_cov000.csv")
    bldg_capa = bldg_info.to_numpy()[:, 1:].astype(dtype_np, copy=False).T  # (B_all, num_gms)

    # drop rows with any NaN
    idx_nan = np.where(np.any(np.isnan(bldg_capa), axis=1))[0]
    if idx_nan.size > 0:
        keep = np.ones(bldg_capa.shape[0], dtype=bool)
        keep[idx_nan] = False
        bldg_capa = bldg_capa[keep]

    # Fit baseline frag params for ALL buildings in this file
    B_all = bldg_capa.shape[0]
    baseline_frag = np.zeros((B_all, 3), dtype=dtype_np)
    for i in range(B_all):
        temp = bldg_capa[i]
        temp = temp[(~np.isnan(temp)) & (temp != 0)]
        if temp.size < 2:
            temp = np.array([1e-6, 2e-6], dtype=dtype_np)
        shape, loc, scale = lognorm.fit(temp, floc=0)
        baseline_frag[i, :] = [dtype_np(shape), dtype_np(loc), dtype_np(scale)]

    # Select target buildings (must align with idx_target)
    baseline_frag = baseline_frag[idx_target, :]

    if baseline_frag.shape[0] != num_bldgs:
        raise ValueError(
            f"num_bldgs mismatch after idx_target: baseline_frag has {baseline_frag.shape[0]} rows, "
            f"but num_bldgs={num_bldgs}. Check idx_target / filtering."
        )
    
    # -----------------------------
    # Loop disorder realizations
    # -----------------------------
    R = int(num_disorder_realizations)

    ds_all = np.zeros((R, num_bldgs, num_sim), dtype=bool)

    for r in range(R):
        frag_r = baseline_frag.copy()

        # perturb ONLY the median scale (col=2) by lognormal noise
        noise = np.random.lognormal(mean=0.0, sigma=float(cov), size=num_bldgs).astype(dtype_np, copy=False)
        frag_r[:, 2] *= noise

        ds_r = _single_ds_from_frag(
            bldg_frag_params=frag_r,
            num_bldgs=num_bldgs,
            num_sim=num_sim,
            X_cov=X_cov,
            ln_dmnd_mean_cp=ln_dmnd_mean_cp,
            dtype_np=dtype_np,
            dtype_cp=dtype_cp,
        )
        ds_all[r, :, :] = ds_r

    return ds_all

def equilibrium_range_from_hist_mstar(
    ds_mean: np.ndarray,
    m_star: float,
    bins: int = 100,
    frac: float = 0.10,
    value_range=(0.0, 1.0),
    snap_to_bin: bool = True,
):
    ds_mean_full = np.asarray(ds_mean, dtype=float)

    # Filter valid values for histogram counts/expansion target
    x = ds_mean_full[np.isfinite(ds_mean_full)]

    xmin, xmax = value_range
    x = x[(x >= xmin) & (x <= xmax)]
    S = x.size
    if S == 0:
        return np.zeros(ds_mean_full.shape, dtype=bool), (np.nan, np.nan), {"reason": "no valid data"}

    # Histogram (counts) and bin geometry
    counts, edges = np.histogram(x, bins=bins, range=value_range, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Determine starting bin index for m_star
    m_star = float(m_star)

    if snap_to_bin:
        # If m_star outside range, clamp to nearest valid value in range
        m_clamped = min(max(m_star, xmin), xmax)
        # np.searchsorted returns insertion index into edges; subtract 1 gives bin index
        i0 = int(np.searchsorted(edges, m_clamped, side="right") - 1)
        i0 = int(np.clip(i0, 0, bins - 1))
    else:
        # choose nearest bin center to m_star (if outside, nearest end)
        i0 = int(np.argmin(np.abs(centers - m_star)))

    target = int(np.ceil(frac * S))

    left = i0
    right = i0
    cum = int(counts[i0])

    # expand symmetrically in bin index
    while (cum < target) and (left > 0 or right < bins - 1):
        if left > 0:
            left -= 1
            cum += int(counts[left])
        if cum >= target:
            break
        if right < bins - 1:
            right += 1
            cum += int(counts[right])

    m_lo = float(edges[left])
    m_hi = float(edges[right + 1])

    # mask over ORIGINAL ds_mean length (keep NaNs as False)
    mask_full = np.isfinite(ds_mean_full) & (ds_mean_full >= m_lo) & (ds_mean_full <= m_hi)

    info = {
        "bins": bins,
        "frac": frac,
        "target_count": target,
        "selected_count": int(mask_full.sum()),
        "hist_counts": counts,
        "hist_edges": edges,
        "hist_centers": centers,
        "m_star": m_star,
        "m_star_clamped": float(min(max(m_star, xmin), xmax)),
        "start_bin": i0,
        "left_bin": left,
        "right_bin": right,
        "m_lo": m_lo,
        "m_hi": m_hi,
    }
    return mask_full, (m_lo, m_hi), info


def equilibrium_range_from_hist_mode(
    ds_mean: np.ndarray,
    bins: int = 100,
    frac: float = 0.10,
    value_range=(0.0, 1.0),
):
    x = np.asarray(ds_mean, dtype=float)
    x = x[np.isfinite(x)]

    xmin, xmax = value_range
    x = x[(x >= xmin) & (x <= xmax)]
    S = x.size
    if S == 0:
        return np.zeros(0, dtype=bool), (np.nan, np.nan), {"reason": "no valid data"}

    counts, edges = np.histogram(x, bins=bins, range=value_range, density=False)
    imax = int(np.argmax(counts))

    target = int(np.ceil(frac * S))

    left = imax
    right = imax
    cum = int(counts[imax])

    # expand symmetrically in bin index
    while (cum < target) and (left > 0 or right < bins - 1):
        if left > 0:
            left -= 1
            cum += int(counts[left])
        if (cum >= target):
            break
        if right < bins - 1:
            right += 1
            cum += int(counts[right])

    m_lo = float(edges[left])
    m_hi = float(edges[right + 1])

    # mask over ORIGINAL ds_mean length (keep NaNs as False)
    ds_mean_full = np.asarray(ds_mean, dtype=float)
    mask_full = np.isfinite(ds_mean_full) & (ds_mean_full >= m_lo) & (ds_mean_full <= m_hi)

    # by default, include right edge; to mimic histogram half-open bins you can use < m_hi
    # but including right edge is often fine for [0,1] bounded m.
    info = {
        "bins": bins,
        "frac": frac,
        "target_count": target,
        "selected_count": int(mask_full.sum()),
        "hist_counts": counts,
        "hist_edges": edges,
        "imax": imax,
        "left_bin": left,
        "right_bin": right,
        "m_lo": m_lo,
        "m_hi": m_hi,
    }
    return mask_full, (m_lo, m_hi), info

def equilibrium_range_from_min_width(
    ds_mean: np.ndarray,
    frac: float = 0.20,
    value_range=(0.0, 1.0),
):
    x_full = np.asarray(ds_mean, dtype=float)
    finite = np.isfinite(x_full)

    xmin, xmax = value_range
    in_range = finite & (x_full >= xmin) & (x_full <= xmax)

    x = x_full[in_range]
    S = x.size
    if S == 0:
        return np.zeros_like(x_full, dtype=bool), (np.nan, np.nan), {"reason": "no valid data"}

    # clamp frac
    frac = float(frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0,1], got {frac}")

    xs = np.sort(x)
    k = int(np.ceil(frac * S))
    k = max(1, min(k, S))

    # sliding window widths
    if k == 1:
        # pick the single point closest to m_star (if provided) else median
        j = S // 2
        lo = hi = float(xs[j])
        best_i = j
        width = 0.0
    else:
        widths = xs[k-1:] - xs[:S-k+1]   # length S-k+1
        best_i = int(np.argmin(widths))
        lo = float(xs[best_i])
        hi = float(xs[best_i + k - 1])
        width = float(hi - lo)

    # mask over original ds_mean length (keep NaNs as False)
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


def susceptibility_from_ds_eq(ds_eq: np.ndarray, scale_by_B: bool = True) -> float:
    """
    ds_eq: (B, S_eq) boolean or 0/1
    returns chi = B*Var(m) by default, where m is damage fraction per realization.
    """
    ds_eq = np.asarray(ds_eq, dtype=np.float64)
    B, S_eq = ds_eq.shape
    m_eq = ds_eq.mean(axis=0)            # (S_eq,)
    var_m = m_eq.var(ddof=0)             # population variance over realizations
    chi = (B * var_m) if scale_by_B else var_m
    return float(chi)

# %%
# ---------- Config ---------- 
target_region     = "Milpitas_all"
target_structure  = "MultiStory"  # "SingleStory", "TwoStory", "MultiStory", "All"

covs              = np.round(np.arange(0.0, 1.001, 0.01), 3)
Mws               = np.round(np.arange(8.50, 8.51, 0.05), 2)

# covs              = np.round(np.array([0.0, 1.0]), 3)
# Mws               = np.round(np.array([7.25]), 2) 

dtype_np          = np.float64
dtype_cp          = cp.float64

# Sampling settings
gmm               = "CY14"
num_sim           = 10_000

# %%
# ---------- Utilities ----------
def free_gpu():
    """Release cached blocks so memory is reusable between iterations."""
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

# %%
# ---------- Load capacities ----------
bldg_info  = pd.read_csv(f"Savio/{target_region}/IDA_results_OpenSeesPy_cov000.csv")
bldg_capa  = bldg_info.to_numpy()[:, 1:].astype(dtype_np, copy=False).T  # (num_bldgs, num_gms)
del bldg_info
free_gpu()

idx_nan = np.where(np.any(np.isnan(bldg_capa), axis=1))[0]
if idx_nan.size > 0:
    keep = np.ones(bldg_capa.shape[0], dtype=bool); keep[idx_nan] = False
    bldg_capa = bldg_capa[keep]

region_info = pd.read_csv(f"TargetRegion_{target_region}.csv")
if idx_nan.size > 0:
    region_info = region_info.drop(idx_nan).reset_index(drop=True)

if target_structure == "SingleStory":
    idx_target = (region_info['NumberOfStories'] == 1).to_numpy()
elif target_structure == "TwoStory":
    idx_target = (region_info['NumberOfStories'] == 2).to_numpy()
elif target_structure == "MultiStory":
    idx_target = (region_info['NumberOfStories'] > 1).to_numpy()
else:
    idx_target = np.ones(len(region_info), dtype=bool)

region_info = region_info[idx_target].reset_index(drop=True)
bldg_capa   = bldg_capa[idx_target, :]

num_bldgs, num_gms = bldg_capa.shape

# %%
# ---------- Estimate baseline fragility parameters ----------
bldg_frag_params = np.zeros((num_bldgs, 3), dtype=dtype_np)
for i in range(num_bldgs):
    temp = bldg_capa[i]
    temp = temp[(~np.isnan(temp)) & (temp != 0)]
    # If too few samples, fallback tiny positive value
    if temp.size < 2:
        temp = np.array([1e-6, 2e-6], dtype=dtype_np)
    shape, loc, scale = lognorm.fit(temp, floc=0)
    bldg_frag_params[i, :] = [dtype_np(shape), dtype_np(loc), dtype_np(scale)]
del temp
free_gpu()

# %%
# ---------- Capacity stats ----------
ln_capa_mean_np     = np.log(bldg_frag_params[:, 2].astype(dtype_np))
ln_capa_std_np      = bldg_frag_params[:, 0].astype(dtype_np)
ln_capa_corr_np     = np.corrcoef(bldg_capa.astype(dtype_np), rowvar=True).astype(dtype_np, copy=False)
ln_capa_cov_np      = np.outer(ln_capa_std_np, ln_capa_std_np) * ln_capa_corr_np
del bldg_capa
free_gpu()

ln_capa_mean_cp     = cp.asarray(ln_capa_mean_np, dtype=dtype_cp)
ln_capa_std_cp      = cp.asarray(ln_capa_std_np, dtype=dtype_cp)
ln_capa_corr_cp     = cp.asarray(ln_capa_corr_np, dtype=dtype_cp)
ln_capa_cov_cp      = cp.asarray(ln_capa_cov_np, dtype=dtype_cp)

# %%
# ---------- Site info ----------
easting, northing, _, _ = utm.from_latlon(region_info['Latitude'].values,
                                          region_info['Longitude'].values)
easting  = easting.astype(dtype_np); northing = northing.astype(dtype_np)

vs30 = region_info['Vs30'].to_numpy().astype(dtype_np)
vs30[vs30 < 180] = 180

assets = pd.DataFrame({'x': easting, 'y': northing, 'Vs30': vs30})
del easting, northing, vs30

repair_cost = cp.asarray(region_info['RepairCost'].to_numpy().astype(dtype_np), dtype=dtype_cp)

# %%
# ---------- Hazard info ----------
# Epicenter UTM
epicenter = (37.666, -122.076)  # (lat, lon)
event_e, event_n, _, _ = utm.from_latlon(*epicenter)
event_xy = (dtype_np(event_e), dtype_np(event_n))

depth_to_top = dtype_np(3.0)
strike, rake, dip = 325, 180, 90

# %%
# ---------- Demand stats ----------
soil_case = 1 
ln_dmnd_corr_cp = intra_residuals_corr_fn_cupy(
    cp.asarray(assets['x'].to_numpy(), dtype=dtype_cp),
    cp.asarray(assets['y'].to_numpy(), dtype=dtype_cp),
    soil_case=soil_case
).astype(dtype_cp)

# %%
# ---------- Output directory ----------
outdir = f"Savio/{target_region}/results_for_[sim_total]"
os.makedirs(outdir, exist_ok=True)

Mw_val = 5.55
# covs = np.round(np.arange(0.0, 1.001, 0.01), 3)

# covs1 = np.round(np.arange(0.80, 1.001, 0.05), 3)
# covs2 = np.round(np.arange(0.76, 1.001, 0.05), 3)
# covs3 = np.round(np.arange(0.77, 1.001, 0.05), 3)
# covs4 = np.round(np.arange(0.78, 1.001, 0.05), 3)
# covs5 = np.round(np.arange(0.79, 1.001, 0.05), 3)
# covs = np.unique(np.concatenate([covs1, covs2, covs3, covs4, covs5]))

covs = np.array([0.6, 0.8, 1.0], dtype=dtype_np)
# covs = np.array([0.57, 0.62, 0.67, 0.72, 0.96, 0.97, 0.98, 0.99], dtype=dtype_np)

num_sim = 10_000
num_disorder_realizations = 50

r_max = 2000
r_max_km = r_max / 1e3

# xi_km = np.zeros_like(covs)
# chi = np.zeros_like(covs)

xi_km = np.full((len(covs), num_disorder_realizations), np.nan, dtype=dtype_np)
chi   = np.full((len(covs), num_disorder_realizations), np.nan, dtype=dtype_np)

base_seed = 12345
t0_all = time.time()
for cov_idx, cov in enumerate(covs):
    t_cov = time.time()

    ds = simulate_ds(
        Mw=Mw_val,  # single Mw for this phase diagram
        cov=cov,
        target_region=target_region,
        idx_target=idx_target,
        num_sim=num_sim,
        num_bldgs=num_bldgs,
        dtype_np=dtype_np,
        dtype_cp=dtype_cp,
        num_disorder_realizations=num_disorder_realizations,
        seed=base_seed + cov_idx,  # different seed for each cov (realizations differ within simulate_ds)
    )  # shape: (R, B, S)

    # np.save(f"{outdir}/ds_cov{int(round(float(cov)*1000)):04d}.npy", ds)

    for nn in range(num_disorder_realizations):
        
        if num_disorder_realizations == 1:
            ds_tmp = ds.copy()  # (B, S)
        else:
            if nn % 10 == 0:
                print(f"    Processing cov={cov:.2f}, realization {nn}...")
            ds_tmp = ds[nn, :, :]           # (B, S)
        ds_mean = ds_tmp.mean(axis=0)   # (S,)

        # mask_eq, (m_lo, m_hi), info = equilibrium_range_from_hist_mode(
        #     ds_mean, bins=50, frac=0.20, value_range=(0.0, 1.0)
        # )
        mask_eq, (m_lo, m_hi), info = equilibrium_range_from_min_width(
            ds_mean, frac=0.10, value_range=(0.0, 1.0)
        )

        ds_eq = ds_tmp[:, mask_eq]      # (B, S_eq)

        if ds_eq.shape[1] < 10:
            print(f"Warning: only {ds_eq.shape[1]} equilibrium samples for cov={cov:.2f}, realization {nn}.")
            continue

        chi[cov_idx, nn] = susceptibility_from_ds_eq(ds_eq, scale_by_B=True)

        r_km, C_r, _ = radial_cov_curve_from_ds(
            ds=ds_eq,
            pairwise_dist_flat=pairwise_dist_flat,
            bin_w_m=20.0,
            rmax_m=r_max,
            residual_cov=True
        )

        try:
            fit = estimate_xi_from_slope(r_km, C_r, rmin_km=0, rmax_km=r_max_km)
            xi_km[cov_idx, nn] = fit["xi_km"]
        except ValueError:
            print(f"Warning: could not estimate xi for cov={cov:.2f}, realization {nn} due to fit error. Setting xi=NaN.")     
    
    # np.save(f"{outdir}/xi_km_cov{int(round(float(cov)*1000)):04d}.npy", xi_km[cov_idx, :])
    # np.save(f"{outdir}/chi_cov{int(round(float(cov)*1000)):04d}.npy", chi[cov_idx, :])
    del ds
    free_gpu()

    print(f"=== cov {cov:.2f} done in {time.time()-t_cov:.2f}s ===")

print(f"\n(Mw={Mw_val:.2f}) All done in {time.time()-t0_all:.2f}s")

# %%
covs = np.round(np.arange(0.0, 1.001, 0.01), 3)
xi_km = np.full((len(covs), num_disorder_realizations), np.nan, dtype=dtype_np)
chi   = np.full((len(covs), num_disorder_realizations), np.nan, dtype=dtype_np)

for cov_idx, cov in enumerate(covs):
    xi_km[cov_idx, :] = np.load(f"{outdir}/xi_km_cov{int(round(float(cov)*1000)):04d}.npy")
    chi[cov_idx, :]   = np.load(f"{outdir}/chi_cov{int(round(float(cov)*1000)):04d}.npy")

# %%
from scipy.ndimage import gaussian_filter
import matplotlib as mpl

def curtain_heatmap_smooth(
    covs,
    Y,                         # (Nsig, R)
    *,
    ylabel,
    yscale=None,               # "log" 가능
    nbins_y=160,
    quantile_clip=(1, 99),
    x_refine=20,
    smooth_sigma_y=1.8,
    smooth_sigma_x=1.0,
    min_density_mode="auto",   # "auto" or "fixed"
    auto_frac=1e-3,
    min_density=0.0,
    savefig=False,
):    
    if "Correlation" in ylabel:
        key = "xi_km"
        color_ = CP_HEX[5]
    else:
        key = "chi"
        color_ = CP_HEX[0]

    covs = np.asarray(covs, float)
    Y = np.asarray(Y, float)

    y_flat = Y[np.isfinite(Y)]
    if y_flat.size == 0:
        raise ValueError("No finite values in Y.")

    # robust y-range
    qlo, qhi = np.percentile(y_flat, quantile_clip)
    if not np.isfinite(qlo) or not np.isfinite(qhi) or qhi <= qlo:
        qlo, qhi = np.nanmin(y_flat), np.nanmax(y_flat)

    # y bins
    if yscale == "log":
        qlo = max(qlo, 1e-12)
        y_edges = np.logspace(np.log10(qlo), np.log10(qhi), nbins_y + 1)
    else:
        y_edges = np.linspace(Y.min(), Y.max(), nbins_y + 1)

    Nsig = len(covs)
    D = np.zeros((nbins_y, Nsig), dtype=float)

    # per-sigma histogram density
    for i in range(Nsig):
        yi = Y[i, :]
        yi = yi[np.isfinite(yi)]
        if yi.size == 0:
            continue
        h, _ = np.histogram(yi, bins=y_edges, density=True)
        D[:, i] = h

    # x refine (interpolate columns)
    if x_refine is not None and x_refine > 1 and Nsig >= 2:
        x_new = np.linspace(covs.min(), covs.max(), (Nsig - 1) * x_refine + 1)
        D_ref = np.empty((nbins_y, x_new.size), dtype=float)
        for k in range(nbins_y):
            D_ref[k, :] = np.interp(x_new, covs, D[k, :])
        x_plot = x_new
        D_plot = D_ref
    else:
        x_plot = covs
        D_plot = D

    # 2D smoothing (y + x)
    sy = float(smooth_sigma_y) if (smooth_sigma_y and smooth_sigma_y > 0) else 0.0
    sx = float(smooth_sigma_x) if (smooth_sigma_x and smooth_sigma_x > 0) else 0.0
    if sy > 0 or sx > 0:
        D_plot = gaussian_filter(D_plot, sigma=(sy, sx), mode="nearest")

    # transparency threshold (after smoothing)
    if min_density_mode == "auto":
        nonzero = D_plot[D_plot > 0]
        thr = (auto_frac * nonzero.max()) if nonzero.size else 0.0
    else:
        thr = float(min_density)

    D_masked = np.ma.masked_less_equal(D_plot, thr)

    # colormap: masked -> transparent
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "CP_custom",
        [(0.0, "white"), (0.5, color_), (1.0, CP_HEX[2])],
        # [(0.0, "white"), (0.33, CP_HEX[0]), (0.66, CP_HEX[2]), (1.0, CP_HEX[1])],
        # [(0.0, "white"), (0.33, CP_HEX[8]), (0.66, CP_HEX[10]), (1.0, CP_HEX[0])],
        # [(0.0, CP_HEX[0]), (0.5, CP_HEX[2]), (1.0, CP_HEX[1])],
        N=256
    )
    cmap.set_bad((0, 0, 0, 0))

    # x edges for pcolormesh
    x = x_plot
    dx0 = x[1] - x[0]
    dx1 = x[-1] - x[-2]
    x_edges = np.concatenate(([x[0] - dx0 / 2], 0.5 * (x[1:] + x[:-1]), [x[-1] + dx1 / 2]))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(covs, Y, linestyle='None', marker='.', markersize=2, color=color_, alpha=1.0, zorder=0)
    pcm = ax.pcolormesh(x_edges, y_edges, D_masked, shading="auto", cmap=cmap, zorder=1, alpha=0.6)

    cb = fig.colorbar(pcm, ax=ax, pad=0.05)
    cb.set_label("Occurrence frequency", fontsize=18)
    vmin, vmax = pcm.get_clim()
    # cb.set_ticks([vmin, vmax])
    # cb.set_ticklabels(["Low", "High"])
    # cb.ax.tick_params(labelsize=10)

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

    plt.tight_layout()
    
    if savefig:
        fname = f"{outdir}/{key}_scatter_with_heatmap.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Saved heatmap: {fname}")

    plt.show()

    return fig, ax

curtain_heatmap_smooth(
    covs, chi,
    ylabel=r"Susceptibility $\chi$",
    yscale="None",
    nbins_y=100,
    x_refine=None,
    smooth_sigma_y=2,
    smooth_sigma_x=1,
    min_density_mode="auto",
    auto_frac=1e-3,   # 3e-4 ~ 3e-3 사이에서 취향 조절
    # savefig=True,
)

curtain_heatmap_smooth(
    covs, xi_km,
    ylabel=r"Correlation length $\xi$ (km)",
    yscale="None",
    nbins_y=100,
    x_refine=None,
    smooth_sigma_y=2,
    smooth_sigma_x=1,
    min_density_mode="auto",
    auto_frac=1e-3,   # 3e-4 ~ 3e-3 사이에서 취향 조절
    # savefig=True,
)

# %%
ds = np.load(f"{outdir}/ds_cov0000.npy")
nn = 3
ds_tmp = ds[nn, :, :]           # (B, S)
ds_mean = ds_tmp.mean(axis=0)   # (S,)
mask_eq, (m_lo, m_hi), info = equilibrium_range_from_min_width(
    ds_mean, frac=0.10, value_range=(0.0, 1.0)
)
ds_eq = ds_tmp[:, mask_eq]      # (B, S_eq)
r_km00, C_r00, _ = radial_cov_curve_from_ds(
    ds=ds_eq,
    pairwise_dist_flat=pairwise_dist_flat,
    bin_w_m=100.0,
    rmax_m=r_max,
    residual_cov=True
)
fit00 = estimate_xi_from_slope(r_km00, C_r00, rmin_km=0, rmax_km=r_max_km)

ds = np.load(f"{outdir}/ds_cov0500.npy")
nn = 2
ds_tmp = ds[nn, :, :]           # (B, S)
ds_mean = ds_tmp.mean(axis=0)   # (S,)
mask_eq, (m_lo, m_hi), info = equilibrium_range_from_min_width(
    ds_mean, frac=0.10, value_range=(0.0, 1.0)
)
ds_eq = ds_tmp[:, mask_eq]      # (B, S_eq)
r_km05, C_r05, _ = radial_cov_curve_from_ds(
    ds=ds_eq,
    pairwise_dist_flat=pairwise_dist_flat,
    bin_w_m=100.0,
    rmax_m=r_max,
    residual_cov=True
)
fit05 = estimate_xi_from_slope(r_km05, C_r05, rmin_km=0, rmax_km=r_max_km)

ds = np.load(f"{outdir}/ds_cov1000.npy")
nn = 3
ds_tmp = ds[nn, :, :]           # (B, S)
ds_mean = ds_tmp.mean(axis=0)   # (S,)
mask_eq, (m_lo, m_hi), info = equilibrium_range_from_min_width(
    ds_mean, frac=0.10, value_range=(0.0, 1.0)
)
ds_eq = ds_tmp[:, mask_eq]      # (B, S_eq)
r_km10, C_r10, _ = radial_cov_curve_from_ds(
    ds=ds_eq,
    pairwise_dist_flat=pairwise_dist_flat,
    bin_w_m=100.0,
    rmax_m=r_max,
    residual_cov=True
)
fit10 = estimate_xi_from_slope(r_km10, C_r10, rmin_km=0, rmax_km=r_max_km)

fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
ax.plot(r_km00, C_r00, linestyle='None', marker='o', markersize=5, color=CP_HEX[0], alpha=0.6, label=r"$\sigma=0.0$")
ax.plot(r_km05, C_r05, linestyle='None', marker='s', markersize=5, color=CP_HEX[2], alpha=0.6, label=r"$\sigma=0.5$")
ax.plot(r_km10, C_r10, linestyle='None', marker='^', markersize=5, color=CP_HEX[3], alpha=0.6, label=r"$\sigma=1.0$")
ax.plot(fit00["rr"], fit00["y_line"], '--', color=CP_HEX[0], linewidth=2)
ax.plot(fit05["rr"], fit05["y_line"], '--', color=CP_HEX[2], linewidth=2)
ax.plot(fit10["rr"], fit10["y_line"], '--', color=CP_HEX[3], linewidth=2)
ax.set_xlabel(r"Distance $r$ (km)", fontsize=16)
ax.set_ylabel(r"Correlation $C(r) / C(0)$", fontsize=16)
ax.tick_params(axis='both',labelsize=12)
ax.set_yscale("log")
ax.legend(
    fontsize=16,
    frameon=False,
    loc="upper right",
    borderaxespad=0.2,   # distance from axes corner
    labelspacing=0.2,    # vertical space between entries
    handlelength=1.0,    # length of legend lines/markers
    handletextpad=0.4,   # space between handle and text
    borderpad=0.1,       # inner padding of legend box
    columnspacing=0.6,   # if you ever use ncol>1
)
# ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
ax.spines[['top', 'right']].set_visible(False)
# fname = f"{outdir}/correlation_fit_examples.png"
# fig.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
# print(f"Saved figure: {fname}")
plt.show()