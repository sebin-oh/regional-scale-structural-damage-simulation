#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 10/14/2025
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

- Estimate mean-field RFIM parameters (a1(Mw), a2(sigma)) from regional DV simulations,
- Plot the implied phase diagram with a bistable regime overlay.
"""


from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import least_squares
from scipy.special import erf, erfinv

import json
import warnings

# =============================================================================
# User settings
# =============================================================================
SaveFig = False
RESOLUTION = 501

TARGET_REGION = "Milpitas"
TARGET_STRUCTURE: Literal["SingleStory", "TwoStory", "MultiStory"] = "MultiStory"

DATA_DIR = Path("../data/damage_simulation_results")
FIG_DIR = Path("../results")

USE_SAVED_PARAMS = True          # if True and file exists -> skip estimation and load coeffs
SAVE_PARAMS_AFTER_FIT = True     # if True -> save coeffs after estimating

RFIM_PARAMS_PATH = Path("../data/rfim_params_est/{TARGET_REGION}_{TARGET_STRUCTURE}_rfim_coeffs.npz")

MWS = np.round(np.arange(3.5, 8.51, 0.05), 2)
SIGMAS = np.round(np.arange(0.0, 1.001, 0.01), 3)

NUM_SIM = 10_000
HIST_BINS = 100
TOP_MASS_FRAC = 0.01  # use bins that collectively contain top 1% of samples
SIGMA_FIT_START_IDX = 60  # match your previous: fit only on sigmas[60:]

# Fitting knobs
RIDGE = 1e-4
MAX_ITERS = 1000
TOL = 1e-8
ETA = 1.0  # damping in alternating updates (1.0 = no damping)

# Width penalty for a2 fitting (encourages wider transitions)
LAM_WIDTH = 1e-2
M_BAND = 0.6
A2_FLOOR = 0.0
ENFORCE_MONOTONE_A2 = True


# =============================================================================
# File/path helpers
# =============================================================================
def mw_label(mw: float) -> str:
    return f"{int(round(float(mw) * 100)):03d}"


def sigma_label(sigma: float) -> str:
    return f"{int(round(float(sigma) * 100)):03d}"


def dv_file(
    *,
    region: str,
    structure: str,
    mw: float,
    sigma: float,
    cost: bool = False,
) -> Path:
    kind = "repair_cost" if cost else "damage_fraction"
    fname = f"{region}_{structure}_{kind}_Mw{mw_label(mw)}_sigma{sigma_label(sigma)}.npy"
    return DATA_DIR / region / kind / fname


def save_mf_params(path: Path, *, coeffs_a1: np.ndarray, coeffs_a2: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        coeffs_a1=np.asarray(coeffs_a1, dtype=float),
        coeffs_a2=np.asarray(coeffs_a2, dtype=float),
        meta=json.dumps(meta),
    )

def load_mf_params(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    obj = np.load(path, allow_pickle=False)
    coeffs_a1 = np.asarray(obj["coeffs_a1"], dtype=float)
    coeffs_a2 = np.asarray(obj["coeffs_a2"], dtype=float)
    meta = json.loads(str(obj["meta"])) if "meta" in obj.files else {}
    return coeffs_a1, coeffs_a2, meta

def make_rfim_meta() -> dict:
    return {
        "target_region": TARGET_REGION,
        "target_structure": TARGET_STRUCTURE,
        "mws_minmax": [float(MWS.min()), float(MWS.max())],
        "sigmas_minmax": [float(SIGMAS.min()), float(SIGMAS.max())],
        "sigma_fit_start_idx": int(SIGMA_FIT_START_IDX),
        "hist_bins": int(HIST_BINS),
        "top_mass_frac": float(TOP_MASS_FRAC),
        "num_sim": int(NUM_SIM),
        "ridge": float(RIDGE),
        "lam_width": float(LAM_WIDTH),
        "m_band": float(M_BAND),
        "a2_floor": float(A2_FLOOR),
        "monotone_a2": bool(ENFORCE_MONOTONE_A2),
        "a1_poly_degree": 2,
        "a2_poly_degree": 1,
    }

def validate_loaded_meta(meta: dict) -> None:
    # Soft checks only (warn but continue)
    if not meta:
        return
    if meta.get("target_region") != TARGET_REGION or meta.get("target_structure") != TARGET_STRUCTURE:
        warnings.warn(
            f"Loaded params were saved for region/structure "
            f"{meta.get('target_region')}/{meta.get('target_structure')} "
            f"but current is {TARGET_REGION}/{TARGET_STRUCTURE}.",
            RuntimeWarning,
        )


# =============================================================================
# DV -> top-bin mean
# =============================================================================
def top_mass_bin_mean(x: np.ndarray, *, bins: int, top_mass_frac: float, num_sim: int) -> float:
    """
    Take histogram bins in decreasing count order until cumulative count exceeds
    (top_mass_frac * num_sim), then return the mean of samples falling in those bins.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan

    counts, edges = np.histogram(x, bins=bins)
    order = np.argsort(counts)[::-1]

    needed = top_mass_frac * num_sim
    cum = 0
    selected_bins = []
    for b in order:
        if cum > needed:
            break
        selected_bins.append(b)
        cum += counts[b]

    selected_bins = np.asarray(selected_bins, dtype=int)
    bmin = edges[selected_bins]
    bmax = edges[selected_bins + 1]

    mask = np.zeros_like(x, dtype=bool)
    for lo, hi in zip(bmin, bmax):
        mask |= (x >= lo) & (x < hi)

    if not np.any(mask):
        return float(np.mean(x))

    return float(np.mean(x[mask]))


def load_top_values_mean(
    *,
    region: str,
    structure: str,
    mws: np.ndarray,
    sigmas: np.ndarray,
    bins: int,
    top_mass_frac: float,
    num_sim: int,
) -> np.ndarray:
    """
    Return top_values_mean[sigma_idx, mw_idx] in [0,1].
    """
    out = np.empty((sigmas.size, mws.size), dtype=float)

    for si, s in enumerate(sigmas):
        print(f"sigma row {si+1}/{sigmas.size}")
        for mi, mw in enumerate(mws):
            p = dv_file(region=region, structure=structure, mw=float(mw), sigma=float(s), cost=False)
            if not p.exists():
                raise FileNotFoundError(f"Missing DV file: {p}")
            dv = np.load(p)
            out[si, mi] = top_mass_bin_mean(dv, bins=bins, top_mass_frac=top_mass_frac, num_sim=num_sim)

    return out


# =============================================================================
# Mean-field fitting utilities
# =============================================================================
def _clip_m(M: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    M2 = M.copy()
    M2[mask] = np.clip(M2[mask], -1 + eps, 1 - eps)
    return M2


def fit_grid_a1_a2(
    M: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    a1_init: Optional[np.ndarray] = None,
    a2_init: Optional[np.ndarray] = None,
    sign_targets: Optional[np.ndarray] = None,
    ridge: float = 0.0,
    sign_rule: Literal["mean", "median"] = "mean",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Estimate column-wise a1[j] and row-wise a2[i] > 0 from:
        M[i,j] ≈ erf((M[i,j] + a1[j]) / (sqrt(2) * a2[i]))

    Constraint:
        a1[j] = s[j] * exp(alpha[j])   where s[j] ∈ {−1,+1}.
    """
    M = np.asarray(M, float)
    R, C = M.shape

    if mask is None:
        mask = np.isfinite(M)
    else:
        mask = mask & np.isfinite(M)

    M_use = _clip_m(M, mask)

    # infer sign targets per column if not given
    if sign_targets is None:
        col_stat = np.nanmean(M_use, axis=0) if sign_rule == "mean" else np.nanmedian(M_use, axis=0)
        s = np.sign(col_stat)
        zero_cols = (s == 0)
        if np.any(zero_cols):
            maj = np.sign(np.nanmean(np.sign(M_use[:, zero_cols]), axis=0))
            s[zero_cols] = np.where(maj == 0, 1.0, maj)
        sign_targets = s.astype(float)
    else:
        sign_targets = np.sign(np.asarray(sign_targets, float))
        sign_targets[sign_targets == 0] = 1.0

    # init a2
    if a2_init is None:
        a2 = np.full(R, 0.6, dtype=float)
    else:
        a2 = np.asarray(a2_init, float)
        a2[~np.isfinite(a2) | (a2 <= 0)] = 0.6

    invM = np.zeros_like(M_use)
    invM[mask] = erfinv(M_use[mask])

    # init a1 from rearrangement using current a2
    if a1_init is None:
        a1_guess = np.zeros(C, dtype=float)
        for j in range(C):
            rows = mask[:, j]
            if np.any(rows):
                vals = np.sqrt(2.0) * a2[rows] * invM[rows, j] - M_use[rows, j]
                vals = vals[np.isfinite(vals)]
                a1_guess[j] = np.median(vals) if vals.size else 0.0
    else:
        a1_guess = np.asarray(a1_init, float)
        a1_guess[~np.isfinite(a1_guess)] = 0.0

    tiny = 1e-9
    alpha0 = np.log(np.maximum(sign_targets * a1_guess, tiny))

    # one refinement for a2 with current a1
    a1_signed = sign_targets * np.exp(alpha0)
    for i in range(R):
        cols = mask[i, :]
        inv = invM[i, cols]
        good = np.abs(inv) > 1e-8
        if np.any(good):
            cand = (M_use[i, cols][good] + a1_signed[cols][good]) / (np.sqrt(2.0) * inv[good])
            cand = cand[(cand > 0) & np.isfinite(cand)]
            if cand.size:
                a2[i] = np.median(cand)

    # pack params: a1 = s*exp(alpha), a2 = exp(b2)
    b2 = np.log(np.clip(a2, 1e-6, None))
    theta0 = np.concatenate([alpha0, b2])

    ii, jj = np.where(mask)
    M_flat = M_use[ii, jj]

    def residuals(theta):
        alpha = theta[:C]
        b2 = theta[C:]
        a1 = sign_targets * np.exp(alpha)
        a2 = np.exp(b2)
        r = erf((M_flat + a1[jj]) / (np.sqrt(2.0) * a2[ii])) - M_flat
        if ridge > 0:
            r = np.concatenate([r, np.sqrt(ridge) * alpha, np.sqrt(ridge) * b2])
        return r

    res = least_squares(
        residuals,
        theta0,
        method="trf",
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=100000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    th = res.x
    a1_hat = sign_targets * np.exp(th[:C])
    a2_hat = np.exp(th[C:])

    if verbose:
        print("=== Fit summary (sign-constrained a1) ===")
        print(f"Rows (a2): {R}, Cols (a1): {C}")
        print(f"Cost: {0.5*np.sum(res.fun**2):.6g} | success={res.success}")

    return a1_hat, a2_hat, res


def fit_cols_a1_given_a2(M: np.ndarray, *, a2: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Fit a1[j] given fixed row-wise a2[i]."""
    M = np.asarray(M, float)
    a2 = np.asarray(a2, float)
    R, C = M.shape
    if a2.shape != (R,):
        raise ValueError(f"a2 must have shape ({R},), got {a2.shape}")
    if np.any(~np.isfinite(a2)) or np.any(a2 <= 0):
        raise ValueError("a2 must be finite and > 0.")

    if mask is None:
        mask = np.isfinite(M)
    else:
        mask = mask & np.isfinite(M)

    M_use = _clip_m(M, mask)
    invM = np.zeros_like(M_use)
    invM[mask] = erfinv(M_use[mask])

    # init a1 per column
    a1_0 = np.zeros(C, dtype=float)
    for j in range(C):
        rows = mask[:, j]
        if np.any(rows):
            vals = np.sqrt(2.0) * a2[rows] * invM[rows, j] - M_use[rows, j]
            vals = vals[np.isfinite(vals)]
            a1_0[j] = np.median(vals) if vals.size else 0.0

    ii, jj = np.where(mask)
    M_flat = M_use[ii, jj]
    a2_flat = a2[ii]

    def residuals(a1_vec):
        r = erf((M_flat + a1_vec[jj]) / (np.sqrt(2.0) * a2_flat)) - M_flat
        return r

    res = least_squares(
        residuals,
        a1_0,
        method="trf",
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=100000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    return res.x


def fit_rows_a2_given_a1_with_width(
    M: np.ndarray,
    *,
    a1: np.ndarray,
    mask: Optional[np.ndarray] = None,
    a2_init: Optional[np.ndarray] = None,
    lam_width: float = 0.0,
    m_band: float = 0.6,
    a2_floor: float = 0.0,
    monotone: bool = False,
) -> np.ndarray:
    """
    Fit row-wise a2[i] > 0 given a1[j], with an optional width penalty
    discouraging steep transitions near |m|≈0.
    """
    M = np.asarray(M, float)
    a1 = np.asarray(a1, float)
    R, C = M.shape
    if a1.shape != (C,):
        raise ValueError(f"a1 must have shape ({C},), got {a1.shape}")

    if mask is None:
        mask = np.isfinite(M)
    else:
        mask = mask & np.isfinite(M)

    M_use = _clip_m(M, mask)
    invM = np.zeros_like(M_use)
    invM[mask] = erfinv(M_use[mask])

    # init a2
    if a2_init is None:
        a2 = np.full(R, a2_floor + 1e-6, float)
        for i in range(R):
            cols = mask[i, :]
            inv = invM[i, cols]
            good = np.abs(inv) > 1e-8
            if np.any(good):
                num = M_use[i, cols][good] + a1[cols][good]
                cand = num / (np.sqrt(2.0) * inv[good])
                cand = cand[(cand > 0) & np.isfinite(cand)]
                if cand.size:
                    a2[i] = max(np.median(cand), a2_floor + 1e-6)
    else:
        a2 = np.asarray(a2_init, float)
        a2[~np.isfinite(a2) | (a2 <= 0)] = a2_floor + 1e-6

    a2 = np.maximum(a2, a2_floor + 1e-9)
    b2_0 = np.log(a2 - a2_floor)

    if monotone and R > 1:
        b2_0 = np.maximum.accumulate(b2_0)

    ii, jj = np.where(mask)
    m_obs = M_use[ii, jj]
    a1_flat = a1[jj]

    def residuals(b2):
        if monotone and R > 1:
            b2 = np.maximum.accumulate(b2 + 1e-8 * np.arange(R))

        a2_rows = a2_floor + np.exp(b2)
        den = np.sqrt(2.0) * a2_rows[ii]
        r_data = erf((m_obs + a1_flat) / den) - m_obs

        if lam_width <= 0:
            return r_data

        z = (m_obs + a1_flat) / den
        K = (np.sqrt(2.0 / np.pi) * np.exp(-z * z)) / a2_rows[ii]
        K = np.clip(K, -0.99, 0.99)
        chi = K / (1.0 - K)
        w = np.exp(-((m_obs / m_band) ** 2))
        r_width = np.sqrt(lam_width) * w * chi

        return np.concatenate([r_data, r_width])

    res = least_squares(
        residuals,
        b2_0,
        method="trf",
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=100000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    return a2_floor + np.exp(res.x)


# =============================================================================
# Polynomial fits with nonnegative slope/intercept
# =============================================================================
def fit_linear_nonneg(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit y ≈ c1*x + c0 with c1>=0 and c0>=0.
    Returns coeffs in np.polyval order: [c1, c0].
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    def resid(c):
        return np.polyval(c, x) - y

    c0 = np.polyfit(x, y, 1).astype(float)
    c0[0] = max(c0[0], 0.0)
    c0[1] = max(c0[1], 0.0)

    lb = [0.0, 0.0]
    ub = [np.inf, np.inf]
    res = least_squares(resid, c0, bounds=(lb, ub))
    return res.x


def find_real_root_in_range(coeffs: np.ndarray, lo: float, hi: float) -> float:
    """Pick a real root of a polynomial within [lo, hi]."""
    roots = np.roots(coeffs)
    roots_real = roots[np.isclose(roots.imag, 0.0, atol=1e-8)].real
    in_rng = roots_real[(roots_real >= lo) & (roots_real <= hi)]
    if in_rng.size:
        # choose the one closest to mid-range (or you can choose closest to 0 crossing)
        mid = 0.5 * (lo + hi)
        return float(in_rng[np.argmin(np.abs(in_rng - mid))])

    # fallback: choose the real root with smallest imaginary part
    idx = int(np.argmin(np.abs(roots.imag)))
    return float(roots[idx].real)


# =============================================================================
# Fixed-point solver for phase diagram
# =============================================================================
def fixed_point_solve(
    a1g: np.ndarray,
    a2g: np.ndarray,
    m0: np.ndarray,
    *,
    tol: float = 1e-8,
    max_iter: int = 10000,
    omega: float = 1.0,
) -> np.ndarray:
    """Vectorized fixed-point iteration for m = erf((m+a1)/(sqrt(2)*a2))."""
    sqrt2 = np.sqrt(2.0)
    m = m0.copy()
    prev_delta = np.inf
    w = float(omega)

    for _ in range(max_iter):
        rhs = erf((m + a1g) / (sqrt2 * a2g))
        m_new = (1 - w) * m + w * rhs
        delta = float(np.max(np.abs(m_new - m)))
        m = m_new

        if delta > prev_delta * 1.05 and w > 0.2:
            w *= 0.9
        prev_delta = delta

        if delta < tol:
            break

    return np.clip(m, -1.0, 1.0)


# =============================================================================
# Main workflow
# =============================================================================
def main() -> None:
    # --- Load saved coeffs if requested ---
    if USE_SAVED_PARAMS and RFIM_PARAMS_PATH.exists():
        coeffs_a1, coeffs_a2, meta = load_mf_params(RFIM_PARAMS_PATH)
        validate_loaded_meta(meta)
        print(f"Loaded mean-field parameters: {RFIM_PARAMS_PATH}")
        a1_prev = None
        a2_prev = None
        sigmas_fit = SIGMAS[SIGMA_FIT_START_IDX:]  # keep for plotting axes/labels
        mws_orig = MWS.copy()
        sigmas_orig = SIGMAS.copy()

    else:
        # --- 1) Load DV summaries ---
        print("Loading damage simulation data...")
        top_values_mean = load_top_values_mean(
            region=TARGET_REGION,
            structure=TARGET_STRUCTURE,
            mws=MWS,
            sigmas=SIGMAS,
            bins=HIST_BINS,
            top_mass_frac=TOP_MASS_FRAC,
            num_sim=NUM_SIM,
        )
        print("Done\n")

        # Keep originals for plotting grids
        top_values_mean_orig = top_values_mean.copy()
        mws_orig = MWS.copy()
        sigmas_orig = SIGMAS.copy()

        # --- 2) Choose subset for fitting (match your earlier behavior) ---
        sigmas_fit = sigmas_orig[SIGMA_FIT_START_IDX:]
        top_fit = top_values_mean_orig[SIGMA_FIT_START_IDX:, :]
        m_data = top_fit * 2.0 - 1.0  # map DV ∈ [0,1] to m ∈ [-1,1]

        # --- 3) Initial joint fit (a1 per Mw column, a2 per sigma row) ---
        a1_hat0, a2_hat0, _ = fit_grid_a1_a2(
            m_data,
            a2_init=np.zeros(m_data.shape[0]),
            ridge=RIDGE,
            verbose=False,
        )

        # --- 4) Initialize smooth parameterizations ---
        coeffs_a1 = np.polyfit(mws_orig, a1_hat0, 2)          # quadratic in Mw
        coeffs_a2 = fit_linear_nonneg(sigmas_fit, a2_hat0)    # linear in sigma, slope>=0, intercept>=0

        # --- 5) Alternating refinement ---
        a1_prev = a1_hat0.copy()
        a2_prev = a2_hat0.copy()

        print("Proceeds parameter fitting...")
        for it in range(MAX_ITERS):
            a2_on_sigma = np.clip(np.polyval(coeffs_a2, sigmas_fit), 1e-8, None)
            a1_new = fit_cols_a1_given_a2(m_data, a2=a2_on_sigma)
            coeffs_a1 = np.polyfit(mws_orig, a1_new, 2)

            a1_on_mw = np.polyval(coeffs_a1, mws_orig)
            a2_new = fit_rows_a2_given_a1_with_width(
                m_data,
                a1=a1_on_mw,
                lam_width=LAM_WIDTH,
                m_band=M_BAND,
                a2_floor=A2_FLOOR,
                monotone=ENFORCE_MONOTONE_A2,
            )
            coeffs_a2 = fit_linear_nonneg(sigmas_fit, a2_new)

            diff1 = float(np.mean((a1_prev - a1_new) ** 2))
            diff2 = float(np.mean((a2_prev - a2_new) ** 2))
            print(it, diff1, diff2)

            a1_prev = (1 - ETA) * a1_prev + ETA * a1_new
            a2_prev = (1 - ETA) * a2_prev + ETA * a2_new

            if max(diff1, diff2) <= TOL:
                break

        print("Done:", it, diff1, diff2)

        # --- Save coeffs if requested ---
        if SAVE_PARAMS_AFTER_FIT:
            meta = make_rfim_meta()
            save_mf_params(RFIM_PARAMS_PATH, coeffs_a1=coeffs_a1, coeffs_a2=coeffs_a2, meta=meta)
            print(f"Saved mean-field RFIM parameters: {RFIM_PARAMS_PATH}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- 6) Plot a1(Mw) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mws_orig, a1_prev, "r.", label=r"Estimated $a_{1}$")
    ax.plot(mws_orig, np.polyval(coeffs_a1, mws_orig), "k--", lw=2.0, label="Quadratic fit")
    ax.set_xticks([4, 5, 6, 7, 8])
    ax.set_yticks([-3, -2, -1, 0, 1, 2])
    ax.set_xlabel(r"Earthquake magnitude $M_{w}$", fontsize=20)
    ax.set_ylabel(r"$a_{1}\,\left(=H/J\right)$", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=18)
    fig.tight_layout()
    if SaveFig:
        out = FIG_DIR / "param_est_Mw_to_a1.png"
        fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()

    # critical Mw: a1(Mw)=0
    Mw_c = find_real_root_in_range(coeffs_a1, float(mws_orig.min()), float(mws_orig.max()))
    print(f"Critical magnitude Mw: {Mw_c:.3f}")

    # --- 7) Plot a2(sigma) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(sigmas_fit, a2_prev, "r.", label=r"Estimated $a_{2}$")
    ax.plot(sigmas_fit, np.polyval(coeffs_a2, sigmas_fit), "k--", lw=2.0, label="Linear fit")
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel(r"Structural diversity $\sigma$", fontsize=20)
    ax.set_ylabel(r"$a_{2}\,\left(=\Delta/J\right)$", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=18)
    fig.tight_layout()
    if SaveFig:
        out = FIG_DIR / "param_est_sigma_to_a2.png"
        fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()

    # critical sigma from a2(sigma)=sqrt(2/pi)
    a2_c = np.sqrt(2 / np.pi)
    c1, c0 = coeffs_a2  # a2 = c1*sigma + c0
    sigma_c = (a2_c - c0) / c1 if c1 > 0 else np.nan
    print(f"Critical diversity σ = {sigma_c:.3f}")

    # --- 8) Phase diagram grid ---
    mws_grid = np.linspace(mws_orig.min(), mws_orig.max(), RESOLUTION)
    sigmas_grid = np.linspace(sigmas_orig.min(), sigmas_orig.max(), RESOLUTION)

    a1_of_mw = np.polyval(coeffs_a1, mws_grid)
    a2_of_sigma = np.clip(np.polyval(coeffs_a2, sigmas_grid), 1e-12, None)

    Mw_grid, Sigma_grid = np.meshgrid(mws_grid, sigmas_grid, indexing="xy")
    a1_grid = np.tile(a1_of_mw, (sigmas_grid.size, 1))
    a2_grid = np.tile(a2_of_sigma[:, None], (1, mws_grid.size))

    # Solve from both branches
    m_plus = fixed_point_solve(a1_grid, a2_grid, np.ones_like(a1_grid))
    m_minus = fixed_point_solve(a1_grid, a2_grid, -np.ones_like(a1_grid))

    damage_plus = 0.5 * (m_plus + 1.0)
    damage_minus = 0.5 * (m_minus + 1.0)

    # bistability mask
    eps_bi = 1e-4
    bistable = (np.abs(m_plus - m_minus) > eps_bi).astype(float)

    # equilibrium damage (choose branch by sign of a1; average at a1≈0)
    tol0 = 1e-12
    damage_eq = np.where(
        a1_grid > tol0,
        damage_plus,
        np.where(a1_grid < -tol0, damage_minus, 0.5 * (damage_plus + damage_minus)),
    )

    # --- 9) Plot phase diagram ---
    SIG, MAG = np.meshgrid(sigmas_grid, mws_grid)

    vmin, vmax = 0.0, 1.0
    base = plt.cm.gray_r(np.linspace(0, 1, 256))
    base[:, -1] = np.linspace(0, 0.6, 256)
    transparent_gray = ListedColormap(base)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    c = ax.contourf(
        SIG,
        MAG,
        damage_eq.T,
        levels=100,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )

    # Transition boundary and critical point
    ax.plot([0.0, sigma_c], [Mw_c, Mw_c], "-", lw=2.0, color="white", zorder=3)
    ax.plot(sigma_c, Mw_c, "o", ms=10, markerfacecolor="black", markeredgecolor="white", zorder=5)

    # Bistable overlay
    ax.contourf(
        SIG,
        MAG,
        bistable.T,
        levels=10,
        cmap=transparent_gray,
        vmin=vmin,
        vmax=vmax,
        zorder=4,
    )

    # Contour labels
    levels = [0.05, 0.1, 0.9, 0.95]
    label_positions = [(0.86, 6.5), (0.87, 6.5), (0.85, 6.5), (0.84, 6.5)]
    for lvl, pos in zip(levels, label_positions):
        cs = ax.contour(SIG, MAG, damage_eq.T, levels=[lvl], colors="black", linewidths=0.5, linestyles="--", zorder=2)
        ax.clabel(cs, fmt=f"{lvl:.2f}", fontsize=14, inline=True, manual=[pos])

    cb = plt.colorbar(c, ax=ax)
    cb.set_label("Damage fraction", fontsize=20)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=16)

    ax.set_xlabel(r"Structural diversity $\sigma$", fontsize=20)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r"Earthquake magnitude $M_w$", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)

    fig.tight_layout()
    if SaveFig:
        out = FIG_DIR / "phase_diagram_RFIM_bistable_clean.png"
        fig.savefig(out, dpi=300, transparent=True, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()