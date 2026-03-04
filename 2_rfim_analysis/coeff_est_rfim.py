#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 10/14/2025
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

This script estimates the relationship between Mw-H and COV-\Delta
using the zero-temperature mean-field solution for the RFIM.
The H and \Delta are represented by a1 and a2 in the self-consistent equation:
    m = erf((m + a1) / (sqrt(2) * a2)),
where a1 = H/J and a2 = \Delta/J.

The estimation is done in two steps:
1) For each (Mw, COV) pair, estimate (a1_hat, a2_hat) from the simulated damage data
   using nonlinear least squares.
2) Fit a1_hat(Mw) and a2_hat(COV) using quadratic polynomials, with the constraint
    that the linear coefficient of a2_hat(COV) is nonnegative (to ensure a2 grows
    with COV).

A fine tuning is further conducted using the estimates from step 2 as initial guesses:
    a) Fix a2(COV) and estimate a1(Mw)
    b) Fix a1(Mw) and estimate a2(COV)
    c) Repeat a) and b) until convergence.

While optimizing a2(COV), a penalty is added to discourage very steep transitions
to make the estimates more realistic. The penalty is based on the susceptibility χ.

The final outputs are the fitted coefficients for a1(Mw) and a2(COV),
along with the reproduced phase diagram.

### NOTE ###
Only 

"""
# %%
import numpy as np
from math import erf, sqrt, isfinite, copysign
from scipy.optimize import newton, root_scalar
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "mathtext.fontset": "cm",
})

SaveFig = False  # set to True to save figures
RESOLUTION = 501  # grid resolution for plots


# %%
from pathlib import Path
from typing import Tuple, Optional, Literal

StructureType = Literal["SingleStory", "TwoStory", "MultiStory"]
ModeType = Literal["1st", "2nd"]

def _dv_filename(
    target_structure: StructureType,
    target_region: str,
    mode: ModeType,
    cost: bool,
    Mw: Optional[float],
    cov: Optional[float],
    corr: Optional[float],
    category: Optional[str],
) -> Tuple[Path, str]:
    """Build DV filename and a short English title."""
    base = f"C:/Users/ohsb1/Desktop/Savio/{target_region}/{target_structure}_dv"
    if cost:
        base += "_cost"

    Mw_label = f"{int(round(Mw*100)):03d}"
    cov_label = f"{int(round(cov*1000)):04d}"

    suffix = f"_Mw{Mw_label}_cov{cov_label}_{target_region}"

    fname = f"{base}{suffix}.npy"
    title = None

    return Path(fname), title

Mws = np.round(np.arange(3.5, 8.51, 0.05), 2)
covs = np.round(np.arange(0.0, 1.001, 0.01), 3)

num_sim = 10_000

top_values_mean = []
for cov_idx, cov_val in enumerate(covs):
    print(cov_idx)

    means_per_Mw = []
    for Mw_idx, Mw_val in enumerate(Mws):
        dv_path, _ = _dv_filename("MultiStory", "Milpitas_all", "1st", None, Mw_val, cov_val, None, None)
        if not dv_path.exists():
            raise FileNotFoundError(f"Missing DV file: {dv_path}")
        dv_temp = np.load(dv_path)

        counts, bin_edges = np.histogram(dv_temp, bins=100)

        sorted_bins = np.argsort(counts)[::-1]

        count = 0
        selected_bin_indices = []
        for bin_idx in sorted_bins:
            if count > 0.01 * num_sim:
                break
            selected_bin_indices.append(bin_idx)
            count += counts[bin_idx]

        selected_bin_indices = np.array(selected_bin_indices)
        bin_mins = bin_edges[selected_bin_indices]
        bin_maxs = bin_edges[selected_bin_indices + 1]

        mask = np.zeros_like(dv_temp, dtype=bool)
        for bmin, bmax in zip(bin_mins, bin_maxs):
            mask |= (dv_temp >= bmin) & (dv_temp < bmax)

        dv_temp_selected = dv_temp[mask]
        means_per_Mw.append(np.mean(dv_temp_selected))

    top_values_mean.append(means_per_Mw)

top_values_mean = np.array(top_values_mean)

# %%
top_values_mean_orig = top_values_mean.copy()
Mws_orig = Mws.copy()
covs_orig = covs.copy()

top_values_mean = top_values_mean_orig[60:,:]
Mws = Mws_orig.copy()
covs = covs_orig[60:]

# %%
from scipy.optimize import least_squares
from scipy.special import erf, erfinv

def mf_self_consistent_eq(m, a1, a2):
    """Mean-field prediction of magnetization m from parameters a1, a2."""
    return erf((m + a1) / (np.sqrt(2.0) * a2))

def mf_self_consistent_eq_res(m, a1, a2):
    """Residuals for mean-field self-consistent equation."""
    return m - mf_self_consistent_eq(m, a1, a2)

def fit_grid_a1_a2(
    M, mask=None, a1_init=None, a2_init=None,
    sign_targets=None,           # optional array of shape (C,) with entries -1 or +1
    ridge=0.0,                   # small L2 on alpha and b2 (log a2)
    sign_rule="mean",            # "mean" or "median" to infer signs from each column of M
    verbose=True
    ):
    """
    Estimate column-wise a1[j] and row-wise a2[i]>0 from:
        M[i,j] ≈ erf( (M[i,j] + a1[j]) / (sqrt(2) * a2[i]) )

    with the constraint that a1[j] has the same sign as the column's magnetization:
        a1[j] = sign_targets[j] * exp(alpha[j])  (so sign(a1[j]) = sign_targets[j])

    Notes
    -----
    - If a column has mixed signs (both + and - magnetization), the sign is inferred
      from the column mean/median. You can pass `sign_targets` explicitly to override.
    - a2[i] is enforced positive via a2[i] = exp(b2[i]).
    """
    M = np.asarray(M, dtype=float)
    R, C = M.shape
    eps_clip = 1e-6

    # mask & clip
    if mask is None:
        mask = np.isfinite(M)
    else:
        mask = mask & np.isfinite(M)

    M_use = M.copy()
    M_use[mask] = np.clip(M_use[mask], -1 + eps_clip, 1 - eps_clip)

    # --------- choose sign targets per column ----------
    if sign_targets is None:
        col_stat = np.nanmean(M_use, axis=0) if sign_rule == "mean" else np.nanmedian(M_use, axis=0)
        s = np.sign(col_stat)
        # fallback: if exactly zero, look at majority sign; default to +1 if still ambiguous
        zero_cols = (s == 0)
        if np.any(zero_cols):
            maj = np.sign(np.nanmean(np.sign(M_use[:, zero_cols]), axis=0))
            s[zero_cols] = np.where(maj == 0, 1.0, maj)
        sign_targets = s.astype(float)
    else:
        sign_targets = np.asarray(sign_targets, dtype=float)
        assert sign_targets.shape == (C,)
        sign_targets[sign_targets == 0] = 1.0
        sign_targets = np.sign(sign_targets)

    # --------- init a2 (rows) ----------
    if a2_init is None:
        a2 = np.full(R, 0.6, dtype=float)
    else:
        a2 = np.asarray(a2_init, dtype=float)
        a2[~np.isfinite(a2) | (a2 <= 0)] = 0.6

    # helper
    invM = np.zeros_like(M_use)
    invM[mask] = erfinv(M_use[mask])

    # --------- init a1 (cols) from rearrangement with current a2 ----------
    if a1_init is None:
        a1_guess = np.zeros(C, dtype=float)
        for j in range(C):
            rows = mask[:, j]
            if np.any(rows):
                vals = np.sqrt(2.0) * a2[rows] * invM[rows, j] - M_use[rows, j]
                vals = vals[np.isfinite(vals)]
                a1_guess[j] = np.median(vals) if vals.size else 0.0
            else:
                a1_guess[j] = 0.0
    else:
        a1_guess = np.asarray(a1_init, dtype=float)
        a1_guess[~np.isfinite(a1_guess)] = 0.0

    # convert a1_guess to alpha init respecting sign: a1 = s*exp(alpha) => alpha = log(max(s*a1, tiny))
    tiny = 1e-9
    alpha0 = np.log(np.maximum(sign_targets * a1_guess, tiny))

    # one refinement for a2 with current a1 (using signed a1)
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

    # pack params: theta = [alpha(0..C-1), b2(0..R-1)] with
    # a1 = s * exp(alpha), a2 = exp(b2)
    b2 = np.log(np.clip(a2, 1e-6, None))
    theta0 = np.concatenate([alpha0, b2])

    # pre-index used entries
    ii, jj = np.where(mask)
    M_flat = M_use[ii, jj]

    # ridge anchors (zeros) for alpha and b2
    alpha_anchor = np.zeros(C)
    b2_anchor    = np.zeros(R)

    def residuals(theta):
        alpha = theta[:C]
        b2    = theta[C:]
        a1    = sign_targets * np.exp(alpha)
        a2    = np.exp(b2)

        num = M_flat + a1[jj]
        den = np.sqrt(2.0) * a2[ii]
        r = erf(num / den) - M_flat

        if ridge > 0:
            r = np.concatenate([
                r,
                np.sqrt(ridge) * (alpha - alpha_anchor),
                np.sqrt(ridge) * (b2    - b2_anchor),
            ])
        return r

    result = least_squares(
        residuals, theta0,
        method="trf",
        loss="soft_l1", f_scale=0.01,
        max_nfev=100000,
        xtol=1e-12, ftol=1e-12, gtol=1e-12
    )

    th = result.x
    alpha_hat = th[:C]
    b2_hat    = th[C:]
    a1_hat    = sign_targets * np.exp(alpha_hat)
    a2_hat    = np.exp(b2_hat)

    if verbose:
        print("=== Fit summary (sign-constrained a1) ===")
        print(f"Shape: {R} rows (a2), {C} cols (a1)")
        print(f"Cost (1/2||r||^2): {0.5*np.sum(result.fun**2):.6g}")
        print(f"Success: {result.success}  Message: {result.message}")

    return a1_hat, a2_hat, result

def fit_cols_a1_given_a2(M, a2, mask=None, a1_init=None, ridge=0.0, verbose=True):
    M = np.asarray(M, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    R, C = M.shape
    assert a2.shape == (R,), "a2 must have shape (R,)"
    if not np.all(np.isfinite(a2)) or np.any(a2 <= 0):
        raise ValueError("All a2 entries must be finite and > 0.")

    # Mask & clip M for numerical stability
    if mask is None:
        mask = np.isfinite(M)
    else:
        mask = mask & np.isfinite(M)
    eps_clip = 1e-6
    M_use = M.copy()
    M_use[mask] = np.clip(M_use[mask], -1 + eps_clip, 1 - eps_clip)

    # Precompute pieces
    sqrt2 = np.sqrt(2.0)
    invM = np.zeros_like(M_use)
    invM[mask] = erfinv(M_use[mask])

    # ----- Initial a1 per column (median of rearranged estimates over available rows) -----
    if a1_init is None:
        a1 = np.zeros(C, dtype=float)
        for j in range(C):
            rows = mask[:, j]
            if np.any(rows):
                # from (M + a1)/(sqrt(2)*a2) = erfinv(M)  ⇒  a1 = sqrt(2)*a2*erfinv(M) - M
                vals = sqrt2 * a2[rows] * invM[rows, j] - M_use[rows, j]
                vals = vals[np.isfinite(vals)]
                a1[j] = np.median(vals) if vals.size else 0.0
            else:
                a1[j] = 0.0
    else:
        a1 = np.asarray(a1_init, dtype=float)
        a1[~np.isfinite(a1)] = 0.0

    # Vectorize used entries for residuals
    ii, jj = np.where(mask)
    M_flat = M_use[ii, jj]
    a2_flat = a2[ii]

    # Optional ridge anchor at 0 for a1
    a1_anchor = np.zeros(C, dtype=float)

    def residuals(a1_vec):
        num = M_flat + a1_vec[jj]
        den = sqrt2 * a2_flat
        r = erf(num / den) - M_flat
        if ridge > 0:
            r = np.concatenate([r, np.sqrt(ridge) * (a1_vec - a1_anchor)])
        return r

    result = least_squares(
        residuals,
        a1,
        method="trf",
        loss="soft_l1",   # robust to outliers
        f_scale=0.01,
        max_nfev=100000,
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )

    a1_hat = result.x

    if verbose:
        print("=== Fit summary (a1 | given a2) ===")
        print(f"Rows (a2 given): {R}, Cols (a1): {C}")
        print(f"Cost (1/2||r||^2): {0.5*np.sum(result.fun**2):.6g}")
        print(f"Success: {result.success}  Message: {result.message}")

    return a1_hat, result

def fit_rows_a2_given_a1(M, a1, mask=None, a2_init=None, ridge=0.0, verbose=True):
    M = np.asarray(M, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    R, C = M.shape
    assert a1.shape == (C,), "a1 must have shape (C,)"

    # Build mask and clip M for stability (erfinv and gradients)
    if mask is None:
        mask = np.isfinite(M)
    else:
        mask = mask & np.isfinite(M)
    eps_clip = 1e-6
    M_use = M.copy()
    M_use[mask] = np.clip(M_use[mask], -1 + eps_clip, 1 - eps_clip)

    # Precompute pieces
    sqrt2 = np.sqrt(2.0)
    invM = np.zeros_like(M_use)
    invM[mask] = erfinv(M_use[mask])

    # ----- Initial a2 per row -----
    if a2_init is None:
        a2 = np.full(R, 0.6, dtype=float)
        for i in range(R):
            cols = mask[i, :]
            inv = invM[i, cols]
            # avoid division near m≈0 where erfinv ≈ 0
            good = np.abs(inv) > 1e-8
            if np.any(good):
                num = M_use[i, cols][good] + a1[cols][good]
                cand = num / (sqrt2 * inv[good])
                cand = cand[(cand > 0) & np.isfinite(cand)]
                if cand.size:
                    a2[i] = np.median(cand)
    else:
        a2 = np.asarray(a2_init, dtype=float)
        a2[~np.isfinite(a2) | (a2 <= 0)] = 0.6

    # Optimize in log-space: a2 = exp(b2)
    b2_0 = np.log(np.clip(a2, 1e-6, None))

    # Vectorize data used in residuals
    ii, jj = np.where(mask)
    M_flat  = M_use[ii, jj]
    a1_flat = a1[jj]

    # Optional ridge anchor at 0 for b2 (i.e., a2 anchored at 1)
    b2_anchor = np.zeros(R, dtype=float)

    def residuals(b2):
        a2 = np.exp(b2)  # (R,)
        den = sqrt2 * a2[ii]
        r = erf((M_flat + a1_flat) / den) - M_flat
        if ridge > 0:
            r = np.concatenate([r, np.sqrt(ridge) * (b2 - b2_anchor)])
        return r

    result = least_squares(
        residuals,
        b2_0,
        method="trf",
        loss="soft_l1",    # robust to a few outliers
        f_scale=0.01,
        max_nfev=100000,
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )

    a2_hat = np.exp(result.x)

    if verbose:
        print("=== Fit summary (a2 | given a1) ===")
        print(f"Rows (a2): {R}, Cols (a1 given): {C}")
        print(f"Cost (1/2||r||^2): {0.5*np.sum(result.fun**2):.6g}")
        print(f"Success: {result.success}  Message: {result.message}")

    return a2_hat, result

def fit_rows_a2_given_a1_with_width(
    M, a1, mask=None, a2_init=None,
    lam_width=0.0,        # strength of width penalty (0 disables)
    m_band=0.6,           # focus the width penalty near |m|≈0; larger = wider band
    a2_floor=0.0,         # hard minimum on a2 (prevents very steep sigmoids)
    monotone=False,       # enforce a2 nondecreasing across rows
    verbose=True,
):
    """
    Estimate row-wise a2[i] > 0 given a1[j] from:
        M[i,j] ≈ erf( (M[i,j] + a1[j]) / (sqrt(2) * a2[i]) )

    Adds a differentiable penalty that discourages *steep* transitions by
    penalizing susceptibility χ = K/(1-K), where K = (√2/√π) e^{-z^2} / a2.
    The penalty is applied mostly near |m|≈0 via a Gaussian window of width m_band.
    """
    M  = np.asarray(M, float)
    a1 = np.asarray(a1, float)
    R, C = M.shape
    assert a1.shape == (C,)

    # mask & clip
    if mask is None:
        mask = np.isfinite(M)
    else:
        mask = mask & np.isfinite(M)
    eps_clip = 1e-6
    M_use = M.copy()
    M_use[mask] = np.clip(M_use[mask], -1 + eps_clip, 1 - eps_clip)

    sqrt2 = np.sqrt(2.0)
    invM = np.zeros_like(M_use)
    invM[mask] = erfinv(M_use[mask])

    # ----- initializer for a2 (then floor & monotone projection on log-scale) -----
    if a2_init is None:
        a2 = np.full(R, a2_floor + 1e-6, float)
        for i in range(R):
            cols = mask[i, :]
            inv  = invM[i, cols]
            good = np.abs(inv) > 1e-8
            if np.any(good):
                num  = M_use[i, cols][good] + a1[cols][good]
                cand = num / (sqrt2 * inv[good])
                cand = cand[(cand > 0) & np.isfinite(cand)]
                if cand.size:
                    a2[i] = max(np.median(cand), a2_floor + 1e-6)
    else:
        a2 = np.asarray(a2_init, float)
        a2[~np.isfinite(a2) | (a2 <= 0)] = a2_floor + 1e-6

    # parameterize a2 = a2_floor + exp(b2)  (positivity + floor)
    a2 = np.maximum(a2, a2_floor + 1e-9)
    b2 = np.log(a2 - a2_floor)

    # optional monotone projection on b2
    if monotone and R > 1:
        b2 = np.maximum.accumulate(b2)

    theta0 = b2.copy()

    # vectorized data
    ii, jj = np.where(mask)
    m_obs   = M_use[ii, jj]
    a1_flat = a1[jj]

    def residuals(theta):
        b2 = theta
        if monotone and R > 1:
            b2 = np.maximum.accumulate(b2 + 1e-8*np.arange(R))

        a2_rows = a2_floor + np.exp(b2)          # (R,)
        den = sqrt2 * a2_rows[ii]

        # data residuals (same as before)
        r_data = erf((m_obs + a1_flat) / den) - m_obs

        regs = [r_data]

        # ---- width penalty (discourage steepness) ----
        if lam_width > 0:
            # Evaluate susceptibility χ at the OBSERVED m (no inner solve needed)
            z  = (m_obs + a1_flat) / den
            K  = (np.sqrt(2.0/np.pi) * np.exp(-z*z)) / a2_rows[ii]
            Kc = np.clip(K, -0.99, 0.99)          # numerical safety near 1
            chi = Kc / (1.0 - Kc)                 # χ = K/(1-K)
            w   = np.exp(-(m_obs / m_band)**2)    # emphasizes the transition band
            r_width = np.sqrt(lam_width) * w * chi
            regs.append(r_width)

        return np.concatenate(regs)

    res = least_squares(
        residuals, theta0,
        method="trf",
        loss="soft_l1", f_scale=0.01,
        max_nfev=100000,
        xtol=1e-12, ftol=1e-12, gtol=1e-12
    )

    a2_hat = a2_floor + np.exp(res.x)

    if verbose:
        print("=== Fit summary (a2 with width penalty) ===")
        print(f"Rows: {R}  Cost: {0.5*np.sum(res.fun**2):.6g}")
        print(f"Success: {res.success}  Message: {res.message}")

    return a2_hat, res

# %%
m_datasets = top_values_mean * 2 - 1

# --- Initial joint fit (optional but helpful)
a1_hat0, a2_hat0, _ = fit_grid_a1_a2(
    m_datasets,
    a2_init=np.zeros(len(m_datasets)),  # rows
    ridge=1e-4,
    verbose=False
)

# Helper: constrained quadratic fit for a2 with c1 >= 0
def fit_quad_c1_nonneg(x, y):
    def resid(c, x_, y_):  # c = [c2, c1, c0] (np.polyfit order: high->low)
        return np.polyval(c, x_) - y_
    c0 = np.polyfit(x, y, 2).astype(float)
    c0[1] = max(c0[1], 0.0)               # project seed inside bounds
    lb = [0.0, 0.0, 0.0]
    ub = [ np.inf,  np.inf,  np.inf]
    res = least_squares(resid, c0, bounds=(lb, ub), args=(x, y))
    return res.x

def fit_lin_c1_nonneg(x, y):
    def resid(c, x_, y_):  # c = [c1, c0] (np.polyfit order: high->low)
        return np.polyval(c, x_) - y_
    c0 = np.polyfit(x, y, 1).astype(float)
    c0[1] = max(c0[1], 0.0)               # project seed inside bounds
    lb = [0.0, 0.0]
    ub = [ np.inf,  np.inf]
    res = least_squares(resid, c0, bounds=(lb, ub), args=(x, y))
    return res.x

# Initialize coeffs from the initial hats
coeffs_a1 = np.polyfit(Mws, a1_hat0, 2)
# coeffs_a2 = fit_quad_c1_nonneg(covs, a2_hat0)
coeffs_a2 = fit_lin_c1_nonneg(covs, a2_hat0)

# Iteration settings
a1_prev = a1_hat0.copy()
a2_prev = a2_hat0.copy()
tol = 1e-8
count = 0
max_count = 1000
eta = 1.0  # 1.0: No damping, 0.0: No update (0 < eta <= 1.0)

while count < max_count:
    # ----- update a1 given a2(poly) -----
    a2_on_cov = np.polyval(coeffs_a2, covs)
    a2_on_cov = np.clip(a2_on_cov, 1e-8, None)  # positivity
    a1_new, _ = fit_cols_a1_given_a2(m_datasets, a2=a2_on_cov, verbose=False)
    coeffs_a1 = np.polyfit(Mws, a1_new, 2)

    # ----- update a2 given a1(poly) -----
    a1_on_Mw = np.polyval(coeffs_a1, Mws)
    # a2_new, _ = fit_rows_a2_given_a1(m_datasets, a1=a1_on_Mw, verbose=False)
    a2_new, _ = fit_rows_a2_given_a1_with_width(
        m_datasets, a1=a1_on_Mw,# a2_init=a2_prev,
        lam_width=1e-2,      # ↑ increases width (try 1e-3 → 1e-1)
        m_band=0.6,          # where to focus the penalty (|m|≈0)
        a2_floor=0.0,        # impose hard minimum value for a2
        monotone=True,       # impose monotonicity
        verbose=False
    )
    # coeffs_a2 = fit_quad_c1_nonneg(covs, a2_new)
    coeffs_a2 = fit_lin_c1_nonneg(covs, a2_new)

    # diffs (MSE)
    diff1 = float(np.mean((a1_prev - a1_new) ** 2))
    diff2 = float(np.mean((a2_prev - a2_new) ** 2))
    print(count, diff1, diff2)

    # numerical damping
    a1_prev = (1 - eta) * a1_prev + eta * a1_new
    a2_prev = (1 - eta) * a2_prev + eta * a2_new

    # convergence check
    if max(diff1, diff2) <= tol:
        break
    count += 1

print("done:", count, diff1, diff2)

# %%
outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
outdir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(Mws, a1_prev, 'r.', label=r'Estimated $a_{1}$')
ax.plot(Mws, coeffs_a1[0]*Mws**2 + coeffs_a1[1]*Mws + coeffs_a1[2], 'k--', label='Fitted curve (quadratic)', linewidth=2.0)
ax.set_xticks([4, 5, 6, 7, 8])
ax.set_yticks([-3, -2, -1, 0, 1, 2])
ax.set_xlabel(r"Earthquake magnitude $M_{w}$", fontsize=20)
ax.set_ylabel(r"$a_{1}\,\left(=H/J\right)$", fontsize=20)
ax.tick_params(axis='both', labelsize=16)
ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.0)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=20)
if SaveFig:
    fname = f"{outdir}/param_est_Mw_to_a1.png"
    fig.savefig(fname, dpi=300, transparent=True, bbox_inches='tight')
    print(f"Saved figure: {fname}")
plt.show()

roots = np.roots(coeffs_a1)
Mw_c = roots[1]
print(f"Critical magnitude Mw: {Mw_c:.3f}")

# plt.figure(figsize=(6,4))
# plt.plot(covs, a2_prev, '.', label='data (a2_hat)')
# plt.plot(covs, coeffs_a2[0]*covs**2 + coeffs_a2[1]*covs + coeffs_a2[2],
#          label='quadratic fit (c1 ≥ 0)', color='C1')
# plt.xlabel("COV")
# plt.ylabel("a2_hat")
# plt.legend()
# plt.show()

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(covs, a2_prev, 'r.', label=r'Estimated $a_{2}$')
ax.plot(covs, coeffs_a2[0]*covs + coeffs_a2[1], 'k--', label='Fitted curve (linear)', linewidth=2.0)
ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
# ax.set_yticks([-3, -2, -1, 0, 1, 2])
ax.set_xlabel(r"Structural diversity $\sigma$", fontsize=20)
ax.set_ylabel(r"$a_{2}\,\left(=\Delta/J\right)$", fontsize=20)
ax.tick_params(axis='both', labelsize=16)
ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.0)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=20)
if SaveFig:
    fname = f"{outdir}/param_est_sigma_to_a2.png"
    fig.savefig(fname, dpi=300, transparent=True, bbox_inches='tight')
    print(f"Saved figure: {fname}")
plt.show()

a2_c = np.sqrt(2/np.pi)
cov_c = (a2_c - coeffs_a2[1]) / coeffs_a2[0]
print(f"Critical diversity σ = {cov_c:.3f}")


# %%

# Mws = Mws_orig.copy()
# covs = covs_orig.copy()

Mws = np.linspace(Mws_orig.min(), Mws_orig.max(), RESOLUTION)
covs = np.linspace(covs_orig.min(), covs_orig.max(), RESOLUTION)

top_values_mean = top_values_mean_orig.copy()
m_datasets = top_values_mean * 2 - 1

a1_of_Mw  = np.polyval(coeffs_a1, Mws)               # (101,)
a2_of_cov = np.clip(np.polyval(coeffs_a2, covs), 1e-12, None)  # (101,)

Mw_grid, cov_grid = np.meshgrid(Mws, covs, indexing="xy")
a1_grid = np.tile(a1_of_Mw, (covs.size, 1))          # (rows, cols)
a2_grid = np.tile(a2_of_cov[:, None], (1, Mws.size)) # (rows, cols)

sqrt2 = np.sqrt(2.0)

def fixed_point_solve(a1g, a2g, m0, tol=1e-8, max_iter=10000, omega=1.0, newton_polish=5):
    m = m0.copy()
    prev_delta = np.inf
    iters = 0
    for it in range(max_iter):
        rhs = erf((m + a1g) / (sqrt2 * a2g))
        m_new = (1 - omega) * m + omega * rhs
        delta = np.max(np.abs(m_new - m))
        m = m_new
        iters = it + 1
        # simple adaptive damping if we start to bounce
        if delta > prev_delta * 1.05 and omega > 0.2:
            omega *= 0.9
        prev_delta = delta
        if delta < tol:
            break

    # # Newton polish on f(m)=erf((m+a1)/s)-m=0 with s=sqrt(2)*a2
    # s = sqrt2 * a2g
    # for _ in range(newton_polish):
    #     z = (m + a1g) / s
    #     f = erf(z) - m
    #     # f'(m) = (2/sqrt(pi)) * exp(-z^2) * (1/s) - 1
    #     fp = (2.0/np.sqrt(np.pi)) * np.exp(-z*z) * (1.0/s) - 1.0
    #     # safe Newton step with clipping
    #     step = np.where(np.abs(fp) > 1e-10, f / fp, 0.0)
    #     step = np.clip(step, -0.5, 0.5)  # tame huge steps near tanh-like steepness
    #     m -= step
    #     if np.max(np.abs(step)) < tol:
    #         break

    return np.clip(m, -1.0, 1.0), iters

# Solve from both branches
m_plus0  = np.ones_like(a1_grid)
m_minus0 = -np.ones_like(a1_grid)

m_plus, it_plus   = fixed_point_solve(a1_grid, a2_grid, m_plus0)
m_minus, it_minus = fixed_point_solve(a1_grid, a2_grid, m_minus0)

damage_plus  = 0.5 * (m_plus  + 1.0)
damage_minus = 0.5 * (m_minus + 1.0)

# Bistability map: cells where branches disagree by > eps
eps = 1e-4
bistable = (np.abs(m_plus - m_minus) > eps).astype(float)

# Equilibrium damage
tol = 1e-12
damage_eq = np.where(a1_grid > tol, damage_plus,
                     np.where(a1_grid < -tol, damage_minus,
                              0.5*(damage_plus + damage_minus)))

# %%
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as colors

MAGS, COVS = np.meshgrid(Mws, covs)

vmin = 0.0
vmax = 1.0

base_cmap = plt.cm.gray_r(np.linspace(0, 1, 256))
base_cmap[:, -1] = np.linspace(0, 0.6, 256)  # last column is alpha channel (0→transparent, 0.5→semi)
transparent_gray = ListedColormap(base_cmap)

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
c = ax.contourf(
    COVS, MAGS, damage_eq, 
    levels=100, 
    cmap="RdYlBu_r", 
    vmin=vmin, vmax=vmax, zorder=1,
    # norm = colors.BoundaryNorm(boundaries=np.linspace(0,1,20), ncolors=256),
)

# c = ax.imshow(
#     top_values_mean.T,
#     origin="lower",
#     extent=[COVS.min(), COVS.max(), MAGS.min(), MAGS.max()],
#     aspect="auto",
#     cmap="RdYlBu_r", 
#     vmin=vmin, vmax=vmax,
#     interpolation="none"   # critical: no blending between pixels
# )


# Transition boundary (critical Mw values)
ax.plot([0.0, cov_c], [Mw_c, Mw_c], '-', linewidth=2.0, color="white", zorder=3, label='Transition boundary')

# Critical sigma value (critical point)
ax.plot(cov_c, Mw_c, 'o', ms=10, markerfacecolor="black", markeredgecolor="white", zorder=5, label='Critical point')

# Bistable regime
c2 = ax.contourf(
    COVS, MAGS, (np.abs(m_plus - m_minus) > 1e-4).astype(float), 
    levels=10, 
    cmap=transparent_gray,
    vmin=vmin, vmax=vmax, zorder=4
)

# Contour levels and label positions
# levels = [0.01, 0.05, 0.1, 0.9, 0.95, 0.99]
# label_positions = [(0.85, 6.5), (0.86, 6.5), (0.87, 6.5), (0.85, 6.5), (0.84, 6.5), (0.84, 6.5)]
levels = [0.05, 0.1, 0.9, 0.95]
label_positions = [(0.86, 6.5), (0.87, 6.5), (0.85, 6.5), (0.84, 6.5)]

# Draw contours and labels
for lvl, pos in zip(levels, label_positions):
    cs = ax.contour(
        COVS, MAGS, damage_eq,
        levels=[lvl],
        colors='black',
        linewidths=0.5,
        linestyles='--',
        zorder=2
    )
    ax.clabel(cs, fmt=f'{lvl:.2f}', fontsize=14, inline=True, manual=[pos])

# --- Custom Legend Elements ---
# bistable_patch = mpatches.Patch(color='gray', alpha=0.8, label='Bistable regime')
bistable_patch = mpatches.Patch(color='gray', alpha=0.8, label='Bistable regime')
transition_line = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Transition boundary')
critical_marker = mlines.Line2D([], [], color='black', marker='o', markerfacecolor='black',
                                markeredgecolor='white', markersize=10, linestyle='None', label='Critical point')

# Add legend (custom handles)
# legend = ax.legend(
#     handles=[bistable_patch, transition_line, critical_marker],
#     fontsize=16,
#     loc='upper left',
#     frameon=True,
# )
# legend.get_frame().set_boxstyle('square', pad=0.0)
# legend.get_frame().set_facecolor('white')
# legend.get_frame().set_edgecolor('black')
# legend.get_frame().set_alpha(1.0)

# # --- Colorbar and labels ---
cb = plt.colorbar(c, ax=ax)
cb.set_label('Damage fraction', fontsize=20)
cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
cb.set_ticklabels(["0.0", "0.5", "1.0"])
# cb.set_ticklabels(["Ordered", "Disordered", "Ordered"])
cb.ax.tick_params(labelsize=16)

ax.set_xlabel(r'Structural diversity $\sigma$', fontsize=20)
ax.set_xticks([0.0, 0.5, 1.0])
ax.set_xticklabels(["0.0", "0.5", "1.0"])
# ax.set_xticklabels(["Low", "High"])
# ax.set_yticks([3.5, 7.7])
# ax.set_yticklabels(["Weak", "Strong"])
ax.set_ylabel(r'Earthquake magnitude $M_w$', fontsize=20)
ax.tick_params(axis="both", labelsize=16)

for label in ax.get_xticklabels():
    if label.get_text() == "Low":
        label.set_ha("left")   # align left edge to tick
    elif label.get_text() == "High":
        label.set_ha("right")  # align right edge to tick

for label in ax.get_yticklabels():
    if label.get_text() == "Weak":
        label.set_va("bottom") # move slightly upward
    elif label.get_text() == "Strong":
        label.set_va("top")    # move slightly downward

fig.tight_layout()

if SaveFig:
    plt.savefig(f"{outdir}/phase_diagram_RFIM_bistable_clean.png", dpi=300, transparent=True, bbox_inches='tight')
    print("Saved phase_diagram_RFIM_bistable.png")

plt.show()



# %% Save the estimated coefficients
# np.save("coeffs_a1.npy", coeffs_a1)
# np.save("coeffs_a2.npy", coeffs_a2)
# print("Saved coeffs_a1.npy and coeffs_a2.npy")

# # %%
# MAGS, COVS = np.meshgrid(Mws, covs)


# fig, ax = plt.subplots(figsize=(10, 6))
# c = ax.contourf(
#     COVS, MAGS, damage_eq-top_values_mean, 
#     levels=100, 
#     cmap="bone",
# )

# # --- Colorbar and labels ---
# cb = plt.colorbar(c, ax=ax)
# cb.set_label('Damage fraction', fontsize=20)
# # cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
# # cb.set_ticklabels(["0.0", "0.5", "1.0"])
# cb.ax.tick_params(labelsize=16)

# ax.set_xlabel('Structural diversity', fontsize=20)
# ax.set_xticks([0.0, 1.0])
# ax.set_xticklabels(["Low", "High"])
# ax.set_ylabel('Earthquake magnitude', fontsize=20)
# ax.tick_params(axis="both", labelsize=16)

# fig.tight_layout()

# if SaveFig:
#     plt.savefig("phase_diagram_RFIM_bistable.png", dpi=300, transparent=True, bbox_inches='tight')
#     print("Saved phase_diagram_RFIM_bistable.png")

# plt.show()
