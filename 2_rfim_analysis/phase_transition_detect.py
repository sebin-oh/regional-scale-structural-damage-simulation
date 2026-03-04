#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 11/03/2025
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],
    "mathtext.fontset": "cm",
})

CP_HEX = [
    "#4056A1", "#F12815", "#D79922", "#14A098",
    "#CB2D6F", "#501F3A", "#116466", "#F76C6C",
    "#EFE2BA", "#C5CBE3", "#8C9AC7", "#0072B5",
]

# %%
Mws = np.round(np.arange(3.5, 8.51, 0.05), 2)
covs = np.round(np.arange(0.0, 1.001, 0.01), 3)
num_sim = 10_000
num_bins_mag = 101

# %%
from pathlib import Path
from typing import Tuple, Optional, Literal

def _dv_filename(
    target_structure: Literal["SingleStory", "TwoStory", "MultiStory"],
    target_region: str,
    Mw: Optional[float],
    cov: Optional[float],
) -> Tuple[Path, str]:
    """Build DV filename and a short English title."""
    base = f"C:/Users/ohsb1/Desktop/Savio/{target_region}/{target_structure}_dv"

    Mw_label = f"{int(round(Mw*100)):03d}"
    cov_label = f"{int(round(cov*1000)):04d}"

    suffix = f"_Mw{Mw_label}_cov{cov_label}_{target_region}"

    fname = f"{base}{suffix}.npy"

    return Path(fname)

def F_model(m, c0, a1, a2, a4, a100):
    return c0 + a1*m + a2*m**2 + a4*m**4 + a100*m**100

def solve_m(popt, tol=1e-12, max_iter=100):
    """
    Solve a2 + 2*a4*m^2 + 50*a100*m^98 = 0 for m (returns ±m*).
    Uses Newton's method with a quadratic-based initial guess.
    """
    a1      = popt[1]
    a2      = popt[2]
    a4      = popt[3]
    a100    = popt[4]
    
    m0 = 1.0 # Initial guess

    def f(m):
        return a2 + 2.0*a4*(m*m) + 50.0*a100*(m**98)

    def f(m):
        return a1 + 2.0*a2*m + 4.0*a4*(m**3) + 100.0*a100*(m**99)

    def df(m):
        return 2.0*a2 + 12.0*a4*m**2 + 9900.0*a100*(m**98)

    # Newton with simple damping for robustness
    m = abs(m0)
    for _ in range(max_iter):
        fm = f(m)
        dfm = df(m)
        if dfm == 0.0:
            # Nudge if derivative is zero
            m = m + (1e-6 if m == 0.0 else 1e-6*abs(m))
            dfm = df(m)

        step = fm / dfm
        m_new = m - step

        # Dampen if residual doesn't decrease
        tries = 0
        while abs(f(m_new)) > abs(fm) and tries < 8:
            step *= 0.5
            m_new = m - step
            tries += 1

        if abs(m_new - m) <= tol * max(1.0, abs(m)):
            m = m_new
            break
        m = m_new

    return m

# %%
# Mws = np.array([5.6, 5.65, 5.7])
Mws = np.array([5.55, 5.6, 5.65])

bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 1e-12], [np.inf, np.inf, np.inf, np.inf, np.inf])

edges = np.linspace(-1.0, 1.0, num_bins_mag + 1)
edges = np.insert(edges, 0, -1.001)
edges = np.insert(edges, len(edges), 1.001)

mag_centers = 0.5 * (edges[:-1] + edges[1:])

chi     = np.zeros((len(Mws),len(covs)))
m_star  = np.zeros((len(Mws),len(covs)))

c0_hat   = np.empty((len(Mws),len(covs)))
a1_hat   = np.empty((len(Mws),len(covs)))
a2_hat   = np.empty((len(Mws),len(covs)))
a4_hat   = np.empty((len(Mws),len(covs)))
a100_hat = np.empty((len(Mws),len(covs)))

for Mw_idx, Mw_val in enumerate(Mws):
    print(f"Selected Mw: {Mw_val}")

    for cov_idx, cov_val in enumerate(covs):
        # print(cov_idx)

        dv = np.zeros((len(Mws),num_sim))
        for Mw_idx_, Mw_val_ in enumerate(Mws):
            dv_path = _dv_filename("MultiStory", "Milpitas_all", Mw_val_, cov_val)
            if not dv_path.exists():
                raise FileNotFoundError(f"Missing DV file: {dv_path}")
            dv_temp = np.load(dv_path)
            dv[Mw_idx_,:] = dv_temp

        magnetization = dv * 2.0 - 1.0
        PARAMS = Mws.copy()
        
        counts = np.zeros((len(PARAMS), len(edges)-1), dtype=int)  # [y=mag_bins, x=PARAMS]
        for i in range(len(PARAMS)):
            counts[i, :], _ = np.histogram(magnetization[i, :], bins=edges)
            counts[i, 0] = 1
            counts[i, -1] = 1
        counts = counts.T

        alpha = 1.0  # pseudocount to avoid zeros
        counts_ps = counts.astype(float) + alpha
        counts_sm = gaussian_filter(counts_ps, sigma=(1.0, 0.0))
        counts_sm[0,:] = np.ones((len(PARAMS)))
        counts_sm[-1,:] = np.ones((len(PARAMS)))
        col_sum = counts_sm.sum(axis=0, keepdims=True)
        p = counts_sm / np.clip(col_sum, 1e-12, None)

        F = -np.log(np.clip(p, 1e-12, None))  # shape [M, P]

        # plt.plot(mag_centers, p[:,Mw_idx])
        # plt.show()

        # plt.plot(mag_centers, F[:,Mw_idx])
        # plt.show()

        pmin = 1e-5

        Fcol = F[:, Mw_idx]
        w = counts_sm[:, Mw_idx].astype(float)
        # Convert weights -> sigma for curve_fit (sigma ~ 1/sqrt(weight))
        w = np.maximum(w, 1e-12)
        sigma = 1.0 / np.sqrt(w)
        p_col = np.exp(-np.clip(Fcol - np.nanmin(Fcol), 0, 100))  # relative (ok for masking)
        mask = np.isfinite(Fcol) & (p_col >= pmin)

        # Ensure enough points
        if np.count_nonzero(mask) < 8:
            # fallback: relax mask
            mask = np.isfinite(Fcol)

        m = mag_centers.copy()
        c0_0, a1_0, a2_0, a4_0, a100_0 = [0,0,0,0,1e-6]

        popt, pcov = curve_fit(
            F_model,
            m[mask],
            Fcol[mask],
            p0=[c0_0, a1_0, a2_0, a4_0, a100_0],
            sigma=sigma[mask],
            absolute_sigma=False,
            bounds=bounds,
            maxfev=20000,
        )
        c0_hat[Mw_idx,cov_idx], a1_hat[Mw_idx,cov_idx], a2_hat[Mw_idx,cov_idx], a4_hat[Mw_idx,cov_idx], a100_hat[Mw_idx,cov_idx] = popt

        # m_plot = np.linspace(-1.01,1.01,100)
        # F_fit = F_model(m_plot, popt[0], popt[1], popt[2], popt[3], popt[4])
        # plt.plot(m, Fcol,'k-')
        # plt.plot(m_plot, F_fit,'r--')
        # plt.show()

        m_eq = solve_m(popt)

        chi[Mw_idx,cov_idx] = 1/(2*popt[2]+12*popt[3]*m_eq**2+9900*popt[4]*m_eq**98)
        m_star[Mw_idx,cov_idx] = m_eq


markers = ['o', 's', '^']
outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
outdir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
for i in range(len(Mws)):
    ax.plot(
        covs, m_star[i, :],
        '-', marker=markers[i],
        linewidth=2, ms=5,
        color=CP_HEX[i],
        label=rf"$M_w={Mws[i]:.2f}$"
    )
ax.set_xlabel(r'Structural diversity $\sigma$', fontsize=20)
ax.set_ylabel(r'Equilibrium magnetization $m^*$', fontsize=20)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.tick_params(axis='both', labelsize=14)
ax.legend(loc='lower left', fontsize=20, frameon=False)
ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.0)
ax.spines[['top', 'right']].set_visible(False) 
fname = f"{outdir}/phase_transition_detect_2nd_mstar.png"
# fig.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
# print(f"Saved figure: {fname}")
plt.show()

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
for i in range(len(Mws)):
    ax.plot(covs, chi[i, :], '-', marker=markers[i], linewidth=2, ms=5, color=CP_HEX[i], label=rf"$M_w={Mws[i]:.2f}$")
ax.set_xlabel(r'Structural diversity $\sigma$', fontsize=20)
ax.set_ylabel(r'Susceptibility $\chi$', fontsize=20)
ax.set_xlim(0.0, 1.0)
ax.tick_params(axis='both', labelsize=14)
ax.legend(loc='upper left', fontsize=20, frameon=False)
ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.0)
ax.spines[['top', 'right']].set_visible(False)
fname = f"{outdir}/phase_transition_detect_2nd_chi.png"
# fig.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
# print(f"Saved figure: {fname}")
plt.show()

# %%
from scipy import special as nps 

coeffs_a1 = np.load("coeffs_a1.npy").astype(np.float64)
coeffs_a2 = np.load("coeffs_a2.npy").astype(np.float64)

def Mw_to_a1(Mw):
    return np.polyval(coeffs_a1, Mw)

def cov_to_a2(cov):
    return np.polyval(coeffs_a2, cov)

def landau_free_energy(m, Mw, cov, eps=1e-12):
    """
    F(m) = 1/2*m^2 - (m+a1) * erf( (m+a1) / (sqrt(2)*a2) )
           - sqrt(2/pi) * a2 * exp( -(m+a1)^2 / (2*a2^2) ) + Const.
    """
    a1 = Mw_to_a1(Mw)
    a2 = cov_to_a2(cov)

    m = np.asarray(m, dtype=float)

    # guard against a2 -> 0
    a2 = np.maximum(a2, eps)
    alpha = 1.0 / (np.sqrt(2.0) * a2)

    term_quad = 0.5 * m**2
    term_erf  = (m + a1) * nps.erf(alpha * (m + a1))
    term_exp  = np.exp(- (alpha * (m + a1))**2) / (alpha * np.sqrt(np.pi))

    return term_quad - term_erf - term_exp

def solve_landau_stationary_m(Mw, cov, eps=1e-12, tol=1e-10, max_iter=200):
    """
    Solve for m in:
        m - erf((m + a1) / (sqrt(2)*a2)) = 0 (equivalent to dF/dm = 0),
    where:
        a1 = Mw_to_a1(Mw)
        a2 = cov_to_a2(cov)

    Uses bisection on [-1, 1] (guaranteed bracket for real solutions).
    Works for scalar or array-like Mw/cov via np.vectorize.

    Returns
    -------
    m_star : float or ndarray
        Root in [-1, 1].
    """

    a1 = Mw_to_a1(Mw)
    a2 = np.maximum(cov_to_a2(cov), eps)

    def g(m, a1_, a2_):
        return m - nps.erf((m + a1_) / (np.sqrt(2.0) * a2_))

    def solve_one(a1_, a2_):
        lo, hi = -1.0, 1.0
        flo, fhi = g(lo, a1_, a2_), g(hi, a1_, a2_)

        # Handle edge roots
        if abs(flo) <= tol:
            return lo
        if abs(fhi) <= tol:
            return hi

        # In practice flo <= 0 and fhi >= 0 on [-1,1].
        # If numerical issues break the sign bracket, expand slightly and retry.
        if flo > 0 or fhi < 0:
            lo2, hi2 = -1.0 - 1e-6, 1.0 + 1e-6
            flo, fhi = g(lo2, a1_, a2_), g(hi2, a1_, a2_)
            lo, hi = lo2, hi2
            if flo > 0 or fhi < 0:
                raise RuntimeError(
                    f"Failed to bracket root: g(lo)={flo}, g(hi)={fhi}, a1={a1_}, a2={a2_}"
                )

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            fmid = g(mid, a1_, a2_)

            if abs(fmid) <= tol or (hi - lo) <= tol:
                return mid

            # Keep the sub-interval that contains the sign change
            if fmid > 0:
                hi = mid
            else:
                lo = mid

        # Best estimate if max_iter reached
        return 0.5 * (lo + hi)

    # Vectorize over a1/a2 (handles scalars and arrays)
    a1_arr = np.asarray(a1, dtype=float)
    a2_arr = np.asarray(a2, dtype=float)

    solver_vec = np.vectorize(solve_one, otypes=[float])
    return solver_vec(a1_arr, a2_arr)

def landau_free_energy_second_derivative(m, Mw, cov, eps=1e-12):
    """
    F''(m) for
    F(m) = 1/2*m^2 - (m+a1)*erf((m+a1)/(sqrt(2)*a2))
           - sqrt(2/pi)*a2*exp(-(m+a1)^2/(2*a2^2)) + C

    Result:
    F''(m) = 1 - sqrt(2/pi) * (1/a2) * exp(-(m+a1)^2/(2*a2^2))
    """
    a1 = Mw_to_a1(Mw)
    a2 = np.maximum(cov_to_a2(cov), eps)

    m = np.asarray(m, dtype=float)
    z = (m + a1) / a2  # used in exponent

    return 1.0 - np.sqrt(2.0 / np.pi) * (1.0 / a2) * np.exp(-0.5 * z**2)

def susceptibility_landau(Mw, cov, eps=1e-12, tol=1e-10, max_iter=200):
    """
    Compute susceptibility chi = 1 / F''(m*) at equilibrium m* from Landau free energy.

    Uses solve_landau_stationary_m to find m*.

    Returns
    -------
    chi : float or ndarray
        Susceptibility at equilibrium m*.
    """
    m_star = solve_landau_stationary_m(Mw, cov, eps=eps, tol=tol, max_iter=max_iter)
    Fpp = landau_free_energy_second_derivative(m_star, Mw, cov, eps=eps)

    # Guard against division by zero
    chi = np.where(Fpp > 0, 1.0 / Fpp, np.inf)

    return chi

# Mw_landau = np.array([5.6, 5.65])
# for i in range(len(Mws)):
#     fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
#     ax.plot(covs, chi[i, :], 'k-', marker=markers[i], linewidth=2, ms=5, label=rf"Numerical")
#     ax.plot(covs,(susceptibility_landau(Mw_landau[i], covs)-1)/(susceptibility_landau(Mw_landau[i], covs).max()-1)*chi[i, :].max(), 'r--',linewidth=2, ms=5, label="Mean-field solution")
#     ax.set_xlabel(r'Structural diversity $\sigma$', fontsize=20)
#     ax.set_ylabel(r'Susceptibility $\chi$', fontsize=20)
#     ax.set_yticks([])
#     ax.set_xlim(0.0, 1.0)
#     ax.tick_params(axis='both', labelsize=14)
#     ax.legend(loc='upper left', fontsize=20, frameon=False)
#     # ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.0)
#     ax.spines[['top', 'right']].set_visible(False)
#     fname = f"{outdir}/phase_transition_detect_2nd_chi.png"
#     # fig.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
#     # print(f"Saved figure: {fname}")
#     plt.show()



import matplotlib.ticker as mticker

Mw_landau = np.array([5.6, 5.61])

fig, ax = plt.subplots(figsize=(5, 6), dpi=300)
ax.plot(covs,susceptibility_landau(Mw_landau[1], covs)/susceptibility_landau(Mw_landau[1], covs).max()*chi[0, :].max(), 'r--',linewidth=2, ms=5, color='gray', label="Mean-field solution")
ax.plot(covs, chi[0, :], linestyle='-', linewidth=2, color=CP_HEX[0], alpha=0.5, label=rf"Numerical")
ax.plot(covs, chi[0, :], linestyle='None', marker=markers[0], markerfacecolor=CP_HEX[0], markeredgecolor='white', alpha=0.7)
ax.set_xlabel(r'Structural diversity $\sigma$', fontsize=20)
ax.set_ylabel(r'Susceptibility $\chi$', fontsize=20)
# ax.set_yticks([])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels([])
ax.yaxis.set_major_locator(mticker.LinearLocator(6))
ax.set_xlim(0.0, 1.0)
ax.tick_params(axis='both', labelsize=14)
# ax.legend(loc='upper left', fontsize=16, frameon=False)
ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
ax.spines[['top', 'right']].set_visible(False)
# fname = f"{outdir}/susceptibility_crit.png"
# fig.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
# print(f"Saved figure: {fname}")
plt.show()


fig, ax = plt.subplots(figsize=(5, 6), dpi=300)
ax.plot(covs,susceptibility_landau(Mw_landau[1], covs)/susceptibility_landau(Mw_landau[1], covs).max()*chi[1, :].max(), 'r--',linewidth=2, ms=5, color='gray', label="Mean-field solution")
ax.plot(covs, chi[1, :], linestyle='-', linewidth=1, color=CP_HEX[3], alpha=0.5, label=rf"Numerical")
ax.plot(covs, chi[1, :], linestyle='None', marker=markers[0], markerfacecolor=CP_HEX[3], markeredgecolor='white', alpha=0.7)
ax.plot(covs, chi[2, :], linestyle='-', linewidth=1, color=CP_HEX[2], alpha=0.5, label=rf"Numerical")
ax.plot(covs, chi[2, :], linestyle='None', marker=markers[0], markerfacecolor=CP_HEX[2], markeredgecolor='white', alpha=0.7)
ax.set_xlabel(r'Structural diversity $\sigma$', fontsize=20)
ax.set_ylabel(r'Susceptibility $\chi$', fontsize=20)
# ax.set_yticks([])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels([])
ax.yaxis.set_major_locator(mticker.LinearLocator(6))
ax.set_xlim(0.0, 1.0)
ax.tick_params(axis='both', labelsize=14)
# ax.legend(loc='upper left', fontsize=16, frameon=False)
ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
ax.spines[['top', 'right']].set_visible(False)
# fname = f"{outdir}/susceptibility_away_crit.png"
# fig.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
# print(f"Saved figure: {fname}")
plt.show()

# %%
import numpy as np

import numpy as np

def fit_piecewise_affine_mf_smooth(
    covs,
    chi_row,
    Mw,
    chi_mf,
    susceptibility_landau_fn,
    sigma0=0.5,
    sigma_left=0.0,          # NEW: anchor left end here (default 0)
    n_dense=600,
    blend_half_width=10,
    join_mode="peak",        # "peak" or "sigma0"
):
    covs = np.asarray(covs, dtype=float)
    chi_row = np.asarray(chi_row, dtype=float)
    chi_mf = np.asarray(chi_mf, dtype=float)

    # Ensure increasing covs for interpolation
    if np.any(np.diff(covs) < 0):
        idx = np.argsort(covs)
        covs = covs[idx]
        chi_row = chi_row[idx]
        chi_mf = chi_mf[idx]

    # Masks
    mask_lo = covs <= sigma0
    mask_hi = covs >= sigma0

    # ---- Anchor values on the DATA (numerical) ----
    yL = float(np.interp(sigma_left, covs, chi_row))  # chi at sigma_left
    y0 = float(np.interp(sigma0,     covs, chi_row))  # chi at sigma0

    # ---- Corresponding MF values (from chi_mf on cov grid) ----
    mL = float(np.interp(sigma_left, covs, chi_mf))   # MF at sigma_left
    m0 = float(np.interp(sigma0,     covs, chi_mf))   # MF at sigma0

    # ---- Left branch parameters fixed by two anchors ----
    denom = (m0 - mL)
    if abs(denom) < 1e-14:
        raise ValueError(
            f"Cannot anchor left branch: MF values too close at sigma_left={sigma_left} and sigma0={sigma0} "
            f"(mL≈m0≈{m0})."
        )
    a_lo = (y0 - yL) / denom
    b_lo = yL - a_lo * mL

    # ---- Right branch: fit a_hi with continuity at sigma0 ----
    # Model: y = a_hi*m + b_hi, with b_hi chosen so y(sigma0)=y0:
    # b_hi = y0 - a_hi*m0  => y - y0 = a_hi*(m - m0)
    y_hi = chi_row[mask_hi]
    m_hi = chi_mf[mask_hi]
    dm = m_hi - m0
    dy = y_hi - y0

    dm2 = np.dot(dm, dm)
    a_hi = (np.dot(dm, dy) / dm2) if dm2 > 0 else 0.0
    b_hi = y0 - a_hi * m0

    # ---- Dense grid for smooth curves ----
    cov_dense = np.linspace(covs.min(), covs.max(), n_dense)
    cov_dense = np.unique(np.append(cov_dense, sigma0))

    chi_mf_dense = susceptibility_landau_fn(Mw, cov_dense)

    chi_fit_lo = a_lo * chi_mf_dense + b_lo
    chi_fit_hi = a_hi * chi_mf_dense + b_hi

    # ---- Choose join index ----
    if join_mode == "sigma0":
        i0 = int(np.argmin(np.abs(cov_dense - sigma0)))
    elif join_mode == "peak":
        i0 = int(np.argmax(chi_fit_lo))
    else:
        raise ValueError("join_mode must be 'peak' or 'sigma0'")

    # ---- Smooth blend around join ----
    w = int(blend_half_width)
    iL = max(i0 - w, 0)
    iR = min(i0 + w, len(cov_dense) - 1)

    chi_fit = np.empty_like(cov_dense, dtype=float)
    chi_fit[:iL] = chi_fit_lo[:iL]
    chi_fit[iR+1:] = chi_fit_hi[iR+1:]

    t = np.linspace(0.0, 1.0, iR - iL + 1)
    s = t * t * (3 - 2 * t)  # smoothstep
    chi_fit[iL:iR+1] = (1 - s) * chi_fit_lo[iL:iR+1] + s * chi_fit_hi[iL:iR+1]

    params = dict(
        a_lo=a_lo, b_lo=b_lo, a_hi=a_hi, b_hi=b_hi,
        sigma_left=sigma_left, sigma0=sigma0,
        yL=yL, y0=y0, mL=mL, m0=m0,
        join_index=i0, blend_half_width=w,
        chi_fit_lo=chi_fit_lo, chi_fit_hi=chi_fit_hi, chi_mf_dense=chi_mf_dense
    )
    return cov_dense, chi_fit, params



sigma0 = 0.5
chi_mf = susceptibility_landau(Mw_landau[0], covs)

cov_dense, chi_fit, params = fit_piecewise_affine_mf_smooth(
    covs=covs,
    chi_row=chi[0, :],
    Mw=Mw_landau[0],
    chi_mf=chi_mf,
    susceptibility_landau_fn=susceptibility_landau,
    sigma0=sigma0,
    n_dense=600,
    blend_half_width=10,
    join_mode="peak",   # or "sigma0"
)

# --- Plot ---
fig, ax = plt.subplots(figsize=(5, 6), dpi=300)

# keep your original MF normalization, but avoid recomputing susceptibility_landau
chi_mf_norm = (chi_mf - 1.0) / (chi_mf.max() - 1.0) * chi_fit.max()
ax.plot(covs, chi_mf_norm, '--', linewidth=2, color='gray', label="Mean-field solution")

ax.plot(cov_dense, chi_fit, '-', linewidth=2, alpha=0.5, color=CP_HEX[0])
ax.plot(covs, chi[0, :], linestyle='None', marker=markers[0], markerfacecolor=CP_HEX[0], markeredgecolor='white', alpha=0.7, label="Numerical")
ax.set_xlabel(r'Structural diversity $\sigma$', fontsize=20)
ax.set_ylabel(r'Susceptibility $\chi$', fontsize=20)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

ax.tick_params(axis="y", labelleft=False)   # cleaner than set_yticklabels([])
ax.yaxis.set_major_locator(mticker.LinearLocator(6))

ax.set_xlim(0.0, 1.0)
ax.tick_params(axis='both', labelsize=14)
ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
ax.spines[['top', 'right']].set_visible(False)

fname = f"{outdir}/susceptibility_crit.png"
fig.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
print(f"Saved figure: {fname}")

plt.show()





sigma0 = 0.53
chi_mf = susceptibility_landau(Mw_landau[1], covs)

cov_dense, chi_fit, params = fit_piecewise_affine_mf_smooth(
    covs=covs,
    chi_row=chi[2, :],
    Mw=Mw_landau[1],
    chi_mf=chi_mf,
    susceptibility_landau_fn=susceptibility_landau,
    sigma0=sigma0,
    n_dense=600,
    blend_half_width=10,
    join_mode="peak",   # or "sigma0"
)

# --- Plot ---
fig, ax = plt.subplots(figsize=(5, 6), dpi=300)

# keep your original MF normalization, but avoid recomputing susceptibility_landau
chi_mf_norm = chi_mf / chi_mf.max() * chi_fit.max()
ax.plot(covs, chi_mf_norm, '--', linewidth=2, color='gray', label="Mean-field solution")

ax.plot(cov_dense, chi_fit, '-', linewidth=2, alpha=0.5, color=CP_HEX[3])
ax.plot(covs, chi[2, :], linestyle='None', marker=markers[0], markerfacecolor=CP_HEX[3], markeredgecolor='white', alpha=0.7, label="Numerical")
ax.set_xlabel(r'Structural diversity $\sigma$', fontsize=20)
ax.set_ylabel(r'Susceptibility $\chi$', fontsize=20)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

ax.tick_params(axis="y", labelleft=False)   # cleaner than set_yticklabels([])
ax.yaxis.set_major_locator(mticker.LinearLocator(6))

ax.set_xlim(0.0, 1.0)
ax.set_ylim(-0.05, 1.5)
ax.tick_params(axis='both', labelsize=14)
ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
ax.spines[['top', 'right']].set_visible(False)

fname = f"{outdir}/susceptibility_near_crit.png"
fig.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
print(f"Saved figure: {fname}")

plt.show()



# %% Supplementary figures

Mws = np.array([5.55, 5.6, 5.65])

bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 1e-12], [np.inf, np.inf, np.inf, np.inf, np.inf])

edges = np.linspace(-1.0, 1.0, num_bins_mag + 1)
edges = np.insert(edges, 0, -1.001)
edges = np.insert(edges, len(edges), 1.001)

mag_centers = 0.5 * (edges[:-1] + edges[1:])

chi     = np.zeros((len(Mws),len(covs)))
m_star  = np.zeros((len(Mws),len(covs)))

c0_hat   = np.empty((len(Mws),len(covs)))
a1_hat   = np.empty((len(Mws),len(covs)))
a2_hat   = np.empty((len(Mws),len(covs)))
a4_hat   = np.empty((len(Mws),len(covs)))
a100_hat = np.empty((len(Mws),len(covs)))

for Mw_idx, Mw_val in enumerate(Mws):
    print(f"Selected Mw: {Mw_val}")

    for cov_idx, cov_val in enumerate(covs):
        # print(cov_idx)

        dv = np.zeros((len(Mws),num_sim))
        for Mw_idx_, Mw_val_ in enumerate(Mws):
            dv_path = _dv_filename("MultiStory", "Milpitas_all", Mw_val_, cov_val)
            if not dv_path.exists():
                raise FileNotFoundError(f"Missing DV file: {dv_path}")
            dv_temp = np.load(dv_path)
            dv[Mw_idx_,:] = dv_temp

        magnetization = dv * 2.0 - 1.0
        PARAMS = Mws.copy()
        
        counts = np.zeros((len(PARAMS), len(edges)-1), dtype=int)  # [y=mag_bins, x=PARAMS]
        for i in range(len(PARAMS)):
            counts[i, :], _ = np.histogram(magnetization[i, :], bins=edges)
            counts[i, 0] = 1
            counts[i, -1] = 1
        counts = counts.T

        alpha = 1.0  # pseudocount to avoid zeros
        counts_ps = counts.astype(float) + alpha
        counts_sm = gaussian_filter(counts_ps, sigma=(1.0, 0.0))
        counts_sm[0,:] = np.ones((len(PARAMS)))
        counts_sm[-1,:] = np.ones((len(PARAMS)))
        col_sum = counts_sm.sum(axis=0, keepdims=True)
        p = counts_sm / np.clip(col_sum, 1e-12, None)

        F = -np.log(np.clip(p, 1e-12, None))  # shape [M, P]

        # plt.plot(mag_centers, p[:,Mw_idx])
        # plt.show()

        # plt.plot(mag_centers, F[:,Mw_idx])
        # plt.show()

        pmin = 1e-5

        Fcol = F[:, Mw_idx]
        w = counts_sm[:, Mw_idx].astype(float)
        # Convert weights -> sigma for curve_fit (sigma ~ 1/sqrt(weight))
        w = np.maximum(w, 1e-12)
        sigma = 1.0 / np.sqrt(w)
        p_col = np.exp(-np.clip(Fcol - np.nanmin(Fcol), 0, 100))  # relative (ok for masking)
        mask = np.isfinite(Fcol) & (p_col >= pmin)

        # Ensure enough points
        if np.count_nonzero(mask) < 8:
            # fallback: relax mask
            mask = np.isfinite(Fcol)

        m = mag_centers.copy()
        c0_0, a1_0, a2_0, a4_0, a100_0 = [0,0,0,0,1e-6]

        popt, pcov = curve_fit(
            F_model,
            m[mask],
            Fcol[mask],
            p0=[c0_0, a1_0, a2_0, a4_0, a100_0],
            sigma=sigma[mask],
            absolute_sigma=False,
            bounds=bounds,
            maxfev=20000,
        )
        c0_hat[Mw_idx,cov_idx], a1_hat[Mw_idx,cov_idx], a2_hat[Mw_idx,cov_idx], a4_hat[Mw_idx,cov_idx], a100_hat[Mw_idx,cov_idx] = popt

        # m_plot = np.linspace(-1.01,1.01,100)
        # F_fit = F_model(m_plot, popt[0], popt[1], popt[2], popt[3], popt[4])
        # plt.plot(m, Fcol,'k-')
        # plt.plot(m_plot, F_fit,'r--')
        # plt.show()

        m_eq = solve_m(popt)

        chi[Mw_idx,cov_idx] = 1/(2*popt[2]+12*popt[3]*m_eq**2+9900*popt[4]*m_eq**98)
        m_star[Mw_idx,cov_idx] = m_eq


markers = ['o', 's', '^']
outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
outdir.mkdir(exist_ok=True)

Mw_landau = np.array([5.6, 5.61, 5.62])

fig, ax = plt.subplots(figsize=(5, 6), dpi=300)
for i in range(len(Mws)):
    # ax.plot(covs, m_star[i, :], '-', linewidth=2, alpha=0.5, color=CP_HEX[i], label=rf"$M_w={Mws[i]:.2f}$")
    ax.plot(covs, m_star[i, :], linestyle='None', marker=markers[i], markerfacecolor=CP_HEX[i], markeredgecolor='white', alpha=1.0)
    ax.plot(covs, solve_landau_stationary_m(Mw_landau[i], covs), '--', linewidth=2, alpha=1.0, color=CP_HEX[i])
ax.set_xlabel(r'Structural diversity $\sigma$', fontsize=20)
ax.set_ylabel(r'Equilibrium magnetization $m^*$', fontsize=20)
ax.set_xlim(0.0, 1.03)
ax.set_ylim(-0.4, 1.03)
ax.tick_params(axis='both', labelsize=14)
ax.legend(loc='lower left', fontsize=20, frameon=False)
ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
ax.spines[['top', 'right']].set_visible(False) 
# fname = f"{outdir}/griffiths_verification_mstar.png"
# fig.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
# print(f"Saved figure: {fname}")
plt.show()

i=0
fig, ax = plt.subplots(figsize=(5, 6), dpi=300)
# ax.plot(covs, chi[i, :], '-', linewidth=2, alpha=0.5, color=CP_HEX[i], label=rf"$M_w={Mws[i]:.2f}$")
ax.plot(covs, chi[i, :], linestyle='None', marker=markers[i], markerfacecolor=CP_HEX[i], markeredgecolor='white', alpha=1.0)
ax.plot(covs, susceptibility_landau(Mw_landau[i], covs)/susceptibility_landau(Mw_landau[i], covs).max()*chi[i, :].max(), '--', linewidth=2, alpha=1.0, color=CP_HEX[i])
ax.set_xlabel(r'Structural diversity $\sigma$', fontsize=20)
ax.set_ylabel(r'Susceptibility $\chi$', fontsize=20)
ax.set_xlim(0.0, 1.03)
ax.tick_params(axis='both', labelsize=14)
ax.legend(loc='upper left', fontsize=20, frameon=False)
ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
ax.spines[['top', 'right']].set_visible(False)
# fname = f"{outdir}/griffiths_verification_chi_1.png"
# fig.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
# print(f"Saved figure: {fname}")
plt.show()

ii = [1, 2]
fig, ax = plt.subplots(figsize=(5, 6), dpi=300)
for i in ii:
    # ax.plot(covs, chi[i, :], '-', linewidth=2, alpha=0.5, color=CP_HEX[i], label=rf"$M_w={Mws[i]:.2f}$")
    ax.plot(covs, chi[i, :], linestyle='None', marker=markers[i], markerfacecolor=CP_HEX[i], markeredgecolor='white', alpha=1.0)
    ax.plot(covs, susceptibility_landau(Mw_landau[i], covs)/susceptibility_landau(Mw_landau[i], covs).max()*chi[i, :].max(), '--', linewidth=2, alpha=1.0, color=CP_HEX[i])
ax.set_xlabel(r'Structural diversity $\sigma$', fontsize=20)
ax.set_ylabel(r'Susceptibility $\chi$', fontsize=20)
ax.set_xlim(0.0, 1.03)
ax.set_ylim(-0.05, 1.3)
ax.tick_params(axis='both', labelsize=14)
ax.legend(loc='upper left', fontsize=20, frameon=False)
ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
ax.spines[['top', 'right']].set_visible(False)
# fname = f"{outdir}/griffiths_verification_chi_2.png"
# fig.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
# print(f"Saved figure: {fname}")
plt.show()
