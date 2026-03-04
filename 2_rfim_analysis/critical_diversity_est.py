#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 11/07/2025
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "mathtext.fontset": "cm",
})

CP_HEX = [
    "#4056A1", "#F12815", "#D79922", "#14A098",
    "#CB2D6F", "#501F3A", "#116466", "#F76C6C",
    "#F0DFAE", "#C5CBE3", "#8C9AC7", "#0072B5",
]

outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
outdir.mkdir(exist_ok=True)

# %%
# Mws = np.round(np.arange(3.5, 8.51, 0.05), 2)
Mw_val = 5.50
covs = np.round(np.arange(0.0, 0.501, 0.01), 3)
Ts   = np.round(np.arange(0.0, 0.501, 0.01), 3)
num_sim = 10_000
num_bins_mag = 101

# %%
from typing import Tuple, Optional, Literal

def _dv_filename(
    Mw: Optional[float],
    cov: Optional[float],
    T: Optional[float],
) -> Tuple[Path, str]:
    """Build DV filename and a short English title."""
    Mw_label = f"{int(round(Mw*100)):03d}"
    cov_label = f"{int(round(cov*1000)):04d}"
    T_label = f"{int(round(T*100)):03d}"

    base = f"C:/Users/ohsb1/Desktop/Savio/Milpitas_all/dv/MultiStory_dv"
    suffix = f"_Mw{Mw_label}_cov{cov_label}_Milpitas_all_T{T_label}"
    fname = f"{base}{suffix}.npy"

    return Path(fname)

def F_model(m, c0, a1, a2, a100):
    return c0 + a1*m + a2*m**2 + a100*m**100

def solve_m(popt, tol=1e-12, max_iter=100):
    """
    Solve a2 + 100*a100*m^99 = 0 for m.
    Uses Newton's method with a quadratic-based initial guess.
    """
    a1      = popt[1]
    a2      = popt[2]
    a100    = popt[3]
    
    m0 = 1.0 # Initial guess

    def f(m):
        return a1 + 2.0*a2*m + 100.0*a100*(m**99)

    def df(m):
        return 2.0*a2 + 9900.0*a100*(m**98)

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
bounds = ([-np.inf, -np.inf, -np.inf, 1e-12], [np.inf, np.inf, np.inf, np.inf])

edges = np.linspace(-1.0, 1.0, num_bins_mag + 1)
edges = np.insert(edges, 0, -1.001)
edges = np.insert(edges, len(edges), 1.001)

mag_centers = 0.5 * (edges[:-1] + edges[1:])

sigma_c  = np.zeros((len(Ts),)) # Critical structural diversity
min_a2   = np.zeros((len(Ts),))

c0_hat   = np.empty((len(Ts), len(covs)))
a1_hat   = np.empty((len(Ts), len(covs)))
a2_hat   = np.empty((len(Ts), len(covs)))
a100_hat = np.empty((len(Ts), len(covs)))

# %%
print(f"Selected Mw: {Mw_val}")

for T_idx, T_val in enumerate(Ts):
    print(T_idx)

    dv = np.zeros((len(covs),num_sim))
    for cov_idx, cov_val in enumerate(covs):
        dv_path = _dv_filename(Mw_val, cov_val, T_val)
        dv_temp = np.load(dv_path)
        dv[cov_idx,:] = dv_temp

    magnetization = dv * 2.0 - 1.0
    PARAMS = covs.copy()
    
    counts = np.zeros((len(PARAMS), len(edges)-1), dtype=int)  # [y=mag_bins, x=PARAMS]
    for i in range(len(PARAMS)):
        counts[i, :], _ = np.histogram(magnetization[i, :], bins=edges)
        counts[i, 0] = 1
        counts[i, -1] = 1
    counts = counts.T

    alpha = 1.0  # pseudocount to avoid zeros
    counts_ps = counts.astype(float) + alpha
    counts_sm = gaussian_filter(counts_ps, sigma=(1.0, 1.0))
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
    m = mag_centers.copy()
    c0_0, a1_0, a2_0, a4_0, a100_0 = [0,0,0,0,1e-6]

    for cov_idx, cov_val in enumerate(covs):
        Fcol = F[:, cov_idx]
        w = counts_sm[:, cov_idx].astype(float)
        w = np.maximum(w, 1e-12)
        sigma = 1.0 / np.sqrt(w)
        p_col = np.exp(-np.clip(Fcol - np.nanmin(Fcol), 0, 100))  # relative (ok for masking)
        mask = np.isfinite(Fcol) & (p_col >= pmin)

        # Ensure enough points
        if np.count_nonzero(mask) < 8:
            # fallback: relax mask
            mask = np.isfinite(Fcol)

        popt, pcov = curve_fit(
            F_model,
            m[mask],
            Fcol[mask],
            p0=[c0_0, a1_0, a2_0, a100_0],
            sigma=sigma[mask],
            absolute_sigma=False,
            bounds=bounds,
            maxfev=20000,
        )
        c0_hat[T_idx,cov_idx], a1_hat[T_idx,cov_idx], a2_hat[T_idx,cov_idx], a100_hat[T_idx,cov_idx] = popt

    # m_plot = np.linspace(-1.01,1.01,100)
    # F_fit = F_model(m_plot, popt[0], popt[1], popt[2], popt[3])
    # plt.plot(m, Fcol,'k-')
    # plt.plot(m_plot, F_fit,'r--')
    # plt.show()
    
    idx_a2_zero     = np.argmin(np.abs(a2_hat[T_idx, :]))
    min_a2[T_idx]   = np.abs(a2_hat[T_idx, idx_a2_zero])
    sigma_c[T_idx]  = covs[idx_a2_zero]


# %%
# plt.plot(Ts, min_a2, '-o', color='black')
# plt.plot(Ts, np.zeros((len(Ts))),'r--')
# plt.xlabel('Effective temperature')
# plt.ylabel('Minimum a2 (should be close to zero)')
# plt.show()

# %%
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(Ts, sigma_c, '-o', color=CP_HEX[0], linewidth=1.5)
ax.fill_between(Ts, sigma_c, 0, color=CP_HEX[0], alpha=0.2)
ax.fill_between(Ts, sigma_c, ax.get_ylim()[1], color="#F1DC9C", alpha=0.3)
ax.set_xlabel(r'Effective temperature $T$', fontsize=18)
ax.set_ylabel(r'Structural diversity $\sigma$', fontsize=18)
x_pad = (Ts.max() - Ts.min()) * 0.02
y_pad = (covs.max() - covs.min()) * 0.02
ax.set_xlim(Ts.min() - x_pad, Ts.max() + x_pad)
ax.set_ylim(covs.min() - y_pad, covs.max() + y_pad)
ax.tick_params(axis='both', labelsize=14)
ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.0)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='both', linewidth=0.5)
fname = f"{outdir}/critical_diversity_est_temp.png"
# plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
# print(f"Saved figure: {fname}")
# fig.tight_layout()
plt.show()


# %%
Mws     = np.round(np.arange(4.5, 5.61, 0.1), 2)
covs    = np.round(np.arange(0.0, 1.001, 0.01), 3)
Ts      = np.round(np.arange(0.0, 1.001, 0.05), 3)

sigma_c_3d  = np.zeros((len(Mws), len(Ts)))

alpha = 1.0  # pseudocount to avoid zeros
pmin = 1e-5
m = mag_centers.copy()
c0_0, a1_0, a2_0, a4_0, a100_0 = [0,0,0,0,1e-6]

for Mw_idx, Mw_val in enumerate(Mws):
    print(f"Selected Mw: {Mw_val}")

    if Mw_val < 5.0:
        covs_ = np.round(np.arange(0.0, 1.001, 0.01), 3)
    else:
        covs_ = np.round(np.arange(0.0, 0.801, 0.01), 3)
    PARAMS = covs_.copy()
    
    for T_idx, T_val in enumerate(Ts):
        # print(T_idx)

        if T_val > 0.5:
            covs_ = np.round(np.arange(0.0, 0.201, 0.01), 3)
            PARAMS = covs_.copy()

        dv = np.zeros((len(covs_),num_sim))
        for cov_idx, cov_val in enumerate(covs_):
            dv_path = _dv_filename(Mw_val, cov_val, T_val)
            dv_temp = np.load(dv_path)
            dv[cov_idx,:] = dv_temp

        magnetization = dv * 2.0 - 1.0
    
        counts = np.zeros((len(PARAMS), len(edges)-1), dtype=int)  # [y=mag_bins, x=PARAMS]
        for i in range(len(PARAMS)):
            counts[i, :], _ = np.histogram(magnetization[i, :], bins=edges)
            counts[i, 0] = 1
            counts[i, -1] = 1
        counts = counts.T
        
        counts_ps = counts.astype(float) + alpha
        counts_sm = gaussian_filter(counts_ps, sigma=(1.0, 1.0))
        counts_sm[0,:] = np.ones((len(PARAMS)))
        counts_sm[-1,:] = np.ones((len(PARAMS)))
        col_sum = counts_sm.sum(axis=0, keepdims=True)
        p = counts_sm / np.clip(col_sum, 1e-12, None)

        F = -np.log(np.clip(p, 1e-12, None))  # shape [M, P]

        # plt.plot(mag_centers, p[:,Mw_idx])
        # plt.show()
        # plt.plot(mag_centers, F[:,Mw_idx])
        # plt.show()

        a2_hat = np.zeros((len(covs_)))
        for cov_idx, cov_val in enumerate(covs_):
            Fcol = F[:, cov_idx]
            w = counts_sm[:, cov_idx].astype(float)
            w = np.maximum(w, 1e-12)
            sigma = 1.0 / np.sqrt(w)
            p_col = np.exp(-np.clip(Fcol - np.nanmin(Fcol), 0, 100))  # relative (ok for masking)
            mask = np.isfinite(Fcol) & (p_col >= pmin)

            # Ensure enough points
            if np.count_nonzero(mask) < 8:
                # fallback: relax mask
                mask = np.isfinite(Fcol)

            popt, pcov = curve_fit(
                F_model,
                m[mask],
                Fcol[mask],
                p0=[c0_0, a1_0, a2_0, a100_0],
                sigma=sigma[mask],
                absolute_sigma=False,
                bounds=bounds,
                maxfev=20000,
            )
            _, _, a2_hat[cov_idx], _ = popt
        
        idx_a2_zero     = np.argmin(np.abs(a2_hat))
        sigma_c_3d[Mw_idx, T_idx]  = covs[idx_a2_zero]

# %%
Mws_original = Mws.copy()
sigma_c_3d_original = sigma_c_3d.copy()

Mws = Mws[1:]
sigma_c_3d = sigma_c_3d[1:,:]

# %%
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import interp1d

Ts_ = Ts.copy()
sigma_c_3d_ = sigma_c_3d.copy()

Ts = Ts_[:12]
sigma_c_3d = sigma_c_3d_[:,:12]

X, Y = np.meshgrid(Mws, Ts)
Z = sigma_c_3d.T   # σ data

# Swap axes: make Temperature the z-axis
X_new = X                  # still M_w
Y_new = Z                  # structural diversity σ becomes Y-axis
Z_new = Y                  # temperature T becomes vertical axis

# --- Figure / axes ---
fig = plt.figure(figsize=(10,10))
ax  = fig.add_subplot(111, projection='3d')

# --- Main surface ---
ax.plot_surface(X_new, Y_new, Z_new,
                facecolor='white', edgecolor='k',
                linewidth=0.15, alpha=0.5, antialiased=True, zorder=1)

# --- Common grids / intersections reused later ---
Mw_grid, T_grid = np.meshgrid(Mws, Ts)

# σ(Mw, T=0) along Mw (T=0 intersection curve on the base plane)
sigma_at_T0 = np.array([
    interp1d(Z_new[:, j], Y_new[:, j], kind='linear', fill_value='extrapolate')(0.0)
    for j in range(X_new.shape[1])
], dtype=float)
ax.plot(Mws, sigma_at_T0, np.zeros_like(Mws), 'k--', lw=2.5, clip_on=False, zorder=9)

# --- Fixed-σ planes ---
# σ = 0.0 (light blue)
ax.plot_surface(Mw_grid, np.full_like(Mw_grid, 0.0), T_grid, color=CP_HEX[9], alpha=0.25, edgecolor='none')
# σ = σ_max (light coral)  ← (fixed value; not Z.max())
# ax.plot_surface(Mw_grid, np.full_like(Mw_grid, Z.max()), T_grid, color="#F1DC9C", alpha=0.25, edgecolor='none')

# --- Helpers ---
def _mw_column_interp(Mw):
    """Return (T_sorted, sigma_sorted, sigma_of_T) on the Mw-constant column."""
    j  = int(np.argmin(np.abs(Mws - Mw)))
    T  = Z_new[:, j].astype(float)
    S  = Y_new[:, j].astype(float)
    o  = np.argsort(T)
    T, S = T[o], S[o]
    f  = interp1d(T, S, kind='linear', fill_value='extrapolate')
    return T, S, f

def plot_Mw_intersection(Mw, color='lightgray', lw=1.0):
    T, S, _ = _mw_column_interp(Mw)
    ax.plot([Mw]*len(T), S, T, color=color, lw=lw)

def fill_Mw_between(Mw, T0, T1, c_under=CP_HEX[9], c_over="#F1DC9C", alpha=0.35, sigma_cap=None):
    T, S, f = _mw_column_interp(Mw)
    T0, T1  = float(T0), float(T1)
    s0, s1  = float(f(T0)), float(f(T1))
    T_seg   = np.linspace(T0, T1, 200)
    S_seg   = f(T_seg)

    # Under-surface (σ: 0 → σ(T))
    verts_under = [(Mw, 0.0, T0), (Mw, s0, T0),
                   *[(Mw, float(s), float(t)) for s, t in zip(S_seg, T_seg)],
                   (Mw, s1, T1), (Mw, 0.0, T1)]
    ax.add_collection3d(Poly3DCollection([verts_under], facecolor=c_under, edgecolor='none', alpha=alpha))

    # Opposite region (σ: σ(T) → σ_cap)
    if sigma_cap is None:
        sigma_cap = float(np.nanmax(Y_new))
    verts_over = [(Mw, s0, T0), (Mw, sigma_cap, T0), (Mw, sigma_cap, T1), (Mw, s1, T1),
                  *[(Mw, float(s), float(t)) for s, t in zip(S_seg[::-1], T_seg[::-1])]]
    ax.add_collection3d(Poly3DCollection([verts_over], facecolor=c_over, edgecolor='none', alpha=alpha))

def fill_T0_strip(Mw_lo, Mw_hi, c_under=CP_HEX[9], c_over="#F1DC9C", alpha=0.35, sigma_cap=None):
    Mw_seg = np.linspace(Mw_lo, Mw_hi, 200)
    sig_seg = np.interp(Mw_seg, Mws, sigma_at_T0)
    s0 = float(np.interp(Mw_lo, Mws, sigma_at_T0))
    s1 = float(np.interp(Mw_hi, Mws, sigma_at_T0))

    # Under-surface on T=0
    verts_under = [(Mw_lo, 0.0, 0.0), (Mw_lo, s0, 0.0),
                   *[(float(m), float(s), 0.0) for m, s in zip(Mw_seg, sig_seg)],
                   (Mw_hi, 0.0, 0.0)]
    ax.add_collection3d(Poly3DCollection([verts_under], facecolor=c_under, edgecolor='none', alpha=alpha))

    # Opposite region on T=0
    if sigma_cap is None:
        sigma_cap = float(np.nanmax(Y_new))
    verts_over = [(Mw_lo, s0, 0.0), (Mw_lo, sigma_cap, 0.0),
                  *[(float(m), sigma_cap, 0.0) for m in Mw_seg],
                  (Mw_hi, sigma_cap, 0.0), (Mw_hi, s1, 0.0),
                  *[(float(m), float(s), 0.0) for m, s in zip(Mw_seg[::-1], sig_seg[::-1])]]
    ax.add_collection3d(Poly3DCollection([verts_over], facecolor=c_over, edgecolor='none', alpha=alpha))

# --- Intersections (Mw sections) ---
plot_Mw_intersection(Mws.min())
plot_Mw_intersection(Mws.max())

# --- Filled regions ---
fill_Mw_between(Mw=Mws.min(), T0=Z_new.min(), T1=Z_new.max())
# fill_Mw_between(Mw=Mws.max(), T0=Ts.min(), T1=Ts.max())
fill_T0_strip(Mw_lo=Mws.min(), Mw_hi=Mws.max())


Mw_star = 5.5
j = int(np.argmin(np.abs(Mws - Mw_star)))
# σ–T cross-section at Mw = 5.5
T_col = Z_new[:, j].astype(float)   # T values
S_col = Y_new[:, j].astype(float)   # σ values
# Sort by T so the line is ordered
o = np.argsort(T_col)
T_line = T_col[o]
S_line = S_col[o]

T_line = T_line[:-1]
S_line = S_line[:-1]

# Curve on the surface (x fixed at Mw = 5.5)
ax.plot([Mw_star]*len(T_line), S_line, T_line, 'o-', color=CP_HEX[0], lw=2.0, clip_on=False, zorder=10)

# --- Legend ---
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0],[0], marker='s', color='none', markerfacecolor=CP_HEX[9],  markersize=12, label='Synchronized (below surface)'),
    Line2D([0],[0], marker='s', color='none', markerfacecolor="#F1DC9C", markersize=12, label='Volatile (above surface)'),
]
# ax.legend(handles=legend_elems, loc='upper left', frameon=False, fontsize=20, ncol=2, columnspacing=1.4, handletextpad=0.5)
ax.legend(
    handles=legend_elems,
    loc='upper left',             # anchor point
    bbox_to_anchor=(-0.07, 0.97),  # (x, y) in axes fraction coordinates
    frameon=False,
    fontsize=20,
    columnspacing=1.0,
    ncol=2
)

# --- Styling ---
ax.view_init(elev=22, azim=42)
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor('white'); axis.pane.set_edgecolor('white')
    axis._axinfo['grid'].update(linestyle='--', linewidth=0.2, color='gray')

ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel(r'Earthquake magnitude $M_w$', fontsize=20, labelpad=10)
ax.set_ylabel(r'Structural diversity $\sigma$', fontsize=20, labelpad=10)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r'Effective temperature $T$', fontsize=20, rotation=92, labelpad=5)

ax.set_xlim([Mws.min(), Mws.max()])
ax.set_ylim([covs.min(), covs.max()])
ax.set_zlim([Z_new.min(), Z_new.max()])

ax.set_xticks([4.6, 4.8, 5.0, 5.2, 5.4, 5.6])

outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
outdir.mkdir(exist_ok=True)
fname = f"{outdir}/phase_transition_3d_temp.png"
plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
print(f"Saved figure: {fname}")

plt.show()

# %%
