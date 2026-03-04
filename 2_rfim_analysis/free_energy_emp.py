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
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "mathtext.fontset": "cm",
})

CP_HEX = [
    "#4056A1", "#F12815", "#D79922", "#14A098",
    "#CB2D6F", "#501F3A", "#116466", "#F76C6C",
    "#EFE2BA", "#C5CBE3", "#8C9AC7", "#0072B5",
]

outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
outdir.mkdir(exist_ok=True)

# %%
Mws = np.round(np.arange(3.5, 8.51, 0.05), 2)
covs = np.round(np.arange(0.0, 1.001, 0.01), 3)
num_sim = 10_000
num_bins_mag = 101

# %%
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
Mws = np.array([5.6, 5.65, 5.7])
# Mws = np.array([6.25, 6.30, 6.35])

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
        # F_fit = F_model(m_plot, popt[0], popt[1], popt[2], popt[3])
        # plt.plot(m, Fcol,'k-')
        # plt.plot(m_plot, F_fit,'r--')
        # plt.show()

        m_eq = solve_m(popt)

        chi[Mw_idx,cov_idx] = 1/(2*popt[2]+12*popt[3]*m_eq**2+9900*popt[4]*m_eq**98)
        m_star[Mw_idx,cov_idx] = m_eq


markers = ['o', 's', '^']

plt.figure(figsize=(8,6), dpi=300)
for i in range(3):
    plt.plot(covs, m_star[i, :], '-',  marker=markers[i], linewidth=2, ms=5, color=CP_HEX[i])
plt.xlabel(r'Structural diversity $\sigma$', fontsize=20)
plt.ylabel(r'Equilibrium magnetization $m^*$', fontsize=20)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.tick_params(axis='both',labelsize=14)
plt.legend([rf"$M_w=5.60$", rf"$M_w=5.65$", rf"$M_w=5.70$"], loc='lower left', fontsize=20, frameon=False)
plt.grid(linewidth=0.5)
# fname = f"{outdir}/phase_transition_detect_2nd_mstar.png"
# plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
# print(f"Saved figure: {fname}")
plt.show()

plt.figure(figsize=(8,6), dpi=300)
for i in range(3):
    plt.plot(covs, chi[i, :], '-',  marker=markers[i], linewidth=2, ms=5, color=CP_HEX[i])
plt.xlabel(r'Structural diversity $\sigma$', fontsize=20)
plt.ylabel(r'Susceptibility $\chi$', fontsize=20)
plt.xlim([0.0,1.0])
plt.tick_params(axis='both',labelsize=14)
plt.legend([rf"$M_w=5.60$", rf"$M_w=5.65$", rf"$M_w=5.70$"], loc='upper left', fontsize=20, frameon=False)
plt.grid(linewidth=0.5)
# fname = f"{outdir}/phase_transition_detect_2nd_chi.png"
# plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
# print(f"Saved figure: {fname}")
plt.show()



# %% Example free energy curves
Mws = np.array([5.55])
Mw_idx = 0

cov_vals = np.array([0.0, 0.5, 1.0])

for cov_idx, cov_val in enumerate(cov_vals):

    dv = np.zeros((len(Mws),num_sim))
    for Mw_idx, Mw_val in enumerate(Mws):
        dv_path = _dv_filename("MultiStory", "Milpitas_all", Mw_val, cov_val)
        if not dv_path.exists():
            raise FileNotFoundError(f"Missing DV file: {dv_path}")
        dv_temp = np.load(dv_path)
        dv[Mw_idx,:] = dv_temp

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

    if cov_idx == 0:
        m_plot = np.linspace(-0.993,0.993,100)
    elif cov_idx == 1:
        m_plot = np.linspace(-0.99,0.99,100)
    else:
        m_plot = np.linspace(-0.99,0.99,100)
    F_fit = F_model(m_plot, popt[0], popt[1], popt[2], popt[3], popt[4])

    plt.figure(figsize=(8,6),dpi=300)
    plt.plot(m, Fcol,'k-',linewidth=1.5, label='Empirical')
    plt.plot(m_plot, F_fit,'r--',linewidth=2.0, label='Fitted curve')
    plt.xticks([-1, -0.5, 0, 0.5, 1],["0.0","0.25","0.5","0.75","1.0"])
    plt.tick_params(axis='y', which='both', left=False, labelleft=False)
    plt.xlabel(r'Damage fraction', fontsize=18)
    plt.ylabel(r'Free energy', fontsize=18)
    plt.grid(axis='both', linewidth=0.5)
    if cov_idx == 1:
        plt.legend(fontsize=18, loc='upper center',frameon=False)
    # fname = f"{outdir}/data_driven_landau_Milpitas_Mw{Mws}_cov{cov_val}.png"
    # plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
    # print(f"Saved figure: {fname}")
    plt.show()


# %%
# %%
TUNE_PARAM = "cov"

if TUNE_PARAM == "Mw":
    Mws = np.round(np.arange(3.5, 8.51, 0.05), 2)
    cov_val = 0.0

    dv = np.zeros((len(Mws),num_sim))
    for Mw_idx, Mw_val in enumerate(Mws):
        dv_path = _dv_filename("MultiStory", "Milpitas_all", Mw_val, cov_val)
        if not dv_path.exists():
            raise FileNotFoundError(f"Missing DV file: {dv_path}")
        dv_temp = np.load(dv_path)
        dv[Mw_idx,:] = dv_temp

    PARAMS = Mws.copy()

else:
    Mw_val = 5.6
    covs = np.round(np.arange(0.0, 1.001, 0.01), 3)

    dv = np.zeros((len(covs),num_sim))
    for cov_idx, cov_val in enumerate(covs):

        dv_path = _dv_filename("MultiStory", "Milpitas_all", Mw_val, cov_val)
        if not dv_path.exists():
            raise FileNotFoundError(f"Missing DV file: {dv_path}")
        dv_temp = np.load(dv_path)
        dv[cov_idx,:] = dv_temp

    PARAMS = covs.copy()

magnetization = dv * 2.0 - 1.0

# %% Empirical (Purely data-driven) free energy landscape
edges = np.linspace(-1.0, 1.0, len(PARAMS) + 1)

# edges = np.linspace(-1.0, 1.0, num_bins_mag + 1)
# edges = np.insert(edges, 0, -1.001)
# edges = np.insert(edges, len(edges), 1.001)

mag_centers = 0.5 * (edges[:-1] + edges[1:])

counts = np.zeros((len(edges)-1, len(PARAMS)), dtype=int)  # [y=mag_bins, x=PARAMS]
for i in range(len(PARAMS)):
    counts[:, i], _ = np.histogram(magnetization[i, :], bins=edges)

alpha = 1.0
counts_ps = counts.astype(float) + alpha
counts_sm = gaussian_filter(counts_ps, sigma=(1.0, 1.0))

col_sum = counts_sm.sum(axis=0, keepdims=True)
p = counts_sm / np.clip(col_sum, 1e-12, None)

F = -np.log(np.clip(p, 1e-12, None)) 

X, Y = np.meshgrid(PARAMS, mag_centers)
Z = (F - F.min())/(F.max() - F.min())

from matplotlib import colors
if TUNE_PARAM == "Mw":
    xlabel = r'Earthquake magnitude $M_w$'
    xticks = [4, 5, 6, 7, 8]
    xticklabels = ["4", "5", "6", "7", "8"]
    view_elev = 30
    view_azim = -55
    norm = colors.PowerNorm(gamma=1.0, vmin=Z.min(), vmax=Z.max()) # gamma < 1 → expands low end, compresses high end
else:
    xlabel = r"Structural diversity $\sigma$"
    xticks = [0.05, 0.95]
    xticklabels = ["Low", "High"]
    view_elev = 25
    view_azim = -80
    norm = colors.PowerNorm(gamma=0.5, vmin=Z.min(), vmax=Z.max()) # gamma < 1 → expands low end, compresses high end

fig = plt.figure(figsize=(10, 7), dpi=300)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='RdYlBu', rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=1.0, shade=False, norm=norm)
# ax.set_xlabel(r'Earthquake magnitude $M_w$', fontsize=14)
# ax.set_ylabel(r'Damage fraction', fontsize=14)
# ax.set_zlabel(r'Free energy', fontsize=14)
# plt.show()

ax.set_xlim([X.min(), X.max()])
ax.set_ylim([Y.min(), Y.max()])
# zmin, zmax = Z.min(), Z.max()
# zrange = zmax - zmin
# margin = 0.50 * zrange 
# ax.set_zlim(zmin-margin, zmax + 0.3*margin)
# z0 = zmin - margin

# ax.contourf(X, Y, Z, zdir="z", offset=z0, levels=num_bins_mag, cmap="RdYlBu", alpha=0.3, norm=norm, )

ax.set_xlabel(xlabel, labelpad=3, fontsize=14)
ax.set_ylabel(r"Damage fraction", labelpad=3, fontsize=14)
# ax.set_zlabel(r"Free energy", labelpad=-10, fontsize=14)
ax.text2D(0.07, 0.5, r"Free energy", transform=ax.transAxes,
          rotation=93, va="top", ha="left",fontsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_yticks([-1.0, 0.0, 1.0])
ax.set_yticklabels(["0.0", "0.5", "1.0"])
ax.set_zticks([])
ax.tick_params(axis="both", which="major", pad=0)
ax.grid(False)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = True
    axis.pane.set_facecolor((0.95, 0.95, 0.95, 0.5))
    axis.pane.set_edgecolor((0.0, 0.0, 0.0, 1.0))
    axis.pane.set_linewidth(0.5)
    axis.pane.set_alpha(1.0)
xmin, xmax = float(ax.get_xlim()[0]), float(ax.get_xlim()[-1])
ymin, ymax = float(ax.get_ylim()[0]), float(ax.get_ylim()[-1])
zmin, zmax = float(ax.get_zlim()[0]), float(ax.get_zlim()[-1])
ax.plot(
    [xmin, xmin],
    [ymin, ymin],
    [zmin, zmax],
    "-", linewidth=0.5, color="black"
)
ax.plot([0.05, 0.05],[-1.0, 1.0],[zmin, zmin], ":", linewidth=0.5, color="black")
ax.plot([0.95, 0.95],[-1.0, 1.0],[zmin, zmin], ":", linewidth=0.5, color="black")
ax.plot([0.5, 0.5],[-1.0, 1.0],[zmin, zmin], ":", linewidth=0.5, color="black")
# ax.set_box_aspect(None, zoom=0.9)
ax.view_init(elev=view_elev, azim=view_azim)
ax.zaxis._axinfo["juggled"] = (1, 2, 0)
outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
outdir.mkdir(exist_ok=True)
# fname = f"{outdir}/data_driven_landau_Milpitas_{TUNE_PARAM}.png"
# plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
# print(f"Saved figure: {fname}")
plt.show()
