#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 10/14/2025
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

"""
# %%
import numpy as np
from scipy import special as nps 
from math import erf, sqrt, isfinite, copysign
from scipy.optimize import newton, root_scalar
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
# plt.rcParams["mathtext.fontset"] = "cm"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],
    "mathtext.fontset": "cm",
})

SaveFig = False  # Set to True to save figures
RESOLUTION = 100  # Grid resolution for plots

# %%

Mws = np.round(np.arange(3.5, 8.51, 0.05), 2)
covs = np.round(np.arange(0.0, 1.001, 0.01), 3)

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

# %% 1D plot of Landau free energy

Mw_val = 5.6
cov_val = 0.0

m_vals = np.linspace(-1, 1, RESOLUTION)

F_vals = landau_free_energy(m_vals, Mw_val, cov_val)
F_vals -= np.min(F_vals)

plt.figure(figsize=(6, 4))
plt.plot(m_vals, F_vals, 'k-')
plt.xlabel(r'$m$', fontsize=16)
plt.ylabel(r'$F(m)$', fontsize=16)
plt.ylim(0, None)
plt.xlim(m_vals[0], m_vals[-1])
plt.tight_layout()
# plt.savefig("Landau_free_energy_example.png")
plt.show()

# %% 2D heatmap of Landau free energy

Mw_vals = np.linspace(3.5, 8.5, RESOLUTION)
cov_val = 0.0

M, m_m = np.meshgrid(Mw_vals, m_vals)
Z_m = landau_free_energy(m_m, M, cov_val)

Z = Z_m.copy()
Z_max_gap = 0.0
for ii in range(Z.shape[1]):
    Z_temp = Z[:, ii]
    Z_temp_gap = np.max(Z_temp) - np.min(Z_temp)
    if Z_temp_gap > Z_max_gap:
        Z_max_gap = Z_temp_gap

    Z_temp_normed = Z_temp - np.min(Z_temp)

    Z[:, ii] = Z_temp_normed
Z /= Z_max_gap

plt.figure(figsize=(8, 6))
plt.contourf(M, m_m, Z, levels=50, cmap='RdYlBu')
plt.colorbar(label=r'$F(m)$')
plt.xlabel(r'$M_w$', fontsize=16)
plt.ylabel(r'$\mathrm{Cov}$', fontsize=16)
plt.title(r'Landau Free Energy Heatmap (Mw={:.2f})'.format(Mw_val), fontsize=16)
plt.tight_layout()
plt.show()

cov_vals = np.linspace(0.0, 1.0, RESOLUTION)
Mw_val = 5.6

C, m_c = np.meshgrid(cov_vals, m_vals)
Z_c = landau_free_energy(m_c, Mw_val, C)

Z = Z_c.copy()
Z_max_gap = 0.0
for ii in range(Z.shape[1]):
    Z_temp = Z[:, ii]
    Z_temp_gap = np.max(Z_temp) - np.min(Z_temp)
    if Z_temp_gap > Z_max_gap:
        Z_max_gap = Z_temp_gap

    Z_temp_normed = Z_temp - np.min(Z_temp)

    Z[:, ii] = Z_temp_normed
Z /= Z_max_gap

plt.figure(figsize=(8, 6))
plt.contourf(C, m_c, Z, levels=50, cmap='RdYlBu')
plt.colorbar(label=r'$F(m)$')
plt.xlabel(r'$\mathrm{Cov}$', fontsize=16)
plt.ylabel(r'$m$', fontsize=16)
plt.title(r'Landau Free Energy Heatmap (Mw={:.2f})'.format(Mw_val), fontsize=16)
plt.tight_layout()
plt.show()

# %% 3D surface plot of Landau free energy
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(M, m_m, Z_m, cmap='RdYlBu')
ax.set_xlabel(r'$M_w$', fontsize=16)
ax.set_ylabel(r'$m$', fontsize=16)
ax.set_zlabel(r'$F(m)$', fontsize=16)
ax.set_title(r'Landau Free Energy Landscape (cov={:.2f})'.format(cov_val), fontsize=16)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(M, m_c, Z_c, cmap='RdYlBu')
ax.set_xlabel(r'$\mathrm{Cov}$', fontsize=16)
ax.set_ylabel(r'$m$', fontsize=16)
ax.set_zlabel(r'$F(m)$', fontsize=16)
ax.set_title(r'Landau Free Energy Landscape (Mw={:.2f})'.format(Mw_val), fontsize=16)
plt.tight_layout()
plt.show()

# %%

select = "Mw" # "cov" or "Mw"

contour_levels = 500

if select == "Mw":
    PARAM = M.copy()
    m = m_m.copy()
    Z = Z_m.copy()
    xlabel = r"Earthquake magnitude $M_w$"
    xticks = [4, 5, 6, 7, 8]
    xticklabels = ["4", "5", "6", "7", "8"]
    title = r"Landau Free Energy Landscape (cov={:.2f})".format(cov_val)
    view_elev = 18
    view_azim = -50
else:
    PARAM = C.copy()
    m = m_c.copy()
    Z = Z_c.copy()
    xlabel = r"Structural diversity $\sigma$"
    # xticks = [0.06, 0.94]
    # xticklabels = ["Low", "High"]
    xticks = [0.02, 0.5, 0.98]
    xticklabels = ["0.0", "0.5", "1.0"]
    title = r"Landau Free Energy Landscape (Mw={:.2f})".format(Mw_val)
    view_elev = 18
    view_azim = -50

Z_max_gap = 0.0
for ii in range(Z.shape[1]):
    Z_temp = Z[:, ii]
    Z_temp_gap = np.max(Z_temp) - np.min(Z_temp)
    if Z_temp_gap > Z_max_gap:
        Z_max_gap = Z_temp_gap

    Z_temp_normed = Z_temp - np.min(Z_temp)

    Z[:, ii] = Z_temp_normed
Z /= Z_max_gap

fig = plt.figure(figsize=(10,7), dpi=300)
ax  = fig.add_subplot(111, projection="3d")
ax.set_proj_type("ortho")


from matplotlib import colors

norm = colors.PowerNorm(gamma=0.3, vmin=Z.min(), vmax=Z.max()) # gamma < 1 → expands low end, compresses high end

surf = ax.plot_surface(
    PARAM, m, Z,
    rstride=1, cstride=1,
    linewidth=0.0,
    antialiased=True,
    cmap="RdYlBu",
    norm=norm,
    alpha=0.95,
    shade=False,
)

# ax.contour3D(
#     PARAM, m, Z,
#     levels=contour_levels, colors="k", linewidths=0.3, alpha=0.7
# )

ax.set_xlim([PARAM.min(), PARAM.max()])
ax.set_ylim([m.min(), m.max()])
zmin, zmax = Z.min(), Z.max()
zrange = zmax - zmin
margin = 0.50 * zrange 
ax.set_zlim(zmin-margin, zmax + 0.3*margin)
z0 = zmin - margin

ax.contourf(PARAM, m, Z, zdir="z", offset=z0, levels=contour_levels, cmap="RdYlBu", alpha=0.3, norm=norm)

ax.set_xlabel(xlabel, labelpad=5, fontsize=14)
ax.set_ylabel(r"Damage fraction", labelpad=5, fontsize=14)
ax.text2D(0.02, 0.61, r"Free energy", transform=ax.transAxes,
          rotation=90, va="top", ha="left",fontsize=14)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_yticks([-0.95, 0.0, 0.95])
ax.set_yticklabels(["0.0", "0.5", "1.0"])
ax.set_zticks([])
ax.tick_params(axis="both", which="major", pad=0)

# --- Grid ---
ax.grid(False)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = True
    axis.pane.set_facecolor((0.98, 0.98, 0.98, 0.3))
    axis.pane.set_edgecolor((0.0, 0.0, 0.0, 1.0))
    axis.pane.set_linewidth(1.0)
    axis.pane.set_alpha(1.0)

from mpl_toolkits.mplot3d.art3d import Line3DCollection

xmin, xmax = float(ax.get_xlim()[0]), float(ax.get_xlim()[-1])
ymin, ymax = float(ax.get_ylim()[0]), float(ax.get_ylim()[-1])
zmin, zmax = float(ax.get_zlim()[0]), float(ax.get_zlim()[-1])

# Use your existing major ticks for a clean grid
xt = [t for t in ax.get_xticks() if xmin <= t <= xmax]
xt.insert(0, xmin)
xt.append(xmax)
yt = [t for t in ax.get_yticks() if ymin <= t <= ymax]
yt.insert(0, ymin)
yt.append(ymax)

ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

# Build line segments along x (for each y tick) and along y (for each x tick)
segs = []
for y in yt:
    if y > yt[0] and y < yt[-1]:   # skip outermost to avoid double-drawing
        segs.append(np.array([[xmin, y, z0], [xmax, y, z0]]))   # horizontal lines
for x in xt:
    if x > xt[0] and x < xt[-1]:   # skip outermost to avoid double-drawing
        segs.append(np.array([[x, ymin, z0], [x, ymax, z0]]))   # vertical lines

floor_grid = Line3DCollection(
    segs,
    linewidths=0.5,
    linestyles="--",
    colors=(0.6, 0.6, 0.6)  # light gray
)
ax.add_collection3d(floor_grid)

# ax.set_box_aspect(None, zoom=0.9)
ax.set_box_aspect((1, 1, 0.4), zoom=0.9)
ax.view_init(elev=view_elev, azim=view_azim)
ax.zaxis._axinfo["juggled"] = (1, 2, 0)

# cb = fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.02)
# cb.set_label("Free energy", fontsize=12)

# ax.set_title(title, fontsize=14)
if SaveFig:
    plt.savefig("landau_RFIM_{}.png".format(select), dpi=300, transparent=True, bbox_inches="tight")
    print("Saved figure: landau_RFIM_{}.png".format(select))

plt.show()
# %%
