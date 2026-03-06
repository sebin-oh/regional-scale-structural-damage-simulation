#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 10/14/2025
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

Plot Landau free-energy landscapes using pre-estimated RFIM mean-field parameters:
  a1(Mw) from coeffs_a1, and a2(sigma) from coeffs_a2.

Supports two views:
  - SELECT="Mw"    : vary Mw, fix sigma
  - SELECT="sigma" : vary sigma, fix Mw

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.special import erf  # vectorized


# =============================================================================
# User settings
# =============================================================================
SaveFig = False
RESOLUTION = 100
CONTOUR_LEVELS = 500

SELECT: Literal["Mw", "sigma"] = "sigma"  # "Mw" or "sigma"

TARGET_REGION = "Milpitas"
TARGET_STRUCTURE: Literal["SingleStory", "TwoStory", "MultiStory"] = "MultiStory"

# Which slice to plot
MW_FIXED = 5.6
SIGMA_FIXED = 0.0

# Grids for plotting
MW_MINMAX = (3.5, 8.5)
SIGMA_MINMAX = (0.0, 1.0)
M_MINMAX = (-1.0, 1.0)

# Paths
RFIM_PARAMS_PATH = Path(f"./data/rfim_params_est/{TARGET_REGION}_{TARGET_STRUCTURE}_rfim_coeffs.npz")
OUT_FIG = Path(f"./results/landau_RFIM_{SELECT}.png")


# =============================================================================
# I/O
# =============================================================================
def load_mf_params(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load coeffs_a1, coeffs_a2, and meta dict from an .npz file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing RFIM params file: {path}")

    obj = np.load(path, allow_pickle=False)
    coeffs_a1 = np.asarray(obj["coeffs_a1"], dtype=float)
    coeffs_a2 = np.asarray(obj["coeffs_a2"], dtype=float)
    meta = json.loads(str(obj["meta"])) if "meta" in obj.files else {}
    return coeffs_a1, coeffs_a2, meta


# =============================================================================
# Model
# =============================================================================
def landau_free_energy(m: np.ndarray, *, a1: np.ndarray, a2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Landau-like free energy (up to an additive constant):

      F(m) = 1/2 m^2
             - (m+a1) * erf( (m+a1)/(sqrt(2)*a2) )
             - [1/(alpha*sqrt(pi))] * exp( -(alpha*(m+a1))^2 )

    where alpha = 1/(sqrt(2)*a2).
    """
    m = np.asarray(m, dtype=float)
    a1 = np.asarray(a1, dtype=float)
    a2 = np.maximum(np.asarray(a2, dtype=float), eps)

    alpha = 1.0 / (np.sqrt(2.0) * a2)
    z = alpha * (m + a1)

    term_quad = 0.5 * m**2
    term_erf = (m + a1) * erf(z)
    term_exp = np.exp(-(z**2)) / (alpha * np.sqrt(np.pi))

    return term_quad - term_erf - term_exp


# =============================================================================
# Plot helpers
# =============================================================================
def normalize_columns(Z: np.ndarray) -> np.ndarray:
    """
    Match your original normalization:
      - subtract column-wise minimum
      - divide by max(column-wise range)
    """
    Z = np.asarray(Z, dtype=float)
    zmin = np.min(Z, axis=0, keepdims=True)
    Z0 = Z - zmin
    gap = np.max(Z, axis=0) - np.min(Z, axis=0)
    z_max_gap = float(np.max(gap)) if gap.size else 1.0
    return Z0 / (z_max_gap if z_max_gap > 0 else 1.0)


def add_floor_grid(ax, *, x0: float, x1: float, y0: float, y1: float, z0: float) -> None:
    """Add dashed grid lines on the z=z0 floor."""
    xt = [t for t in ax.get_xticks() if x0 <= t <= x1]
    yt = [t for t in ax.get_yticks() if y0 <= t <= y1]

    xt = [x0] + xt + [x1]
    yt = [y0] + yt + [y1]

    segs = []
    for y in yt[1:-1]:
        segs.append(np.array([[x0, y, z0], [x1, y, z0]]))
    for x in xt[1:-1]:
        segs.append(np.array([[x, y0, z0], [x, y1, z0]]))

    floor_grid = Line3DCollection(
        segs,
        linewidths=0.5,
        linestyles="--",
        colors=(0.6, 0.6, 0.6),
    )
    ax.add_collection3d(floor_grid)


def style_3d_axes(ax) -> None:
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor((0.98, 0.98, 0.98, 0.3))
        axis.pane.set_edgecolor((0.0, 0.0, 0.0, 1.0))
        axis.pane.set_linewidth(1.0)
        axis.pane.set_alpha(1.0)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    coeffs_a1, coeffs_a2, meta = load_mf_params(RFIM_PARAMS_PATH)

    def Mw_to_a1(Mw: np.ndarray) -> np.ndarray:
        return np.polyval(coeffs_a1, Mw)

    def sigma_to_a2(sigma: np.ndarray) -> np.ndarray:
        return np.polyval(coeffs_a2, sigma)

    m_vals = np.linspace(M_MINMAX[0], M_MINMAX[1], RESOLUTION)

    if SELECT == "Mw":
        param_vals = np.linspace(MW_MINMAX[0], MW_MINMAX[1], RESOLUTION)
        sigma_fixed = float(SIGMA_FIXED)

        PARAM, m_grid = np.meshgrid(param_vals, m_vals, indexing="xy")
        a1_grid = Mw_to_a1(PARAM)
        a2_grid = sigma_to_a2(sigma_fixed) * np.ones_like(PARAM)

        xlabel = r"Earthquake magnitude $M_w$"
        xticks = [4, 5, 6, 7, 8]
        xticklabels = [str(x) for x in xticks]
        title = rf"Landau free energy (sigma={sigma_fixed:.2f})"
        view_elev, view_azim = 18, -50

    elif SELECT == "sigma":
        param_vals = np.linspace(SIGMA_MINMAX[0], SIGMA_MINMAX[1], RESOLUTION)
        mw_fixed = float(MW_FIXED)

        PARAM, m_grid = np.meshgrid(param_vals, m_vals, indexing="xy")
        a1_grid = Mw_to_a1(mw_fixed) * np.ones_like(PARAM)
        a2_grid = sigma_to_a2(PARAM)

        xlabel = r"Structural diversity $\sigma$"
        xticks = [0.0, 0.5, 1.0]
        xticklabels = ["0.0", "0.5", "1.0"]
        title = rf"Landau free energy (Mw={mw_fixed:.2f})"
        view_elev, view_azim = 18, -50

    else:
        raise ValueError(f"Unknown SELECT={SELECT!r}. Use 'Mw' or 'sigma'.")

    Z = landau_free_energy(m_grid, a1=a1_grid, a2=a2_grid)
    Z = normalize_columns(Z)

    # 3D plot
    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")

    norm = colors.PowerNorm(gamma=0.3, vmin=float(Z.min()), vmax=float(Z.max()))
    surf = ax.plot_surface(
        PARAM,
        m_grid,
        Z,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        cmap="RdYlBu",
        norm=norm,
        alpha=0.95,
        shade=False,
    )

    # Floor contour (z offset)
    zmin, zmax = float(Z.min()), float(Z.max())
    zrange = zmax - zmin
    margin = 0.50 * zrange
    z0 = zmin - margin

    ax.contourf(PARAM, m_grid, Z, zdir="z", offset=z0, levels=CONTOUR_LEVELS, cmap="RdYlBu", alpha=0.3, norm=norm)

    # Limits
    ax.set_xlim([float(PARAM.min()), float(PARAM.max())])
    ax.set_ylim([float(m_grid.min()), float(m_grid.max())])
    ax.set_zlim(zmin - margin, zmax + 0.3 * margin)

    # Labels
    ax.set_xlabel(xlabel, labelpad=5, fontsize=14)
    ax.set_ylabel("Damage fraction", labelpad=5, fontsize=14)
    ax.text2D(0.02, 0.61, "Free energy", transform=ax.transAxes, rotation=90, va="top", ha="left", fontsize=14)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    m_ticks = [-1.0, 0.0, 1.0]
    ax.set_yticks(m_ticks)
    ax.set_yticklabels(["0.0", "0.5", "1.0"])  # damage fraction = (m+1)/2

    ax.set_zticks([])
    ax.tick_params(axis="both", which="major", pad=0)

    style_3d_axes(ax)
    add_floor_grid(
        ax,
        x0=float(ax.get_xlim()[0]),
        x1=float(ax.get_xlim()[1]),
        y0=float(ax.get_ylim()[0]),
        y1=float(ax.get_ylim()[1]),
        z0=float(z0),
    )

    ax.set_box_aspect((1, 1, 0.4), zoom=0.9)
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)

    # ax.set_title(title, fontsize=14)
    fig.tight_layout()

    if SaveFig:
        OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_FIG, dpi=300, transparent=True, bbox_inches="tight")
        print(f"Saved figure: {OUT_FIG}")

    plt.show()


if __name__ == "__main__":
    main()