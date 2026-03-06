#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 09/26/2025
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

Plotting utilities for regional simulations:
- Heatmap: (Mw × damage-fraction) or (sigma × damage-fraction) density
- Phase diagram: summary statistic + phase-coexistence overlay
- Engineering comparison plots
- Histogram: DV distribution at a fixed Mw or sigma (optional)
- Fragility curves (optional)

"""


from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.stats import lognorm


# =============================================================================
# Styling / palettes
# =============================================================================
CP_HEX = [
    "#4056A1", "#F12815", "#D79922", "#14A098",
    "#CB2D6F", "#501F3A", "#116466", "#F76C6C",
    "#EFE2BA", "#C5CBE3", "#8C9AC7", "#0072B5",
]

CP_STRUCT_TYPES = {
    "C1": "#6A8CAF",
    "C2": "#4E79A7",
    "C3": "#2E4A62",
    "RM1": "#E15759",
    "RM2": "#FF9D9A",
    "S1": "#F28E2B",
    "S2": "#EDC948",
    "URM": "#9C755F",
    "W1": "#59A14F",
    "W2": "#8CD17D",
}

# =============================================================================
# Types
# =============================================================================
StructureType   = Literal["SingleStory", "TwoStory", "MultiStory", "All"]
ModeType        = Literal["1st", "2nd"]  # 1st: Mw sweep, 2nd: sigma sweep
EngVariant      = Literal["base", "eng"]

# =============================================================================
# User config
# =============================================================================
FIGURE_TYPE: Literal["heatmap", "histogram", "phase_diagram", "fragility", "eng"] = "phase_diagram"
SAVE_FIG = False

# Directories (override via env vars if you want)
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
FIG_DIR = Path(
    os.environ.get(
        "FIG_DIR",
        r"./results",
    )
)


# =============================================================================
# Label helpers
# =============================================================================
def mw_label(mw: float) -> str:
    return f"{int(round(float(mw) * 100)):03d}"


def sigma_label(sigma: float, *, scale: int = 100, width: int = 3) -> str:
    # sigma=0.01 -> "001" when scale=100, width=3
    return f"{int(round(float(sigma) * scale)):0{width}d}"


# =============================================================================
# I/O helpers
# =============================================================================
def dv_path(
    *,
    target_structure: StructureType,
    target_region: str,
    cost: bool,
    mw: float,
    sigma: float,
    gmm: Optional[str] = None,
) -> Path:
    """
    Default file pattern:
    """
    kind = "repair_cost" if cost else "damage_fraction"
    fname = f"{target_region}_{target_structure}_{kind}_Mw{mw_label(mw)}_sigma{sigma_label(sigma)}"
    if gmm:
        fname += f"_{gmm}"
    fname += ".npy"
    return DATA_DIR / "damage_simulation_results" / target_region / kind / fname


def eng_dv_path(
    *,
    target_region: str,
    target_structure: StructureType,
    cost: bool,
    mw: float,
    sigma: float,
    category: Optional[str],
    corr_eng: str,
    variant: EngVariant,
) -> Path:
    """
    Engineering-practice DV file path helper.
    """
    kind = "repair_cost" if cost else "damage_fraction"
    subdir = Path("eng")

    mw_lab = mw_label(mw)
    sig_lab = sigma_label(sigma, scale=100, width=3)  # matches your eng filenames

    if category is not None:
        subdir = subdir / "categorization" / kind
        tag = "No" if variant == "base" else str(category)
        suffix = f"_cate{tag}"
    else:
        subdir = subdir / "corr" / kind
        tag = "Dpdn" if variant == "base" else str(corr_eng)
        suffix = f"_corr{tag}"

    fname = f"{target_region}_{target_structure}_{kind}_Mw{mw_lab}_sigma{sig_lab}{suffix}.npy"
    return DATA_DIR / "damage_simulation_results" / target_region / kind / fname


def load_region_info(target_region: str) -> pd.DataFrame:
    p = DATA_DIR / "building_inventories" / f"RegionalInventory_{target_region}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing region_info CSV: {p}")
    df = pd.read_csv(p)

    required = {"NumberOfStories", "RepairCost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"region_info is missing required columns: {sorted(missing)}")

    return df


def load_capacity_matrix(target_region: str, sigma: float) -> np.ndarray:
    """
    Capacity file name varies across projects. This preserves your original pattern:
      data/IDA_results_OpenSeesPy_sigma{XXX}_{target_region}.csv
    where XXX used 2-decimal labeling in the old script.

    If your capacity files use a different naming convention, change CAPA_SCALE/CAPA_WIDTH.
    """
    CAPA_SCALE, CAPA_WIDTH = 100, 3  # original script used *100 -> 3 digits
    lab = sigma_label(sigma, scale=CAPA_SCALE, width=CAPA_WIDTH)
    p = DATA_DIR / "IDA_results" / target_region / f"IDA_results_sigma{lab}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing capacity CSV: {p}")

    # First column is an index; remaining columns are samples per GM (transpose to (n_bldgs, n_gms))
    return pd.read_csv(p).to_numpy()[:, 1:].T


def filter_structure(
    df: pd.DataFrame,
    cap: Optional[np.ndarray],
    structure: StructureType,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    if structure == "All":
        return df.reset_index(drop=True), cap

    ns = df["NumberOfStories"].to_numpy()
    if structure == "SingleStory":
        mask = ns == 1
    elif structure == "TwoStory":
        mask = ns == 2
    else:  # MultiStory
        mask = ns > 1

    df2 = df.loc[mask].reset_index(drop=True)
    cap2 = cap[mask, :] if cap is not None else None
    return df2, cap2


# =============================================================================
# Core compute
# =============================================================================
def choose_cutoff(sc: np.ndarray, method: str = "p99.5") -> float:
    sc = np.asarray(sc, dtype=float)
    sc_pos = sc[sc > 0]
    if sc_pos.size == 0:
        return 1.0

    if method.startswith("p"):
        p = float(method[1:])
        return float(np.percentile(sc_pos, p))

    if method == "iqr":
        q1, q3 = np.percentile(sc_pos, [25, 75])
        iqr = q3 - q1
        return float(q3 + 1.5 * iqr)

    if method == "permag":
        per_mag = np.percentile(sc, 99.0, axis=0)
        return float(np.median(per_mag))

    raise ValueError(f"Unknown cutoff method: {method!r}")


def compute_histogram(
    tuning_params: np.ndarray,
    dv: np.ndarray,
    *,
    step_fraction: float = 1 / 200,
    smooth_sigma: float = 1.0,
    cutoff: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smoothed 2D histogram over (tuning_param × fraction) with max-norm to [0,1].
    dv shape must be (len(tuning_params), num_sim).
    """
    tuning_params = np.asarray(tuning_params, dtype=float).ravel()
    dv = np.asarray(dv, dtype=float)

    if dv.ndim != 2:
        raise ValueError(f"`dv` must be 2D (num_tune, num_sim). Got shape {dv.shape}.")
    if dv.shape[0] != tuning_params.size:
        raise ValueError(f"dv rows ({dv.shape[0]}) must match tuning_params ({tuning_params.size}).")

    # Parameter bins
    diffs = np.diff(tuning_params)
    step_param = float(np.round(np.min(diffs[diffs > 0]), 6)) if diffs.size else 1.0
    grid_param = np.r_[tuning_params, tuning_params[-1] + step_param]

    # Fraction bins (assumes dv is a fraction in [0,1])
    grid_fraction = np.arange(0.0, 1.0 + step_fraction / 2, step_fraction)

    # Bin centers for plotting
    mesh_param, mesh_fraction = np.meshgrid(
        (grid_param[:-1] + grid_param[1:]) / 2.0,
        (grid_fraction[:-1] + grid_fraction[1:]) / 2.0,
        indexing="xy",
    )

    # 2D histogram over all samples
    x = np.repeat(tuning_params, dv.shape[1])
    y = dv.reshape(-1)
    scen_count, _, _ = np.histogram2d(x, y, bins=[grid_param, grid_fraction])
    scen_count = scen_count.T  # (fraction_bins, param_bins)

    # Smooth
    scen_sm = gaussian_filter(scen_count, sigma=smooth_sigma) if smooth_sigma and smooth_sigma > 0 else scen_count

    # Cutoff and normalize
    if isinstance(cutoff, str) and cutoff.startswith("p"):
        cut = choose_cutoff(scen_sm, method=cutoff)
    elif isinstance(cutoff, str):
        cut = float(cutoff)
    elif cutoff is None:
        cut = float(np.max(scen_sm))
    else:
        cut = choose_cutoff(scen_sm, method="p99")

    if cut <= 0:
        scen_norm = np.zeros_like(scen_sm, dtype=float)
    else:
        scen_clip = np.clip(scen_sm, 0, cut) / cut
        mn, mx = float(scen_clip.min()), float(scen_clip.max())
        scen_norm = (scen_clip - mn) / (mx - mn) if mx > mn else np.zeros_like(scen_clip)

    return mesh_param, mesh_fraction, scen_norm


# =============================================================================
# Plotting
# =============================================================================
def _cmap_from_cp_hex() -> LinearSegmentedColormap:
    key_colors = [CP_HEX[0], CP_HEX[8], CP_HEX[1]]
    positions = [0.0, 0.5, 1.0]
    return LinearSegmentedColormap.from_list("cp_linear", list(zip(positions, key_colors)), N=256)


def plot_heatmap(
    mesh_param: np.ndarray,
    mesh_fraction: np.ndarray,
    scen_norm: np.ndarray,
    *,
    mode: ModeType,
    title: Optional[str] = None,
    cmap: str = "RdYlBu_r",
    figsize: Tuple[float, float] = (9, 6),
    dpi: int = 150,
    savefig: bool = False,
    out_name: Optional[str] = None,
) -> None:
    cmap_obj = _cmap_from_cp_hex() if cmap == "CP_HEX" else plt.get_cmap(cmap)

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    c = ax.contourf(mesh_param, mesh_fraction, scen_norm, levels=100, cmap=cmap_obj)

    cb = plt.colorbar(c, ax=ax)
    cb.set_label("Occurrence frequency", fontsize=18)
    cb.set_ticks([])
    cb.ax.text(0.5, -0.01, "Low", ha="center", va="top", fontsize=14, transform=cb.ax.transAxes)
    cb.ax.text(0.5, 1.01, "High", ha="center", va="bottom", fontsize=14, transform=cb.ax.transAxes)

    # X axis
    if mode == "1st":
        ax.set_xlabel(r"Earthquake magnitude $M_{w}$", fontsize=18)
        ax.set_xticks([4, 5, 6, 7, 8])
        ax.set_xticklabels([f"{x:.1f}" for x in [4, 5, 6, 7, 8]])
    else:
        ax.set_xlabel(r"Structural diversity $\sigma$", fontsize=18)
        xt = [float(mesh_param.min()), float((mesh_param.min() + mesh_param.max()) / 2), float(mesh_param.max())]
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{x:.1f}" for x in xt])

    # Y axis
    yt = np.linspace(float(mesh_fraction.min()), float(mesh_fraction.max()), 3)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{y:.1f}" for y in yt])
    ax.set_ylabel("Damage fraction", fontsize=18)

    ax.set_ylim([float(mesh_fraction.min()), float(mesh_fraction.max())])
    ax.tick_params(axis="both", labelsize=16)

    if title:
        ax.set_title(title, fontsize=20)

    fig.tight_layout()

    if savefig:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fname = out_name or "damage_fraction_heatmap.png"
        out_path = FIG_DIR / fname
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=True)
        print(f"Saved: {out_path}")

    plt.show()


def plot_histogram(
    dv_sel: np.ndarray,
    *,
    cost_normalized: bool,
    title: Optional[str] = None,
    cutoff: str = "p97",
    color_max: Optional[float] = None,
    num_bins: int | str = 100,
    cmap: str = "RdYlBu_r",
    savefig: bool = False,
    out_name: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    dv_sel = np.asarray(dv_sel, dtype=float)
    dv_sel = dv_sel[np.isfinite(dv_sel)]
    if dv_sel.size == 0:
        raise ValueError("No finite DV samples to plot.")

    fig, ax = plt.subplots(dpi=300, figsize=(6, 4))
    bins = num_bins if isinstance(num_bins, int) else "fd"
    hist_range = (0.0, 1.0) if cost_normalized else None

    counts, bin_edges, bar_patches = ax.hist(dv_sel, bins=bins, range=hist_range, alpha=0.95, edgecolor="black")

    # Cutoff for colors
    cut = choose_cutoff(counts, method=cutoff) if cutoff.startswith("p") else float(cutoff)
    counts_color = np.clip(counts, 0, cut) if cut > 0 else np.zeros_like(counts)

    cmap_obj = _cmap_from_cp_hex() if cmap == "CP_HEX" else plt.get_cmap(cmap)
    max_count = float(color_max) if color_max is not None else float(np.nanmax(counts_color)) if counts_color.size else 0.0
    rel = counts_color / max_count if max_count > 0 else np.zeros_like(counts_color)

    for r, patch in zip(rel, bar_patches):
        patch.set_facecolor(cmap_obj(float(r)))

    ax.set_xlim([-0.03, 1.03])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="major", labelsize=32)

    if title:
        ax.set_title(title, fontsize=24)

    fig.tight_layout()

    if savefig:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fname = out_name or "damage_fraction_histogram.png"
        out_path = FIG_DIR / fname
        fig.savefig(out_path, dpi=300, transparent=False)
        print(f"Saved: {out_path}")

    return fig, ax


# =============================================================================
# Analysis drivers
# =============================================================================
def main_heatmap(
    *,
    mode: ModeType = "1st",
    target_region: str = "Milpitas",
    target_structure: StructureType = "MultiStory",
    mw_fixed: float = 5.6,           # used when mode="2nd"
    sigma_fixed: float = 0.0,        # used when mode="1st"
    num_sim: int = 10_000,
    smooth_sigma: float = 1.0,
    cutoff: str = "p97",
    step_fraction: float = 1 / 200,
    cmap: str = "RdYlBu_r",
    title: Optional[str] = None,
    savefig: bool = False,
) -> None:
    mws = np.round(np.arange(3.50, 8.51, 0.05), 2)
    sigmas = np.round(np.arange(0.0, 1.001, 0.01), 3)

    if mode == "1st":
        dv = np.empty((mws.size, num_sim), dtype=float)
        for i, mw in enumerate(mws):
            p = dv_path(
                target_structure=target_structure,
                target_region=target_region,
                cost=False,
                mw=float(mw),
                sigma=float(sigma_fixed),
            )
            if not p.exists():
                raise FileNotFoundError(f"Missing DV file: {p}")
            dv[i, :] = np.load(p)
        tuning_params = mws
    else:
        dv = np.empty((sigmas.size, num_sim), dtype=float)
        for i, s in enumerate(sigmas):
            p = dv_path(
                target_structure=target_structure,
                target_region=target_region,
                cost=False,
                mw=float(mw_fixed),
                sigma=float(s),
            )
            if not p.exists():
                raise FileNotFoundError(f"Missing DV file: {p}")
            dv[i, :] = np.load(p)
        tuning_params = sigmas

    mesh_param, mesh_fraction, scen_norm = compute_histogram(
        tuning_params=tuning_params,
        dv=dv,
        step_fraction=step_fraction,
        smooth_sigma=smooth_sigma,
        cutoff=cutoff,
    )

    mode_tag = "1st" if mode == "1st" else "2nd"
    out_name = f"{target_region}_{target_structure}_{mode_tag}_dv_heatmap.png"

    plot_heatmap(
        mesh_param,
        mesh_fraction,
        scen_norm,
        mode=mode,
        title=title,
        cmap=cmap,
        figsize=(9, 6),
        savefig=savefig,
        out_name=out_name,
    )


def main_histogram(
    *,
    mode: ModeType = "1st",
    target_region: str = "Milpitas",
    target_structure: StructureType = "MultiStory",
    cost: bool = False,
    mw: float = 5.6,
    sigma: float = 0.0,
    cutoff: str = "p97",
    color_max: Optional[float] = None,
    num_bins: int | str = 100,
    cmap: str = "RdYlBu_r",
    title: Optional[str] = None,
    savefig: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    p = dv_path(
        target_structure=target_structure,
        target_region=target_region,
        cost=cost,
        mw=float(mw),
        sigma=float(sigma),
    )
    if not p.exists():
        raise FileNotFoundError(f"Missing DV file: {p}")

    dv = np.load(p)
    dv_sel = dv.reshape(-1)  # allow either (num_sim,) or (num_tune, num_sim)

    cost_total = None
    if cost:
        region_info = load_region_info(target_region)
        cap = load_capacity_matrix(target_region, sigma=0.0)  # usually baseline; change if needed

        # Align by dropping NaN capacity rows (only for normalization)
        keep = ~np.any(np.isnan(cap), axis=1)
        region_info = region_info.loc[keep].reset_index(drop=True)
        region_info, _ = filter_structure(region_info, None, target_structure)

        cost_total = float(region_info["RepairCost"].sum())
        if cost_total <= 0:
            raise ValueError("Total repair cost is non-positive after filtering.")

        dv_sel = dv_sel / cost_total

    cost_normalized = bool(cost)
    out_name = f"{target_region}_{target_structure}_sigma{sigma_label(sigma)}_Mw{mw:.2f}_histogram.png"

    return plot_histogram(
        dv_sel,
        cost_normalized=cost_normalized,
        title=title,
        cutoff=cutoff,
        color_max=color_max,
        num_bins=num_bins,
        cmap=cmap,
        savefig=savefig,
        out_name=out_name,
    )


def phase_coexistence_detect(
    dv_temp: np.ndarray,
    *,
    bins: int = 200,
    win: int = 9,
    prominence_frac: float = 0.03,
    tol: float = 0.10,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> int:
    """Return 1 if histogram has peaks near both edges (0 and 1), else 0."""
    x = np.asarray(dv_temp, dtype=float)
    x = x[np.isfinite(x)]

    xmin, xmax = value_range
    x = x[(x >= xmin) & (x <= xmax)]
    if x.size == 0:
        return 0

    counts, edges = np.histogram(x, bins=bins, range=value_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    win = win + 1 if win % 2 == 0 else win
    kernel = np.ones(win) / win
    smooth = np.convolve(counts, kernel, mode="same")

    peaks, _ = find_peaks(smooth, prominence=prominence_frac * smooth.max(), distance=bins // 10)
    if peaks.size == 0:
        return 0

    peak_locs = centers[peaks]
    has_left = np.any(peak_locs <= xmin + tol)
    has_right = np.any(peak_locs >= xmax - tol)
    return int(has_left and has_right)


def main_phase_diagram(
    *,
    target_region: str = "Milpitas",
    target_structure: StructureType = "MultiStory",
    num_bins: int = 100,
    cmap: str = "RdYlBu_r",
    savefig: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    cmap_obj = _cmap_from_cp_hex() if cmap == "CP_HEX" else plt.get_cmap(cmap)

    mws = np.round(np.arange(3.5, 8.51, 0.05), 2)
    sigmas = np.round(np.arange(0.0, 1.001, 0.01), 3)
    num_sim = 10_000

    top_values_mean = np.zeros((sigmas.size, mws.size), dtype=float)
    phase_coexist = np.zeros((sigmas.size, mws.size), dtype=int)

    for si, s in enumerate(sigmas):
        for mi, mw in enumerate(mws):
            p = dv_path(
                target_structure=target_structure,
                target_region=target_region,
                cost=False,
                mw=float(mw),
                sigma=float(s),
            )
            if not p.exists():
                raise FileNotFoundError(f"Missing DV file: {p}")
            dv_temp = np.load(p)

            counts, edges = np.histogram(dv_temp, bins=num_bins)
            sorted_bins = np.argsort(counts)[::-1]

            # Take top ~1% mass bins and average samples inside them
            count = 0
            selected = []
            for b in sorted_bins:
                if count > 0.01 * num_sim:
                    break
                selected.append(b)
                count += counts[b]

            selected = np.asarray(selected, dtype=int)
            bmin = edges[selected]
            bmax = edges[selected + 1]

            mask = np.zeros_like(dv_temp, dtype=bool)
            for lo, hi in zip(bmin, bmax):
                mask |= (dv_temp >= lo) & (dv_temp < hi)

            top_values_mean[si, mi] = float(np.mean(dv_temp[mask])) if np.any(mask) else float(np.mean(dv_temp))
            phase_coexist[si, mi] = phase_coexistence_detect(dv_temp)

    SIG, MAG = np.meshgrid(sigmas, mws)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(0.5, 5.55, "o", ms=10, markerfacecolor="black", markeredgecolor="white", zorder=5)

    c = ax.contourf(SIG, MAG, top_values_mean.T, levels=100, cmap=cmap_obj, vmin=0.0, vmax=1.0, zorder=1)

    # Overlay phase coexistence region
    phase_mask = (phase_coexist == 1).astype(int)
    ax.contourf(SIG, MAG, phase_mask.T, levels=[0.5, 1.5], colors=["black"], alpha=0.5, zorder=3)

    cb = plt.colorbar(c, ax=ax)
    cb.set_label("Damage fraction", fontsize=20)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=16)

    ax.set_xlabel(r"Structural diversity $\sigma$", fontsize=20)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r"Earthquake magnitude $M_w$", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)

    fig.tight_layout()

    if savefig:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIG_DIR / f"{target_region}_{target_structure}_phase_diagram.png"
        fig.savefig(out_path, dpi=300, transparent=True)
        print(f"Saved: {out_path}")

    plt.show()
    return fig, ax


def main_eng(
    *,
    target_region: str = "Milpitas_all",
    target_structure: StructureType = "MultiStory",
    category: Optional[str] = None,   # e.g., "StructureType" (categorization case); None -> correlation case
    corr: str = "Indp",               # used only when category is None (engineering corr tag)
    cost: bool = True,
    legend: bool = True,
    savefig: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare baseline vs engineering-practice DV distributions using split violin plots.

    Baseline vs Eng:
      - If category is not None:
          baseline = cateNo, eng = cate{category}
      - If category is None:
          baseline = corrDpdn (dependent), eng = corr{corr} (e.g., Indp)

    If cost=True, values are assumed to be repair costs in dollars and plotted in million $.
    """
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    # Colors (kept from your original logic)
    if target_region == "SanFrancisco_NE":
        cL, cR = CP_HEX[3], CP_HEX[2]
        mw_vals = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5], dtype=float)
    else:
        cL, cR = CP_HEX[3], CP_HEX[4]
        mw_vals = np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0], dtype=float)

    # Mw grid (must match the simulation grid used to save dv files)
    mws = np.round(np.arange(3.5, 8.51, 0.05), 2)
    mw_idxs = [int(np.argmin(np.abs(mws - t))) for t in mw_vals]
    print("Exact Mws used for violins:", mws[mw_idxs])

    # For the eng outputs you used sigma=0.0 (and *100 labeling)
    sigma0 = 0.0

    # Load dv arrays across all Mw (baseline + engineering)
    dv_base = []
    dv_eng = []
    for mw in mws:
        p_base = eng_dv_path(
            target_region=target_region,
            target_structure=target_structure,
            cost=cost,
            mw=float(mw),
            sigma=sigma0,
            category=category,
            corr_eng=corr,
            variant="base",
        )
        p_eng = eng_dv_path(
            target_region=target_region,
            target_structure=target_structure,
            cost=cost,
            mw=float(mw),
            sigma=sigma0,
            category=category,
            corr_eng=corr,
            variant="eng",
        )
        if not p_base.exists():
            raise FileNotFoundError(f"Missing baseline DV file: {p_base}")
        if not p_eng.exists():
            raise FileNotFoundError(f"Missing engineering DV file: {p_eng}")

        dv_base.append(np.load(p_base))
        dv_eng.append(np.load(p_eng))

    dv_base = np.asarray(dv_base, dtype=float)  # (n_mw, n_sim)
    dv_eng = np.asarray(dv_eng, dtype=float)

    # Optional: scale cost for “Slight” vs “Moderate” mismatch (kept as your original behavior)
    if cost:
        dv_base = dv_base / 5.0
        dv_eng = dv_eng / 5.0

    def clean_values(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if cost:
            return x / 1e6  # million $
        return x

    data_left = [clean_values(dv_base[i, :]) for i in mw_idxs]
    data_right = [clean_values(dv_eng[i, :]) for i in mw_idxs]

    # --- Split violin helper ---
    def split_violin(ax, left, right, x, *, width=0.8, alpha=0.5, bw=None, points=200):
        vL = ax.violinplot([left], positions=[x], widths=width, bw_method=bw,
                           points=points, showmeans=False, showmedians=False, showextrema=False)
        for pc in vL["bodies"]:
            verts = pc.get_paths()[0].vertices
            m = np.mean(verts[:, 0])
            verts[:, 0] = np.minimum(verts[:, 0], m)
            pc.set_alpha(None)
            pc.set_facecolor(mcolors.to_rgba(cL, alpha))
            pc.set_edgecolor((0, 0, 0, 1))
            pc.set_linewidth(1.0)
            pc.set_zorder(3)

        vR = ax.violinplot([right], positions=[x], widths=width, bw_method=bw,
                           points=points, showmeans=False, showmedians=False, showextrema=False)
        for pc in vR["bodies"]:
            verts = pc.get_paths()[0].vertices
            m = np.mean(verts[:, 0])
            verts[:, 0] = np.maximum(verts[:, 0], m)
            pc.set_alpha(None)
            pc.set_facecolor(mcolors.to_rgba(cR, alpha))
            pc.set_edgecolor((0, 0, 0, 1))
            pc.set_linewidth(1.0)
            pc.set_zorder(3)

    # Less smoothing than Scott’s rule (kept from your original intent)
    bw_less = lambda kde: kde.scotts_factor() * 0.4

    fig, ax = plt.subplots(figsize=(8, 6))
    positions = np.arange(1, len(mw_vals) + 1, dtype=float)

    for x, L, R in zip(positions, data_left, data_right):
        split_violin(ax, L, R, x, width=0.8, alpha=0.5, bw=bw_less, points=200)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{t:.1f}" for t in mw_vals])
    ax.tick_params(axis="both", labelsize=17)
    ax.set_xlabel(r"Earthquake magnitude $M_w$", fontsize=20)

    if cost:
        ax.set_ylabel("Repair cost (million $)", fontsize=20)
    else:
        ax.set_ylabel("Damage fraction", fontsize=20)

    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    if legend:
        if category is not None:
            lab_left = "Baseline (no categorization)"
            lab_right = f"Categorized ({category})"
        else:
            lab_left = "Baseline (dependent)"
            lab_right = f"Engineering ({corr})"

        handles = [
            mpatches.Patch(facecolor=mcolors.to_rgba(cL, 0.5), edgecolor="black", linewidth=1.0, label=lab_left),
            mpatches.Patch(facecolor=mcolors.to_rgba(cR, 0.5), edgecolor="black", linewidth=1.0, label=lab_right),
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.1),
                  ncol=2, frameon=False, fontsize=16, columnspacing=0.8, handlelength=1.2)

        plt.subplots_adjust(top=0.88)

    fig.tight_layout()

    if savefig:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"cate_{category}" if category is not None else f"corr_{corr}"
        out_path = FIG_DIR / f"{target_region}_{target_structure}_eng_{tag}_violin.png"
        fig.savefig(out_path, dpi=300, transparent=True)
        print(f"Saved: {out_path}")

    plt.show()
    return fig, ax

# =============================================================================
# (Optional) Fragility curves (kept minimal + cleaned)
# =============================================================================
def main_fragility_curves(
    *,
    target_region: str = "SanFrancisco_NE",
    target_structure: StructureType = "All",
    sigma: float = 0.0,
    category: Optional[str] = None,
    savefig: bool = False,
) -> None:
    # These paths are project-specific; keep as-is but centralized.
    base = Path(
        r"C:\Users\ohsb1\OneDrive\UCB\Research\Phase transitions in regional seismic responses\Codes\Benchmark"
    )

    capa_csv = base / "Savio" / target_region / "IDA_results_OpenSeesPy_sigma000.csv"
    region_csv = base / f"TargetRegion_{target_region}.csv"

    bldg_capa = pd.read_csv(capa_csv).to_numpy()[:, 1:].T
    idx_nan = np.where(np.any(np.isnan(bldg_capa), axis=1))[0]
    if idx_nan.size > 0:
        keep = np.ones(bldg_capa.shape[0], dtype=bool)
        keep[idx_nan] = False
        bldg_capa = bldg_capa[keep]

    region_info = pd.read_csv(region_csv)
    if idx_nan.size > 0:
        region_info = region_info.drop(idx_nan).reset_index(drop=True)

    region_info, bldg_capa = filter_structure(region_info, bldg_capa, target_structure)

    lab = sigma_label(sigma)
    frag_path = DATA_DIR / "fragility_params" / target_region / "frag_params_sigma{lab}.npy"
    frag_params = np.load(frag_path)
    frag_params = frag_params[: bldg_capa.shape[0], :]  # safe alignment if needed

    x = np.linspace(1e-5, 0.6, 1000) if target_region == "Milpitas" else np.linspace(1e-5, 1.0, 1000)
    fig, ax = plt.subplots(figsize=(6, 6))

    if category is None:
        for i in range(frag_params.shape[0]):
            cdf = lognorm.cdf(x, s=frag_params[i, 0], scale=frag_params[i, 2])
            ax.plot(x, cdf, "-", alpha=0.1, linewidth=0.1, color="black")
    else:
        col = region_info[category].astype("string").str.strip()
        for name in sorted(col.dropna().unique()):
            mask = (col == name).to_numpy()
            vals = bldg_capa[mask, :].ravel()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size == 0:
                continue

            s, loc, scale = lognorm.fit(vals, floc=0)
            y_fit = lognorm.cdf(x, s, loc=loc, scale=scale)
            ax.plot(x, y_fit, "-", linewidth=2.0, color=CP_STRUCT_TYPES.get(str(name), "black"), label=str(name))

        ax.legend(fontsize=16, loc="lower right", ncol=2, columnspacing=0.5, frameon=False)

    ax.set_xlabel(r"PGA ($g$)", fontsize=20)
    ax.set_ylabel("Damage probability", fontsize=20)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim([0, x.max()])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    if savefig:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIG_DIR / f"{target_region}_{target_structure}_sigma{lab}_fragility_curves_cate{category}.png"
        fig.savefig(out_path, dpi=300, transparent=True)
        print(f"Saved: {out_path}")

    plt.show()


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    if FIGURE_TYPE == "heatmap":
        main_heatmap(
            mode="1st", # 1st/2nd: Mw/sigma sweep
            target_region="Milpitas",
            target_structure="MultiStory",
            mw_fixed=5.6,
            sigma_fixed=0.0,
            smooth_sigma=0.0,     # Gaussian smoothing; set 0 to disable
            cutoff="p95",        # Recommended: "p95" for the "1st" mode, "p99" for the "2nd" mode
            step_fraction=1 / 1000,
            cmap="RdYlBu_r",
            title=None,
            savefig=SAVE_FIG,
        )

    elif FIGURE_TYPE == "histogram":
        main_histogram(
            mode="2nd",
            target_region="Milpitas",
            target_structure="MultiStory",
            cost=False,
            mw=5.6,
            sigma=0.3,
            cutoff="p95",
            color_max=None,
            num_bins=100,
            cmap="RdYlBu_r",
            title=None,
            savefig=SAVE_FIG,
        )

    elif FIGURE_TYPE == "phase_diagram":
        main_phase_diagram(
            target_region="Milpitas",
            target_structure="MultiStory",
            num_bins=100,
            cmap="RdYlBu_r",
            savefig=SAVE_FIG,
        )

    elif FIGURE_TYPE == "fragility":
        main_fragility_curves(
            target_region="SanFrancisco_NE",
            target_structure="All",
            sigma=0.1,
            category=None,  # e.g., "StructureType"
            savefig=SAVE_FIG,
        )

    elif FIGURE_TYPE == "eng":
        main_eng(
            target_region="Milpitas",
            target_structure="MultiStory",
            category=None,

            # target_region="SanFrancisco_NE",
            # target_structure="All",
            # category="StructureType",

            corr="Indp",
            cost=True,
            legend=True,
            savefig=SAVE_FIG,
        )

    else:
        raise ValueError(f"Unknown FIGURE_TYPE: {FIGURE_TYPE!r}")