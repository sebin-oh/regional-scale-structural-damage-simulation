#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Created     :   09/26/2025
@Author      :   Sebin Oh
@Contact     :   sebin.oh@berkeley.edu
@Description :   

Plot (magnitude × fraction) density heatmap for regional simulations.

- Loads DV arrays (cost or failure-fraction) with either correlation or category tag.
- Cleans capacity table + aligns with region_info.
- Optionally filters by structure class (SingleStory / TwoStory / MultiStory).
- Normalizes cost by total repair cost when cost=True.
- Smooths 2D histogram with a Gaussian filter and max-norms to [0,1].

"""
# %%
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import lognorm
from pathlib import Path
from typing import Tuple, Optional, Literal
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns

CP = np.array([
    [ 64,  86, 161],
    [241,  40,  21],    
    [215, 153,  34],
    [ 20, 160, 152],
    [203,  45, 111],
    [ 80,  31,  58],
    [ 17, 100, 102],
    [247, 108, 108],
    [239, 226, 186],
    [197, 203, 227],
    [140, 154, 199],
    [  0, 114, 181],
]) / 255.0

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

# ---------- Matplotlib defaults ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "mathtext.fontset": "cm",
})

# ---------- Types ----------
StructureType = Literal["SingleStory", "TwoStory", "MultiStory"]
ModeType = Literal["1st", "2nd"]

FigureType = "heatmap" # heatmap / histogram / phase_diagram / fragility / eng
SaveFig = False

# %%
# ---------- I/O helpers ----------
def _dv_filename(
    target_structure: StructureType,
    target_region: str,
    cost: bool,
    Mw: Optional[float],
    cov: Optional[float],
) -> Tuple[Path, str]:
    """Build DV filename and a short English title."""
    base = f"C:/Users/ohsb1/Desktop/Savio/{target_region}/{target_structure}_dv"
    if cost:
        base += "_cost"

    Mw_label = f"{int(round(Mw*100)):03d}"
    cov_label = f"{int(round(cov*1000)):04d}"

    suffix = f"_Mw{Mw_label}_cov{cov_label}_{target_region}"

    # base = f"C:/Users/ohsb1/Desktop/Savio/{target_region}/temp_Mw/{target_structure}_dv"
    # Mw_label = f"{int(round(Mw*100)):03d}"
    # T_label = f"{int(round(T*100)):03d}"
    # suffix = f"_Mw{Mw_label}_T{T_label}_{target_region}"

    fname = f"{base}{suffix}.npy"
    title = None

    return Path(fname), title

def _load_region_info(target_region: str) -> pd.DataFrame:
    p = Path(f"data/TargetRegion_{target_region}.csv")
    if not p.exists():
        raise FileNotFoundError(f"Missing region_info CSV: {p}")
    df = pd.read_csv(p)
    if "NumberOfStories" not in df.columns or "RepairCost" not in df.columns:
        raise ValueError("`region_info` must include 'NumberOfStories' and 'RepairCost' columns.")
    return df


def _load_capacity_matrix(target_region: str, cov: float) -> np.ndarray:
    cov_label = f"{int(np.round(cov*100,2)):03d}"
    p = Path(f"data/IDA_results_OpenSeesPy_cov{cov_label}_{target_region}.csv")
    if not p.exists():
        raise FileNotFoundError(f"Missing capacity CSV: {p}")
    # First col is an index; the rest are capacity samples per building (rows → buildings after transpose).
    arr = pd.read_csv(p).to_numpy()[:, 1:].T  # shape: (num_bldgs, num_samples)
    return arr


def _filter_structure(df: pd.DataFrame, cap: np.ndarray, structure: StructureType) -> Tuple[pd.DataFrame, np.ndarray]:
    cond: np.ndarray
    if structure == "SingleStory":
        cond = df["NumberOfStories"].values == 1
    elif structure == "TwoStory":
        cond = df["NumberOfStories"].values == 2
    else:  # "MultiStory"
        cond = df["NumberOfStories"].values > 1

    df2 = df.loc[cond].reset_index(drop=True)
    cap2 = cap[cond, :]
    return df2, cap2

def figure_with_content_size(
    content_w_in=6.0, content_h_in=4.0,  # desired data-area (axes) size in inches
    left_in=1.2, right_in=0.3,           # margins in inches reserved for labels/ticks
    bottom_in=1.0, top_in=0.3,
    dpi=300,
):
    fig_w = content_w_in + left_in + right_in
    fig_h = content_h_in + bottom_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # Convert inch margins to subplot fractions
    fig.subplots_adjust(
        left   = left_in   / fig_w,
        right  = 1 - right_in / fig_w,
        bottom = bottom_in / fig_h,
        top    = 1 - top_in / fig_h,
    )

    ax = fig.add_subplot(111)
    return fig, ax

# ---------- Core compute ----------
def compute_histogram(
    tuning_params: np.ndarray,
    dv: np.ndarray,
    step_fraction: float = 1 / 200,
    sigma: float = 1.0,
    cutoff: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a smoothed 2D histogram (param × fraction) and max-norm to [0,1].

    Returns:
        mesh_param, mesh_fraction, scen_count_normed (shape: [len(frac_bins)-1, len(mag_bins)-1])
    """
    if dv.ndim != 2:
        raise ValueError(f"`dv` expected shape (num_tune, num_sim); got {dv.shape}")

    num_tune, num_sim = dv.shape
    if num_tune != len(tuning_params):
        raise ValueError(f"tuning_params length ({len(tuning_params)}) must match dv rows ({num_tune}).")

    # Parameter (magnitude) bins — right-closed extension by step size
    step_param = float(np.round(np.min(np.diff(tuning_params)), 3))
    grid_param = np.concatenate([tuning_params, [tuning_params[-1] + step_param]])

    # Fraction bins
    grid_fraction = np.arange(0.0, 1.0 + step_fraction / 2, step_fraction)

    # Mesh for plotting at bin centers
    mesh_param, mesh_fraction = np.meshgrid(
        (grid_param[:-1] + grid_param[1:]) / 2.0,
        (grid_fraction[:-1] + grid_fraction[1:]) / 2.0,
        indexing="xy",
    )

    # 2D histogram of all samples across magnitudes
    # x: repeated magnitudes, y: flattened fractions
    x = tuning_params.repeat(num_sim)
    y = dv.reshape(-1)

    scen_count, _, _ = np.histogram2d(x, y, bins=[grid_param, grid_fraction])
    scen_count = scen_count.T  # rows: fraction bins, cols: mag bins (match mesh axes)
    scen_count_filtered = gaussian_filter(scen_count, sigma=sigma) if sigma and sigma > 0 else scen_count.copy()

    # Cutoff + max-norm
    # if cutoff starts by "p", use choose_cutoff with that percentile
    if isinstance(cutoff, str) and cutoff.startswith("p"):
        cut = choose_cutoff(scen_count_filtered, method=cutoff)
    elif isinstance(cutoff, str):
        cut = float(cutoff)
    elif cutoff is None:
        cut = np.max(scen_count_filtered)
    else:
        cut = choose_cutoff(scen_count_filtered, method="p99")
    print(f"Cutoff for histogram counts: {cut:.3f}")
    # cut = np.max(scen_count_filtered) if (cutoff is None or cutoff is False) else float(cutoff)
    if cut <= 0:
        # Avoid division by zero; return zeros if nothing recorded
        scen_norm = np.zeros_like(scen_count_filtered, dtype=float)
    else:
        scen_clip = np.clip(scen_count_filtered, 0, cut) / cut
        # Stretch to [0,1] robustly (guard against flat arrays)
        mn, mx = float(scen_clip.min()), float(scen_clip.max())
        scen_norm = (scen_clip - mn) / (mx - mn) if mx > mn else np.zeros_like(scen_clip)

    return mesh_param, mesh_fraction, scen_norm

def choose_cutoff(sc, method="p99.5"):
    sc_pos = sc[sc > 0]
    if sc_pos.size == 0:
        return 1.0
    if method.startswith("p"):
        p = float(method[1:])
        return float(np.percentile(sc_pos, p))
    if method == "iqr":
        q1, q3 = np.percentile(sc_pos, [25, 75]); iqr = q3 - q1
        return float(q3 + 1.5 * iqr)
    if method == "permag":
        per_mag = np.percentile(sc, 99.0, axis=0)
        return float(np.median(per_mag))
    raise ValueError("Unknown method")

# ---------- Plot ----------
def plot_heatmap(
    mesh_param: np.ndarray,
    mesh_fraction: np.ndarray,
    scen_norm: np.ndarray,
    *,
    target_region: str,
    target_structure: StructureType,
    title: bool,
    mode: ModeType,
    cmap: str = "RdYlBu_r",
    savefig: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
) -> None:
    if cmap == "CP_HEX":
        key_colors = [CP_HEX[0], CP_HEX[8], CP_HEX[1]]
        positions  = [0.0, 0.5, 1.0]
        cmap = LinearSegmentedColormap.from_list("cp_linear", list(zip(positions, key_colors)), N=256)

        # key_colors = ["#FFFFFF", CP_HEX[8], CP_HEX[7], CP_HEX[1]]
        # cmap = LinearSegmentedColormap.from_list("custom_cmap", key_colors, N=256)

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    c = ax.contourf(mesh_param, mesh_fraction, scen_norm, levels=100, cmap=cmap)

    cbar = plt.colorbar(c, ax=ax)
    # cbar.set_label("# of simulations (max-normed)", fontsize=18)
    # cbar.set_ticks(np.linspace(0, 1, 11))
    cbar.set_label("Occurrence frequency", fontsize=18)

    # cbar.set_ticks([0.0, 1.0])
    # cbar.set_ticklabels(["Low","High"])

    cbar.set_ticks([])
    cbar.ax.text(0.5, -0.01, "Low", ha="center", va="top", fontsize=14, transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, 1.01, "High", ha="center", va="bottom", fontsize=14, transform=cbar.ax.transAxes)

    # X axis (Magnitude)
    # xtick_positions = np.linspace(mesh_param.min(), mesh_param.max(), 6)
    if mode == "1st":
        # xtick_positions = np.arange(mesh_param.min(), mesh_param.max() + 0.1, 0.5)
        xtick_positions = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([f"{x:.1f}" for x in xtick_positions])
    else:  # "2nd"
        # xtick_positions = np.arange(mesh_param.min(), mesh_param.max() + 0.1, 0.2)
        # ax.set_xticks(xtick_positions)
        # ax.set_xticklabels([f"{x:.1f}" for x in xtick_positions])

        # xtick_positions = np.array([mesh_param.min(), mesh_param.max()])
        # ax.set_xticks(xtick_positions)
        # ax.set_xticklabels(["Low", "High"])

        xtick_positions = np.array([mesh_param.min(), (mesh_param.min()+mesh_param.max())/2, mesh_param.max()])
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([f"{x:.1f}" for x in xtick_positions])
        
    # ax.set_xticks([])
    # ax.set_xlabel(r"Earthquake magnitude $M_{w}$" if mode == "1st" else r"Structural diversity $\sigma$", fontsize=18)
    ax.set_xlabel(r"Effective temperature $T$", fontsize=18)

    # Y axis (Cost or Damage fraction)
    ytick_positions = np.linspace(mesh_fraction.min(), mesh_fraction.max(), 3)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([f"{y:.1f}" for y in ytick_positions])
    ax.set_ylabel("Damage fraction", fontsize=18)

    ax.set_ylim([mesh_fraction.min(), mesh_fraction.max()])
    ax.tick_params(axis="both", labelsize=16)

    if title:
        ax.set_title(title, fontsize=20)
    plt.tight_layout()
    if savefig:
        outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
        outdir.mkdir(exist_ok=True)
        mode_tag = "1st" if mode == "1st" else "2nd"
        fname = f"{outdir}/{target_region}_{target_structure}_{mode_tag}_dv_heatmap.png"
        plt.savefig(fname, dpi=dpi, bbox_inches="tight", transparent=True)
        print(f"Saved heatmap: {fname}")
    plt.show()


# ---------- Main ----------
def main_heatmap(
    *,
    mode: ModeType = "1st",
    title: bool = False,
    target_region: str = "SanFrancisco_NE",
    target_structure: StructureType = "MultiStory",
    Mw: Optional[float] = 5.6,
    cov: Optional[float] = 0.0,
    corr: Optional[float] = 0.0,
    category: Optional[str] = None,
    sigma: float = 1.0,
    cutoff: Optional[str] = "p97",  # None or False ⇒ auto max
    step_fraction: float = 1 / 200,
    cmap: str = "RdYlBu_r",
    savefig: bool = False,
) -> None:
    
    # Tuning parameters (Magnitudes, C.O.Vs)
    Mws = np.round(np.arange(3.5, 8.51, 0.05), 2)
    covs = np.round(np.arange(0.0, 1.001, 0.01), 3)

    num_sim = 10_000

    # DV array load
    if mode == "1st":
        dv = np.zeros((len(Mws), num_sim))
        
        for Mw_idx, Mw_val in enumerate(Mws):
            dv_path, eng_title = _dv_filename(target_structure, target_region, None, Mw_val, cov)
            if not dv_path.exists():
                raise FileNotFoundError(f"Missing DV file: {dv_path}")
            dv[Mw_idx, :] = np.load(dv_path)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            # Mw_label = f"{int(round(Mw_val*100)):03d}"
            # cov_label = f"{int(round(cov*100)):03d}"
            # if isinstance(corr, str):
            #     filepath_base = f"C:/Users/ohsb1/Desktop/Savio/{target_region}/eng/corr/dv/{target_structure}_dv_Mw{Mw_label}_cov{cov_label}_{target_region}_corr{corr}.npy"
            # else:
            #     if category == None:
            #         category = "No"
            #     filepath_base = f"C:/Users/ohsb1/Desktop/Savio/{target_region}/eng/categorization/dv/{target_structure}_dv_Mw{Mw_label}_cov{cov_label}_{target_region}_cate{category}.npy"
            # dv[Mw_idx, :] = np.load(filepath_base)

        # # Optional: slice range
        # dv = dv[50:201, :]
        # Mws = Mws[50:201]
        # assert dv.shape[0] == len(Mws), "Rows of dv must equal len(Mws)"

        tuning_params = Mws
    elif mode == "2nd":
        dv = np.zeros((len(covs), num_sim))
        
        for cov_idx, cov_val in enumerate(covs):
            dv_path, eng_title = _dv_filename(target_structure, target_region, None, Mw, cov_val)
            if not dv_path.exists():
                raise FileNotFoundError(f"Missing DV file: {dv_path}")
            dv[cov_idx, :] = np.load(dv_path)

        tuning_params = covs
    else:
        print("Invalid mode selected")
        return None

    # Build histogram + plot
    mesh_param, mesh_fraction, scen_norm = compute_histogram(
        tuning_params=tuning_params, dv=dv, step_fraction=step_fraction, sigma=sigma, cutoff=cutoff
    )

    plot_heatmap(
        mesh_param,
        mesh_fraction,
        scen_norm,
        target_region=target_region,
        target_structure=target_structure,
        title=title,
        mode=mode,
        cmap=cmap,
        figsize=(9, 6),
        savefig=savefig,
    )

def main_histogram(
    *,
    mode: ModeType = "1st",
    title: bool = False,
    target_region: str = "SanFrancisco_NE",
    target_structure: StructureType = "MultiStory",
    cost: bool = True,
    Mw: Optional[float] = 4.5,
    cov: Optional[float] = 0.0,
    cutoff: Optional[str] = "p97",
    color_max: Optional[float] = None,
    num_bins: Optional[int] = 100,   # None -> "fd"
    cmap: str = "RdYlBu_r",
    savefig: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    
    if mode == "1st":
        tuning_param    = Mw
        tuning_params   = np.round(np.arange(3.50, 8.51, 0.05), 2)
    else:  # "2nd"
        tuning_param    = cov
        tuning_params   = np.round(np.arange(0.0, 1.001, 0.01), 3)

    # --- DV array load ---
    dv_path, _ = _dv_filename(target_structure, target_region, cost, Mw, cov)
    if not dv_path.exists():
        raise FileNotFoundError(f"Missing DV file: {dv_path}")
    dv = np.load(dv_path)  # (num_mw, num_sim) or (num_sim,)

    if cost:
        # --- Capacity + region_info (for cost normalization) ---
        region_info = _load_region_info(target_region)
        bldg_capa = _load_capacity_matrix(target_region, cov)

        # Drop NaN rows and align
        keep = ~np.any(np.isnan(bldg_capa), axis=1)
        bldg_capa = bldg_capa[keep]
        region_info = region_info.loc[keep].reset_index(drop=True)

        # Filter by structure class
        region_info, bldg_capa = _filter_structure(region_info, bldg_capa, target_structure)

        # --- Cost normalization (if applicable) ---

        cost_total = float(region_info["RepairCost"].values.sum())
        if cost_total <= 0:
            raise ValueError("Total repair cost is non-positive after filtering; cannot normalize.")

    # --- Select DV row by tuning parameter (if 2-D) ---
    if dv.ndim == 2:
        tuning_params = np.asarray(tuning_params).ravel()
        idx = int(np.argmin(np.abs(tuning_params - tuning_param)))
        dv_sel = dv[idx, :]
    elif dv.ndim == 1:
        dv_sel = dv
    else:
        raise ValueError(f"Unexpected dv.ndim={dv.ndim}; expected 1 or 2.")

    # Normalize by total cost if requested
    if cost:
        dv_sel = dv_sel / cost_total

    # Clean NaN/Inf
    dv_sel = dv_sel[np.isfinite(dv_sel)]
    if dv_sel.size == 0:
        raise ValueError("No finite DV samples to plot after filtering/normalizing.")

    # --- Figure ---
    fig, ax = plt.subplots(dpi=300, figsize=(6, 4))
    # fig, ax = plt.subplots(dpi=300, figsize=(6.5, 4))
    # fig, ax = plt.subplots(dpi=300, figsize=(8, 6.65))
    # fig, ax = plt.subplots(dpi=300, figsize=(8.7, 6))    # fig, ax = figure_with_content_size(content_w_in=6, content_h_in=4, left_in=1.2, bottom_in=1.0)

    bins = num_bins if isinstance(num_bins, int) else "fd"
    hist_range = (0.0, 1.0) if cost else None

    # Draw histogram and get bars
    counts, bin_edges, patches = ax.hist(
        dv_sel, bins=bins, range=hist_range, alpha=0.95, edgecolor="black"
    )

    # Cutoff + max-norm
    # if cutoff starts by "p", use choose_cutoff with that percentile
    if isinstance(cutoff, str) and cutoff.startswith("p"):
        cut = choose_cutoff(counts, method=cutoff)
    elif isinstance(cutoff, str):
        cut = float(cutoff)
    else:
        cut = choose_cutoff(counts, method="p99")
    print(f"Cutoff for histogram counts: {cut:.3f}")
    # cut = np.max(scen_count_filtered) if (cutoff is None or cutoff is False) else float(cutoff)
    
    if cut <= 0:
        # Avoid division by zero; return zeros if nothing recorded
        counts_color = np.zeros_like(counts, dtype=float)
    else:
        counts_color = np.clip(counts, 0, cut)

    # --- Color bars by relative counts ---
    if cmap == "CP_HEX":
        key_colors = [CP_HEX[0], CP_HEX[8], CP_HEX[1]]
        positions  = [0.0, 0.5, 1.0]
        cmap_obj = LinearSegmentedColormap.from_list("cp_linear", list(zip(positions, key_colors)), N=256)
    else:
        cmap_obj = plt.get_cmap(cmap)
    
    if color_max:
        max_count = color_max
    else:
        max_count = float(np.nanmax(counts_color)) if counts_color.size else 0.0

    rel = counts_color / max_count if max_count > 0 else np.zeros_like(counts_color)
    for r, patch in zip(rel, patches):
        patch.set_facecolor(cmap_obj(r))

    # # (Optional) colorbar to show mapping — uncomment if desired
    # sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=max_count), cmap=cmap_obj)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    # cbar.set_label("Count", fontsize=14)
    # cbar.ax.tick_params(labelsize=12)

    # Labels & title
    xlab = "Damage fraction (cost-normalized)" if cost else "Damage fraction"
    # ax.set_xlabel(xlab, fontsize=36)
    # ax.set_ylabel("Count", fontsize=36)
    ax.set_xlim([-0.03, 1.03])
    # ax.set_ylim([0, 220])

    if title:
        ax.set_title(title, fontsize=24)

    # --- Bigger tick labels ---
    ax.set_xticks([0.0, 0.5, 1.0])
    # ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="major", labelsize=32)
    ax.tick_params(axis="both", which="minor", labelsize=32)

    # ax.grid(True, alpha=0.3)
    fig.tight_layout()    

    if savefig:
        cov_label = f"{int(round((cov or 0.0) * 1000)):04d}"
        mw_label = f"{Mw:.2f}" if Mw is not None else "NA"

        outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
        outdir.mkdir(exist_ok=True)
        fname = f"{outdir}/{target_region}_{target_structure}_cov{cov_label}_Mw{mw_label}_histogram.png"
        fig.savefig(fname, dpi=300, transparent=False)
        print(f"Saved histogram: {fname}")

    return fig, ax


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

def main_phase_diagram(
    *,
    target_region: str = "SanFrancisco_NE",
    target_structure: StructureType = "MultiStory",
    num_bins: int = 100,
    cmap: str = "RdYlBu_r",
    savefig: bool = False,
) -> None:
    if cmap == "CP_HEX":
        key_colors = [CP_HEX[0], CP_HEX[8], CP_HEX[1]]
        positions  = [0.0, 0.5, 1.0]
        # key_colors = [CP_HEX[0], CP_HEX[8], CP_HEX[2], CP_HEX[8], CP_HEX[1]]
        # positions  = [0.0, 0.25, 0.5, 0.75, 1.0]
        cmap = LinearSegmentedColormap.from_list("cp_linear", list(zip(positions, key_colors)), N=256)

    Mws = np.round(np.arange(3.5, 8.51, 0.05), 2)
    covs = np.round(np.arange(0.0, 1.001, 0.01), 3)

    num_sim = 10_000

    top_values_mean = []
    phase_coexist = np.zeros((len(covs), len(Mws)), dtype=int)
    for cov_idx, cov_val in enumerate(covs):
        # print(cov_idx)

        means_per_Mw = []        
        for Mw_idx, Mw_val in enumerate(Mws):
            dv_path, _ = _dv_filename(target_structure, target_region, None, Mw_val, cov_val)
            if not dv_path.exists():
                raise FileNotFoundError(f"Missing DV file: {dv_path}")
            dv_temp = np.load(dv_path)

            counts, bin_edges = np.histogram(dv_temp, bins=num_bins)

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

            flag_temp = phase_coexistence_detect(dv_temp)
            phase_coexist[cov_idx, Mw_idx] = flag_temp

        top_values_mean.append(means_per_Mw)

    top_values_mean = np.array(top_values_mean)

    MAGS, COVS = np.meshgrid(Mws, covs)


    # import matplotlib.colors as mcolors
    # # Colors from RdYlBu
    # c0 = CP_HEX[0]   # at x=0
    # c05 = CP_HEX[8]  # used at x=0.25 and 0.75
    # c1 = CP_HEX[1]   # at x=1
    # custom_mid = CP_HEX[3]

    # # Define the new colormap
    # cmap = mcolors.LinearSegmentedColormap.from_list(
    #     "custom_rdyblu",
    #     [
    #         (0.00, c0),
    #         (0.25, c05),
    #         (0.50, custom_mid),
    #         (0.75, c05),
    #         (1.00, c1),
    #     ]
    # )


    fig, ax = plt.subplots(figsize=(7, 6))

    # ax.scatter(COVS, 5.55*np.ones_like(COVS), color="black", s=3, label="Mw=5.55", zorder=5)
    ax.plot(0.5, 5.55, 'o', ms=10, markerfacecolor="black", markeredgecolor="white", zorder=5)

    vmin = 0.0
    vmax = 1.0

    c = ax.contourf(
        COVS, MAGS, top_values_mean, 
        levels=100, 
        cmap=cmap, 
        vmin=vmin, vmax=vmax,
        zorder=1
    )

    phase_mask = (phase_coexist == 1).astype(int)  # shape: (len(covs), len(Mws))
    ax.contourf(
        COVS, MAGS, phase_mask,
        levels=[0.5, 1.5],          # draw only the "1" region
        colors=["black"],
        alpha=0.5,                 # adjust transparency
        zorder=3                    # above background contourf; below black contour lines (zorder=4)
    )

    # extent = [COVS.min(), COVS.max(), MAGS.min(), MAGS.max()]

    # c = ax.imshow(
    #     top_values_mean.T,
    #     origin="lower",
    #     extent=[COVS.min(), COVS.max(), MAGS.min(), MAGS.max()],
    #     aspect="auto",
    #     cmap=cmap,
    #     vmin=vmin, vmax=vmax,
    #     interpolation="none"   # critical: no blending between pixels
    # )

    # Contour levels and label positions
    # levels = [0.01, 0.05, 0.1, 0.9, 0.95, 0.99]
    # label_positions = [(0.93, 6.5), (0.80, 6.5), (0.70, 6.5), (0.80, 6.5), (0.70, 6.5), (0.60, 6.5)]
    levels = [0.05, 0.1, 0.9, 0.95]
    label_positions = [(0.80, 6.5), (0.70, 6.5), (0.80, 6.5), (0.70, 6.5)]

    # Draw contours and labels
    for lvl, pos in zip(levels, label_positions):
        cs = ax.contour(
            COVS, MAGS, top_values_mean,
            levels=[lvl],
            colors='black',
            linewidths=0.5,
            linestyles='--',
            zorder=4
        )
        # ax.clabel(cs, fmt=f'{lvl:.2f}', fontsize=14, inline=False, manual=[pos])


    # Draw gridlines at cell boundaries
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5, alpha=0.6)
    
    cb = plt.colorbar(c, ax=ax)
    cb.set_label('Damage fraction', fontsize=20)
    cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    # cb.set_ticklabels(["0.0", "0.5", "1.0"])
    cb.ax.tick_params(labelsize=16)


    # import matplotlib as mpl
    # norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    # sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])

    # # Add the standalone colorbar
    # cb = fig.colorbar(sm, ax=ax)
    # cb.set_label("Damage fraction", fontsize=20)
    # cb.set_ticks([0.0, 0.5, 1.0])
    # cb.ax.tick_params(labelsize=16)

    # cb = plt.colorbar(c, ax=ax, location='right', pad=0.12)
    # cb.set_label('Collective state', fontsize=20, family='Times New Roman')
    # cb.set_ticks([])
    # cb.ax.text(0, -0.01, "Undamaged", ha="center", va="top", fontsize=16, transform=cb.ax.transAxes)
    # cb.ax.text(0, 1.01, "Complete damage", ha="center", va="bottom", fontsize=16, transform=cb.ax.transAxes)

    ax.set_xlabel(r'Structural diversity $\sigma$', fontsize=20)
    # ax.set_xticks([0.0, 1.0])
    # ax.set_xticklabels(["Low", "High"])

    ax.set_xticks([0.0, 0.5, 1.0])

    ax.set_ylabel(r'Earthquake magnitude $M_w$', fontsize=20)
    ax.tick_params(axis="both", labelsize=16)

    fig.tight_layout()

    if savefig:
        outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
        outdir.mkdir(exist_ok=True)
        fname = f"{outdir}/{target_region}_{target_structure}_phase_diagram.png"
        fig.savefig(fname, dpi=300, transparent=True)
        print(f"Saved phase diagram: {fname}")
    
    plt.show()

    return fig, ax


def main_fragility_curves(
    *,
    target_region: str = "SanFrancisco_NE",
    target_structure: StructureType = "MultiStory",
    cov: float = 0.0,
    category: str = None,
    savefig: bool = False,
) -> None:

    bldg_capa  = pd.read_csv(f"C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Codes/Benchmark/Savio/{target_region}/IDA_results_OpenSeesPy_cov000.csv")
    bldg_capa  = bldg_capa.to_numpy()[:, 1:].T
    idx_nan = np.where(np.any(np.isnan(bldg_capa), axis=1))[0]
    if idx_nan.size > 0:
        keep = np.ones(bldg_capa.shape[0], dtype=bool); keep[idx_nan] = False
        bldg_capa = bldg_capa[keep]

    region_info = pd.read_csv(f"C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Codes/Benchmark/TargetRegion_{target_region}.csv")
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

    num_bldgs, _ = bldg_capa.shape

    cov_label = f"{int(round(cov*1000)):04d}"
    fname = f"C:/Users/ohsb1/Desktop/Savio/{target_region}/bldg_frag_params_for_[sim_total]/bldg_frag_params_cov{cov_label}.npy"
    frag_params = np.load(fname)
    frag_params = frag_params[idx_target, :]

    x = np.linspace(0.00001, 0.6, 1000) if target_region == "Milpitas_all" else np.linspace(0.00001, 1.0, 1000)
    fig, ax = plt.subplots(figsize=(6,6))

    if category is None:
        for i in range(num_bldgs):
            cdf = lognorm.cdf(x, s=frag_params[i, 0], scale=frag_params[i, 2])
            ax.plot(x, cdf, '-', alpha=0.1, linewidth=0.1, color='black')
    else:
        # for i in range(num_bldgs):
        #     cdf = lognorm.cdf(x, s=frag_params[i, 0], scale=frag_params[i, 2])
        #     ax.plot(x, cdf, '-', alpha=0.1, linewidth=0.05, color='black')

        def get_clean_vals(arr2d):
            vals = np.asarray(arr2d).ravel()
            vals = vals[np.isfinite(vals) & (vals > 0)]  # remove NaN/inf and non-positive
            return vals

        def ecdf(x, samples_sorted):
            # ECDF(x) = (# samples <= x) / N, vectorized via searchsorted
            return np.searchsorted(samples_sorted, x, side="right") / samples_sorted.size

        col = (region_info[category]
            .astype('string')
            .str.strip())

        str_types = {k: (col == k) for k in col.dropna().unique()}
        str_types = dict(sorted(str_types.items(), key=lambda x: x[0]))

        frag_params_categorized = {}

        # colors = plt.get_cmap("tab10").colors
        # colors = sns.color_palette("Set3", 10)

        count = 0
        for name, mask in str_types.items():
            data2d = bldg_capa[mask, :]
            vals = get_clean_vals(data2d)
            if vals.size == 0:
                continue

            vals_sorted = np.sort(vals)
            y_emp = ecdf(x, vals_sorted)

            # plot empirical and grab its color
            # emp_line, = ax.plot(x, y_emp, '--', linewidth=1.0, label=f"{name} (Empirical)")
            # color = emp_line.get_color()

            # fit lognormal and plot with the SAME color, dashed
            s, loc, scale = lognorm.fit(vals, floc=0)
            y_fit = lognorm.cdf(x, s, loc=loc, scale=scale)
            ax.plot(x, y_fit, '-', linewidth=2.0, color=CP_STRUCT_TYPES[name], label=f"{name}")

            # store params and assign back to all buildings of this type
            frag_params_categorized[name] = (s, loc, scale)

            count += 1
            
        ax.legend(fontsize=20, loc='lower right', ncol=2, columnspacing=0.5, frameon=False)

    ax.set_xlabel(r'PGA ($g$)', fontsize=20)
    ax.set_ylabel('Damage probability', fontsize=20)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlim([0, x.max()])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.0)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()

    if savefig:
        outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
        outdir.mkdir(exist_ok=True)
        fname = f"{outdir}/{target_region}_{target_structure}_cov{cov_label}_fragilty_curves_cate{category}.png"
        fig.savefig(fname, dpi=300, transparent=True)
        print(f"Saved phase diagram: {fname}")


def main_eng(
    *,
    target_region: str = "SanFrancisco_NE",
    target_structure: StructureType = "MultiStory",
    category: str = "StructureType",
    corr: str = "Indp",
    cost: bool = True,
    legend: bool = False,
    savefig: bool = False,
) -> None:
    import matplotlib.patches as mpatches

    if target_region == "SanFrancisco_NE":
        cL = CP_HEX[3]
        cR = CP_HEX[2]
    else:
        cL = CP_HEX[3]
        cR = CP_HEX[4]

    Mws = np.round(np.arange(3.5, 8.51, 0.05), 2)
    num_sim = 10_000

    cov = 0.0
    cov_label = f"{int(round(cov*100)):03d}"

    dv_base = np.zeros((len(Mws), num_sim))
    dv_eng = np.zeros((len(Mws), num_sim))

    if cost:
        filepath_dv = "dv_cost"
    else:
        filepath_dv = "dv"
        
    for Mw_idx, Mw_val in enumerate(Mws):
        Mw_label = f"{int(round(Mw_val*100)):03d}"

        if category is not None:
            filepath_base = f"C:/Users/ohsb1/Desktop/Savio/{target_region}/eng/categorization/{filepath_dv}/{target_structure}_{filepath_dv}_Mw{Mw_label}_cov{cov_label}_{target_region}_cateNo.npy"
            filepath_eng = f"C:/Users/ohsb1/Desktop/Savio/{target_region}/eng/categorization/{filepath_dv}/{target_structure}_{filepath_dv}_Mw{Mw_label}_cov{cov_label}_{target_region}_cate{category}.npy"
        else:
            filepath_base = f"C:/Users/ohsb1/Desktop/Savio/{target_region}/eng/corr/{filepath_dv}/{target_structure}_{filepath_dv}_Mw{Mw_label}_cov{cov_label}_{target_region}_corrDpdn.npy"
            filepath_eng = f"C:/Users/ohsb1/Desktop/Savio/{target_region}/eng/corr/{filepath_dv}/{target_structure}_{filepath_dv}_Mw{Mw_label}_cov{cov_label}_{target_region}_corrIndp.npy"
        
        dv_base[Mw_idx, :] = np.load(filepath_base)
        dv_eng[Mw_idx, :] = np.load(filepath_eng)
    
    if cost:
        dv_base /= 5   # Calculated repair cost is probably for at least "Moderate" damage state
                        #(I've calculated the ratio between the replacement cost and the repair cost
                        # and it turns out to be around 3%,
                        # which corresponds to the typical moderate damage repair cost ratio),
                        # while we targeted "Slight" damage state.
                        # Typically, repair cost ratio for slight damage is around 1/5 of that of moderate damage.
        dv_eng /= 5

    # --- choose magnitudes and nearest indices for violins ---
    Mw_vals = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5], dtype=float) if target_region == "SanFrancisco_NE" else np.array([4.5, 5.0, 5.5, 6.0, 6.5, 7.0], dtype=float)
    Mw_idxs = [int(np.argmin(np.abs(Mws - t))) for t in Mw_vals]
    print("Exact Mws used for violins:", Mws[Mw_idxs])

    def clean_to_millions(x):
        y = x / 1e6  # convert to million $
        return y[np.isfinite(y)]
    def clean_to_billions(x):
        y = x / 1e9  # convert to Billion $
        return y[np.isfinite(y)]

    data_left  = [clean_to_millions(dv_base[i, :]) for i in Mw_idxs]  # No categorization (left)
    data_right = [clean_to_millions(dv_eng[i, :]) for i in Mw_idxs]  # By StructureType (right)

    # --- 1. Violin plots ---
    fig, ax = plt.subplots(figsize=(8, 6))

    positions = np.arange(1, len(Mw_vals) + 1, dtype=float)
    width = 0.8  # total violin width

    import matplotlib.colors as mcolors
    def split_violin(ax, left, right, x, width=0.8, cL=cL, cR=cR,
                    alpha=0.7, bw=0.1, points=100, edgecolor="black", lw=1.5):
       
        vL = ax.violinplot([left], positions=[x], widths=width,
                        bw_method=bw, points=points,
                        showmeans=False, showmedians=False, showextrema=False)
        for pc in vL["bodies"]:
            verts = pc.get_paths()[0].vertices
            m = np.mean(verts[:, 0])
            verts[:, 0] = np.minimum(verts[:, 0], m)

            pc.set_alpha(None)          # or pc.set_alpha(1.0)
            # face: per-face alpha
            pc.set_facecolor(mcolors.to_rgba(cL, alpha))      # (use cR on the right)
            # edge: fully opaque black
            pc.set_edgecolor((0, 0, 0, 1))
            pc.set_linewidth(1.0)
            pc.set_zorder(3)

        vR = ax.violinplot([right], positions=[x], widths=width,
                        bw_method=bw, points=points,
                        showmeans=False, showmedians=False, showextrema=False)
        for pc in vR["bodies"]:
            verts = pc.get_paths()[0].vertices
            m = np.mean(verts[:, 0])
            verts[:, 0] = np.maximum(verts[:, 0], m)

            pc.set_alpha(None)          # or pc.set_alpha(1.0)
            # face: per-face alpha
            pc.set_facecolor(mcolors.to_rgba(cR, alpha))      # (use cR on the right)
            # edge: fully opaque black
            pc.set_edgecolor((0, 0, 0, 1))
            pc.set_linewidth(1.0)
            pc.set_zorder(3)

    # 60% less smoothing than the default smoothing (Scott's rule)
    bw_less = lambda kde: kde.scotts_factor() * (1 - 0.6)

    for x, L, R in zip(positions, data_left, data_right):
        split_violin(ax, L, R, x, width=width, cL=cL, cR=cR, alpha=0.5, bw=bw_less, points=200, edgecolor="black", lw=1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{t:.1f}" for t in Mw_vals])
    ax.tick_params(axis='both',labelsize=17)
    ax.set_xlabel(r"Earthquake magnitude $M_w$", fontsize=20)
    ax.set_ylabel("Repair cost (million $)", fontsize=20)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)

    if legend:
        legend_patches = [
            mpatches.Patch(facecolor=mcolors.to_rgba(cL, 0.5), edgecolor='black', linewidth=1.0,
                        label='Individual' if category else 'Dependent'),
            mpatches.Patch(facecolor=mcolors.to_rgba(cR, 0.5), edgecolor='black', linewidth=1.0,
                        label='Categorized' if category else 'Conditionally\nIndependent'),
        ]
        legend_markers = [
            Line2D([0], [0], marker='X', linestyle='None', markersize=12,
                markerfacecolor='black', markeredgecolor='black', label='Mean'),
            Line2D([0], [0], marker='^', linestyle='None', markersize=12,
                markerfacecolor='black', markeredgecolor='black', label='10% quantile'),
        ]

        first_legend = ax.legend(
            handles=legend_patches,
            loc='lower center',
            bbox_to_anchor=(0.20, 0.97),
            ncol=2, frameon=False, fontsize=17,
            labelspacing=0.3, handletextpad=0.4, handlelength=1.0,
        )
        ax.add_artist(first_legend)

        ax.legend(
            handles=legend_markers,
            loc='lower center',
            bbox_to_anchor=(0.75, 0.97),
            ncol=2, frameon=False, fontsize=17,
            labelspacing=0.3, handletextpad=0.4, handlelength=1.0,
        )

        plt.subplots_adjust(top=0.85)

    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    
    if savefig:
        outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
        outdir.mkdir(exist_ok=True)
        fname = f"{outdir}/{target_region}_{target_structure}_eng_categorization_violin.png" if category else f"{outdir}/{target_region}_{target_structure}_eng_corr_violin.png"
        fig.savefig(fname, dpi=300, transparent=True)
        print(f"Saved violin plots: {fname}")
    plt.show()

    # --- 2. Scatter plots ---
    dv_mean_base    = np.mean(dv_base, axis=1)
    dv_mean_eng     = np.mean(dv_eng, axis=1)
    dv_q01_base     = np.quantile(dv_base, q=0.01, axis=1)
    dv_q01_eng      = np.quantile(dv_eng, q=0.01, axis=1)
    dv_q05_base     = np.quantile(dv_base, q=0.05, axis=1)
    dv_q05_eng      = np.quantile(dv_eng, q=0.05, axis=1)
    dv_q10_base     = np.quantile(dv_base, q=0.10, axis=1)
    dv_q10_eng      = np.quantile(dv_eng, q=0.10, axis=1)
    dv_med_base     = np.quantile(dv_base, q=0.50, axis=1)
    dv_med_eng      = np.quantile(dv_eng, q=0.50, axis=1)

    dv_mean_base    = clean_to_millions(dv_mean_base)
    dv_mean_eng     = clean_to_millions(dv_mean_eng)
    dv_q01_base     = clean_to_millions(dv_q01_base)
    dv_q01_eng      = clean_to_millions(dv_q01_eng)
    dv_q05_base     = clean_to_millions(dv_q05_base)
    dv_q05_eng      = clean_to_millions(dv_q05_eng)
    dv_q10_base     = clean_to_millions(dv_q10_base)
    dv_q10_eng      = clean_to_millions(dv_q10_eng)
    dv_med_base     = clean_to_millions(dv_med_base)
    dv_med_eng      = clean_to_millions(dv_med_eng)

    dv_mean_gap   = np.abs(dv_mean_base - dv_mean_eng)
    dv_q01_gap    = np.abs(dv_q01_base - dv_q01_eng)
    dv_q05_gap    = np.abs(dv_q05_base - dv_q05_eng)
    dv_q10_gap    = np.abs(dv_q10_base - dv_q10_eng)
    dv_med_gap    = np.abs(dv_med_base - dv_med_eng)

    # dv_mean_gap     = dv_mean_base - dv_mean_eng
    # dv_q01_gap      = dv_q01_base - dv_q01_eng
    # dv_q05_gap      = dv_q05_base - dv_q05_eng
    # dv_q10_gap      = dv_q10_base - dv_q10_eng
    # dv_med_gap      = dv_med_base - dv_med_eng

    print("Mean estimated repair costs (million $) at Mw=6.0:", dv_mean_base[50], "(base),", dv_mean_eng[50], "(eng practice)")
    print("1% quantile estimated repair costs (million $) at Mw=6.0:", dv_q01_base[50], "(base),", dv_q01_eng[50], "(eng practice)")
    print("Mean estimated repair costs (million $) at Mw=6.5:", dv_mean_base[60], "(base),", dv_mean_eng[60], "(eng practice)")
    print("1% quantile estimated repair costs (million $) at Mw=6.5:", dv_q01_base[60], "(base),", dv_q01_eng[60], "(eng practice)")
    print("Mean estimated repair costs (million $) at Mw=7.0:", dv_mean_base[70], "(base),", dv_mean_eng[70], "(eng practice)")
    print("1% quantile estimated repair costs (million $) at Mw=7.0:", dv_q01_base[70], "(base),", dv_q01_eng[70], "(eng practice)")

    # fig,ax = plt.subplots(figsize=(6,6))
    # ax.plot(Mws[np.arange(0,len(Mws),5)], dv_mean_gap[np.arange(0,len(Mws),5)], '-o', color='#4056A1', label='Mean', ms=8)
    # ax.plot(Mws[np.arange(0,len(Mws),5)], dv_q01_gap[np.arange(0,len(Mws),5)], '-s', color='#116466', label='Lower 1%', ms=8)
    # ax.plot(Mws[np.arange(0,len(Mws),5)], dv_q05_gap[np.arange(0,len(Mws),5)], '-^', color='#59A14F', label='Lower 5%', ms=8)
    # ax.plot(Mws[np.arange(0,len(Mws),5)], dv_q10_gap[np.arange(0,len(Mws),5)], '-X', color='#8CD17D', label='Lower 10%', ms=8)

    # ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.7)
    # # ax.plot(Mws, dv_q05_gap, '-o', ms=5, color=CP_HEX[2], linewidth=2.0, label="5%", alpha=0.8)    
    # ax.set_xlim([3.5, 8.5])
    # ax.set_xlabel('Earthquake magnitude', fontsize=20)
    # ax.set_ylabel('Difference in estimated repair costs\n(million $)', fontsize=20)
    # ax.legend(fontsize=18, frameon=False)
    # ax.tick_params(axis='both', labelsize=17)
    # ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.0)
    # ax.spines[['top', 'right']].set_visible(False)
    # fig.tight_layout()
    # if savefig:
    #     outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
    #     outdir.mkdir(exist_ok=True)
    #     fname = f"{outdir}/{target_region}_{target_structure}_eng_categorization_quantile.png" if category else f"{outdir}/{target_region}_{target_structure}_eng_corr_quantile.png"
    #     fig.savefig(fname, dpi=300, transparent=True)
    #     print(f"Saved quantile-comparison plots: {fname}")
    # plt.show()

    # --- 3. CDF plots ---
    # dv_thres = 1/10 * np.max((dv_base.max(), dv_eng.max()))/1e6

    if target_region == "Milpitas_all":
        dv_thres1=10
        dv_thres2=100
    else:
        dv_thres1=10
        dv_thres2=100

    # print(f"Target repair cost: ${dv_thres:.2f}M")

    p_exceed_base1 = np.mean(dv_base/1e6 > dv_thres1, axis=1)
    p_exceed_eng1 = np.mean(dv_eng/1e6 > dv_thres1, axis=1)
    p_exceed_base2 = np.mean(dv_base/1e6 > dv_thres2, axis=1)
    p_exceed_eng2 = np.mean(dv_eng/1e6 > dv_thres2, axis=1)

    # fig,ax = plt.subplots(figsize=(6.5,6))
    # plt.plot(Mws[np.arange(0,len(Mws),5)],p_exceed_base1[np.arange(0,len(Mws),5)],'-o',ms=8, color=cL, linewidth=2.0, alpha=0.8)
    # plt.plot(Mws[np.arange(0,len(Mws),5)],p_exceed_eng1[np.arange(0,len(Mws),5)],'-o',ms=8, color=cR, linewidth=2.0, alpha=0.8)
    # plt.plot(Mws[np.arange(0,len(Mws),5)],p_exceed_base2[np.arange(0,len(Mws),5)],'-X',ms=8, color=cL, linewidth=2.0, alpha=0.8)
    # plt.plot(Mws[np.arange(0,len(Mws),5)],p_exceed_eng2[np.arange(0,len(Mws),5)],'-X',ms=8, color=cR, linewidth=2.0, alpha=0.8)
    # ax.set_xlim(Mws[0], Mws[-1])
    # ax.set_ylim([-0.05, 1.05])
    # ax.set_xlabel('Earthquake magnitude', fontsize=20)
    # ax.set_ylabel('Exceedance probability', fontsize=20)

    # legend_handles = [
    #     Line2D([0], [0], marker='o', color='black', markerfacecolor='black',
    #         markersize=8, linestyle='None', label=f'Ref. cost ${dv_thres1}M'),
    #     Line2D([0], [0], marker='X', color='black', markerfacecolor='black',
    #         markersize=8, linestyle='None', label=f'Ref. cost ${dv_thres2}M'),
    #     Line2D([0], [0], color=cL, linewidth=2.0, label='Baseline'),
    #     Line2D([0], [0], color=cR, linewidth=2.0, label='Engineering practice'),
    # ]

    # # --- Single, compact legend inside lower-left corner ---
    # ax.legend(
    #     handles=legend_handles,
    #     loc='lower right',
    #     bbox_to_anchor=(1.1, 0.02),
    #     frameon=False,
    #     fontsize=18,
    #     ncol=1,              # single-column layout
    #     handlelength=1.3,    # shorten line handle
    #     handletextpad=0.5,   # space between symbol and text
    #     labelspacing=0.50,   # vertical spacing between entries
    #     borderaxespad=0.4,   # distance from axes border
    # )

    # ax.tick_params(axis='both', labelsize=17)
    # ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.0)
    # ax.spines[['top', 'right']].set_visible(False)    
    # fig.tight_layout()

    # if savefig:
    #     outdir = Path("C:/Users/ohsb1/OneDrive/UCB/Research/Phase transitions in regional seismic responses/Paper/Figures")
    #     outdir.mkdir(exist_ok=True)
    #     fname = f"{outdir}/{target_region}_{target_structure}_eng_categorization_cdf.png" if category else f"{outdir}/{target_region}_{target_structure}_eng_corr_cdf.png"
    #     fig.savefig(fname, dpi=300, transparent=True)
    #     print(f"Saved cdf plots: {fname}")
    # plt.show()


# %%
if __name__ == "__main__":
    if FigureType == "heatmap":
        main_heatmap(
            mode="2nd",
            title=False,
            target_region="Milpitas_all",
            target_structure="MultiStory",
            Mw=5.6,
            cov=0.0,
            corr=None,         # None or 0.0–1.0
            category=None,     # None or "StructureType"
            sigma=0.0,         # Gaussian smoothing; set 0 to disable
            cutoff="p99",        # None/False ⇒ auto (max of smoothed counts) # 1st: "p97", 2nd: "p99"
            step_fraction=1/1000,
            cmap="RdYlBu_r",
            # cmap="CP_HEX",
            savefig=SaveFig,
        )
    
    if FigureType == "histogram":
        main_histogram(
            mode="2nd",
            # title=r"$M_{w}$ = 5.6",
            title=None,
            target_region="Milpitas_all",
            target_structure="MultiStory",
            cost=False,
            Mw=5.6,
            cov=0.3,
            cutoff="p95",
            color_max=202.35,
            num_bins=100,
            cmap="RdYlBu_r",
            # cmap="CP_HEX",
            savefig=SaveFig,
        )

    if FigureType == "phase_diagram":
        main_phase_diagram(
            target_region="Milpitas_all",
            target_structure="MultiStory",
            num_bins=100,
            cmap="RdYlBu_r",
            # cmap="CP_HEX",
            savefig=SaveFig,
        )

    if FigureType == "fragility":
        main_fragility_curves(
            target_region="SanFrancisco_NE",
            target_structure="All",
            cov=0.1,
            category=None,       # None     StructureType
            savefig=SaveFig,
        )

    if FigureType == "eng":
        main_eng(
            target_region="Milpitas_all",   # Milpitas_all   SanFrancisco_NE
            target_structure="MultiStory",  # MultiStory     All
            category=None,                  # None           StructureType

            # target_region="SanFrancisco_NE",
            # target_structure="All",
            # category="StructureType", 

            corr="Indp",                    # Indp
            cost=True,
            legend=True,
            savefig=SaveFig,
        )
# %%
