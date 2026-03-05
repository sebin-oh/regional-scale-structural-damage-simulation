#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 02/25/2026
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

- Computes GMPE-based ln(PGA) mean and standard deviation at multiple sites
- Supports batch evaluation for Monte Carlo hazard workflows

"""


from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import cupy as cp

from pygmm.model import Scenario
from pygmm.chiou_youngs_2014 import ChiouYoungs2014
from pygmm.abrahamson_silva_kamai_2014 import AbrahamsonSilvaKamai2014
from pygmm.boore_stewart_seyhan_atkinson_2014 import BooreStewartSeyhanAtkinson2014
from pygmm.campbell_bozorgnia_2014 import CampbellBozorgnia2014


# =============================================================================
# Geometry utilities
# =============================================================================
def _rupture_length_wc94(mw: float) -> float:
    """Wells & Coppersmith (1994) rupture length [m]."""
    return 10 ** (-2.44 + 0.59 * mw) * 1000.0


def _rupture_width_wc94(mw: float) -> float:
    """Wells & Coppersmith (1994) rupture width [m]."""
    return 10 ** (-1.01 + 0.32 * mw) * 1000.0


def _strike_normal_unit_vectors(strike_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (strike_unit_vec, normal_unit_vec) in the XY plane."""
    strike_rad = np.deg2rad(strike_deg)
    strike_vec = np.array([np.cos(strike_rad), np.sin(strike_rad)], dtype=np.float64)
    normal_vec = np.array([-np.sin(strike_rad), np.cos(strike_rad)], dtype=np.float64)
    return strike_vec, normal_vec


def _project_distances_km(
    x: float,
    y: float,
    *,
    event_x: float,
    event_y: float,
    strike_vec: np.ndarray,
    normal_vec: np.ndarray,
    rupture_length_m: float,
) -> Tuple[float, float, float]:
    """
    Compute (Rjb, Rx, horiz_dist) in km using a finite-length surface projection:
      - Along-strike projection is clipped to +/- L/2
      - Rjb is distance to the finite segment in the horizontal plane
      - Rx is signed normal projection (positive on one side)
    """
    rel_xy = np.array([x - event_x, y - event_y], dtype=np.float64)
    proj_strike = float(rel_xy @ strike_vec)
    proj_normal = float(rel_xy @ normal_vec)

    half_L = 0.5 * rupture_length_m
    if abs(proj_strike) <= half_L:
        horiz_dist_m = abs(proj_normal)
    else:
        delta_strike = abs(proj_strike) - half_L
        horiz_dist_m = float(np.sqrt(delta_strike**2 + proj_normal**2))

    rjb_km = horiz_dist_m / 1000.0
    rx_km = proj_normal / 1000.0
    return rjb_km, rx_km, horiz_dist_m / 1000.0


def _iter_assets_xy_vs30(assets: pd.DataFrame):
    """Fast-ish iterator over x, y, Vs30 without constructing Series each time."""
    for row in assets.itertuples(index=False):
        yield float(row.x), float(row.y), float(row.Vs30)


# =============================================================================
# GMPE wrappers (PGA)
# =============================================================================
def gmm_CY14(assets: pd.DataFrame, event_xy: Tuple[float, float], eq_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chiou & Youngs (2014) ln(PGA) mean/std for each asset.

    Required eq_info keys:
      Mw, strike, dip, depth_to_top, mechanism, region, on_hanging_wall, vs_source
    Optional:
      L, W, hypo_down_dip
    """
    event_x, event_y = event_xy
    strike_vec, normal_vec = _strike_normal_unit_vectors(eq_info["strike"])

    L_m = float(eq_info.get("L", _rupture_length_wc94(eq_info["Mw"]) / 1000.0)) * 1000.0 if "L" in eq_info else _rupture_length_wc94(eq_info["Mw"])
    W_m = float(eq_info.get("W", _rupture_width_wc94(eq_info["Mw"]) / 1000.0)) * 1000.0 if "W" in eq_info else _rupture_width_wc94(eq_info["Mw"])

    depth_top_m = float(eq_info["depth_to_top"]) * 1000.0
    depth_center_m = depth_top_m + 0.5 * W_m  # vertically planar assumption

    ln_mean = np.empty(len(assets), dtype=np.float64)
    ln_std = np.empty(len(assets), dtype=np.float64)

    for i, (x, y, vs30) in enumerate(_iter_assets_xy_vs30(assets)):
        rjb_km, rx_km, _ = _project_distances_km(
            x, y,
            event_x=event_x,
            event_y=event_y,
            strike_vec=strike_vec,
            normal_vec=normal_vec,
            rupture_length_m=L_m,
        )

        rrup_km = float(np.sqrt((rjb_km * 1000.0) ** 2 + depth_center_m**2) / 1000.0)

        sc = Scenario(
            mag=eq_info["Mw"],
            dist_rup=rrup_km,
            dist_x=rx_km,
            dist_jb=rjb_km,
            v_s30=vs30,
            dip=eq_info["dip"],
            depth_tor=depth_top_m / 1000.0,  # km
            depth_1_0=None,
            dpp_centered=False,
            mechanism=eq_info["mechanism"],
            region=eq_info["region"],
            on_hanging_wall=(False if eq_info["dip"] == 90 else eq_info["on_hanging_wall"]),
            vs_source=eq_info["vs_source"],
        )

        model = ChiouYoungs2014(sc)
        ln_mean[i] = model._ln_resp[model.INDEX_PGA]
        ln_std[i] = model._ln_std[model.INDEX_PGA]

    return ln_mean, ln_std


def gmm_ASK14(assets: pd.DataFrame, event_xy: Tuple[float, float], eq_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Abrahamson, Silva, & Kamai (2014) ln(PGA) mean/std.
    Required eq_info keys:
      Mw, strike, dip, depth_to_top, mechanism, region, on_hanging_wall, vs_source
    Optional:
      L
    """
    event_x, event_y = event_xy
    strike_vec, normal_vec = _strike_normal_unit_vectors(eq_info["strike"])

    L_m = float(eq_info["L"]) * 1000.0 if "L" in eq_info else _rupture_length_wc94(eq_info["Mw"])
    depth_top_m = float(eq_info["depth_to_top"]) * 1000.0

    ln_mean = np.empty(len(assets), dtype=np.float64)
    ln_std = np.empty(len(assets), dtype=np.float64)

    for i, (x, y, vs30) in enumerate(_iter_assets_xy_vs30(assets)):
        rjb_km, rx_km, _ = _project_distances_km(
            x, y,
            event_x=event_x,
            event_y=event_y,
            strike_vec=strike_vec,
            normal_vec=normal_vec,
            rupture_length_m=L_m,
        )

        rrup_km = float(np.sqrt((rjb_km * 1000.0) ** 2 + depth_top_m**2) / 1000.0)
        z1p0 = AbrahamsonSilvaKamai2014.calc_depth_1_0(vs30, eq_info["region"])

        sc = Scenario(
            mag=eq_info["Mw"],
            dist_rup=rrup_km,
            dist_x=rx_km,
            dist_jb=rjb_km,
            depth_1_0=z1p0,
            v_s30=vs30,
            dip=eq_info["dip"],
            depth_tor=depth_top_m / 1000.0,  # km
            mechanism=eq_info["mechanism"],
            region=eq_info["region"],
            on_hanging_wall=eq_info["on_hanging_wall"],
            vs_source=eq_info["vs_source"],
        )

        model = AbrahamsonSilvaKamai2014(sc)
        ln_mean[i] = model._ln_resp[model.INDEX_PGA]
        ln_std[i] = model._ln_std[model.INDEX_PGA]

    return ln_mean, ln_std


def gmm_BSSA14(assets: pd.DataFrame, event_xy: Tuple[float, float], eq_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Boore, Stewart, Seyhan, & Atkinson (2014) ln(PGA) mean/std.
    Required eq_info keys:
      Mw, strike, mechanism, region
    Optional:
      L
    """
    event_x, event_y = event_xy
    strike_vec, normal_vec = _strike_normal_unit_vectors(eq_info["strike"])
    L_m = float(eq_info["L"]) * 1000.0 if "L" in eq_info else _rupture_length_wc94(eq_info["Mw"])

    ln_mean = np.empty(len(assets), dtype=np.float64)
    ln_std = np.empty(len(assets), dtype=np.float64)

    for i, (x, y, vs30) in enumerate(_iter_assets_xy_vs30(assets)):
        rjb_km, _, _ = _project_distances_km(
            x, y,
            event_x=event_x,
            event_y=event_y,
            strike_vec=strike_vec,
            normal_vec=normal_vec,
            rupture_length_m=L_m,
        )

        z1p0 = ChiouYoungs2014.calc_depth_1_0(vs30, eq_info["region"])

        sc = Scenario(
            mag=eq_info["Mw"],
            dist_jb=rjb_km,
            depth_1_0=z1p0,
            v_s30=vs30,
            mechanism=eq_info["mechanism"],
            region=eq_info["region"],
        )

        model = BooreStewartSeyhanAtkinson2014(sc)
        ln_mean[i] = model._ln_resp[model.INDEX_PGA]
        ln_std[i] = model._ln_std[model.INDEX_PGA]

    return ln_mean, ln_std


def gmm_CB14(assets: pd.DataFrame, event_xy: Tuple[float, float], eq_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Campbell & Bozorgnia (2014) ln(PGA) mean/std.
    Required eq_info keys:
      Mw, strike, dip, depth_to_top, mechanism, region
    Optional:
      L
    """
    event_x, event_y = event_xy
    strike_vec, normal_vec = _strike_normal_unit_vectors(eq_info["strike"])

    L_m = float(eq_info["L"]) * 1000.0 if "L" in eq_info else _rupture_length_wc94(eq_info["Mw"])
    depth_top_m = float(eq_info["depth_to_top"]) * 1000.0

    ln_mean = np.empty(len(assets), dtype=np.float64)
    ln_std = np.empty(len(assets), dtype=np.float64)

    for i, (x, y, vs30) in enumerate(_iter_assets_xy_vs30(assets)):
        rjb_km, rx_km, _ = _project_distances_km(
            x, y,
            event_x=event_x,
            event_y=event_y,
            strike_vec=strike_vec,
            normal_vec=normal_vec,
            rupture_length_m=L_m,
        )

        rrup_km = float(np.sqrt((rjb_km * 1000.0) ** 2 + depth_top_m**2) / 1000.0)

        sc = Scenario(
            mag=eq_info["Mw"],
            dist_jb=rjb_km,
            dist_rup=rrup_km,
            dist_x=rx_km,
            v_s30=vs30,
            dip=eq_info["dip"],
            depth_tor=float(eq_info["depth_to_top"]),  # km
            region=eq_info["region"],
            mechanism=eq_info["mechanism"],
        )

        model = CampbellBozorgnia2014(sc)
        ln_mean[i] = model._ln_resp[model.INDEX_PGA]
        ln_std[i] = model._ln_std[model.INDEX_PGA]

    return ln_mean, ln_std


################################################################################
# Intra-event residual correlation
################################################################################

def intra_residuals_corr(
    x_data,
    y_data,
    *,
    soil_case: int = 2,
    indp_intra_residuals: bool = False,
    dtype=cp.float64,
) -> cp.ndarray:
    """
    Intra-event residual correlation matrix for PGA (Jayaram & Baker, 2009).

    Parameters
    ----------
    x_data, y_data : array-like
        Site coordinates in meters (e.g., UTM easting/northing). Must be same length.
    soil_case : int
        - soil_case == 2 : RangeSemiv = 40.7 (at T=0)
        - else           : RangeSemiv = 8.5  (at T=0)
    indp_intra_residuals : bool
        If True, return identity (independent intra-event residuals).
    dtype : cupy dtype
        Output dtype.

    Returns
    -------
    corr : (n, n) cupy.ndarray
        Correlation matrix.
    """
    x = cp.asarray(x_data, dtype=dtype).ravel()
    y = cp.asarray(y_data, dtype=dtype).ravel()

    if x.size != y.size:
        raise ValueError(f"x_data and y_data must have the same length, got {x.size} and {y.size}.")

    n = int(x.size)
    if n == 0:
        return cp.empty((0, 0), dtype=dtype)

    if indp_intra_residuals:
        return cp.eye(n, dtype=dtype)

    # Pairwise distances (km)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    h_km = cp.sqrt(dx * dx + dy * dy) / dtype(1000.0)

    # PGA only (T = 0)
    T = dtype(0.0)
    range_semiv = (dtype(40.7) - dtype(15.0) * T) if soil_case == 2 else (dtype(8.5) + dtype(17.2) * T)

    # Correlation model
    corr = cp.exp(-dtype(3.0) * h_km / range_semiv)
    return corr