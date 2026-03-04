#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Last updated : 02/25/2025
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 
This script is mostly based on the "ground_motion_models.py" by Raul Rincon at Rice University.
Major edits include:
+ GPU-compatible functions using CuPy.
+ Vectorized computation for faster processing.
+ Near Positive Semi-Definite (PSD) correction for covariance matrices.
"""
import cupy as cp

def compute_seismic_im_cupy_vec(assets_lat, assets_lon,
                                event_lat, event_lon,
                                Mw, depth=10000,
                                soil_type='bedrock', soil_var='clusters',
                                inter_residuals=None, intra_residuals=None,
                                indp_intra_residuals=False,
                                seed=12345):
    """
    Optimized version for Mw as a float64 scalar.
    """

    # Ensure inputs are CuPy arrays
    assets_lat = cp.asarray(assets_lat)
    assets_lon = cp.asarray(assets_lon)
    event_lat = cp.asarray(event_lat)
    event_lon = cp.asarray(event_lon)
    Mw = cp.asarray(Mw, dtype=cp.float64)
    depth = cp.asarray(depth, dtype=cp.float64)

    num_buildings = assets_lat.shape[0]
    num_simulation = inter_residuals.shape[0] if inter_residuals is not None else 1

    # Hypocentral Distance
    R_hyp = cp.sqrt((assets_lat - event_lat) ** 2 + (assets_lon - event_lon) ** 2 + depth ** 2) / 1000 # Unit: km
    # # For the 2nd-order phase transition, let us centralize the R_hyp
    # R_hyp = cp.mean(R_hyp) + cp.random.normal(0, 0.1, num_buildings)

    # Determine soil variation
    soil_case = 2 if soil_var == 'clusters' else 1

    # Compute Intra Residuals Covariance Matrix
    intra_residuals_cov_matrix = intra_residuals_corr_fn_cupy(assets_lat,
                                                             assets_lon,
                                                             soil_case,
                                                             indp_intra_residuals=indp_intra_residuals)

    # Preallocate result array (no iterate dimension)
    im_events_assets = cp.zeros((num_simulation, num_buildings), dtype=cp.float32)

    # Compute GMM parameters (only once as Mw is scalar)
    log_median_im, sigma_intra, sigma_inter, fact_soil = gmm_AB95_cupy_vec(Mw, R_hyp, soil_type)
        
    # Check which residuals to include
    include_inter_residuals = True
    include_intra_residuals = True
    if inter_residuals is None:
        include_inter_residuals = False
    if intra_residuals is None:
        include_intra_residuals = False
    
    # Expand the inter residuals over the region
    inter_residuals_unscaled = inter_residuals[:, 0][:, None] * cp.ones((num_buildings,))

    # Generate intra residuals based on the covariance matrix
    if indp_intra_residuals:
        intra_residuals_unscaled = intra_residuals
    else:
        intra_residuals_unscaled = residuals_generator_cupy(intra_residuals_cov_matrix,
                                                            num_simulation,
                                                            seed=seed)

    # Compute log IM components considering residuals
    log_inter = sigma_inter * inter_residuals_unscaled if include_inter_residuals else cp.zeros_like(inter_residuals_unscaled)
    log_intra = sigma_intra * intra_residuals_unscaled if include_intra_residuals else cp.zeros_like(intra_residuals_unscaled)

    # Final Intensity Measure calculation
    log_im = log_median_im + log_inter + log_intra
    # log_im = log_median_im
    im = 10 ** log_im
    im_events_assets = im * fact_soil / 980.7  # Convert to g

    # # Save results
    # im_events_assets = im_event

    return im_events_assets


def gmm_AB95_cupy_vec(Mw, R_hyp, soil_type='bedrock'):
    """
    Optimized GMM for scalar Mw and vectorized R_hyp.
    Mw: float64
    R_hyp: (n_assets,)
    """
    # Ensure R_hyp is a CuPy array
    R_hyp = cp.asarray(R_hyp)

    # Coefficients for PGA
    coef = cp.array([3.79, 0.298, -0.0536, 0.00135])  # PGA coefficients

    # Compute log median IM directly without broadcasting over Mw
    log_median_im = (coef[0] + coef[1] * (Mw - 6) +
                     coef[2] * cp.power((Mw - 6), 2) -
                     cp.log10(R_hyp + 1e-10) - coef[3] * R_hyp)

    # Intra-event and inter-event residuals
    sigma_intra = cp.full_like(log_median_im, 0.20)
    sigma_inter = cp.full_like(log_median_im, 0.18)

    # Soil amplification factor
    if soil_type not in ['bedrock', 'deep-soil']:
        raise ValueError("Invalid soil_type. Use 'bedrock' or 'deep-soil'")
    fact_soil = 1.0 if soil_type == 'bedrock' else 2 #0.93
    fact_soil = cp.full_like(log_median_im, fact_soil)

    return log_median_im, sigma_intra, sigma_inter, fact_soil


def intra_residuals_corr_fn_cupy(x_data, y_data, soil_case=2, indp_intra_residuals=False):
    if indp_intra_residuals:
        return cp.eye(len(x_data))
    else:
        # Ensure CuPy arrays
        x_data = cp.asarray(x_data)
        y_data = cp.asarray(y_data)

        # Compute pairwise distances using broadcasting
        dx = x_data[:, None] - x_data[None, :]
        dy = y_data[:, None] - y_data[None, :]
        H = cp.sqrt(dx ** 2 + dy ** 2)/1000 # Unit: km

        # Correlation model range
        Tunique = cp.array([0.0])  # Only PGA
        b1 = (40.7 - 15 * Tunique) if soil_case == 2 else (8.5 + 17.2 * Tunique)
        RangeSemiv = b1

        return cp.exp(-3 * H / RangeSemiv[0])
        

def intra_residuals_corr_fn_cupy_block(x_data,
                                 y_data,
                                 soil_case=2,
                                 indp_intra_residuals=False,
                                 block=2048,
                                 dtype=cp.float64):
    """
    Memory-safe intra-event residual correlation on GPU.
    Computes in (block x N) tiles; avoids full N×N intermediates.

    Parameters
    ----------
    x_data, y_data : array-like (cupy or numpy)
        Site coordinates in *meters* (e.g., UTM). Length N.
    soil_case : int
        2 → b1 = 40.7 - 15*T ; else → b1 = 8.5 + 17.2*T  (with T=0 for PGA)
    indp_intra_residuals : bool
        If True, return identity.
    block : int
        Row tile size; lower if you still hit OOM (e.g., 1024/512).
    dtype : cupy dtype
        Usually cp.float64. Change it to cp.float32 if the OOM issue arises.

    Returns
    -------
    R : (N, N) cp.ndarray[dtype]
        Correlation matrix with ones on the diagonal.
    """
    # Ensure CuPy arrays (float32)
    x = cp.asarray(x_data, dtype=dtype).ravel()
    y = cp.asarray(y_data, dtype=dtype).ravel()
    N = x.size

    if indp_intra_residuals:
        return cp.eye(N, dtype=dtype)

    # --- Your original range model (PGA only: T=0) ---
    T = 0.0
    b1 = (40.7 - 15.0 * T) if soil_case == 2 else (8.5 + 17.2 * T)
    range_semiv = dtype(b1)  # scalar

    R = cp.empty((N, N), dtype=dtype)

    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)

        xb = x[i0:i1][:, None]   # (B,1)
        yb = y[i0:i1][:, None]   # (B,1)

        # Distance in km, no separate dx/dy squares allocated
        H = cp.hypot(xb - x[None, :], yb - y[None, :]) * dtype(1e-3)

        # Your correlation formula: exp(-3 * H / RangeSemiv)
        R_block = cp.exp((-3.0 / range_semiv) * H).astype(dtype, copy=False)

        R[i0:i1, :] = R_block

        # Cleanup tile (helps on tight VRAM)
        del xb, yb, H, R_block
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()

    # Exact ones on the diagonal
    cp.fill_diagonal(R, dtype(1.0))
    return R
    

def residuals_generator_cupy(intra_residuals_cov_matrix, num_simulation_per_Mw, seed=1234):
    """
    Generate residuals using CuPy for GPU acceleration (vectorized).

    Args:
        intra_residuals_cov_matrix (cp.ndarray): Covariance matrix (GPU array).
        num_simulation_per_Mw (int): Number of simulations.
        seed (int): Random seed for reproducibility.

    Returns:
        inter_residuals_unscaled (cp.ndarray): Inter-event residuals.
        intra_residuals_unscaled (cp.ndarray): Intra-event residuals.
    """
    # Set random seed for reproducibility
    rng = cp.random.RandomState(seed)

    # Number of points from the covariance matrix
    num_points = intra_residuals_cov_matrix.shape[0]
    mean_intra = cp.zeros(num_points)

    # Vectorized multivariate normal sampling
    try:
        # Cholesky decomposition (vectorized for all samples)
        L = cp.linalg.cholesky(intra_residuals_cov_matrix)

        # Generate standard normal samples (vectorized)
        z = rng.standard_normal((num_simulation_per_Mw, num_points))

        # Apply transformation: mean + L @ z.T (vectorized)
        intra_residuals_unscaled = cp.matmul(z, L.T) + mean_intra

    except cp.linalg.LinAlgError as e:
        print(f"Cholesky failed: {e}. Applying PSD correction...")
        
        cov_matrix_psd = nearcorr_clip_cupy(intra_residuals_cov_matrix)
        L = cp.linalg.cholesky(cov_matrix_psd)

        # Retry multivariate sampling
        z = rng.standard_normal((num_simulation_per_Mw, num_points))
        intra_residuals_unscaled = cp.matmul(z, L.T) + mean_intra

    return intra_residuals_unscaled


import numpy as np
import pandas as pd
from pygmm.model import Scenario
from pygmm.chiou_youngs_2014 import ChiouYoungs2014

def gmm_CY14(assets: pd.DataFrame, event_xy: tuple, eq_info: dict):

    strike_rad  = np.deg2rad(eq_info['strike'])
    strike_vec  = np.array([np.cos(strike_rad), np.sin(strike_rad)])    # unit vector in strike direction
    normal_vec  = np.array([-np.sin(strike_rad), np.cos(strike_rad)])   # unit vector perpendicular to strike

    def rupture_length(mw):   # Wells & Coppersmith 1994
        return 10 ** (-2.44 + 0.59 * mw) * 1000 # m

    def rupture_width(mw):
        return 10 ** (-1.01 + 0.32 * mw) * 1000 #m

    L = eq_info['L'] * 1000 if 'L' in eq_info else rupture_length(eq_info['Mw'])  # m
    W = eq_info['W'] * 1000 if 'W' in eq_info else rupture_width(eq_info['Mw'])  # m

    depth_top       = eq_info['depth_to_top'] * 1000  # m
    depth_bottom    = depth_top + W    # Assuming a vertically planar fault
    depth_center    = (depth_top + depth_bottom) / 2
    if 'hypo_down_dip' in eq_info:
        depth_hypo      = depth_top + eq_info['hypo_down_dip'] * 1000

    ln_pga_mean   = np.empty(len(assets))
    ln_pga_std = np.empty(len(assets))

    event_x, event_y = event_xy

    for i, row in assets.iterrows():
        rel_xy = np.array([
            float(row.x) - float(event_x),
            float(row.y) - float(event_y)
        ], dtype=np.float64)

        proj_strike  = rel_xy @ strike_vec
        proj_normal  = rel_xy @ normal_vec

        if abs(proj_strike) <= L / 2:
            horiz_dist = abs(proj_normal)
        else:
            delta_strike = abs(proj_strike) - L / 2
            horiz_dist = np.sqrt(delta_strike**2 + proj_normal**2)

        dist_jb  = horiz_dist / 1000   # km
        dist_x   = proj_normal / 1000  # km
        dist_rup = np.sqrt(horiz_dist**2 + depth_center**2) / 1000  # km

        sc = Scenario(
            mag         = eq_info['Mw'],
            dist_rup    = dist_rup,
            dist_x      = dist_x,
            dist_jb     = dist_jb,
            v_s30       = row.Vs30,
            dip         = eq_info['dip'],
            depth_tor   = depth_top / 1000, # km
            depth_1_0   = None,
            dpp_centered= False,
            mechanism   = eq_info['mechanism'],
            region      = eq_info['region'],
            on_hanging_wall = False if eq_info['dip'] == 90 else eq_info['on_hanging_wall'],
            vs_source   = eq_info['vs_source']
        )

        cy14 = ChiouYoungs2014(sc)
        ln_pga_mean[i]   = cy14._ln_resp[cy14.INDEX_PGA]
        ln_pga_std[i] = cy14._ln_std[cy14.INDEX_PGA]

    return ln_pga_mean, ln_pga_std
    
    
from pygmm.abrahamson_silva_kamai_2014 import AbrahamsonSilvaKamai2014
def gmm_ASK14(assets: pd.DataFrame, event_xy: tuple, eq_info: dict):

    strike_rad  = np.deg2rad(eq_info['strike'])
    strike_vec  = np.array([np.cos(strike_rad), np.sin(strike_rad)])    # unit vector in strike direction
    normal_vec  = np.array([-np.sin(strike_rad), np.cos(strike_rad)])   # unit vector perpendicular to strike

    def rupture_length(mw):   # Wells & Coppersmith 1994
        return 10 ** (-2.44 + 0.59 * mw) * 1000 # m

    L = eq_info['L'] * 1000 if 'L' in eq_info else rupture_length(eq_info['Mw'])  # m

    depth_top       = eq_info['depth_to_top'] * 1000  # m

    ln_pga_mean   = np.empty(len(assets))
    ln_pga_std = np.empty(len(assets))

    event_x, event_y = event_xy

    for i, row in assets.iterrows():
        rel_xy = np.array([
            float(row.x) - float(event_x),
            float(row.y) - float(event_y)
        ], dtype=np.float64)

        proj_strike  = rel_xy @ strike_vec
        proj_normal  = rel_xy @ normal_vec

        if abs(proj_strike) <= L / 2:
            horiz_dist = abs(proj_normal)
        else:
            delta_strike = abs(proj_strike) - L / 2
            horiz_dist = np.sqrt(delta_strike**2 + proj_normal**2)

        dist_jb  = horiz_dist / 1000   # km
        dist_x   = proj_normal / 1000  # km
        dist_rup = np.sqrt(horiz_dist**2 + depth_top**2) / 1000.0

        z1p0 = AbrahamsonSilvaKamai2014.calc_depth_1_0(row.Vs30, eq_info['region'])
        sc = Scenario(
            mag         = eq_info['Mw'],
            dist_rup    = dist_rup,
            dist_x      = dist_x,
            dist_jb     = dist_jb,
            depth_1_0   = z1p0,
            v_s30       = row.Vs30,
            dip         = eq_info['dip'],
            depth_tor   = depth_top / 1000,
            mechanism   = eq_info['mechanism'],
            region      = eq_info['region'],
            on_hanging_wall = eq_info['on_hanging_wall'],
            vs_source   = eq_info['vs_source'],
        )

        ask14 = AbrahamsonSilvaKamai2014(sc)
        ln_pga_mean[i]  = ask14._ln_resp[ask14.INDEX_PGA]
        ln_pga_std[i]   = ask14._ln_std[ask14.INDEX_PGA]

    return ln_pga_mean, ln_pga_std


from pygmm.boore_stewart_seyhan_atkinson_2014 import BooreStewartSeyhanAtkinson2014
def gmm_BSSA14(assets: pd.DataFrame, event_xy: tuple, eq_info: dict):

    strike_rad  = np.deg2rad(eq_info['strike'])
    strike_vec  = np.array([np.cos(strike_rad), np.sin(strike_rad)])    # unit vector in strike direction
    normal_vec  = np.array([-np.sin(strike_rad), np.cos(strike_rad)])   # unit vector perpendicular to strike

    def rupture_length(mw):   # Wells & Coppersmith 1994
        return 10 ** (-2.44 + 0.59 * mw) * 1000 # m

    L = eq_info['L'] * 1000 if 'L' in eq_info else rupture_length(eq_info['Mw'])  # m

    ln_pga_mean   = np.empty(len(assets))
    ln_pga_std = np.empty(len(assets))

    event_x, event_y = event_xy

    for i, row in assets.iterrows():
        rel_xy = np.array([
            float(row.x) - float(event_x),
            float(row.y) - float(event_y)
        ], dtype=np.float64)

        proj_strike  = rel_xy @ strike_vec
        proj_normal  = rel_xy @ normal_vec

        if abs(proj_strike) <= L / 2:
            horiz_dist = abs(proj_normal)
        else:
            delta_strike = abs(proj_strike) - L / 2
            horiz_dist = np.sqrt(delta_strike**2 + proj_normal**2)

        dist_jb  = horiz_dist / 1000   # km

        z1p0 = ChiouYoungs2014.calc_depth_1_0(row.Vs30, eq_info["region"])

        sc = Scenario(
            mag         = eq_info['Mw'],
            dist_jb     = dist_jb,
            depth_1_0   = z1p0,
            v_s30       = row.Vs30,
            mechanism   = eq_info['mechanism'],
            region      = eq_info['region'],
        )

        bssa14 = BooreStewartSeyhanAtkinson2014(sc)
        ln_pga_mean[i]  = bssa14._ln_resp[bssa14.INDEX_PGA]
        ln_pga_std[i]   = bssa14._ln_std[bssa14.INDEX_PGA]

    return ln_pga_mean, ln_pga_std


from pygmm.campbell_bozorgnia_2014 import CampbellBozorgnia2014
def gmm_CB14(assets: pd.DataFrame, event_xy: tuple, eq_info: dict):

    strike_rad  = np.deg2rad(eq_info['strike'])
    strike_vec  = np.array([np.cos(strike_rad), np.sin(strike_rad)])    # unit vector in strike direction
    normal_vec  = np.array([-np.sin(strike_rad), np.cos(strike_rad)])   # unit vector perpendicular to strike

    def rupture_length(mw):   # Wells & Coppersmith 1994
        return 10 ** (-2.44 + 0.59 * mw) * 1000 # m

    L = eq_info['L'] * 1000 if 'L' in eq_info else rupture_length(eq_info['Mw'])  # m

    depth_top       = eq_info['depth_to_top'] * 1000  # m

    ln_pga_mean   = np.empty(len(assets))
    ln_pga_std = np.empty(len(assets))

    event_x, event_y = event_xy

    for i, row in assets.iterrows():
        rel_xy = np.array([
            float(row.x) - float(event_x),
            float(row.y) - float(event_y)
        ], dtype=np.float64)

        proj_strike  = rel_xy @ strike_vec
        proj_normal  = rel_xy @ normal_vec

        if abs(proj_strike) <= L / 2:
            horiz_dist = abs(proj_normal)
        else:
            delta_strike = abs(proj_strike) - L / 2
            horiz_dist = np.sqrt(delta_strike**2 + proj_normal**2)

        dist_jb = horiz_dist / 1000.0           # km
        dist_x  = proj_normal / 1000.0          # km (signed)
        dist_rup = np.sqrt(horiz_dist**2 + depth_top**2) / 1000.0  # km

        sc = Scenario(
            mag       = eq_info["Mw"],
            dist_jb   = dist_jb,
            dist_rup  = dist_rup,
            dist_x    = dist_x,
            v_s30     = row.Vs30,
            dip       = eq_info["dip"],
            depth_tor = eq_info["depth_to_top"],  # km
            region    = eq_info["region"],        # 'california', 'japan', ...
            mechanism = eq_info["mechanism"],     # 'SS', 'NS', 'RS'
        )

        cb14 = CampbellBozorgnia2014(sc)
        ln_pga_mean[i]  = cb14._ln_resp[cb14.INDEX_PGA]
        ln_pga_std[i]   = cb14._ln_std[cb14.INDEX_PGA]

    return ln_pga_mean, ln_pga_std
    

#####################################################################################
def nearcorr_clip_cupy(A):
    """
    Find the nearest Positive Semi-Definite (PSD) matrix using CuPy.
    Automatically converts NumPy array to CuPy array if needed.    
    """
    # Ensure input is a CuPy array
    A = cp.asarray(A)

    # Step 1: Symmetrize the matrix (vectorized)
    A_sym = 0.5 * (A + A.T)

    # Step 2: Eigenvalue decomposition on GPU
    eigvals, eigvecs = cp.linalg.eigh(A_sym)

    # Step 3: Clip negative eigenvalues (in-place for memory efficiency)
    eigvals = cp.clip(eigvals, 0, None)

    # Step 4: Reconstruct PSD matrix (vectorized matrix multiplication)
    # Using broadcasting to avoid creating intermediate diagonal matrices
    A_psd = (eigvecs * eigvals) @ eigvecs.T

    # Step 5: Ensure symmetry again (due to numerical errors)
    A_psd = 0.5 * (A_psd + A_psd.T)

    return A_psd


def nearest_psd_cupy(A, epsilon=1e-8):
    # Accurate but slower method for near PSD correction
    # Once warmed up,
    # e.g., ~10s for a 10,000 x 10,000 matrix
    # e.g., ~100s for a 15,000 x 15,000 matrix

    A = cp.asarray(A)

    A_sym = (A + A.T) / 2
    eigvals, eigvecs = cp.linalg.eigh(A_sym)
    eigvals_clipped = cp.clip(eigvals, epsilon, None)
    A_psd = eigvecs @ cp.diag(eigvals_clipped) @ eigvecs.T
    return (A_psd + A_psd.T) / 2

