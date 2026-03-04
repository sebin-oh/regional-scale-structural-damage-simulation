#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import cupy as cp
import utm
import time, os, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from fn_GMPE_parallel_gpu_vec import (
    gmm_CY14,
    intra_residuals_corr_fn_cupy,
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "cm"

# %%
# ---------- Config ---------- 
target_region     = "Milpitas_all"
target_structure  = "MultiStory"  # "SingleStory", "TwoStory", "MultiStory", "All"

covs              = np.round(np.arange(0.0, 1.001, 0.01), 3)
Mws               = np.round(np.arange(8.50, 8.51, 0.05), 2)

# covs              = np.round(np.array([0.0, 1.0]), 3)
# Mws               = np.round(np.array([7.25]), 2) 

dtype_np          = np.float64
dtype_cp          = cp.float64

# Sampling settings
gmm               = "CY14"
num_sim           = 10_000

# %%
# ---------- Utilities ----------
def free_gpu():
    """Release cached blocks so memory is reusable between iterations."""
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

# %%
# ---------- Load capacities ----------
bldg_info  = pd.read_csv(f"Savio/{target_region}/IDA_results_OpenSeesPy_cov000.csv")
bldg_capa  = bldg_info.to_numpy()[:, 1:].astype(dtype_np, copy=False).T  # (num_bldgs, num_gms)
del bldg_info
free_gpu()

idx_nan = np.where(np.any(np.isnan(bldg_capa), axis=1))[0]
if idx_nan.size > 0:
    keep = np.ones(bldg_capa.shape[0], dtype=bool); keep[idx_nan] = False
    bldg_capa = bldg_capa[keep]

region_info = pd.read_csv(f"TargetRegion_{target_region}.csv")
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

num_bldgs, num_gms = bldg_capa.shape

# %%
# ---------- Estimate baseline fragility parameters ----------
bldg_frag_params = np.zeros((num_bldgs, 3), dtype=dtype_np)
for i in range(num_bldgs):
    temp = bldg_capa[i]
    temp = temp[(~np.isnan(temp)) & (temp != 0)]
    # If too few samples, fallback tiny positive value
    if temp.size < 2:
        temp = np.array([1e-6, 2e-6], dtype=dtype_np)
    shape, loc, scale = lognorm.fit(temp, floc=0)
    bldg_frag_params[i, :] = [dtype_np(shape), dtype_np(loc), dtype_np(scale)]
del temp
free_gpu()

# %%
# ---------- Capacity stats ----------
ln_capa_mean_np     = np.log(bldg_frag_params[:, 2].astype(dtype_np))
ln_capa_std_np      = bldg_frag_params[:, 0].astype(dtype_np)
ln_capa_corr_np     = np.corrcoef(bldg_capa.astype(dtype_np), rowvar=True).astype(dtype_np, copy=False)
ln_capa_cov_np      = np.outer(ln_capa_std_np, ln_capa_std_np) * ln_capa_corr_np
del bldg_capa
free_gpu()

ln_capa_mean_cp     = cp.asarray(ln_capa_mean_np, dtype=dtype_cp)
ln_capa_std_cp      = cp.asarray(ln_capa_std_np, dtype=dtype_cp)
ln_capa_corr_cp     = cp.asarray(ln_capa_corr_np, dtype=dtype_cp)
ln_capa_cov_cp      = cp.asarray(ln_capa_cov_np, dtype=dtype_cp)

# %%
# ---------- Site info ----------
easting, northing, _, _ = utm.from_latlon(region_info['Latitude'].values,
                                          region_info['Longitude'].values)
easting  = easting.astype(dtype_np); northing = northing.astype(dtype_np)

vs30 = region_info['Vs30'].to_numpy().astype(dtype_np)
vs30[vs30 < 180] = 180

assets = pd.DataFrame({'x': easting, 'y': northing, 'Vs30': vs30})
del easting, northing, vs30

repair_cost = cp.asarray(region_info['RepairCost'].to_numpy().astype(dtype_np), dtype=dtype_cp)

# %%
# ---------- Hazard info ----------
# Epicenter UTM
epicenter = (37.666, -122.076)  # (lat, lon)
event_e, event_n, _, _ = utm.from_latlon(*epicenter)
event_xy = (dtype_np(event_e), dtype_np(event_n))

depth_to_top = dtype_np(3.0)
strike, rake, dip = 325, 180, 90

# %%
# ---------- Demand stats ----------
soil_case = 1 
ln_dmnd_corr_cp = intra_residuals_corr_fn_cupy(
    cp.asarray(assets['x'].to_numpy(), dtype=dtype_cp),
    cp.asarray(assets['y'].to_numpy(), dtype=dtype_cp),
    soil_case=soil_case
).astype(dtype_cp)

# %%
# ---------- Output directory ----------
outdir = f"Savio/{target_region}/results_for_[sim_total]"
os.makedirs(outdir, exist_ok=True)

t0_all = time.time()

for Mw_val in Mws:
    print(f"\n=== Mw = {Mw_val:.2f} ===")
    t0 = time.time()

    Mw_label = f"{int(round(Mw_val*100)):03d}"  # e.g., 556 for Mw=5.56

    # GMPE mean/std for this Mw
    eq_info = dict(
        depth_to_top     = depth_to_top,
        strike           = strike,
        rake             = rake,
        dip              = dip,
        mechanism        = 'SS',
        region           = 'california',
        on_hanging_wall  = False,
        vs_source        = 'inferred',
        Mw               = float(Mw_val)   # ensure scalar, not array
    )
    ln_dmnd_mean_np, ln_dmnd_std_np = gmm_CY14(assets, event_xy, eq_info)  # numpy arrays

    # Move to GPU
    ln_dmnd_mean_cp = cp.asarray(ln_dmnd_mean_np, dtype=dtype_cp)
    ln_dmnd_std_cp  = cp.asarray(ln_dmnd_std_np,  dtype=dtype_cp)

    # Demand covariance for this Mw
    ln_dmnd_cov_cp = cp.outer(ln_dmnd_std_cp, ln_dmnd_std_cp) * ln_dmnd_corr_cp

    # Total covariance of X = ln(C) - ln(D): Var(X) = Var(C) + Var(D)
    X_cov = ln_dmnd_cov_cp + ln_capa_cov_cp

    # Prepare output array for all covs
    dv      = cp.zeros((num_sim,), dtype=dtype_cp)
    dv_cost = cp.zeros((num_sim,), dtype=dtype_cp)

    for cov_idx, cov in enumerate(covs):
        t_cov = time.time()

        cov_label = f"{int(round(cov*1000)):04d}"
        bldg_frag_params  = np.load(f"Savio/{target_region}/bldg_frag_params_for_[sim_total]/bldg_frag_params_cov{cov_label}.npy")
        bldg_frag_params = bldg_frag_params[idx_target, :]

        # Capacity stats (per-building)
        ln_capa_mean_cp = cp.asarray(np.log(bldg_frag_params[:, 2].astype(dtype_np)), dtype=dtype_cp)
        X_mean = ln_capa_mean_cp - ln_dmnd_mean_cp
        X_samples = cp.random.multivariate_normal(X_mean, X_cov, num_sim)

        nan_num = 0

        # ds_all = cp.zeros((num_bldgs, num_sim), dtype=dtype_cp)
        for j in range(num_sim):
            valid = ~np.isnan(X_samples[j])
            nan_num += np.sum(~valid)

            ds = X_samples[j, valid] < 0

            # ds_all[:,j] = ds

        # cov_label = f"{int(round(cov*100)):03d}"
        # outfile_ds_npy = os.path.join(outdir, f"ds/{target_structure}_ds_Mw{Mw_label}_cov{cov_label}_{target_region}.npy")
        # np.save(outfile_ds_npy, cp.asnumpy(ds_all))

            dv[j] = ds.mean()
            dv_cost[j] = (repair_cost[valid] * ds).sum()

        if nan_num > 0.01*X_samples.size:
            print(f"Warning: Too many NaNs ({nan_num}) for Mw={Mw_val:.2f}, cov={cov:.3f}.")

        if cov_idx % 25 == 0:
            print(f"  cov={cov:.3f}: {time.time()-t0:.2f}s")

        dv_np       = cp.asnumpy(dv)
        dv_cost_np  = cp.asnumpy(dv_cost)

        outfile_dv_npy = os.path.join(outdir, f"dv/{target_structure}_dv_Mw{Mw_label}_cov{cov_label}_{target_region}.npy")
        outfile_dv_cost_npy = os.path.join(outdir, f"dv_cost/{target_structure}_dv_cost_Mw{Mw_label}_cov{cov_label}_{target_region}.npy")
        
        np.save(outfile_dv_npy, dv_np)
        np.save(outfile_dv_cost_npy, dv_cost_np)
    
    del ln_dmnd_mean_cp, ln_dmnd_std_cp, ln_dmnd_cov_cp, X_cov #, dv, dv_cost, dv_np, dv_cost_np
    free_gpu()

    print(f"=== Mw {Mw_val:.2f} done in {time.time()-t0:.2f}s ===")

print(f"\nAll done in {time.time()-t0_all:.2f}s")






# ## Prepare the bldg_frag_params for different c.o.v values
# bldg_info  = pd.read_csv(f"Savio/{target_region}/IDA_results_OpenSeesPy_cov000.csv")
# bldg_capa  = bldg_info.to_numpy()[:, 1:].astype(dtype_np, copy=False).T  # (num_bldgs, num_gms)

# # drop rows with any NaN (track dropped indices for later alignment)
# idx_nan = np.where(np.any(np.isnan(bldg_capa), axis=1))[0]
# if idx_nan.size > 0:
#     keep = np.ones(bldg_capa.shape[0], dtype=bool); keep[idx_nan] = False
#     bldg_capa = bldg_capa[keep]

# num_bldgs = len(bldg_capa)

# bldg_frag_params = np.zeros((num_bldgs, 3), dtype=dtype_np)
# for i in range(num_bldgs):
#     temp = bldg_capa[i]
#     temp = temp[(~np.isnan(temp)) & (temp != 0)]
#     if temp.size < 2:
#         temp = np.array([1e-6, 2e-6], dtype=dtype_np)
#     shape, loc, scale = lognorm.fit(temp, floc=0)
#     bldg_frag_params[i, :] = [dtype_np(shape), dtype_np(loc), dtype_np(scale)]

# covs = np.round(np.arange(0.0, 1.001, 0.01), 3)

# for cov_idx, cov in enumerate(covs):
#     print(f"{cov_idx}/{len(covs)}")

#     cov_label = f"{int(round(cov*1000)):04d}"    
#     bldg_frag_params_temp = bldg_frag_params.copy()

#     noise = np.random.lognormal(mean=0.0, sigma=float(cov), size=bldg_frag_params_temp.shape[0])
#     bldg_frag_params_temp[:, 2] *= noise

#     np.save(f"Savio/{target_region}/bldg_frag_params_for_[sim_total]/bldg_frag_params_cov{cov_label}.npy", bldg_frag_params_temp)
# %% Expand "ln_dmnd_mean_cp" to a matrix (repeat it 10,000 times)
# ln_dmnd_mean_matrix = cp.tile(ln_dmnd_mean_cp, (num_sim, 1))  # (num_sim, num_bldgs)
