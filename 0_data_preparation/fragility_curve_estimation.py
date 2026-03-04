#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 07/20/2026
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm

TARGET_REGION = "Milpitas"
IDA_CSV = Path(f"../data/IDA_results/{TARGET_REGION}/IDA_results_sigma000.csv")

DTYPE_NP = np.float64

bldg_capa = pd.read_csv(IDA_CSV)
bldg_capa = bldg_capa.to_numpy()[:, 1:].astype(DTYPE_NP, copy=False).T


num_bldgs = bldg_capa.shape[0]
frag_params = np.empty((num_bldgs, 3), dtype=DTYPE_NP)

for i in range(num_bldgs):
    temp = bldg_capa[i]
    temp = temp[np.isfinite(temp) & (temp > 0)]

    # Robust fallback if too few samples
    if temp.size < 2:
        temp = np.array([1e-6, 2e-6], dtype=DTYPE_NP)

    shape, loc, scale = lognorm.fit(temp, floc=0)
    frag_params[i] = (shape, loc, scale)


## Prepare the frag_params for different c.o.v values
bldg_info  = pd.read_csv(f"../data/IDA_results/IDA_results_sigma000.csv")
bldg_capa  = bldg_info.to_numpy()[:, 1:].astype(DTYPE_NP, copy=False).T  # (num_bldgs, num_gms)

# drop rows with any NaN (track dropped indices for later alignment)
idx_nan = np.where(np.any(np.isnan(bldg_capa), axis=1))[0]
if idx_nan.size > 0:
    keep = np.ones(bldg_capa.shape[0], dtype=bool); keep[idx_nan] = False
    bldg_capa = bldg_capa[keep]

num_bldgs = len(bldg_capa)

frag_params = np.zeros((num_bldgs, 3), dtype=DTYPE_NP)
for i in range(num_bldgs):
    temp = bldg_capa[i]
    temp = temp[(~np.isnan(temp)) & (temp != 0)]
    if temp.size < 2:
        temp = np.array([1e-6, 2e-6], dtype=DTYPE_NP)
    shape, loc, scale = lognorm.fit(temp, floc=0)
    frag_params[i, :] = [DTYPE_NP(shape), DTYPE_NP(loc), DTYPE_NP(scale)]

sigmas = np.round(np.arange(0.0, 1.001, 0.01), 3)

for sigma_idx, sigma in enumerate(sigmas):
    print(f"{sigma_idx}/{len(sigmas)}")

    sigma_lab = f"{int(round(sigma*100)):03d}"    
    frag_params_temp = frag_params.copy()

    noise = np.random.lognormal(mean=0.0, sigma=float(sigma), size=frag_params_temp.shape[0])
    frag_params_temp[:, 2] *= noise

    np.save(f"../data/fragility_params/{TARGET_REGION}/frag_params_sigma{sigma_lab}.npy", frag_params_temp)