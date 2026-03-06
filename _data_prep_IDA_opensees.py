#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 07/03/2026
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

Incremental Dynamic Analysis (IDA) using OpenSeesPy.

For each ground motion and each building in a BIM CSV:
  - Find the smallest scale factor such that max interstory drift ratio > DS threshold.
  - Record collapse PGA = (scale * PGA_of_unscaled_GM) in units of g.

"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import openseespy.opensees as ops


# =============================================================================
# Settings
# =============================================================================
MAX_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

REGION = "Milpitas"
DAMAGE_STATE = "DS1"  # "DS1", "DS2", "DS3", "DS4"

NUM_GMS = 100
SIGMAS = np.array([0.0], dtype=float)  # e.g., np.round(np.arange(0.0, 0.051, 0.01), 2)

SCALES = np.arange(0.1, 20.0, 0.1)  # scale factors for binary search
G_TO_MPS2 = 9.811

# =============================================================================
# Paths (relative to this script)
# =============================================================================
# ROOT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# BIM_DIR = ROOT_DIR / "OpenSees_input" / REGION
# GM_DIR = ROOT_DIR / "Ground_motion"
# OUT_DIR = ROOT_DIR / "Results" / REGION

BIM_DIR = Path(f"./data/OpenSees_input/{REGION}" )
GM_DIR = Path("./data/Ground_motion")
OUT_DIR = Path(f"./data/IDA_results/{REGION}")


# =============================================================================
# CSV parsing utilities
# =============================================================================
def parse_csv_array(val) -> np.ndarray:
    """Parse comma-separated numeric strings into a float array (robust to NaNs, lists, etc.)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.array([], dtype=float)

    if isinstance(val, np.ndarray):
        return val.astype(float, copy=False)

    if isinstance(val, list):
        return np.asarray(val, dtype=float)

    if isinstance(val, (int, float, np.integer, np.floating)):
        return np.array([float(val)], dtype=float)

    s = str(val).strip()
    if s.lower() in ("", "nan", "none"):
        return np.array([], dtype=float)
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()

    parts = [p.strip() for p in s.split(",") if p.strip()]
    return np.array([float(p) for p in parts], dtype=float)


def load_bim_arrays(bim_csv: Path) -> Dict[str, object]:
    """
    Load BIM CSV and return arrays needed for analysis.
    Arrays ys/yd/us/ud are object arrays containing numpy arrays per building.
    """
    cols_vec = ["ys", "yd", "us", "ud"]

    df = pd.read_csv(
        bim_csv,
        converters={c: parse_csv_array for c in cols_vec},
    )

    # Basic scalar columns
    out: Dict[str, object] = {
        "ds_idr": df[f"{DAMAGE_STATE}_idr"].to_numpy(dtype=float),
        "NS": df["NS"].to_numpy(dtype=int),
        "story_height": df["story_height"].to_numpy(dtype=float),
        "mass": df["mass"].to_numpy(dtype=float),
        "pinch": df["pinch"].to_numpy(dtype=float),
        "beta": df["beta"].to_numpy(dtype=float),
        # Per-story vectors as object arrays
        "ys": df["ys"].to_numpy(dtype=object),
        "yd": df["yd"].to_numpy(dtype=object),
        "us": df["us"].to_numpy(dtype=object),
        "ud": df["ud"].to_numpy(dtype=object),
        "n_bldgs": len(df),
    }
    return out


# =============================================================================
# OpenSees model + analysis
# =============================================================================
def build_model_max_idr(
    NS: int,
    mass: float,
    story_height: float,
    ys: np.ndarray,
    yd: np.ndarray,
    us: np.ndarray,
    ud: np.ndarray,
    pinch: float,
    beta: float,
    *,
    dt: float,
    n_steps: int,
    gm_file: Path,
    scale: float,
) -> float:
    """
    Build a shear-type multi-story model (zeroLength + Hysteretic) and run one transient analysis.

    Returns
    -------
    float
        Maximum interstory drift ratio across all stories and time steps.
        Returns np.inf if analysis fails (treated as collapse).
    """
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # Nodes / masses / fixities
    # Node tags: 11, 21, 31, ... (10*(j+1)+1)
    for j in range(NS + 1):
        ops.node(10 * (j + 1) + 1, 0.0, 0.0)

    for j in range(NS):
        ops.mass(10 * (j + 2) + 1, mass, mass, 0.0)

    for j in range(NS + 1):
        if j == 0:
            ops.fix(10 * (j + 1) + 1, 1, 1, 1)
        else:
            ops.fix(10 * (j + 1) + 1, 0, 0, 1)

    # Materials
    for j in range(NS):
        mat_tag = 10 * (j + 1) + 1
        ops.uniaxialMaterial(
            "Hysteretic",
            mat_tag,
            ys[j], yd[j], us[j], ud[j], us[j], 1,
            -ys[j], -yd[j], -us[j], -ud[j], -us[j], -1,
            pinch, pinch, 0.0, 0.0, beta,
        )

    # Elements
    for j in range(NS):
        n1 = 10 * (j + 1) + 1
        n2 = 10 * (j + 2) + 1
        ele_tag = 10 + j
        mat_tag = 10 * (j + 1) + 1
        ops.element("zeroLength", ele_tag, n1, n2, "-mat", mat_tag, mat_tag, "-dir", 1, 2)

    # Ground motion (file-based Path timeSeries, scaled by factor)
    ts_tag = 1
    ops.timeSeries(
        "Path",
        ts_tag,
        "-dt",
        dt,
        "-filePath",
        str(gm_file),
        "-factor",
        float(scale * G_TO_MPS2),
    )
    ops.pattern("UniformExcitation", 1, 1, "-accel", ts_tag)

    # Analysis options
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.algorithm("Newton")
    ops.analysis("Transient")

    max_idr = 0.0

    # Precompute node tags for drift checks
    tops = [10 * (j + 2) + 1 for j in range(NS)]
    bots = [10 * (j + 1) + 1 for j in range(NS)]

    for _ in range(n_steps):
        if ops.analyze(1, dt) != 0:
            return float("inf")

        # Update peak drift across stories at this step
        for top, bot in zip(tops, bots):
            drift = abs(ops.nodeDisp(top, 1) - ops.nodeDisp(bot, 1)) / story_height
            if drift > max_idr:
                max_idr = drift

    return float(max_idr)


def binary_search_collapse_pga(
    ii: int,
    *,
    ds_idr: float,
    NS: int,
    story_height: float,
    mass: float,
    ys: np.ndarray,
    yd: np.ndarray,
    us: np.ndarray,
    ud: np.ndarray,
    pinch: float,
    beta: float,
    dt: float,
    n_steps: int,
    gm_file: Path,
    gm_peak_g: float,
    scales: np.ndarray,
) -> float:
    """
    Find smallest scale such that max_idr > ds_idr.
    Return collapse PGA in g (scale * gm_peak_g). Returns NaN if not found in scale range.
    """
    lo, hi = 0, len(scales) - 1
    collapse_pga_g = np.nan

    while lo <= hi:
        mid = (lo + hi) // 2
        scale = float(scales[mid])

        max_idr = build_model_max_idr(
            NS, mass, story_height, ys, yd, us, ud, pinch, beta,
            dt=dt, n_steps=n_steps, gm_file=gm_file, scale=scale,
        )

        if max_idr > ds_idr:
            collapse_pga_g = scale * gm_peak_g
            hi = mid - 1
        else:
            lo = mid + 1

    return float(collapse_pga_g)


# =============================================================================
# Worker globals + entrypoint
# =============================================================================
_BIM = {}
_GM_FILE: Path
_DT: float
_N_STEPS: int
_GM_PEAK_G: float


def _init_worker(bim_csv: str, gm_file: str, dt: float, n_steps: int, gm_peak_g: float) -> None:
    global _BIM, _GM_FILE, _DT, _N_STEPS, _GM_PEAK_G
    _BIM = load_bim_arrays(Path(bim_csv))
    _GM_FILE = Path(gm_file)
    _DT = float(dt)
    _N_STEPS = int(n_steps)
    _GM_PEAK_G = float(gm_peak_g)


def run_one_building(ii: int) -> Tuple[int, float]:
    """Worker: compute collapse PGA (g) for building ii."""
    ds_idr = float(_BIM["ds_idr"][ii])
    NS = int(_BIM["NS"][ii])
    story_height = float(_BIM["story_height"][ii])
    mass = float(_BIM["mass"][ii])

    ys = _BIM["ys"][ii]
    yd = _BIM["yd"][ii]
    us = _BIM["us"][ii]
    ud = _BIM["ud"][ii]

    pinch = float(_BIM["pinch"][ii])
    beta = float(_BIM["beta"][ii])

    pga_g = binary_search_collapse_pga(
        ii,
        ds_idr=ds_idr,
        NS=NS,
        story_height=story_height,
        mass=mass,
        ys=ys,
        yd=yd,
        us=us,
        ud=ud,
        pinch=pinch,
        beta=beta,
        dt=_DT,
        n_steps=_N_STEPS,
        gm_file=_GM_FILE,
        gm_peak_g=_GM_PEAK_G,
        scales=SCALES,
    )
    return ii, pga_g


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    for sigma in np.asarray(SIGMAS, dtype=float):
        sigma_label = f"{int(round(sigma * 100)):03d}"
        bim_csv = (BIM_DIR / f"BIM_sigma{sigma_label}.csv").resolve()

        if not bim_csv.exists():
            raise FileNotFoundError(f"BIM CSV not found: {bim_csv}")

        n_bldgs = load_bim_arrays(bim_csv)["n_bldgs"]
        collapse_pga = np.full((NUM_GMS, n_bldgs), np.nan, dtype=float)

        for gm_id in range(NUM_GMS):
            gm_file = (GM_DIR / f"GM{gm_id}.txt").resolve()
            dt_file = (GM_DIR / f"time{gm_id}.txt").resolve()

            if not gm_file.exists():
                raise FileNotFoundError(f"GM file not found: {gm_file}")
            if not dt_file.exists():
                raise FileNotFoundError(f"dt file not found: {dt_file}")

            print(f"[sigma={sigma:.2f}] Processing GM {gm_id} ({gm_id + 1}/{NUM_GMS})")

            # Load GM once here just to compute its PGA (in g) and length for n_steps
            gm_arr = np.loadtxt(gm_file, dtype=float)
            gm_peak_g = float(np.max(np.abs(gm_arr)))
            n_steps = int(gm_arr.size)
            dt = float(np.loadtxt(dt_file))

            start = time.time()

            with ProcessPoolExecutor(
                max_workers=MAX_WORKERS,
                initializer=_init_worker,
                initargs=(str(bim_csv), str(gm_file), dt, n_steps, gm_peak_g),
            ) as ex:
                future_to_ii = {ex.submit(run_one_building, ii): ii for ii in range(n_bldgs)}

                for fut in as_completed(future_to_ii):
                    ii = future_to_ii[fut]
                    try:
                        bldg_i, pga_g = fut.result()
                        collapse_pga[gm_id, bldg_i] = pga_g
                    except Exception as e:
                        print(f"[GM {gm_id} | bldg {ii}] Failed: {e}")

            print(f"  Completed GM {gm_id} in {time.time() - start:.2f}s")

        df_out = pd.DataFrame(
            collapse_pga,
            columns=[f"Bldg{i}" for i in range(n_bldgs)],
            index=pd.Index(range(NUM_GMS), name="GM_ID"),
        )

        out_path = OUT_DIR / f"IDA_results_sigma{sigma_label}.csv"
        df_out.to_csv(out_path, float_format="%.4f")
        print(f"Wrote: {out_path}")

    print(f"Total elapsed: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()