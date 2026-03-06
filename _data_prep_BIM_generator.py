#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File Created : 07/12/2025
@Author       : Sebin Oh
@Contact      : sebin.oh@berkeley.edu
@Description  : 

Generate per-building BIM CSVs using HAZUS tables.
User may enable uncertainty and generate BIM for one or multiple sigma values.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd

from fn_find_str_type import find_str_type, cal_lambda

# =============================================================================
# User settings
# =============================================================================
REGION = "Milpitas"
BUILDING_CSV = Path(f"./data/building_inventories/RegionalInventory_{REGION}.csv")
HAZUS_TXT = Path("./data/HazusData.txt")
OUT_DIR = Path(f"./data/OpenSees_input/{REGION}")

# --- Uncertainty control (DEFAULT: OFF) ---
ENABLE_UNCERTAINTY = False  # <- set True to enable

SEED = 42
UNC_TYPE = "lognormal"      # "lognormal" or "uniform" (used only if ENABLE_UNCERTAINTY=True)
SIGMAS: Union[float, Iterable[float]] = np.round(np.arange(0, 1.01, 0.01), 2) # coeffcient of variance
STORY_HEIGHT_sigma = 0.05      # used only if ENABLE_UNCERTAINTY=True

# --- Constraints ---
UD_MAX = 0.999
UD_RESAMPLE_ITERS = 100

if ENABLE_UNCERTAINTY is False:
    SIGMAS = [0.0]  # just one case with no uncertainty

# =============================================================================
# Helpers
# =============================================================================
def load_hazus_tables(path: Path) -> Dict[str, np.ndarray]:
    """Load HazusData.txt and return tables keyed by code level."""
    data = pd.read_table(path, header=None).to_numpy()
    if data.shape[0] < 144:
        raise ValueError(
            f"HAZUS table looks too short: got {data.shape[0]} rows, expected >= 144."
        )
    return {
        "Hig": data[0:36, :],
        "Mod": data[36:72, :],
        "Low": data[72:108, :],
        "Pre": data[108:144, :],
    }


def select_hazus_code(*, year_built: int, structure_type: str, tables: Dict[str, np.ndarray]) -> np.ndarray:
    """Match the original selection logic for HAZUS code-level tables."""
    st = str(structure_type).strip().upper()

    if year_built <= 1941 and st == "W1":
        return tables["Mod"]
    if year_built <= 1941:
        return tables["Pre"]
    if year_built <= 1975:
        return tables["Mod"]
    return tables["Hig"]


def safe_int(x, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def fmt_vec(vec: np.ndarray, fmt: str) -> str:
    """Format a 1D vector as a comma-separated string."""
    vec = np.atleast_1d(vec)
    return ",".join(format(float(x), fmt) for x in vec)


def to_sigma_list(sigmas: Union[float, Iterable[float]], *, enabled: bool) -> List[float]:
    """Normalize sigma input to a list. If uncertainty is disabled, return [0.0]."""
    if not enabled:
        return [0.0]

    if isinstance(sigmas, (int, float, np.integer, np.floating)):
        return [float(sigmas)]

    sigma_list = [float(c) for c in sigmas]
    if len(sigma_list) == 0:
        raise ValueError("SIGMAS is empty. Provide a float or a non-empty iterable.")
    return sigma_list


@dataclass(frozen=True)
class UncertaintyModel:
    enabled: bool
    unc_type: str
    rng: np.random.Generator

    def factor(self, sigma: float, size=None) -> np.ndarray:
        """
        Sample multiplicative factors.
        NOTE: For 'lognormal', `sigma` is treated as sigma of the underlying normal.
        """
        if (not self.enabled) or sigma == 0.0:
            return np.ones(size) if size is not None else np.array(1.0)

        if self.unc_type == "lognormal":
            return self.rng.lognormal(mean=0.0, sigma=sigma, size=size)

        if self.unc_type == "uniform":
            low = max(0.0, 1.0 - sigma)
            high = 1.0 + sigma
            return self.rng.uniform(low=low, high=high, size=size)

        raise ValueError("UNC_TYPE must be 'lognormal' or 'uniform'.")

    def scale(self, val, sigma: float):
        """Scale val by random factors matching its shape (or return val if disabled)."""
        if (not self.enabled) or sigma == 0.0:
            return val

        arr = np.asarray(val, dtype=float)
        f = self.factor(sigma, size=arr.shape)
        out = arr * f
        return out.item() if np.isscalar(val) else out


def enforce_ud_limit(
    unc: UncertaintyModel,
    ud_base: np.ndarray,
    *,
    sigma: float,
    ud_max: float = UD_MAX,
    resample_iters: int = UD_RESAMPLE_ITERS,
) -> np.ndarray:
    """
    If uncertainty is enabled and sigma>0:
      - scale ud
      - resample entries > 1.0 (up to resample_iters)
    Always:
      - clamp to ud_max
    """
    ud_base = np.asarray(ud_base, dtype=float)

    ud_unc = unc.scale(ud_base, sigma)
    ud_unc = np.asarray(ud_unc, dtype=float)

    if unc.enabled and sigma > 0.0:
        for _ in range(resample_iters):
            mask = ud_unc > 1.0
            if not np.any(mask):
                break
            ud_unc[mask] = ud_base[mask] * unc.factor(sigma, size=int(mask.sum()))

    return np.minimum(ud_unc, ud_max)


def output_filename(*, sigma: float) -> str:
    sigma_label = f"{int(round(sigma * 100)):03d}"
    return f"BIM_sigma{sigma_label}.csv"

# =============================================================================
# Main
# =============================================================================
def main() -> None:
    if ENABLE_UNCERTAINTY and UNC_TYPE not in ("lognormal", "uniform"):
        raise ValueError("UNC_TYPE must be 'lognormal' or 'uniform'.")

    rng = np.random.default_rng(SEED)
    unc = UncertaintyModel(enabled=ENABLE_UNCERTAINTY, unc_type=UNC_TYPE, rng=rng)

    building_info = pd.read_csv(BUILDING_CSV)
    tables = load_hazus_tables(HAZUS_TXT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sigma_list = to_sigma_list(SIGMAS, enabled=ENABLE_UNCERTAINTY)

    excluded_at_sigma0: List[int] = []

    for cc, sigma in enumerate(sigma_list, start=1):
        print(f"Processing sigma={sigma:.2f} ({cc}/{len(sigma_list)}) | uncertainty={'ON' if ENABLE_UNCERTAINTY else 'OFF'}")

        rows = []

        for ii, bldg in enumerate(building_info.itertuples(index=False), start=0):
            year_built = safe_int(getattr(bldg, "YearBuilt", np.nan), default=9999)
            stype = str(getattr(bldg, "StructureType", "")).strip()
            ns = safe_int(getattr(bldg, "NumberOfStories", np.nan), default=0)

            if ns < 1 or stype == "":
                if sigma == 0.0:
                    excluded_at_sigma0.append(ii)
                continue

            code = select_hazus_code(year_built=year_built, structure_type=stype, tables=tables)

            try:
                idx = find_str_type(stype, ns)
            except Exception:
                idx = -1

            if idx == -1:
                if sigma == 0.0:
                    print(f"  Skipping building {ii}: unsupported structure type {stype!r} (NS={ns}).")
                    excluded_at_sigma0.append(ii)
                continue

            # --- geometry ---
            height_m = float(code[idx, 22]) * 0.0254  # inches -> meters
            story_height = height_m / ns
            story_height_sigma = STORY_HEIGHT_sigma if ENABLE_UNCERTAINTY else 0.0
            story_height = float(unc.scale(story_height, story_height_sigma))

            # --- mechanics ---
            period = float(ns * code[idx, 18])
            if period <= 0:
                if sigma == 0.0:
                    excluded_at_sigma0.append(ii)
                continue

            plan_area = float(getattr(bldg, "PlanArea", np.nan))
            if not np.isfinite(plan_area) or plan_area <= 0:
                if sigma == 0.0:
                    excluded_at_sigma0.append(ii)
                continue

            mass = plan_area * 1000.0
            k0 = float(cal_lambda(ns)) * mass * (2.0 * np.pi / period) ** 2

            fy = float(code[idx, 5]) * mass * 9.81 * ns
            uy = fy / k0
            fu = fy * float(code[idx, 10])
            alpha = float(code[idx, 6])
            pinch = float(code[idx, 8])
            beta = float(code[idx, 12])
            uu = uy + (fu - fy) / (alpha * k0)

            floors = np.arange(1, ns + 1, dtype=float)
            gamma = 1.0 - (floors * (floors - 1.0)) / ((ns + 1.0) * ns)

            ys, yd = fy * gamma, uy * gamma
            us, ud = fu * gamma, uu * gamma

            ds1, ds2, ds3, ds4 = (
                float(code[idx, 14]),
                float(code[idx, 15]),
                float(code[idx, 16]),
                float(code[idx, 17]),
            )

            # --- uncertainty (optional) ---
            mass_out = float(unc.scale(mass, sigma))
            ys_out = unc.scale(ys, sigma)
            yd_out = unc.scale(yd, sigma)
            us_out = unc.scale(us, sigma)
            ud_out = enforce_ud_limit(unc, ud, sigma=sigma, ud_max=UD_MAX, resample_iters=UD_RESAMPLE_ITERS)
            pinch_out = float(unc.scale(pinch, sigma))
            beta_out = float(unc.scale(beta, sigma))

            rows.append(
                {
                    "BuildingIndex": ii,
                    "DS1_idr": ds1,
                    "DS2_idr": ds2,
                    "DS3_idr": ds3,
                    "DS4_idr": ds4,
                    "NS": ns,
                    "story_height": story_height,
                    "mass": mass_out,
                    "ys": fmt_vec(ys_out, ".4e"),
                    "yd": fmt_vec(yd_out, ".5f"),
                    "us": fmt_vec(us_out, ".4e"),
                    "ud": fmt_vec(ud_out, ".5f"),
                    "pinch": pinch_out,
                    "beta": beta_out,
                }
            )

        df = pd.DataFrame(rows)

        out_name = output_filename(sigma=sigma)
        out_path = OUT_DIR / out_name
        df.to_csv(out_path, index=False)
        print(f"  Wrote: {out_path}")

    if excluded_at_sigma0:
        print(f"Excluded {len(set(excluded_at_sigma0))} buildings at sigma=0 due to missing/invalid fields or unsupported types.")


if __name__ == "__main__":
    main()