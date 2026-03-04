#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Created     :   04/17/2025
@Author      :   Sebin Oh
@Contact     :   sebin.oh@berkeley.edu
@Description :   

Extract a target-region building subset from a Bay Area inventory GeoJSON and
attach Vs30 values from a raster (Wills/Thompson hybrid), then export as CSV.

"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import rasterio
from pyproj import Transformer

# Optional plotting deps (only used if PLOT=True)
import matplotlib.pyplot as plt
import contextily as ctx

# =============================================================================
# User settings
# =============================================================================
INVENTORY_GEOJSON = Path("../data/SF Bay Area Building Inventory Data/sf_inventory.geojson")

TARGET_REGION = "Milpitas"
TARGET_REGION_FULL = "Milpitas, California, USA"

VS30_TIF = Path("../data/California_vs30_Wills15_hybrid/California_vs30_Wills15_hybrid.tif")

OUT_CSV = Path(f"../data/building_inventories/RegionalInventory_{TARGET_REGION}.csv")

PLOT = True
PLOT_BASEMAP = True  # requires internet for tiles

# Only needed for TARGET_REGION == "NorthBerkeley"
BERKELEY_DISTRICTS_SHP = Path("../data/CoB_CouncilDistricts/CoB_CouncilDistricts.shp")
BERKELEY_DISTRICT_IDS = {"5", "6"}  # North Berkeley districts


# =============================================================================
# Helpers
# =============================================================================
def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["mathtext.fontset"] = "cm"


def load_inventory(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Inventory file not found: {path}")
    print(f"Loading inventory: {path} (this may take a while)")
    return gpd.read_file(path)


def geocode_boundary(place: str) -> gpd.GeoDataFrame:
    print(f"Geocoding region boundary: {place}")
    return ox.geocode_to_gdf(place)


def subset_to_boundary(
    gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Subset inventory to features fully within boundary polygon (matches original behavior)."""
    gdf_ll = gdf.to_crs(boundary.crs)
    geom = boundary.geometry.iloc[0]
    return gdf_ll[gdf_ll.geometry.within(geom)].copy()


def subset_north_berkeley(gdf_region_ll: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    For NorthBerkeley only:
    intersect region inventory with Berkeley council districts 5 & 6.
    """
    if not BERKELEY_DISTRICTS_SHP.exists():
        raise FileNotFoundError(f"Berkeley districts shapefile not found: {BERKELEY_DISTRICTS_SHP}")

    berkeley = gpd.read_file(BERKELEY_DISTRICTS_SHP)

    # dissolve districts 5 & 6 into one polygon
    north = berkeley[berkeley["DISTRICT"].astype(str).isin(BERKELEY_DISTRICT_IDS)].dissolve()
    north = north.to_crs(gdf_region_ll.crs)

    # intersection keeps original attributes
    return gpd.overlay(gdf_region_ll, north, how="intersection")


def get_lon_lat(gdf_ll: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return lon/lat arrays. Uses existing Longitude/Latitude columns if available;
    otherwise uses geometry centroids (in EPSG:4326).
    """
    if {"Longitude", "Latitude"}.issubset(gdf_ll.columns):
        lon = pd.to_numeric(gdf_ll["Longitude"], errors="coerce").to_numpy()
        lat = pd.to_numeric(gdf_ll["Latitude"], errors="coerce").to_numpy()
        return lon, lat

    # fallback: centroid in lon/lat CRS
    gdf_ll = gdf_ll.to_crs(epsg=4326)
    cent = gdf_ll.geometry.centroid
    return cent.x.to_numpy(), cent.y.to_numpy()


def transform_points(
    xs: np.ndarray,
    ys: np.ndarray,
    src_crs,
    dst_crs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform coordinate arrays from src_crs to dst_crs."""
    if src_crs == dst_crs:
        return xs, ys

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x2, y2 = transformer.transform(xs, ys)
    return np.asarray(x2), np.asarray(y2)


def sample_raster_at_points(
    raster_path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    progress: bool = True,
) -> np.ndarray:
    """
    Sample raster values at lon/lat points.
    Handles CRS conversion if raster is not EPSG:4326.
    """
    if not raster_path.exists():
        raise FileNotFoundError(f"Vs30 raster not found: {raster_path}")

    n = len(lon)
    out = np.full(n, np.nan, dtype=float)

    # skip NaN coords early
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(valid):
        return out

    lon_v = lon[valid]
    lat_v = lat[valid]

    with rasterio.open(raster_path) as src:
        # Determine input CRS (we assume lon/lat are EPSG:4326)
        src_xy_crs = "EPSG:4326"
        dst_xy_crs = src.crs
        if dst_xy_crs is None:
            raise ValueError("Raster has no CRS; cannot sample reliably.")

        xs, ys = transform_points(lon_v, lat_v, src_xy_crs, dst_xy_crs)

        # Sample (masked=True gives masked arrays for nodata/out-of-bounds)
        coords = zip(xs, ys)
        samples = src.sample(coords, indexes=1, masked=True)

        nodata = src.nodata
        total = xs.shape[0]
        step = max(1, total // 10)  # 10% increments

        for k, s in enumerate(samples, start=1):
            val = s[0]
            if np.ma.is_masked(val):
                v = np.nan
            else:
                v = float(val)
                if nodata is not None and np.isfinite(v) and v == nodata:
                    v = np.nan

            out_idx = np.flatnonzero(valid)[k - 1]
            out[out_idx] = v

            if progress and (k % step == 0 or k == total):
                pct = int(round(100 * k / total))
                print(f"  Vs30 sampling: {pct}%")

    return out


def plot_region(gdf_ll: gpd.GeoDataFrame, *, title: str) -> None:
    """Plot region footprint over basemap."""
    gdf_3857 = gdf_ll.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_3857.plot(ax=ax, alpha=1.0)
    if PLOT_BASEMAP:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    if PLOT:
        configure_matplotlib()

    gdf_all = load_inventory(INVENTORY_GEOJSON)

    boundary = geocode_boundary(TARGET_REGION_FULL)
    gdf_region = subset_to_boundary(gdf_all, boundary)

    if TARGET_REGION == "NorthBerkeley":
        gdf_region = subset_north_berkeley(gdf_region)

    if PLOT:
        plot_region(gdf_region, title=f"Target region: {TARGET_REGION}")

    # Lon/lat for Vs30 sampling
    gdf_ll = gdf_region.to_crs(epsg=4326)
    lon, lat = get_lon_lat(gdf_ll)

    print("Sampling Vs30 for the region...")
    vs30 = sample_raster_at_points(VS30_TIF, lon, lat, progress=True)

    gdf_out = gdf_ll.copy()
    gdf_out["Vs30"] = vs30

    # Export as CSV (drop geometry)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    gdf_out.drop(columns=["geometry"]).to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()