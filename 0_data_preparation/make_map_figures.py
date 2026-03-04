#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Created     :   02/18/2026
@Author      :   Sebin Oh
@Contact     :   sebin.oh@berkeley.edu
@Description :  

Interactive examples for map-based figures.

"""

from __future__ import annotations

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path

import fn_map_based_figures as fn_map

# %% Working directory (optional but helps in VS Code Interactive)
# If __file__ is unavailable (some notebook contexts), fall back to cwd.
try:
    HERE = Path(__file__).resolve().parent
except NameError:  # __file__ not defined
    HERE = Path.cwd()


# %% User settings (edit these)
# -----------------------------------------------------------------------------
# Core
TARGET_REGION = "Milpitas"  # e.g. "Milpitas", "SanFrancisco_NE", "NorthBerkeley"
PLACE_SUFFIX = "California, USA"

# I/O
DATA_DIR = HERE / "Data"
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

SAVEFIG = False
SHOW = True

# Inputs (set to existing files on your machine)
BUILDINGS_PATH = DATA_DIR / "SF Bay Area Building Inventory Data/sf_inventory.geojson"               # GeoJSON/Shapefile/GPKG/etc.
VS30_PATH = DATA_DIR / "California_vs30_Wills15_hybrid/California_vs30_Wills15_hybrid.tif"    # Vs30 GeoTIFF
SF_NEIGHBORHOODS_PATH = DATA_DIR / "SanFrancisco_Neighborhoods_Boundary.geojson"
NORTHBERK_DISTRICTS_PATH = DATA_DIR / "CoB_CouncilDistricts/CoB_CouncilDistricts.shp"
NORTHBERK_DISTRICTS_COL = "DISTRICT"

# Epicenter (optional)
EPI_LON = -122.076
EPI_LAT = 37.666
USE_EPICENTER = True

# Buildings plotting
BLDG_PLOT_TYPE = "footprint_colored"  # "point" | "footprint" | "footprint_colored"
STORIES_COL = "NumberOfStories"
STORY_ORDER = ["1", "2", "3+"]  # buckets for footprint_colored

# Apply consistent Matplotlib style
fn_map.apply_mpl_style()


# %% Build a boundary for TARGET_REGION
boundary_gdf = None
boundary_outline = None  # optional extra outline layer (e.g., SF selected neighborhoods)
region_label = TARGET_REGION
neigh_sel = None

if TARGET_REGION == "SanFrancisco_NE":
    if not SF_NEIGHBORHOODS_PATH.exists():
        raise FileNotFoundError(f"SF neighborhoods file not found: {SF_NEIGHBORHOODS_PATH}")

    SF_TARGETS = [
        "Marina",
        "Pacific Heights",
        "Japantown",
        "Western Addition",
        "Hayes Valley",
        "Russian Hill",
        "Nob Hill",
        "Tenderloin",
        "North Beach",
        "Chinatown",
        "Financial District/South Beach",
        "South of Market",
        "Mission Bay",
        "Mission",
        "Potrero Hill",
    ]

    boundary_gdf, neigh_sel = fn_map.sf_neighborhood_union(
        SF_NEIGHBORHOODS_PATH,
        targets=SF_TARGETS,
        city_query="San Francisco, California, USA",
        name_col=None,
    )
    boundary_outline = neigh_sel
    region_label = "San Francisco (Selected NE)"

elif TARGET_REGION == "NorthBerkeley":
    if not NORTHBERK_DISTRICTS_PATH.exists():
        raise FileNotFoundError(f"North Berkeley districts file not found: {NORTHBERK_DISTRICTS_PATH}")

    boundary_gdf = fn_map.north_berkeley_union(
        NORTHBERK_DISTRICTS_PATH,
        districts=("5", "6"),
        district_col=NORTHBERK_DISTRICTS_COL,
        city_query="Berkeley, California, USA",
    )
    region_label = "North Berkeley"

else:
    boundary_gdf = fn_map.geocode_boundary(f"{TARGET_REGION}, {PLACE_SUFFIX}")


# %% Load buildings (optional) + clip to boundary
gdf_all = None
gdf_region = None
region_main_gmt = None
region_all_gmt = None
points_df = None
palette = {}

if BUILDINGS_PATH.exists():
    gdf_all = gpd.read_file(BUILDINGS_PATH)

    # Clip to boundary (fast + deterministic)
    gdf_region = fn_map.clip_buildings_within(gdf_all, boundary_gdf, to_crs="EPSG:4326")

    # GMT regions for PyGMT
    gdf_all_ll = gdf_all.to_crs("EPSG:4326")
    region_main_gmt = fn_map.bounds_to_region_gmt(gdf_region.total_bounds)
    region_all_gmt = fn_map.bounds_to_region_gmt(gdf_all_ll.total_bounds)

    # Points for bldg_plot_type="point"
    points_df = fn_map.building_centroids_lonlat(gdf_region)

    # Buckets/palette for bldg_plot_type="footprint_colored"
    if (BLDG_PLOT_TYPE == "footprint_colored") and (STORIES_COL in gdf_region.columns):

        def bucket_story(x):
            try:
                n = int(float(x))
            except Exception:
                return "Other"
            if str(n) in {s for s in STORY_ORDER if not s.endswith("+")}:
                return str(n)
            # find overflow bucket like "3+"
            overflow = next((s for s in STORY_ORDER if s.endswith("+")), None)
            if overflow is not None:
                thr = int(overflow[:-1])
                return overflow if n >= thr else "Other"
            return "Other"

        gdf_region = gdf_region.copy()
        gdf_region["_bucket"] = gdf_region[STORIES_COL].apply(bucket_story)

        palette = {"Other": "dimgray"}
        if "1" in STORY_ORDER:
            palette["1"] = "#7f7f7f@30"
        for s in STORY_ORDER:
            if s != "1":
                palette[s] = fn_map.CP_HEX[0]

else:
    print(f"[INFO] BUILDINGS_PATH not found, skipping building-based figures: {BUILDINGS_PATH}")


# %% Epicenter object (optional)
epicenter = None
if USE_EPICENTER:
    epicenter = fn_map.Epicenter(lon=EPI_LON, lat=EPI_LAT)


# %% PyGMT region map: tilemap + buildings + inset
if (gdf_region is not None) and (region_main_gmt is not None) and (region_all_gmt is not None):
    outpath = OUTDIR / f"{TARGET_REGION}_region_map.png"

    inset_highlights = [
        {
            "boundary_df": boundary_gdf,
            "label": region_label,
            "label_region_gmt": region_main_gmt,
            "fill": "red",
            "pen": "1.0p,black",
            "offset": "-1.4c/-0.8c",
        }
    ]

    fn_map.plot_region_with_inset_pygmt(
        region_main_gmt=region_main_gmt,
        boundary_main=boundary_gdf,
        buildings_gdf=gdf_region,
        bldg_plot_type=BLDG_PLOT_TYPE,
        points_df=points_df,
        palette=palette,
        bucket_attr="_bucket",
        add_inset=True,
        region_inset_base_gmt=region_all_gmt,
        inset_highlights=inset_highlights,
        epicenter=epicenter,
        savefig=SAVEFIG,
        outpath=outpath,
        show=SHOW,
    )
else:
    print("[INFO] Skipping PyGMT region map (need buildings + regions).")


# %% PyGMT overview map: broader view + highlighted boundary + epicenter
if (gdf_all is not None) and (region_main_gmt is not None) and (region_all_gmt is not None):
    outpath = OUTDIR / f"{TARGET_REGION}_overview.png"

    fn_map.plot_overview_with_epicenter_pygmt(
        region_all_gmt=region_all_gmt,
        boundary_df=boundary_gdf,
        region_label_gmt=region_main_gmt,
        region_label=region_label,
        region_fill=fn_map.CP_HEX[0],
        epicenter=epicenter,
        savefig=SAVEFIG,
        outpath=outpath,
        show=SHOW,
    )
else:
    print("[INFO] Skipping overview map (need buildings for global extent).")


# %% SF neighborhood selection map (only for SanFrancisco_NE)
if TARGET_REGION == "SanFrancisco_NE" and neigh_sel is not None:
    neigh = gpd.read_file(SF_NEIGHBORHOODS_PATH)
    neigh = neigh.to_crs(neigh_sel.crs)

    name_col = fn_map.find_name_col(neigh, regex=r"nhood")

    outpath = OUTDIR / f"{TARGET_REGION}_selected_neighborhoods.png"
    fn_map.plot_selected_neighborhoods_pygmt(
        neigh=neigh,
        neigh_sel=neigh_sel,
        targets=None,  # inferred from neigh_sel
        name_col=name_col,
        savefig=SAVEFIG,
        outpath=outpath,
        show=SHOW,
    )


# %% Vs30 map (Matplotlib + rasterio + contextily)
if VS30_PATH.exists():
    outpath = OUTDIR / f"{TARGET_REGION}_vs30map.png"

    # Use the boundary you already built above for reproducibility
    fn_map.plot_vs30_map_from_boundary(
        vs30_path=VS30_PATH,
        target_boundary=boundary_gdf,
        boundary_outline=boundary_outline,
        savefig=SAVEFIG,
        outpath=outpath,
        show=SHOW,
    )
else:
    print(f"[INFO] VS30_PATH not found, skipping Vs30 map: {VS30_PATH}")


# %% If you generated multiple Matplotlib figures and want them to display together
if SHOW:
    plt.show()
