#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Created     :   02/18/2026
@Author      :   Sebin Oh
@Contact     :   sebin.oh@berkeley.edu
@Description :  

Function library for map-based figures (PyGMT + Matplotlib).

Dependencies
------------
Core:
  geopandas, pandas, numpy, shapely, matplotlib

Maps:
  pygmt, contextily (tile provider), osmnx (geocode boundaries)

Raster/Vs30:
  rasterio

Quickstart examples (copy/paste)
--------------------------------
These are intended as *templates* for your own data paths.

Generic region boundary + buildings + PyGMT region map
  import geopandas as gpd
  import fn_map_based_figures as fn_map

  fn_map.apply_mpl_style()

  # 1) Boundary from OSM (requires osmnx)
  boundary = fn_map.geocode_boundary("Milpitas, California, USA")

  # 2) Buildings (any vector format supported by GeoPandas)
  gdf_all = gpd.read_file("data/buildings.geojson").to_crs("EPSG:4326")
  gdf_region = fn_map.clip_buildings_within(gdf_all, boundary, to_crs="EPSG:4326")

  # 3) Regions for PyGMT
  region_main = fn_map.bounds_to_region_gmt(gdf_region.total_bounds)
  region_all  = fn_map.bounds_to_region_gmt(gdf_all.total_bounds)

  # 4) Optional: bucket by story count for colored footprints
  story_order = ["1", "2", "3+"]
  def _bucket_story(x):
      try:
          n = int(float(x))
      except Exception:
          return "Other"
      if str(n) in {"1", "2"}:
          return str(n)
      return "3+" if n >= 3 else "Other"

  gdf_region = gdf_region.copy()
  gdf_region["_bucket"] = gdf_region["NumberOfStories"].apply(_bucket_story)
  palette = {"1": "#7f7f7f@30", "2": fn_map.CP_HEX[0], "3+": fn_map.CP_HEX[0], "Other": "dimgray"}

  epi = fn_map.Epicenter(lon=-122.076, lat=37.666)

  fn_map.plot_region_with_inset_pygmt(
      region_main_gmt=region_main,
      boundary_main=boundary,
      buildings_gdf=gdf_region,
      bldg_plot_type="footprint_colored",
      palette=palette,
      bucket_attr="_bucket",
      add_inset=True,
      region_inset_base_gmt=region_all,
      inset_highlights=[{"boundary_df": boundary, "label": "Milpitas", "label_region_gmt": region_main}],
      epicenter=epi,
      savefig=True,
      outpath="outputs/milpitas_region_map.png",
  )

Vs30 map (Matplotlib + rasterio + contextily)
  fn_map.plot_vs30_map_from_boundary(
      vs30_path="data/California_vs30_Wills15_hybrid.tif",
      target_boundary=boundary,
      savefig=True,
      outpath="outputs/milpitas_vs30.png",
  )

SanFrancisco_NE boundary from selected neighborhoods
  targets = ["Marina", "Pacific Heights", "Japantown", "..."]
  boundary_sfne, neigh_sel = fn_map.sf_neighborhood_union(
      "data/SanFrancisco_Neighborhoods_Boundary.geojson",
      targets=targets,
  )

  neigh = gpd.read_file("data/SanFrancisco_Neighborhoods_Boundary.geojson")
  name_col = fn_map.find_name_col(neigh, regex=r"nhood")

  fn_map.plot_selected_neighborhoods_pygmt(
      neigh=neigh,
      neigh_sel=neigh_sel,
      targets=None,   # inferred from neigh_sel
      name_col=name_col,
      savefig=True,
      outpath="outputs/sf_selected_neighborhoods.png",
  )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import io
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Optional deps (used by specific functions). We import them lazily where needed,
# but keep sentinel imports for friendlier error messages.
try:  # PyGMT mapping
    import pygmt  # type: ignore
except Exception:  # pragma: no cover
    pygmt = None  # type: ignore

try:  # tile providers (xyzservices)
    import contextily as ctx  # type: ignore
except Exception:  # pragma: no cover
    ctx = None  # type: ignore

try:  # boundary geocoding
    import osmnx as ox  # type: ignore
except Exception:  # pragma: no cover
    ox = None  # type: ignore

try:  # raster clipping for Vs30
    import rasterio  # type: ignore
    from rasterio.mask import mask as rio_mask  # type: ignore
    from rasterio.plot import plotting_extent  # type: ignore
except Exception:  # pragma: no cover
    rasterio = None  # type: ignore
    rio_mask = None  # type: ignore
    plotting_extent = None  # type: ignore


# -----------------------------------------------------------------------------
# Style + color palettes
# -----------------------------------------------------------------------------
DEFAULT_MPL_STYLE: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "mathtext.fontset": "cm",
}


def apply_mpl_style(style: Mapping[str, Any] | None = None) -> None:
    """Apply a consistent Matplotlib style for figure reproducibility."""
    plt.rcParams.update(dict(DEFAULT_MPL_STYLE if style is None else style))


# Color palette used in the original notebook/script (kept for consistency)
CP = np.array(
    [
        [64, 86, 161],
        [241, 40, 21],
        [215, 153, 34],
        [20, 160, 152],
        [203, 45, 111],
        [80, 31, 58],
        [17, 100, 102],
        [247, 108, 108],
        [239, 226, 186],
        [197, 203, 227],
        [140, 154, 199],
        [0, 114, 181],
    ],
    dtype=float,
) / 255.0

CP_HEX: list[str] = [
    "#4056A1",
    "#F12815",
    "#D79922",
    "#14A098",
    "#CB2D6F",
    "#116466",
    "#501F3A",
    "#F76C6C",
    "#EFE2BA",
    "#C5CBE3",
    "#8C9AC7",
    "#0072B5",
]


# ------------------

# Public API (re-exported names)
__all__ = [
    "DEFAULT_MPL_STYLE",
    "CP",
    "CP_HEX",
    "apply_mpl_style",
    "default_tile_source",
    "bounds_to_region_gmt",
    "region_center",
    "expand_region",
    "reduce_region",
    "polygon_to_lonlat_df",
    "polygons_to_multiseg",
    "polygons_to_segments_with_attr",
    "geocode_boundary",
    "find_name_col",
    "sf_neighborhood_union",
    "north_berkeley_union",
    "clip_buildings_within",
    "building_centroids_lonlat",
    "plot_attribute_histograms",
    "value_counts_capped",
    "story_counts_for_donut",
    "YEAR_ORDER",
    "bin_years_to_code_eras",
    "year_counts_for_donut",
    "palette_for",
    "plot_donut",
    "Epicenter",
    "plot_region_with_inset_pygmt",
    "plot_selected_neighborhoods_pygmt",
    "plot_overview_with_epicenter_pygmt",
    "plot_vs30_map_from_boundary",
    "plot_vs30_map",
]
# -----------------------------------------------------------------------------
# Small helpers (I/O, regions, dependency checks)
# -----------------------------------------------------------------------------
def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_matplotlib(fig: plt.Figure, outpath: str | Path, *, dpi: int = 300) -> Path:
    outpath = Path(outpath)
    _ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    return outpath


def _save_pygmt(fig: Any, outpath: str | Path, *, dpi: int = 300, transparent: bool = True) -> Path:
    outpath = Path(outpath)
    _ensure_dir(outpath.parent)
    fig.savefig(str(outpath), dpi=dpi, transparent=transparent)
    return outpath


def _require(module: Any, name: str) -> None:
    if module is None:
        raise ImportError(f"Missing dependency '{name}'. Install it to use this function.")


def default_tile_source(tile_source: Any = None) -> Any:
    """Return a reasonable default XYZ tile provider for PyGMT/contextily."""
    if tile_source is not None:
        return tile_source
    _require(ctx, "contextily")
    return ctx.providers.OpenStreetMap.Mapnik


def bounds_to_region_gmt(bounds: Sequence[float]) -> list[float]:
    """Convert (minx, miny, maxx, maxy) bounds to GMT region [W, E, S, N]."""
    minx, miny, maxx, maxy = bounds
    return [float(minx), float(maxx), float(miny), float(maxy)]


def region_center(region_gmt: Sequence[float]) -> tuple[float, float]:
    """Return (lon_center, lat_center) for a [W, E, S, N] region."""
    W, E, S, N = region_gmt
    return float((W + E) / 2.0), float((S + N) / 2.0)


def expand_region(region_gmt: Sequence[float], *, lon_pad_frac: float, lat_pad_frac: float) -> list[float]:
    """Pad a [W, E, S, N] region outward by a fraction of width/height."""
    W, E, S, N = region_gmt
    lon_pad = (E - W) * lon_pad_frac
    lat_pad = (N - S) * lat_pad_frac
    return [W - lon_pad, E + lon_pad, S - lat_pad, N + lat_pad]


def reduce_region(
    region_gmt: Sequence[float],
    *,
    lon_pad_frac: float,
    lat_pad_frac: float,
    north_extra_latpad: float = 1.0,
    east_mode: str = "shrink",
    east_custom: float | None = None,
) -> list[float]:
    """
    Pad a [W, E, S, N] region inward by a fraction of width/height.

    Parameters
    ----------
    east_mode:
      - "shrink": east = E - lon_pad
      - "keep":   east = E
      - "custom": east = east_custom
    north_extra_latpad:
      north = N - north_extra_latpad * lat_pad
    """
    W, E, S, N = region_gmt
    lon_pad = (E - W) * lon_pad_frac
    lat_pad = (N - S) * lat_pad_frac

    if east_mode == "shrink":
        E2 = E - lon_pad
    elif east_mode == "keep":
        E2 = E
    elif east_mode == "custom":
        if east_custom is None:
            raise ValueError("east_custom must be provided when east_mode='custom'.")
        E2 = float(east_custom)
    else:
        raise ValueError("east_mode must be one of {'shrink','keep','custom'}.")

    return [W + lon_pad, float(E2), S + lat_pad, N - north_extra_latpad * lat_pad]


def polygon_to_lonlat_df(geom: Any) -> pd.DataFrame:
    """
    Convert a (Multi)Polygon geometry to a lon/lat DataFrame suitable for PyGMT.

    Returns
    -------
    DataFrame with columns ['lon','lat'] and NaN separators between parts.
    """
    if geom is None:
        return pd.DataFrame(columns=["lon", "lat"])

    segs: list[pd.DataFrame] = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        polys = list(getattr(geom, "geoms", []))

    for poly in polys:
        x, y = poly.exterior.xy
        segs.append(pd.DataFrame({"lon": x, "lat": y}))
        segs.append(pd.DataFrame({"lon": [np.nan], "lat": [np.nan]}))

    if not segs:
        return pd.DataFrame(columns=["lon", "lat"])
    return pd.concat(segs, ignore_index=True)


# -----------------------------------------------------------------------------
# Geometry helpers for PyGMT footprints
# -----------------------------------------------------------------------------
def polygons_to_multiseg(gdf: gpd.GeoDataFrame, include_holes: bool = False) -> pd.DataFrame:
    """
    Convert (Multi)Polygon geometries to a single DataFrame with NaN separators,
    so PyGMT can plot multiple outlines efficiently.

    Output columns: ['lon', 'lat'].
    """
    segs: list[pd.DataFrame] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            x, y = poly.exterior.xy
            segs.append(pd.DataFrame({"lon": x, "lat": y}))

            if include_holes:
                for ring in poly.interiors:
                    xi, yi = ring.xy
                    segs.append(pd.DataFrame({"lon": xi, "lat": yi}))

            segs.append(pd.DataFrame({"lon": [np.nan], "lat": [np.nan]}))

    if not segs:
        return pd.DataFrame(columns=["lon", "lat"])
    return pd.concat(segs, ignore_index=True)


def polygons_to_segments_with_attr(
    gdf: gpd.GeoDataFrame, *, attr: str | None = None, include_holes: bool = False, hole_tag: str = "__HOLE__"
) -> Iterator[tuple[pd.DataFrame, Any]]:
    """
    Yield (df_segment, attr_value) for each (Multi)Polygon in gdf.

    Notes
    -----
    - df_segment has columns ['lon', 'lat'] and includes a closed ring.
    - Rings are oriented consistently (outer CCW, holes CW) for stable plotting.
    """
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        val = row.get(attr) if attr else None

        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            p = orient(poly, sign=1.0)
            x, y = p.exterior.xy
            yield pd.DataFrame({"lon": x, "lat": y}), val

            if include_holes:
                for ring in p.interiors:
                    xi, yi = ring.xy
                    yield pd.DataFrame({"lon": xi, "lat": yi}), hole_tag


# -----------------------------------------------------------------------------
# Region boundary builders (optional helpers)
# -----------------------------------------------------------------------------
def geocode_boundary(place: str, *, crs: str | None = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Geocode a place name into a polygon boundary using OSMnx.
    """
    _require(ox, "osmnx")
    gdf = ox.geocode_to_gdf(place)
    return gdf.to_crs(crs) if crs else gdf


def find_name_col(gdf: gpd.GeoDataFrame, *, regex: str = r"nhood") -> str:
    """Find a plausible neighborhood/name column by regex (case-insensitive)."""
    for c in gdf.columns:
        if re.search(regex, c, re.I):
            return c
    raise KeyError(f"No name column found (regex='{regex}'). Columns: {list(gdf.columns)}")


def sf_neighborhood_union(
    neighborhoods_path: str | Path,
    *,
    targets: Sequence[str] | None = None,
    city_query: str = "San Francisco, California, USA",
    name_col: str | None = None,
    name_regex: str = r"nhood",
    buffer0: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build a single union polygon from selected SF neighborhoods.

    Returns
    -------
    region_gdf:
        GeoDataFrame with one merged polygon row (union of targets).
    neigh_sel:
        GeoDataFrame of selected neighborhoods (useful for dashed outlines).
    """
    _require(ox, "osmnx")

    neigh = gpd.read_file(str(neighborhoods_path))
    sf_city = geocode_boundary(city_query, crs=None)
    neigh = neigh.to_crs(sf_city.crs)

    if name_col is None:
        name_col = find_name_col(neigh, regex=name_regex)

    neigh_sel = neigh[neigh[name_col].isin(list(targets))].copy()
    if buffer0:
        neigh_sel["geometry"] = neigh_sel.buffer(0)

    union_geom = unary_union(neigh_sel.geometry)
    if buffer0:
        union_geom = union_geom.buffer(0)

    region_gdf = gpd.GeoDataFrame(geometry=[union_geom], crs=neigh_sel.crs)
    return region_gdf, neigh_sel


def north_berkeley_union(
    districts_path: str | Path,
    *,
    districts: Sequence[str] = ("5", "6"),
    district_col: str = "DISTRICT",
    city_query: str = "Berkeley, California, USA",
    buffer0: bool = True,
) -> gpd.GeoDataFrame:
    """Dissolve council districts into a single boundary polygon."""
    _require(ox, "osmnx")

    districts_gdf = gpd.read_file(str(districts_path))
    city = geocode_boundary(city_query, crs=None)

    nb = districts_gdf[districts_gdf[district_col].isin(list(districts))].copy()
    nb = nb.to_crs(city.crs).dissolve()

    if buffer0:
        nb["geometry"] = nb.buffer(0)

    return nb


def clip_buildings_within(
    gdf_all: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame, *, to_crs: str | None = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Clip buildings to a single boundary polygon using `.within(...)` (fast and deterministic).
    """
    poly = boundary_gdf.geometry.iloc[0]
    if to_crs is not None:
        gdf_all = gdf_all.to_crs(to_crs)
        boundary_gdf = boundary_gdf.to_crs(to_crs)
        poly = boundary_gdf.geometry.iloc[0]
    return gdf_all[gdf_all.geometry.within(poly)].copy()


def building_centroids_lonlat(gdf: gpd.GeoDataFrame, *, lonlat_crs: str = "EPSG:4326") -> pd.DataFrame:
    """
    Compute building centroids in a planar CRS (UTM) then return lon/lat DataFrame for plotting.
    """
    utm_crs = gdf.estimate_utm_crs()
    cent_proj = gdf.to_crs(utm_crs).geometry.centroid
    cent_ll = cent_proj.to_crs(lonlat_crs)
    return pd.DataFrame({"lon": cent_ll.x.values, "lat": cent_ll.y.values})


# -----------------------------------------------------------------------------
# Matplotlib: attribute histograms
# -----------------------------------------------------------------------------
def plot_attribute_histograms(
    data: pd.DataFrame,
    *,
    cols: Sequence[str] | None = None,
    figsize: tuple[int, int] = (18, 10),
    bins: int = 20,
    savefig: bool = False,
    outpath: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray, Path | None]:
    """
    Plot histograms (numeric) or bar charts (categorical) for selected columns.
    Adds inset zoom plots for PlanArea, ResidentialUnits, and NumberOfStories.
    """
    if cols is None:
        cols = [
            "PlanArea",
            "NumberOfStories",
            "YearBuilt",
            "ResidentialUnits",
            "OccupancyClass",
            "StructureType",
        ]

    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).ravel()

    for i, col in enumerate(cols):
        ax = axes[i]
        if col not in data.columns:
            ax.set_visible(False)
            continue

        series = data[col].dropna()

        if pd.api.types.is_numeric_dtype(series):
            ax.hist(series, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
        else:
            counts = series.value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, color="steelblue", edgecolor="black", alpha=0.7)
            ax.tick_params(axis="x", rotation=45)

        ax.tick_params(axis="both", labelsize=16)
        ax.set_ylabel("Count", fontsize=20)
        ax.set_title(r"PlanArea (ft$^2$)" if col == "PlanArea" else col, fontsize=20)

        if col in {"PlanArea", "ResidentialUnits", "NumberOfStories"} and pd.api.types.is_numeric_dtype(series):
            axins = inset_axes(ax, width="40%", height="40%", loc="upper right", borderpad=1)

            if col == "PlanArea":
                subset = series[series < series.quantile(0.1)]
                axins.hist(subset, bins=15, color="steelblue", edgecolor="black", alpha=0.8)

            elif col == "ResidentialUnits":
                subset = series[series <= 5]
                axins.hist(
                    subset,
                    bins=np.arange(-0.5, 5.5, 1),
                    color="steelblue",
                    edgecolor="black",
                    alpha=0.8,
                )
                axins.set_xticks(np.arange(0, 6, 1))

            elif col == "NumberOfStories":
                subset = series[series <= 6]
                axins.hist(
                    subset,
                    bins=np.arange(0.5, 6.5, 1),
                    color="steelblue",
                    edgecolor="black",
                    alpha=0.8,
                )
                axins.set_xticks(np.arange(1, 7, 1))

            axins.tick_params(axis="both", labelsize=14)

    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()

    saved = None
    if savefig:
        saved = _save_matplotlib(fig, outpath or "attribute_histograms.png")

    if show:
        plt.show()

    return fig, axes, saved


# -----------------------------------------------------------------------------
# Matplotlib: donut charts (categorical distributions)
# -----------------------------------------------------------------------------
def value_counts_capped(
    series: Iterable[Any],
    *,
    topn: int = 5,
    other_label: str = "Other",
    sort_by_category: bool = False,
) -> pd.Series:
    """Return capped value counts with optional numeric-aware sorting."""
    s = pd.Series(series, dtype="string").fillna("Unknown").replace({"": "Unknown"})
    vc = s.value_counts(dropna=False)

    if sort_by_category:

        def sort_key(x: Any):
            xs = str(x)
            if xs.replace(".", "", 1).isdigit():
                return float(xs)
            if xs.endswith("s") and xs[:-1].isdigit():
                return int(xs[:-1])
            if xs.lower() in {"unknown", "other"}:
                return float("inf")
            return xs

        vc = vc.loc[sorted(vc.index, key=sort_key)]

    if len(vc) > topn:
        vc = pd.concat([vc.iloc[:topn], pd.Series({other_label: int(vc.iloc[topn:].sum())})])

    return vc.astype(int)


def story_counts_for_donut(story_series: Iterable[Any], *, story_order: Sequence[str]) -> pd.Series:
    """
    Bucket story counts into a user-specified order like:
      ["1", "2", "3", "4+"] or ["1", "2", "3+"]

    Any unparsable values go to "Other".
    """
    order = list(story_order)
    overflow_label = next((x for x in order if str(x).endswith("+")), None)
    overflow_threshold = int(str(overflow_label)[:-1]) if overflow_label else None
    discrete_labels = {x for x in order if not str(x).endswith("+")}

    def bucket(x: Any) -> str:
        try:
            n = int(float(x))
        except Exception:
            return "Other"
        s = str(n)
        if s in discrete_labels:
            return s
        if overflow_threshold is not None and n >= overflow_threshold:
            return str(overflow_label)
        return "Other"

    mapped = pd.Series(story_series).apply(bucket)
    vc = mapped.value_counts()
    return vc.reindex(order).fillna(0).astype(int)


YEAR_ORDER: list[str] = ["-1941", "1941-1975", "1975-2000", "2000-"]


def bin_years_to_code_eras(series: Iterable[Any], *, other_label: str = "Unknown") -> pd.Series:
    """Bin construction years into broad code-era categories (kept consistent with original)."""

    def to_bin(y: Any) -> str:
        try:
            y = int(float(y))
            if y < 1500 or y > 2100:
                return other_label
            if y < 1941:
                return "-1941"
            if y < 1975:
                return "1941-1975"
            if y < 2000:
                return "1975-2000"
            return "2000-"
        except Exception:
            return other_label

    return pd.Series(series).apply(to_bin)


def year_counts_for_donut(year_series: Iterable[Any]) -> pd.Series:
    """Return construction-year counts in fixed chronological order + 'Unknown' at the end (if present)."""
    binned = bin_years_to_code_eras(year_series, other_label="Unknown")
    vc = binned.value_counts()

    ordered = vc.reindex([c for c in YEAR_ORDER if c in vc.index]).fillna(0).astype(int)
    if "Unknown" in vc.index:
        ordered = pd.concat([ordered, pd.Series({"Unknown": int(vc["Unknown"])})])
    return ordered.astype(int)


def palette_for(categories: Sequence[Any], *, colors: Sequence[Any] | None = None, cmap_name: str | None = None) -> dict[Any, Any]:
    """Return a deterministic mapping {category: color}."""
    cats = list(categories)
    if colors is not None:
        pal = list(colors)
    elif cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        k = max(len(cats), 3)
        pal = [cmap(i / (k - 1)) for i in range(k)]
    else:
        raise ValueError("Provide either 'colors' or 'cmap_name'.")
    return {c: pal[i % len(pal)] for i, c in enumerate(cats)}


def plot_donut(
    counts: pd.Series,
    *,
    title: str,
    outpath: str | Path,
    donut_width: float = 0.38,
    min_pct_label: float = 3.0,
    use_legend: bool = True,
    legend_cols: int = 2,
    dpi: int = 300,
    savefig: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, Path | None]:
    """Render a donut chart with percentage labels and an optional legend."""
    counts = counts.dropna()
    if counts.sum() == 0:
        counts = pd.Series({"No data": 1})

    cats = list(counts.index)
    vals = counts.values.astype(float)

    pal = palette_for(cats, colors=CP_HEX)
    facecolors = [pal[c] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi, constrained_layout=False)

    wedges, _, autotexts = ax.pie(
        vals,
        radius=1.0,
        startangle=90,
        counterclock=False,
        colors=facecolors,
        wedgeprops=dict(width=donut_width, edgecolor="white", linewidth=0.8, antialiased=True),
        labels=None if use_legend else cats,
        labeldistance=1.10 if not use_legend else None,
        autopct=lambda p: f"{p:.1f}%" if p >= min_pct_label else "",
        pctdistance=0.8,
        normalize=True,
        textprops={"fontsize": 16, "fontweight": "bold"},
    )

    plt.setp(autotexts, color="black", fontsize=14)

    ax.text(
        0,
        0,
        title,
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        linespacing=1.1,
        wrap=True,
    )
    ax.set_aspect("equal", adjustable="box")

    if use_legend:
        fig.subplots_adjust(bottom=0.20)
        ax.legend(
            wedges,
            cats,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),
            frameon=False,
            fontsize=18,
            ncol=legend_cols,
            handlelength=1.2,
            handletextpad=0.6,
            borderaxespad=0.0,
        )
    else:
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    saved = None
    if savefig:
        saved = _save_matplotlib(fig, outpath, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)

    return fig, ax, saved


# -----------------------------------------------------------------------------
# PyGMT: tilemap figures (region + buildings + optional inset)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Epicenter:
    lon: float
    lat: float
    label: str = "Epicenter"
    style: str = "a0.5c"
    fill: str = "#FDB813"
    pen: str = "1.0p,black"
    font: str = "14p,Helvetica-Bold,black"
    offset: str = "-0.85c/0.5c"


def plot_region_with_inset_pygmt(
    *,
    region_main_gmt: Sequence[float],
    boundary_main: Any | None = None,
    buildings_gdf: gpd.GeoDataFrame | None = None,
    polygons_to_segments: Any = polygons_to_segments_with_attr,
    bldg_plot_type: str = "footprint_colored",  # "point" | "footprint" | "footprint_colored"
    points_df: pd.DataFrame | None = None,      # required if bldg_plot_type == "point"
    bucket_attr: str = "_bucket",
    palette: Mapping[Any, str] | None = None,
    default_bucket_color: str = "#7f7f7f@30",
    # region padding
    main_lon_pad_frac: float = 0.10,
    main_lat_pad_frac: float = 0.15,
    # basemap
    projection_main: str = "M20c",
    frame_main: str = "ag",
    tile_source: Any = None,
    config_main: Mapping[str, str] | None = None,
    # optional overlay to lighten basemap
    lighten_overlay: bool = False,
    overlay_alpha: int = 70,
    # boundary styling (if boundary_main is provided)
    boundary_pen: str = "0.5p,black",
    boundary_fill: str = "lightyellow@30",
    # footprint styling
    footprint_pen: str | None = "0.25p,black",
    footprint_fill: str = "dimgray",
    point_style: str = "c0.05c",
    point_transparency: int = 20,
    # scale bar
    map_scale_main: str = "jTR+w2k+o0.5c/0.5c",
    # inset
    add_inset: bool = False,
    region_inset_base_gmt: Sequence[float] | None = None,
    inset_lon_pad_frac: float = 0.05,
    inset_lat_pad_frac: float = 0.10,
    inset_north_extra_latpad: float = 2.0,
    inset_position: str = "jBR+o0.5c/0.5c",
    inset_box: str = "+p1.5p,black",
    projection_inset: str = "M4.5c",
    coast_land: str = "lightgrey",
    coast_borders: str = "3/thin",
    coast_shorelines: str = "1/thin",
    coast_water: str = "lightblue",
    coast_resolution: str = "h",
    coast_frame: str = "af",
    inset_highlights: Sequence[Mapping[str, Any]] | None = None,
    epicenter: Epicenter | None = None,
    map_scale_inset: str = "jTR+w20k+o0.2c/0.2c",
    # output
    savefig: bool = False,
    outpath: str | Path = "region_figure.png",
    dpi: int = 300,
    transparent: bool = True,
    show: bool = True,
    show_width: int | None = 1000,
) -> tuple[Any, list[float], list[float] | None]:
    """
    Generic PyGMT tilemap plot:
      - tilemap background
      - optional lightening overlay
      - optional boundary polygon
      - optional building points/footprints (colored by bucket)
      - optional inset with coast + highlights + epicenter

    Returns
    -------
    fig, region_main_expanded, region_inset_reduced
    """
    _require(pygmt, "pygmt")
    tile_source = default_tile_source(tile_source)

    if palette is None:
        palette = {}

    # Default config chosen to match your original "region + inset" figure
    if config_main is None:
        config_main = {
            "FONT": "24p",
            "MAP_GRID_PEN_PRIMARY": "0.5p,gray20,--",
        }

    if bldg_plot_type == "point" and points_df is None:
        raise ValueError("points_df must be provided when bldg_plot_type='point'.")

    region_main_expanded = expand_region(region_main_gmt, lon_pad_frac=main_lon_pad_frac, lat_pad_frac=main_lat_pad_frac)

    region_inset_reduced = None
    if add_inset:
        if region_inset_base_gmt is None:
            raise ValueError("region_inset_base_gmt must be provided when add_inset=True.")
        region_inset_reduced = reduce_region(
            region_inset_base_gmt,
            lon_pad_frac=inset_lon_pad_frac,
            lat_pad_frac=inset_lat_pad_frac,
            north_extra_latpad=inset_north_extra_latpad,
            east_mode="shrink",
        )

    cfg = dict(config_main or {})
    fig = pygmt.Figure()

    def _plot_buildings():
        if buildings_gdf is None:
            return

        if bldg_plot_type == "point":
            fig.plot(x=points_df["lon"], y=points_df["lat"], style=point_style, transparency=point_transparency)
            return

        if bldg_plot_type == "footprint":
            for df_poly, _ in polygons_to_segments(buildings_gdf):
                if footprint_pen is None:
                    fig.plot(data=df_poly, fill=footprint_fill)
                else:
                    fig.plot(data=df_poly, pen=footprint_pen, fill=footprint_fill)
            return

        if bldg_plot_type == "footprint_colored":
            for df_poly, val in polygons_to_segments(buildings_gdf, attr=bucket_attr):
                fill = palette.get(val, default_bucket_color)
                if footprint_pen is None:
                    fig.plot(data=df_poly, fill=fill)
                else:
                    fig.plot(data=df_poly, pen=footprint_pen, fill=fill)
            return

        raise ValueError("bldg_plot_type must be one of {'point','footprint','footprint_colored'}.")

    # Apply config as a context manager
    with pygmt.config(**cfg):
        fig.tilemap(region=region_main_expanded, projection=projection_main, source=tile_source, zoom="auto", frame=frame_main)

        if lighten_overlay:
            w, e, s, n = region_main_expanded
            fig.plot(x=[w, e, e, w, w], y=[s, s, n, n, s], pen="0p", fill=f"white@{int(overlay_alpha)}")

        if boundary_main is not None:
            fig.plot(data=boundary_main, pen=boundary_pen, fill=boundary_fill)

        _plot_buildings()
        fig.basemap(map_scale=map_scale_main)

        if add_inset and region_inset_reduced is not None:
            with fig.inset(position=inset_position, box=inset_box, region=region_inset_reduced, projection=projection_inset):
                fig.coast(
                    land=coast_land,
                    borders=coast_borders,
                    shorelines=coast_shorelines,
                    water=coast_water,
                    resolution=coast_resolution,
                    frame=coast_frame,
                )

                if inset_highlights:
                    for h in inset_highlights:
                        fig.plot(data=h["boundary_df"], fill=h.get("fill", "red"), pen=h.get("pen", "1.0p,black"))
                        if "label" in h and "label_region_gmt" in h:
                            lon_c, lat_c = region_center(h["label_region_gmt"])
                            fig.text(
                                x=lon_c,
                                y=lat_c,
                                text=h["label"],
                                font=h.get("font", "14p,Helvetica-Bold,black"),
                                offset=h.get("offset", "-1.4c/-0.8c"),
                                justify=h.get("justify", "LB"),
                                clearance=h.get("clearance", "3p/2p"),
                                fill=h.get("fillbox", None),
                            )

                if epicenter is not None:
                    fig.plot(x=epicenter.lon, y=epicenter.lat, style=epicenter.style, fill=epicenter.fill, pen=epicenter.pen)
                    fig.text(
                        x=epicenter.lon,
                        y=epicenter.lat,
                        text=epicenter.label,
                        font=epicenter.font,
                        offset=epicenter.offset,
                        justify="LB",
                        clearance="3p/2p",
                    )

                fig.basemap(map_scale=map_scale_inset)

    if savefig:
        _save_pygmt(fig, outpath, dpi=dpi, transparent=transparent)

    if show:
        if show_width is None:
            fig.show()
        else:
            fig.show(width=show_width)

    return fig, region_main_expanded, region_inset_reduced


def plot_selected_neighborhoods_pygmt(
    *,
    neigh: gpd.GeoDataFrame,
    neigh_sel: gpd.GeoDataFrame,
    targets: Sequence[str] | None = None,
    name_col: str,
    selected_label: str = "Selected neighborhoods",
    projection: str = "M20c",
    frame: str = "ag",
    tile_source: Any = None,
    config: Mapping[str, str] | None = None,
    # styles
    fill_unsel: str = "lightyellow@60",
    pen_unsel: str = "0.5p,black",
    fill_sel: str = "royalblue@40",
    pen_sel: str = "0.75p,black",
    # legend
    legend: bool = True,
    legend_position: str = "JTL+jTL+o0.5c/0.5c",
    legend_box: str = "+gwhite+p1p",
    legend_font_annot_primary: str = "24p,Helvetica,black",
    # scale bar
    map_scale: str = "jTR+w2k+o0.5c/0.5c",
    # output
    savefig: bool = False,
    outpath: str | Path = "selected_neighbors.png",
    dpi: int = 300,
    show: bool = True,
    show_width: int | None = None,
) -> tuple[Any, list[float]]:
    """
    Neighborhood selection plot (fixes PyGMT legend spec handling).

    Notes
    -----
    - If `targets` is None, they are inferred from `neigh_sel[name_col]`.

    Default `config` matches your original neighborhood figure.
    """
    _require(pygmt, "pygmt")
    tile_source = default_tile_source(tile_source)

    if config is None:
        config = {
            "FONT_ANNOT_PRIMARY": "17p,Helvetica,black",
            "MAP_GRID_PEN_PRIMARY": "1p,gray50,--",
            "MAP_FRAME_AXES": "WSne",
        }

    # If targets aren't provided, infer them from neigh_sel (useful for driver scripts).
    if targets is None:
        if name_col not in neigh_sel.columns:
            raise KeyError(f"name_col='{name_col}' not found in neigh_sel columns: {list(neigh_sel.columns)}")
        targets = list(pd.Series(neigh_sel[name_col]).dropna().unique())

    neigh_unsel = neigh[~neigh[name_col].isin(list(targets))].copy()
    neigh_unsel["geometry"] = neigh_unsel.buffer(0)

    region_gmt = bounds_to_region_gmt(neigh.total_bounds)

    fig = pygmt.Figure()
    with pygmt.config(**dict(config)):
        fig.tilemap(region=region_gmt, projection=projection, source=tile_source, zoom="auto", frame=frame)
        fig.plot(data=neigh_unsel, pen=pen_unsel, fill=fill_unsel)
        fig.plot(data=neigh_sel, pen=pen_sel, fill=fill_sel)
        fig.basemap(map_scale=map_scale)

        if legend:
            legend_text = f"S 0.4c s 1.2c {fill_sel},black 0.75p 1.0c {selected_label}\n"
            with pygmt.config(FONT_ANNOT_PRIMARY=legend_font_annot_primary):
                fig.legend(spec=io.StringIO(legend_text), position=legend_position, box=legend_box)

    if savefig:
        _save_pygmt(fig, outpath, dpi=dpi, transparent=True)

    if show:
        if show_width is None:
            fig.show()
        else:
            fig.show(width=show_width)

    return fig, region_gmt


def plot_overview_with_epicenter_pygmt(
    *,
    region_all_gmt: Sequence[float],
    boundary_df: Any,
    region_label_gmt: Sequence[float],
    region_label: str,
    region_fill: str = "red",
    region_pen: str = "0.7p,black",
    region_transparency: int = 20,
    epicenter: Epicenter | None = None,
    # padding/cropping (kept consistent with original logic)
    lon_pad_frac: float = 0.10,
    lat_pad_frac: float = 0.20,
    east_mode: str = "keep",
    east_custom: float | None = None,
    north_latpad_multiplier: float = 1.15,
    # basemap
    projection: str = "M13c",
    frame: Any | None = None,
    tile_source: Any = None,
    config: Mapping[str, str] | None = None,
    # text styling
    region_font: str = "18p,Helvetica-Bold,black",
    region_offset: str = "-1.0c/1.0c",
    # scale bar
    map_scale: str = "jTR+w20k+o0.2c/0.2c",
    # output
    savefig: bool = False,
    outpath: str | Path = "region_overview.png",
    dpi: int = 300,
    transparent: bool = True,
    show: bool = True,
    show_width: int | None = 1000,
) -> tuple[Any, list[float]]:
    """
    Overview tilemap with one highlighted region + optional epicenter.
    (Cleaned-up version of the Bay Area overview figure.)
    """
    _require(pygmt, "pygmt")
    tile_source = default_tile_source(tile_source)

    if config is None:
        config = {
            "FONT_ANNOT_PRIMARY": "12p,Helvetica",
            "MAP_GRID_PEN_PRIMARY": "1p,gray30,--",
        }

    if frame is None:
        frame = [
            "xag0.2f0.1+lLongitude",
            "yag0.2f0.1+lLatitude",
            "WSne",
        ]

    region_reduced = reduce_region(
        region_all_gmt,
        lon_pad_frac=lon_pad_frac,
        lat_pad_frac=lat_pad_frac,
        north_extra_latpad=north_latpad_multiplier,
        east_mode=east_mode,
        east_custom=east_custom,
    )

    fig = pygmt.Figure()
    with pygmt.config(**dict(config)):
        fig.tilemap(region=region_reduced, projection=projection, source=tile_source, zoom="auto", frame=frame)

        fig.plot(data=boundary_df, fill=region_fill, pen=region_pen, transparency=region_transparency)

        lon_c, lat_c = region_center(region_label_gmt)
        fig.text(
            x=lon_c,
            y=lat_c,
            text=region_label,
            font=region_font,
            offset=region_offset,
            justify="LB",
            clearance="3p/2p",
        )

        if epicenter is not None:
            fig.plot(x=epicenter.lon, y=epicenter.lat, style=epicenter.style, fill=epicenter.fill, pen=epicenter.pen)
            fig.text(
                x=epicenter.lon,
                y=epicenter.lat,
                text=epicenter.label,
                font=epicenter.font,
                offset=epicenter.offset,
                justify="LB",
                clearance="3p/2p",
            )

        fig.basemap(map_scale=map_scale)

    if savefig:
        _save_pygmt(fig, outpath, dpi=dpi, transparent=transparent)

    if show:
        if show_width is None:
            fig.show()
        else:
            fig.show(width=show_width)

    return fig, region_reduced


# -----------------------------------------------------------------------------
# Vs30 raster map (Matplotlib + rasterio + contextily)
# -----------------------------------------------------------------------------
def plot_vs30_map_from_boundary(
    *,
    vs30_path: str | Path,
    target_boundary: gpd.GeoDataFrame,
    boundary_outline: gpd.GeoDataFrame | None = None,
    boundary_outline_style: dict[str, Any] | None = None,
    cap_max: float | None = 800.0,
    global_minmax: bool = True,
    cmap: str = "BrBG_r",
    alpha_vs30: float = 0.7,
    basemap_alpha: float = 0.8,
    margin: float = 0.10,
    figsize: tuple[int, int] = (8, 8),
    cbar_shrink: float = 0.65,
    savefig: bool = False,
    outpath: str | Path = "vs30_map.png",
    dpi: int = 300,
    show: bool = True,
    verbose: bool = True,
) -> tuple[plt.Figure, plt.Axes, dict[str, Any]]:
    """
    Plot a Vs30 raster clipped to a boundary on top of an OSM basemap.

    Parameters
    ----------
    target_boundary:
        GeoDataFrame with boundary polygon(s) used for clipping.
    boundary_outline:
        Optional second outline layer (e.g., selected neighborhoods) plotted with dashed lines.
    """
    _require(rasterio, "rasterio")
    _require(ctx, "contextily")

    vs30_path = Path(vs30_path)

    with rasterio.open(vs30_path) as src:
        target_proj = target_boundary.to_crs(src.crs)
        shapes = list(target_proj.geometry)

        vs30_clip, out_transform = rio_mask(src, shapes=shapes, crop=True)
        vs30_clip = vs30_clip[0].astype(float)

        if src.nodata is not None:
            vs30_clip[vs30_clip == src.nodata] = np.nan

        if global_minmax:
            band = src.read(1).astype(float)
            if src.nodata is not None:
                band[band == src.nodata] = np.nan
            vmin = float(np.nanmin(band))
            vmax = float(np.nanmax(band))
        else:
            vmin = float(np.nanmin(vs30_clip))
            vmax = float(np.nanmax(vs30_clip))

        if cap_max is not None:
            vmax = min(vmax, float(cap_max))

        raster_crs = src.crs

    if verbose:
        print("Minimum vs30 value:", vmin)
        print("Maximum vs30 value:", vmax)

    fig, ax = plt.subplots(figsize=figsize)

    extent = plotting_extent(vs30_clip, out_transform)
    xmin, xmax, ymin, ymax = extent

    dx = (xmax - xmin) * margin
    dy = (ymax - ymin) * margin
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

    img = ax.imshow(
        vs30_clip,
        extent=extent,
        origin="upper",
        cmap=cmap,
        zorder=2,
        alpha=alpha_vs30,
        vmin=vmin,
        vmax=vmax,
    )

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=raster_crs, zorder=1, alpha=basemap_alpha)

    target_proj.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, zorder=3)

    if boundary_outline is not None:
        style = {"edgecolor": "black", "linewidth": 0.5, "linestyle": "--", "zorder": 3}
        if boundary_outline_style:
            style.update(boundary_outline_style)
        boundary_outline.to_crs(raster_crs).boundary.plot(ax=ax, **style)

    cbar = plt.colorbar(img, ax=ax, shrink=cbar_shrink)
    cbar.set_label(r"$V_{s30}$ (m/s)", fontsize=20)
    cbar.ax.tick_params(labelsize=14)

    ax.set_xlabel("Longitude", fontsize=20)
    ax.set_ylabel("Latitude", fontsize=20)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", labelsize=14)

    plt.tight_layout()

    saved = None
    if savefig:
        saved = _save_matplotlib(fig, outpath, dpi=dpi)

    if show:
        plt.show()

    info = {
        "target_proj": target_proj,
        "vs30_clip": vs30_clip,
        "out_transform": out_transform,
        "vs30_min": vmin,
        "vs30_max": vmax,
        "saved_path": str(saved) if saved else None,
        "raster_crs": raster_crs,
    }
    return fig, ax, info


def plot_vs30_map(
    *,
    target_region: str,
    vs30_path: str | Path,
    place_suffix: str = "California, USA",
    # SF NE special case
    sf_neighborhoods_geojson: str | Path | None = None,
    sf_city_query: str = "San Francisco, California, USA",
    sf_targets: Sequence[str] | None = None,
    sf_name_col: str | None = None,
    # North Berkeley special case
    northberk_districts_shp: str | Path | None = None,
    northberk_district_values: Sequence[str] = ("5", "6"),
    use_northberk_boundary: bool = True,
    # plotting
    cap_max: float | None = 800.0,
    global_minmax: bool = True,
    cbar_shrink: float | None = None,
    cbar_shrink_by_region: Mapping[str, float] | None = None,
    savefig: bool = False,
    outdir: str | Path = ".",
    dpi: int = 300,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, dict[str, Any]]:
    """
    High-level wrapper matching the original workflow:
    build a boundary from `target_region` (with a couple of special cases),
    then call `plot_vs30_map_from_boundary`.
    """
    _require(ox, "osmnx")

    boundary_outline = None

    if target_region == "SanFrancisco_NE":
        if sf_neighborhoods_geojson is None:
            raise ValueError("sf_neighborhoods_geojson is required for target_region='SanFrancisco_NE'.")
        if sf_targets is None:
            sf_targets = [
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
        boundary_gdf, neigh_sel = sf_neighborhood_union(
            sf_neighborhoods_geojson,
            targets=sf_targets,
            city_query=sf_city_query,
            name_col=sf_name_col,
        )
        boundary_outline = neigh_sel  # dashed outline layer

    elif target_region == "NorthBerkeley":
        if northberk_districts_shp is None:
            raise ValueError("northberk_districts_shp is required for target_region='NorthBerkeley'.")
        nb = north_berkeley_union(northberk_districts_shp, districts=northberk_district_values)
        boundary_gdf = nb if use_northberk_boundary else geocode_boundary(f"Berkeley, {place_suffix}")

    else:
        boundary_gdf = geocode_boundary(f"{target_region}, {place_suffix}")

    if cbar_shrink is None:
        default_map = {"SanFrancisco_NE": 0.85, "Belmont": 0.5, "Livermore": 0.6, "Milpitas": 0.66}
        if cbar_shrink_by_region:
            default_map.update(dict(cbar_shrink_by_region))
        cbar_shrink = default_map.get(target_region, 0.65)

    outdir = Path(outdir)
    outpath = outdir / f"{target_region}_vs30map.png"

    return plot_vs30_map_from_boundary(
        vs30_path=vs30_path,
        target_boundary=boundary_gdf,
        boundary_outline=boundary_outline,
        cap_max=cap_max,
        global_minmax=global_minmax,
        cbar_shrink=float(cbar_shrink),
        savefig=savefig,
        outpath=outpath,
        dpi=dpi,
        show=show,
    )
