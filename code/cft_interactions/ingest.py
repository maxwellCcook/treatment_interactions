"""
Colorado Forest Tracker — ingest, geometry cleaning, and acre-mismatch flagging.

author: maxwell.cook@colostate.edu
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
from scipy import stats
from shapely.validation import make_valid


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CFT_URL   = "https://services3.arcgis.com/unh8qj8OkZd7wlfi/arcgis/rest/services/Forest_Tracker_v2/FeatureServer"
CFT_LAYER = 10
M2_TO_ACRES = 0.000247105
MIN_AREA_ACRES = (30 * 30) * M2_TO_ACRES  # ~0.22 ac — one 30 m pixel


# ---------------------------------------------------------------------------
# REST download
# ---------------------------------------------------------------------------

def download_cft(
    url: str = CFT_URL,
    layer: int = CFT_LAYER,
    out_fp: str | Path | None = None,
    crs: int = 26913,
) -> gpd.GeoDataFrame:
    """
    Download all features from the CFT ArcGIS Feature Service.

    Parameters
    ----------
    url : str
        FeatureServer root URL.
    layer : int
        Layer index (10 = treatment polygons).
    out_fp : path-like, optional
        If provided, save the raw result as a GeoPackage before returning.
    crs : int
        EPSG code to reproject to (default NAD83 UTM 13N).

    Returns
    -------
    GeoDataFrame
    """
    s_info = requests.get(url + "?f=pjson", timeout=30).json()
    srn = s_info["spatialReference"]["wkid"]
    sr  = f"EPSG:{srn}"

    url1 = f"{url}/{layer}"
    try:
        l_info = requests.get(url1 + "?f=pjson", timeout=30).json()
        maxrcn = int(l_info.get("maxRecordCount", 1000))
    except Exception:
        maxrcn = 1000

    url2 = url1 + "/query?"
    o_info = requests.get(url2, params={
        "where": "1=1",
        "returnIdsOnly": "True",
        "f": "pjson",
    }, timeout=60).json()

    oid_name = o_info["objectIdFieldName"]
    oids     = o_info["objectIds"]
    n        = len(oids)
    print(f"Fetching {n} features from CFT layer {layer}…")

    batches = []
    for i in range(0, n, maxrcn):
        batch = oids[i: i + maxrcn]
        idstr = f"{oid_name} in ({str(batch)[1:-1]})"
        prm   = {
            "where": idstr,
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": srn,
            "f": "pgeojson",
        }
        resp = requests.post(url2, data=prm, timeout=120)
        try:
            ftrs = resp.json()["features"]
        except Exception:
            prm["f"] = "geojson"
            ftrs = requests.post(url2, data=prm, timeout=120).json()["features"]
        batches.append(gpd.GeoDataFrame.from_features(ftrs, crs=sr))
        if (i // maxrcn + 1) % 10 == 0:
            print(f"  …{i + len(batch):,} / {n:,}")

    valid = [g for g in batches if not g.empty]
    if not valid:
        return gpd.GeoDataFrame(columns=["geometry"])

    gdf = pd.concat(
        [g.dropna(axis=1, how="all") for g in valid], ignore_index=True
    )
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=sr).to_crs(epsg=crs)
    print(f"Downloaded {len(gdf):,} treatment features.")

    if out_fp is not None:
        out_fp = Path(out_fp)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_fp)
        print(f"Saved → {out_fp}")

    return gdf


def load_or_download(
    out_fp: str | Path,
    url: str = CFT_URL,
    layer: int = CFT_LAYER,
    crs: int = 26913,
) -> gpd.GeoDataFrame:
    """Load from disk if the file exists, otherwise download and save."""
    out_fp = Path(out_fp)
    if out_fp.exists():
        gdf = gpd.read_file(out_fp)
        print(f"Loaded {len(gdf):,} features from {out_fp}")
        return gdf
    return download_cft(url=url, layer=layer, out_fp=out_fp, crs=crs)


# ---------------------------------------------------------------------------
# Geometry cleaning
# ---------------------------------------------------------------------------

def clean_geometry(
    gdf: gpd.GeoDataFrame,
    snap_m: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    Repair invalid geometries, apply precision snap, and drop empties.

    Parameters
    ----------
    gdf : GeoDataFrame
    snap_m : float
        Precision grid size in map units (metres). Default 1 m.

    Returns
    -------
    GeoDataFrame — copy with repaired geometry column.
    """
    out = gdf.copy()
    out["geometry"] = out.geometry.apply(
        lambda g: make_valid(g.buffer(0)) if g is not None else g
    )
    out["geometry"] = out.geometry.apply(
        lambda g: shapely.set_precision(g, snap_m)
        if (g is not None and not g.is_empty) else g
    )
    # Two-step filter: is_empty first (None → False, so Nones pass through),
    # then notna() on a series that no longer contains empty geometries → no warning.
    out = out[~out.geometry.is_empty].copy()
    out = out[out.geometry.notna()].copy()
    out["gis_acres"] = out.geometry.area * M2_TO_ACRES
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Acre-mismatch flagging
# ---------------------------------------------------------------------------

def flag_acre_mismatch(
    gdf: gpd.GeoDataFrame,
    n_std: float = 3.0,
    min_area_acres: float = MIN_AREA_ACRES,
) -> gpd.GeoDataFrame:
    """
    Flag records where the GIS polygon is much larger than reported management
    acres (likely NEPA boundary captures or data entry errors).

    Adds columns:
        gis_acres       — recalculated from geometry
        acres_diff      — ACRES_GIS - ACRES_MGT
        log_ac_resid    — residual from log-log OLS fit
        log_ac_pred     — OLS predicted log10(ACRES_MGT)
        ac_flag         — True where GIS >> MGT (residual < -n_std·σ)
        ac_flag_abs     — True where acres_diff exceeds 99th-pctile by activity

    Does NOT filter — callers decide which records to exclude.

    Parameters
    ----------
    gdf : GeoDataFrame  (must have ACRES_GIS, ACRES_MGT, OBJECTID, ACTIVITY)
    n_std : float       Residual threshold in standard deviations (default 3.0)
    min_area_acres : float  Absolute floor for the GIS-inflation flag

    Returns
    -------
    GeoDataFrame with flag columns added.
    """
    out = gdf.copy()

    if "gis_acres" not in out.columns:
        out["gis_acres"] = out.geometry.area * M2_TO_ACRES

    out["acres_diff"] = out["ACRES_GIS"] - out["ACRES_MGT"]

    # --- log-log OLS
    mask = (
        (out["gis_acres"] > 0)
        & (out["ACRES_MGT"] > 0)
        & out["ACRES_MGT"].notna()
    )
    df_v = out[mask].copy()
    log_gis = np.log10(df_v["gis_acres"])
    log_mgt = np.log10(df_v["ACRES_MGT"])

    slope, intercept, r, *_ = stats.linregress(log_gis, log_mgt)
    print(f"Log-log OLS: slope={slope:.3f}, intercept={intercept:.3f}, R²={r**2:.3f}")

    df_v["log_ac_pred"]  = intercept + slope * log_gis
    df_v["log_ac_resid"] = log_mgt - df_v["log_ac_pred"]

    resid_std = df_v["log_ac_resid"].std()
    df_v["ac_flag"] = (
        (df_v["log_ac_resid"] < -n_std * resid_std)
        & (df_v["acres_diff"] > min_area_acres)
    )

    flag_cols = ["ac_flag", "log_ac_pred", "log_ac_resid"]
    out = out.merge(df_v[["OBJECTID"] + flag_cols], on="OBJECTID", how="left")
    out["ac_flag"]      = out["ac_flag"].fillna(False).astype(bool)
    out["log_ac_pred"]  = out["log_ac_pred"].fillna(np.nan)
    out["log_ac_resid"] = out["log_ac_resid"].fillna(np.nan)

    # --- absolute flag: per-activity 99th percentile of acres_diff
    p99 = out.groupby("ACTIVITY")["acres_diff"].transform(lambda x: x.quantile(0.99))
    out["ac_flag_abs"] = out["acres_diff"] > p99

    n_flag     = out["ac_flag"].sum()
    n_flag_abs = out["ac_flag_abs"].sum()
    print(f"Flagged (residual): {n_flag:,}  |  Flagged (absolute): {n_flag_abs:,}")

    return out.reset_index(drop=True)
