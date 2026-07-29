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
# Shared ArcGIS REST download
# ---------------------------------------------------------------------------

def query_feature_service(
    url: str,
    layer: int,
    where: str = "1=1",
    out_fields: str = "*",
    crs: int = 26913,
    label: str = "features",
    aoi: "gpd.GeoDataFrame | gpd.GeoSeries | None" = None,
) -> gpd.GeoDataFrame:
    """
    Download all matching features from an ArcGIS Feature/Map Service layer.

    OID-batched paging (respecting the layer's ``maxRecordCount``) so large
    layers download reliably. Shared by :func:`download_cft` and
    :func:`treatment_interactions.twig.download_twig` — keeping the package free of
    any ``tealom`` import (portability goal) while both sources use identical
    request handling.

    Parameters
    ----------
    url : str
        FeatureServer / MapServer root URL.
    layer : int
        Layer index.
    where : str
        SQL ``where`` clause (default ``"1=1"`` = all features).
    out_fields : str
        Comma-delimited field list or ``"*"``.
    crs : int
        EPSG code to reproject the result to (default NAD83 UTM 13N).
    label : str
        Human-readable noun for progress messages.
    aoi : GeoDataFrame / GeoSeries, optional
        If given, only features whose envelope intersects this AOI's bounding box
        are requested (a server-side pre-filter; caller should still do an exact
        ``sjoin`` clip afterwards). The AOI is reprojected to the service SR.

    Returns
    -------
    GeoDataFrame in ``EPSG:crs`` (empty with a ``geometry`` column if nothing matched).
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
    id_params = {"where": where, "returnIdsOnly": "True", "f": "pjson"}
    if aoi is not None:
        xmin, ymin, xmax, ymax = aoi.to_crs(epsg=srn).total_bounds
        id_params.update({
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": str(srn),
            "spatialRel": "esriSpatialRelIntersects",
        })
    o_info = requests.get(url2, params=id_params, timeout=60).json()

    oid_name = o_info["objectIdFieldName"]
    oids     = o_info.get("objectIds") or []
    n        = len(oids)
    print(f"Fetching {n} {label} from layer {layer}…")

    batches = []
    for i in range(0, n, maxrcn):
        batch = oids[i: i + maxrcn]
        idstr = f"{oid_name} in ({str(batch)[1:-1]})"
        prm   = {
            "where": idstr,
            "outFields": out_fields,
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
        return gpd.GeoDataFrame(columns=["geometry"], crs=sr).to_crs(epsg=crs)

    gdf = pd.concat(
        [g.dropna(axis=1, how="all") for g in valid], ignore_index=True
    )
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=sr).to_crs(epsg=crs)


def download_cft(
    url: str = CFT_URL,
    layer: int = CFT_LAYER,
    out_fp: str | Path | None = None,
    crs: int = 26913,
    where: str = "1=1",
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
    where : str
        Optional SQL filter (e.g. ``"YEAR_COMP >= 2014"``); default all features.

    Returns
    -------
    GeoDataFrame
    """
    gdf = query_feature_service(url, layer, where=where, crs=crs, label="CFT treatment features")
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
    gis_col: str = "ACRES_GIS",
    mgt_col: str = "ACRES_MGT",
    id_col: str = "OBJECTID",
    group_col: str = "ACTIVITY",
) -> gpd.GeoDataFrame:
    """
    Flag records where the GIS polygon is much larger than reported management
    acres (likely NEPA boundary captures or data entry errors).

    Adds columns:
        gis_acres       — recalculated from geometry
        acres_diff      — <gis_col> - <mgt_col>
        log_ac_resid    — residual from log-log OLS fit
        log_ac_pred     — OLS predicted log10(<mgt_col>)
        ac_flag         — True where GIS >> MGT (residual < -n_std·σ)
        ac_flag_abs     — True where acres_diff exceeds 99th-pctile by group

    Does NOT filter — callers decide which records to exclude.

    Parameters
    ----------
    gdf : GeoDataFrame  (must have <gis_col>, <mgt_col>, <id_col>, <group_col>)
    n_std : float       Residual threshold in standard deviations (default 3.0)
    min_area_acres : float  Absolute floor for the GIS-inflation flag
    gis_col, mgt_col : str
        Reported GIS / management acreage columns. TWIG uses ``acres`` for both
        the reported value; CFT carries distinct ``ACRES_GIS`` / ``ACRES_MGT``.
    id_col : str
        Unique record identifier used to merge the flags back.
    group_col : str
        Column the per-group 99th-percentile absolute flag is computed within.

    Returns
    -------
    GeoDataFrame with flag columns added. If ``mgt_col`` is absent (a source that
    reports only one acreage), the residual flag is skipped and only the
    per-group absolute flag is computed.

    Notes
    -----
    Defaults reproduce the original CFT behaviour byte-for-byte.
    """
    out = gdf.copy()

    if "gis_acres" not in out.columns:
        out["gis_acres"] = out.geometry.area * M2_TO_ACRES

    has_mgt = mgt_col in out.columns
    if not has_mgt:
        print(f"[warn] '{mgt_col}' not in source columns — skipping the log-log residual "
              f"flag; only the per-'{group_col}' absolute flag is computed.")
        out["acres_diff"]   = out["gis_acres"]
        out["ac_flag"]      = False
        out["log_ac_pred"]  = np.nan
        out["log_ac_resid"] = np.nan
    else:
        out["acres_diff"] = out[gis_col] - out[mgt_col]

        # --- log-log OLS
        mask = (
            (out["gis_acres"] > 0)
            & (out[mgt_col] > 0)
            & out[mgt_col].notna()
        )
        df_v = out[mask].copy()
        log_gis = np.log10(df_v["gis_acres"])
        log_mgt = np.log10(df_v[mgt_col])

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
        out = out.merge(df_v[[id_col] + flag_cols], on=id_col, how="left")
        out["ac_flag"]      = out["ac_flag"].fillna(False).astype(bool)
        out["log_ac_pred"]  = out["log_ac_pred"].fillna(np.nan)
        out["log_ac_resid"] = out["log_ac_resid"].fillna(np.nan)

    # --- absolute flag: per-group 99th percentile of acres_diff
    p99 = out.groupby(group_col)["acres_diff"].transform(lambda x: x.quantile(0.99))
    out["ac_flag_abs"] = out["acres_diff"] > p99

    n_flag     = out["ac_flag"].sum()
    n_flag_abs = out["ac_flag_abs"].sum()
    print(f"Flagged (residual): {n_flag:,}  |  Flagged (absolute): {n_flag_abs:,}")

    return out.reset_index(drop=True)
