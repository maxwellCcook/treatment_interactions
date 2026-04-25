"""
Atomic decomposition of CFT treatment polygons into a non-overlapping
treatment interactions database.

author: maxwell.cook@colostate.edu
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import polygonize, unary_union


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

M2_TO_ACRES = 0.000247105
MIN_AREA_M2 = (30 * 30) / M2_TO_ACRES  # ~1 30 m pixel in m²

THIN_TYPES    = {"Manual", "Mechanical"}
BURNING_TYPES = {"Broadcast Burn"}

# Attributes preserved from each source CFT polygon into EVENTS records
EVENT_FIELDS = [
    "ACTIVITY", "YEAR_COMP", "AGENCY_C",
    "FUND_SOURCE", "FUND_TYPE", "PARTNERS", "LANDOWNER", "MGT_TYPE",
]


# ---------------------------------------------------------------------------
# Geometry helpers  (vectorized via Shapely 2.0 array API)
# ---------------------------------------------------------------------------

def make_valid_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Repair invalid geometries and drop empty/null rows."""
    out = gdf.copy()
    out["geometry"] = shapely.make_valid(out.geometry.values)
    # Two-step filter: is_empty first (None → False, so Nones pass through),
    # then notna() on a series that no longer contains empty geometries → no warning.
    out = out[~out.geometry.is_empty].copy()
    return out[out.geometry.notna()].copy()


def drop_slivers(
    gdf: gpd.GeoDataFrame,
    min_area_m2: float = MIN_AREA_M2,
) -> gpd.GeoDataFrame:
    return gdf[gdf.geometry.area > min_area_m2].copy()


def snap_to_network(
    gdf: gpd.GeoDataFrame,
    tolerance: float = 0.5,
) -> gpd.GeoDataFrame:
    """
    Snap all geometries to a shared boundary reference to suppress near-
    identical boundary drift before atomic decomposition. Snapping to the
    full unary_union boundary forces all vertices toward the same shared
    edge set, which is more stable than pairwise snapping at n=40k+.

    Uses Shapely 2.0's vectorized snap/make_valid (no Python-level loop).
    """
    gdf = make_valid_gdf(gdf)
    reference = unary_union(gdf.geometry)

    snapped = shapely.snap(gdf.geometry.values, reference, tolerance)
    snapped = shapely.make_valid(snapped)

    out = gdf.copy()
    out["geometry"] = snapped
    out = out[~out.geometry.is_empty].copy()
    return out[out.geometry.notna()].copy()


# ---------------------------------------------------------------------------
# Atomic decomposition
# ---------------------------------------------------------------------------

def build_atomic_zones(
    gdf: gpd.GeoDataFrame,
    snap_tolerance: float = 0.5,
    min_area_m2: float = MIN_AREA_M2,
) -> gpd.GeoDataFrame:
    """
    Decompose all treatment polygons into non-overlapping atomic zones.

    Each output polygon is the unique area covered by a specific combination
    of source treatments. Builds from the union of all polygon boundaries,
    then polygonizes to produce the atomic partition — avoiding the O(n²)
    pairwise overlay for large datasets.

    Parameters
    ----------
    gdf : GeoDataFrame in projected CRS
    snap_tolerance : float
        Pre-snap tolerance in metres (default 0.5 m).
    min_area_m2 : float
        Sliver filter threshold in m².
    """
    assert gdf.crs.is_projected, "Input must be in a projected CRS."

    gdf = snap_to_network(gdf, tolerance=snap_tolerance)
    gdf = drop_slivers(gdf, min_area_m2)

    all_boundaries = unary_union(shapely.get_parts(shapely.boundary(gdf.geometry.values)))
    atomic_faces   = list(polygonize(all_boundaries))

    if not atomic_faces:
        raise ValueError("polygonize returned no faces — check input geometry validity.")

    atomic = gpd.GeoDataFrame(geometry=atomic_faces, crs=gdf.crs)
    atomic = drop_slivers(atomic, min_area_m2).reset_index(drop=True)
    atomic["_atom_id"] = atomic.index
    return atomic


def assign_treatments_to_atoms(
    atomic: gpd.GeoDataFrame,
    gdf: gpd.GeoDataFrame,
    source_id_col: str = "OBJECTID",
) -> gpd.GeoDataFrame:
    """
    Spatial join: for each atomic zone find all source CFT polygons that
    contain it. Uses representative_point() (guaranteed interior) to avoid
    boundary edge cases.

    Returns atomic GDF with SOURCE_IDS column (list of OBJECTIDs).
    """
    atomic_pts = atomic.copy()
    atomic_pts["geometry"] = atomic.geometry.representative_point()

    joined = gpd.sjoin(
        atomic_pts[["_atom_id", "geometry"]],
        gdf[[source_id_col, "geometry"]],
        how="left",
        predicate="within",
    )

    atom_sources = (
        joined.groupby("_atom_id")[source_id_col]
        .apply(lambda s: sorted(s.dropna().astype(int).tolist()))
        .reset_index()
        .rename(columns={source_id_col: "SOURCE_IDS"})
    )

    return atomic.merge(atom_sources, on="_atom_id", how="left")


# ---------------------------------------------------------------------------
# Attribute derivation helpers
# ---------------------------------------------------------------------------

def _is_complete(events: list[dict]) -> bool:
    """True if a burn followed (or co-occurred with) a thin in this atom."""
    thin_yrs = [
        e["YEAR_COMP"] for e in events
        if e.get("ACTIVITY") in THIN_TYPES and e.get("YEAR_COMP") is not None
    ]
    burn_yrs = [
        e["YEAR_COMP"] for e in events
        if e.get("ACTIVITY") in BURNING_TYPES and e.get("YEAR_COMP") is not None
    ]
    if not thin_yrs or not burn_yrs:
        return False
    return min(burn_yrs) >= min(thin_yrs)


def _complete_type(events: list[dict]) -> str | None:
    """Return the complete treatment label, or None."""
    activities = {e.get("ACTIVITY") for e in events}
    has_burn   = bool(activities & BURNING_TYPES)
    has_mech   = "Mechanical" in activities
    has_manual = "Manual" in activities

    if not has_burn:
        return None
    if has_mech:
        return "Mechanical + Broadcast Burn"
    if has_manual:
        return "Manual + Broadcast Burn"
    return None


def _pipe_set(values: list) -> str:
    """Sorted pipe-delimited set of non-null values."""
    return "|".join(sorted({str(v) for v in values if v is not None and str(v).strip()}))


# ---------------------------------------------------------------------------
# Parallel attribute aggregation
# ---------------------------------------------------------------------------

def _aggregate_atom_chunk(args: tuple) -> list[dict]:
    """
    Process a chunk of atomic zones — module-level so it is picklable
    by multiprocessing across platforms (spawn / fork).

    Parameters (packed as a single tuple for executor.map compatibility)
    ----------
    chunk_df       : DataFrame slice of atomic_with_sources
    src_lookup     : {OBJECTID: {field: value}} pre-built dict
    present_fields : list of EVENT_FIELDS that exist in the source data
    """
    chunk_df, src_lookup, present_fields = args
    records = []

    for _, atom_row in chunk_df.iterrows():
        source_ids = atom_row.get("SOURCE_IDS") or []
        if not source_ids:
            continue

        events = []
        for sid in source_ids:
            src = src_lookup.get(int(sid), {})
            event = {"SOURCE_ID": int(sid)}
            for field in present_fields:
                val = src.get(field)
                # Normalise NaN → None for clean JSON serialisation
                if val is None or (isinstance(val, float) and val != val):
                    val = None
                event[field] = val
            if "YEAR_COMP" in event and event["YEAR_COMP"] is not None:
                try:
                    event["YEAR_COMP"] = int(event["YEAR_COMP"])
                except (ValueError, TypeError):
                    event["YEAR_COMP"] = None
            events.append(event)

        all_years      = [e["YEAR_COMP"] for e in events if e.get("YEAR_COMP") is not None]
        trt_activities = sorted({e["ACTIVITY"] for e in events if e.get("ACTIVITY")})
        complete       = _is_complete(events)
        c_type         = _complete_type(events) if complete else None

        records.append({
            "_atom_id":       atom_row["_atom_id"],
            "SOURCE_IDS":     json.dumps(source_ids),
            "EVENTS":         json.dumps(events),
            "N_EVENTS":       len(events),
            "TRT_ACTIVITIES": "|".join(trt_activities),
            "FIRST_TRT_YEAR": int(min(all_years)) if all_years else None,
            "LAST_TRT_YEAR":  int(max(all_years)) if all_years else None,
            "COMPLETE":       complete,
            "COMPLETE_TYPE":  c_type,
            "AGENCIES":       _pipe_set([e.get("AGENCY_C")   for e in events]),
            "FUND_SOURCES":   _pipe_set([e.get("FUND_SOURCE") for e in events]),
            "geometry":       atom_row["geometry"],
        })

    return records


def aggregate_atom_attributes(
    atomic_with_sources: gpd.GeoDataFrame,
    gdf: gpd.GeoDataFrame,
    source_id_col: str = "OBJECTID",
    n_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    For each atomic zone, build a list of event records from all source CFT
    polygons that cover it. Derives summary columns from the merged events.

    Parallelises across *n_workers* processes by splitting the atom table
    into equal chunks. Each worker receives a pre-built plain-dict lookup
    (fast to pickle, avoids per-row DataFrame indexing overhead).

    Parameters
    ----------
    atomic_with_sources : output of assign_treatments_to_atoms()
    gdf : original CFT GeoDataFrame (unflagged, with EVENT_FIELDS columns)
    source_id_col : str
        Column in gdf that matches SOURCE_IDS (default 'OBJECTID').
    n_workers : int, optional
        Number of parallel worker processes. Defaults to cpu_count - 1.
        Pass 1 to run sequentially (useful for debugging).

    Returns
    -------
    GeoDataFrame with ATOM_ID, SOURCE_IDS (JSON), EVENTS (JSON), and
    derived summary columns. See module docstring for full schema.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    present_fields = [f for f in EVENT_FIELDS if f in gdf.columns]

    # Build a plain-dict lookup: {OBJECTID: {field: value}}
    # Plain dict is far faster to pickle and index than a DataFrame.
    attr_df = gdf.set_index(source_id_col)[present_fields]
    src_lookup = {
        int(idx): row.to_dict()
        for idx, row in attr_df.iterrows()
    }

    # Split atom table into chunks — one per worker.
    # np.array_split on a GeoDataFrame triggers __array__ and returns ndarrays,
    # so split integer indices instead and use iloc to get proper GDF slices.
    idx_chunks = np.array_split(np.arange(len(atomic_with_sources)), n_workers)
    chunks = [atomic_with_sources.iloc[idx] for idx in idx_chunks if len(idx) > 0]
    args   = [(chunk, src_lookup, present_fields) for chunk in chunks]

    if n_workers == 1:
        all_records = _aggregate_atom_chunk(args[0])
    else:
        # ProcessPoolExecutor works on both fork (Linux) and spawn (macOS/Win)
        # because _aggregate_atom_chunk is a module-level function.
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_aggregate_atom_chunk, args))
        all_records = [r for chunk_result in results for r in chunk_result]

    result = gpd.GeoDataFrame(all_records, geometry="geometry", crs=atomic_with_sources.crs)
    result["ATOM_ID"]   = result.index + 1
    result["ACRES_GIS"] = result.geometry.area * M2_TO_ACRES

    cols = [
        "ATOM_ID", "SOURCE_IDS", "EVENTS",
        "N_EVENTS", "TRT_ACTIVITIES",
        "FIRST_TRT_YEAR", "LAST_TRT_YEAR",
        "COMPLETE", "COMPLETE_TYPE",
        "AGENCIES", "FUND_SOURCES",
        "ACRES_GIS", "geometry",
    ]
    return result[cols].reset_index(drop=True)


def first_last_year(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convenience: add FIRST_TRT_YEAR / LAST_TRT_YEAR if not already present."""
    if "FIRST_TRT_YEAR" not in gdf.columns:
        def _fl(events_json):
            events = json.loads(events_json) if isinstance(events_json, str) else events_json
            yrs = [e["YEAR_COMP"] for e in events if e.get("YEAR_COMP") is not None]
            return (int(min(yrs)), int(max(yrs))) if yrs else (None, None)
        gdf = gdf.copy()
        gdf["FIRST_TRT_YEAR"], gdf["LAST_TRT_YEAR"] = zip(*gdf["EVENTS"].apply(_fl))
    return gdf


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def build_treatment_interactions(
    cft_clean: gpd.GeoDataFrame,
    source_id_col: str = "OBJECTID",
    snap_tolerance: float = 0.5,
    min_acres: float = 0.11,
    target_crs: int = 26913,
    n_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """
    Full pipeline: clean CFT polygons → non-overlapping atomic treatment
    interactions database.

    Replaces the multi-tier geometry match and erase-and-stamp approach
    with a topologically clean atomic decomposition. Complete treatment
    identification (thin → burn) is derived from event attributes rather
    than spatial overlay.

    Parameters
    ----------
    cft_clean : GeoDataFrame
        Pre-cleaned, unflagged CFT polygons. Must contain OBJECTID plus the
        columns listed in EVENT_FIELDS.
    source_id_col : str
        Unique row identifier from the REST source (default 'OBJECTID').
    snap_tolerance : float
        Pre-snap tolerance in metres (default 0.5 m).
    min_acres : float
        Sliver filter threshold in acres (default 0.11 ac ≈ ½ pixel).
    target_crs : int
        EPSG code for all distance operations (default NAD83 UTM 13N).
    n_workers : int, optional
        Worker processes for attribute aggregation. Defaults to cpu_count - 1.
        Pass 1 to disable parallelism (useful for debugging in notebooks).

    Returns
    -------
    GeoDataFrame in the original CRS of cft_clean.
    """
    original_crs = cft_clean.crs
    min_area_m2  = min_acres / M2_TO_ACRES

    gdf = cft_clean.to_crs(epsg=target_crs).copy()
    gdf = make_valid_gdf(gdf)
    print(f"Input polygons:  {len(gdf):,}")

    # Step 1 — atomic decomposition
    atomic = build_atomic_zones(gdf, snap_tolerance, min_area_m2)
    print(f"Atomic zones:    {len(atomic):,}")

    # Step 2 — assign source OBJECTIDs to atoms
    atomic_src = assign_treatments_to_atoms(atomic, gdf, source_id_col)

    # Step 3 — aggregate full event records + derive summary fields (parallel)
    result = aggregate_atom_attributes(atomic_src, gdf, source_id_col, n_workers=n_workers)
    result = drop_slivers(result, min_area_m2)
    print(f"Final features:  {len(result):,}")

    n_complete = result["COMPLETE"].sum()
    print(f"Complete (thin+burn): {n_complete:,}  ({n_complete/len(result)*100:.1f}%)")
    if result["COMPLETE_TYPE"].notna().any():
        print(result["COMPLETE_TYPE"].value_counts().to_string())

    return result.to_crs(original_crs)
