"""
Atomic decomposition of CFT treatment polygons into a non-overlapping
treatment interactions database.

author: maxwell.cook@colostate.edu
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.ops import polygonize, unary_union


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

M2_TO_ACRES = 0.000247105
MIN_AREA_M2 = 30 * 30  # 900 m² ≈ one 30 m pixel (~0.22 ac)

THIN_TYPES    = {"Manual", "Mechanical"}
BURNING_TYPES = {"Broadcast Burn"}

TRT_PRIORITY = [
    ("Mechanical", "Broadcast Burn"),
    ("Manual", "Broadcast Burn"),
    ("Broadcast Burn",),
    ("Mechanical",),
    ("Manual",),
    ("Mastication",),
    ("Removal",),
    ("Pile Burn",),
    ("Pile Fuels",),
    ("Chipping",),
    ("Mulching",),
    ("Lop and Scatter",),
]

# Management group membership (Broadcast Burn spans both CANOPY and SURFACE)
MGT_GROUPS: dict[str, set[str]] = {
    "CANOPY":  {"Manual", "Mechanical", "Mastication", "Broadcast Burn"},
    "SURFACE": {"Pile Fuels", "Removal", "Lop and Scatter", "Mulching",
                "Pile Burn", "Broadcast Burn"},
}

# Attributes preserved from each source CFT polygon into EVENTS records
EVENT_FIELDS = [
    "ACTIVITY", "YEAR_COMP", "AGENCY_C",
    "FUND_SOURCE", "FUND_TYPE", "PARTNERS",
    "LANDOWNER", "MGT_TYPE",
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
    tolerance: float = 1,
) -> gpd.GeoDataFrame:
    """
    Align near-coincident polygon boundaries before atomic decomposition by
    snapping every vertex to a shared precision grid.

    Collapsing all coordinates onto one grid (Shapely 2.0's vectorized
    ``set_precision``) forces coincident-but-drifting boundaries from
    different source polygons onto identical coordinates — which is what
    ``polygonize`` needs to produce clean, shared atomic edges.

    This replaces an earlier snap-to-unioned-boundary approach
    (``shapely.snap(geoms, unary_union(geoms), tol)``) that was effectively
    O(n × |union|) and stopped scaling past a few thousand polygons.

    Parameters
    ----------
    tolerance : float
        Precision grid size in map units (metres). Vertices are snapped to
        this grid; boundaries closer than ``tolerance`` become coincident.
    """
    gdf = make_valid_gdf(gdf)

    snapped = shapely.set_precision(gdf.geometry.values, tolerance)
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
    snap_tolerance: float = 1,
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

def _is_complete(trt_years: dict) -> bool:
    """
    True if this atom received both a thinning and a broadcast burn.

    Presence-based, not sequence-based: YEAR_COMP is annual, so within-year
    ordering is unknowable, and burn/thin/re-burn units are common. An atom
    treated by both methods is a complete treatment regardless of the order
    the records happen to carry.
    """
    present = set(trt_years)
    return bool(present & THIN_TYPES) and bool(present & BURNING_TYPES)



def _pipe_set(values: list) -> str:
    """Sorted pipe-delimited set of non-null, whitespace-stripped values."""
    return "|".join(sorted({str(v).strip() for v in values if v is not None and str(v).strip()}))


def _events_to_trt_years(events: list[dict]) -> dict[str, tuple]:
    """Convert list of event dicts to {activity: (sorted unique years)} for label helpers."""
    trt_years: dict[str, set] = {}
    for e in events:
        act = e.get("ACTIVITY")
        yr  = e.get("YEAR_COMP")
        if act and yr is not None:
            trt_years.setdefault(act, set()).add(int(yr))
    return {k: tuple(sorted(v)) for k, v in trt_years.items()}


def _label_priority(trt_years: dict, priority: list) -> str:
    """Assign TRT_EFF: highest-priority treatment type (or combo) present across all years."""
    present = set(trt_years.keys())
    if not present:
        return "Unknown"
    for combo in priority:
        if all(t in present for t in combo):
            return " + ".join(combo)
    return " + ".join(sorted(present))


def _trt_set(trt_years: dict, priority: list) -> str:
    """All treatment types present, joined in priority order."""
    if not trt_years:
        return "Unknown"
    present   = set(trt_years.keys())
    ordered   = [combo[0] for combo in priority if len(combo) == 1 and combo[0] in present]
    remainder = sorted(present - set(ordered))
    return " + ".join(ordered + remainder)


def _first_last_act(trt_years: dict, priority: list, which: str = "first") -> str | None:
    """Activity type at the earliest ('first') or most recent ('last') year."""
    if not trt_years:
        return None
    all_years      = [y for ys in trt_years.values() for y in ys]
    boundary_year  = min(all_years) if which == "first" else max(all_years)
    boundary_types = {t for t, ys in trt_years.items() if boundary_year in ys}
    for combo in priority:
        if len(combo) == 1 and combo[0] in boundary_types:
            return combo[0]
    return sorted(boundary_types)[0]


def _last_thin_year(trt_years: dict, analysis_start: int = 2014) -> int | None:
    """Most recent thinning year >= analysis_start, or None."""
    thin_yrs = [
        y for t in THIN_TYPES
        for y in trt_years.get(t, ())
        if y >= analysis_start
    ]
    return max(thin_yrs) if thin_yrs else None


def _mgt_group_cols(trt_years: dict) -> dict[str, str | None]:
    """
    Pipe-delimited activity types present per management group, or None.

    Broadcast Burn is a member of both groups, so it appears in CANOPY and
    SURFACE whenever the atom was burned.
    """
    return {
        group: ("|".join(sorted(act for act in trt_years if act in members)) or None)
        for group, members in MGT_GROUPS.items()
    }


# ---------------------------------------------------------------------------
# Attribute aggregation
# ---------------------------------------------------------------------------

def aggregate_atom_attributes(
    atomic_with_sources: gpd.GeoDataFrame,
    gdf: gpd.GeoDataFrame,
    source_id_col: str = "OBJECTID",
    extra_event_fields: list[str] | None = None,
    event_dedupe=None,
) -> gpd.GeoDataFrame:
    """
    For each atomic zone, build event records from all overlapping source CFT
    polygons and derive summary columns.

    Uses explode → merge → groupby.apply rather than iterrows + process pool:
    the vectorized merge eliminates the per-row dict lookup, and groupby
    handles chunking in C, avoiding Python-level iteration over ~40k atoms.

    Parameters
    ----------
    atomic_with_sources : output of assign_treatments_to_atoms()
    gdf : original CFT GeoDataFrame with EVENT_FIELDS columns
    source_id_col : str
        Unique row identifier in gdf (default 'OBJECTID').
    extra_event_fields : list[str], optional
        Additional source-GDF columns to carry into each per-event dict in
        the EVENTS JSON (e.g. ['PROJECT_NAME', 'MGT_REGION']). Unknown column
        names are skipped with a warning.
    event_dedupe : callable, optional
        ``events -> events`` hook applied to each atom's event list **before** the
        derived columns (TRT_SET, TRT_EFF, COMPLETE, CANOPY/SURFACE, year bounds)
        are computed. Used to reconcile cross-source duplicate events for a
        multi-source (CFT+TWIG) build — see
        :func:`treatment_interactions.reconcile.dedupe_events`. Default ``None`` leaves
        the event list untouched (a CFT-only statewide run is unaffected).

    Returns
    -------
    GeoDataFrame with ATOM_ID, SOURCE_IDS (JSON), EVENTS (JSON), and
    derived summary columns.
    """
    requested = list(EVENT_FIELDS) + list(extra_event_fields or [])
    seen: set[str] = set()
    requested = [f for f in requested if not (f in seen or seen.add(f))]
    present_fields = [f for f in requested if f in gdf.columns]
    missing = [f for f in (extra_event_fields or []) if f not in gdf.columns]
    if missing:
        print(f"[warn] extra_event_fields not in source GDF, skipped: {missing}")

    # ── 1. Explode SOURCE_IDS list → long-form (one row per atom–source pair) ──
    long = (
        atomic_with_sources[["_atom_id", "SOURCE_IDS"]]
        .explode("SOURCE_IDS")
        .dropna(subset=["SOURCE_IDS"])
        .copy()
    )
    long["SOURCE_IDS"] = long["SOURCE_IDS"].astype(int)

    # ── 2. Merge source attributes — vectorized, no dict lookup ───────────────
    attrs = gdf[[source_id_col] + present_fields].copy()
    long  = long.merge(attrs, left_on="SOURCE_IDS", right_on=source_id_col, how="left")
    if source_id_col in long.columns and source_id_col != "SOURCE_IDS":
        long = long.drop(columns=[source_id_col])

    # ── 3. Per-atom aggregation — groupby replaces iterrows + ProcessPoolExecutor
    def _agg(grp: pd.DataFrame) -> pd.Series:
        src_ids = sorted(grp["SOURCE_IDS"].tolist())

        # to_dict("records") is much faster than iterrows for small per-atom groups
        rec_df = grp[["SOURCE_IDS"] + present_fields].rename(
            columns={"SOURCE_IDS": "SOURCE_ID"}
        )
        events = []
        for row in rec_df.to_dict("records"):
            event = {}
            for k, v in row.items():
                try:
                    event[k] = None if pd.isna(v) else v
                except (TypeError, ValueError):
                    event[k] = v  # non-scalar types (lists, dicts) pass through
            if "YEAR_COMP" in event and event["YEAR_COMP"] is not None:
                try:
                    event["YEAR_COMP"] = int(event["YEAR_COMP"])
                except (ValueError, TypeError):
                    event["YEAR_COMP"] = None
            events.append(event)

        if event_dedupe is not None:
            events = event_dedupe(events)

        all_years = [e["YEAR_COMP"] for e in events if e.get("YEAR_COMP") is not None]
        trt_years = _events_to_trt_years(events)
        mgt_cols  = _mgt_group_cols(trt_years)

        return pd.Series({
            "SOURCE_IDS":     json.dumps(src_ids),
            "EVENTS":         json.dumps(events),
            "N_EVENTS":       len(events),
            "TRT_ACTIVITIES": _trt_set(trt_years, TRT_PRIORITY).replace(" + ", "|"),
            "TRT_EFF":        _label_priority(trt_years, TRT_PRIORITY),
            "TRT_SET":        _trt_set(trt_years, TRT_PRIORITY),
            "FIRST_TRT":      _first_last_act(trt_years, TRT_PRIORITY, "first"),
            "LAST_TRT":       _first_last_act(trt_years, TRT_PRIORITY, "last"),
            "FIRST_TRT_YEAR": int(min(all_years)) if all_years else None,
            "LAST_TRT_YEAR":  int(max(all_years)) if all_years else None,
            "COMPLETE":       _is_complete(trt_years),
            "LAST_THIN":      _last_thin_year(trt_years),
            "CANOPY":         mgt_cols["CANOPY"],
            "SURFACE":        mgt_cols["SURFACE"],
            "AGENCIES":       _pipe_set([e.get("AGENCY_C")   for e in events]),
            "FUND_SOURCES":   _pipe_set([e.get("FUND_SOURCE") for e in events]),
        })

    agg = long.groupby("_atom_id", sort=False).apply(_agg, include_groups=False)

    # ── 4. Re-attach geometry ─────────────────────────────────────────────────
    geom_idx = atomic_with_sources.set_index("_atom_id")["geometry"]
    result = gpd.GeoDataFrame(
        agg.join(geom_idx), geometry="geometry", crs=atomic_with_sources.crs
    )
    result = result.reset_index(drop=True)
    result["ATOM_ID"]   = result.index + 1
    result["ACRES_GIS"] = result.geometry.area * M2_TO_ACRES

    cols = [
        "ATOM_ID", "SOURCE_IDS", "EVENTS",
        "N_EVENTS", "TRT_ACTIVITIES",
        "TRT_EFF", "TRT_SET",
        "FIRST_TRT", "LAST_TRT",
        "FIRST_TRT_YEAR", "LAST_TRT_YEAR",
        "COMPLETE", "LAST_THIN",
        "CANOPY", "SURFACE",
        "AGENCIES", "FUND_SOURCES",
        "ACRES_GIS", "geometry",
    ]
    return result[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Duplicate-atom merge
# ---------------------------------------------------------------------------

# Attribute columns that define atom identity for merge_duplicate_atoms().
# FUND_SOURCES is deliberately excluded: CFT and TWIG use different funding-
# source vocabularies for the same program (e.g. TWIG 'BIL' vs CFT
# 'Bipartisan Infrastructure Law (BIL)'), so requiring it to match would leave
# otherwise-identical adjacent atoms unmerged. SOURCE_IDS/EVENTS/N_EVENTS/
# ATOM_ID/ACRES_GIS are excluded too — they're recomputed after the merge.
DISSOLVE_KEY = [
    "TRT_ACTIVITIES", "TRT_EFF", "TRT_SET",
    "FIRST_TRT", "LAST_TRT", "FIRST_TRT_YEAR", "LAST_TRT_YEAR",
    "COMPLETE", "LAST_THIN", "CANOPY", "SURFACE", "AGENCIES",
]


def merge_duplicate_atoms(
    atoms: gpd.GeoDataFrame,
    key_cols: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Merge geometrically adjacent atoms that share identical derived attributes
    into a single polygon and regenerate ATOM_ID.

    build_atomic_zones() partitions on raw geometry boundaries only, so a
    boundary between (e.g.) a CFT-sourced and a TWIG-sourced record can leave
    two attribute-identical atoms side by side — visible as thin slivers.
    Dissolving on ``key_cols`` and exploding back to contiguous parts merges
    those pairs while leaving non-adjacent atoms that happen to share the same
    key separate.

    SOURCE_IDS and EVENTS are unioned (not dropped) across every atom that
    feeds into a merged polygon, so provenance back to the original CFT/TWIG
    records is fully preserved. FUND_SOURCES is re-derived via ``_pipe_set()``
    over the unioned events rather than used as part of the merge key, so a
    funding-source label difference never blocks a merge.

    Uses a representative-point spatial rejoin against the dissolved geometry
    (mirroring ``assign_treatments_to_atoms()``) rather than ``dissolve()``'s
    built-in ``aggfunc``, since the latter would aggregate SOURCE_IDS/EVENTS
    across *every* atom sharing a key statewide, not just the spatially
    contiguous ones actually being merged.

    Parameters
    ----------
    atoms : output of aggregate_atom_attributes()
    key_cols : list[str], optional
        Attribute columns that define atom identity (default DISSOLVE_KEY).

    Returns
    -------
    GeoDataFrame with the same columns/order as aggregate_atom_attributes(),
    ATOM_ID and ACRES_GIS regenerated.
    """
    key_cols = list(key_cols or DISSOLVE_KEY)

    # ── 1. Dissolve by attribute key, explode back to contiguous parts ────────
    #      dropna=False: several key columns (SURFACE, LAST_THIN, ...) are
    #      legitimately None for many atoms (e.g. a canopy-only atom has no
    #      SURFACE label) — the groupby default would silently drop every atom
    #      with any null key column instead of grouping them together.
    merged = (
        atoms[key_cols + ["geometry"]]
        .dissolve(by=key_cols, dropna=False)
        .explode(index_parts=False)
        .reset_index()
    )
    merged = merged.reset_index(drop=True)
    merged["_merge_id"] = merged.index

    # ── 2. Re-derive SOURCE_IDS / EVENTS / N_EVENTS / FUND_SOURCES per merged
    #      polygon by rejoining the original atoms (safe: each original atom's
    #      geometry is a strict subset of its merged group's unioned geometry).
    pts = atoms.copy()
    pts["geometry"] = atoms.geometry.representative_point()
    joined = gpd.sjoin(
        pts[["SOURCE_IDS", "EVENTS", "geometry"]],
        merged[["_merge_id", "geometry"]],
        how="left",
        predicate="within",
    )

    def _union(grp: pd.DataFrame) -> pd.Series:
        src_ids = sorted({sid for s in grp["SOURCE_IDS"] for sid in json.loads(s)})
        events  = [e for s in grp["EVENTS"] for e in json.loads(s)]
        return pd.Series({
            "SOURCE_IDS":   json.dumps(src_ids),
            "EVENTS":       json.dumps(events),
            "N_EVENTS":     len(events),
            "FUND_SOURCES": _pipe_set([e.get("FUND_SOURCE") for e in events]),
        })

    agg = (
        joined.groupby("_merge_id")[["SOURCE_IDS", "EVENTS"]]
        .apply(_union, include_groups=False)
        .reset_index()
    )

    # ── 3. Reassemble, regenerate ATOM_ID / ACRES_GIS ──────────────────────────
    result = merged.merge(agg, on="_merge_id", how="left").drop(columns="_merge_id")
    result = result.reset_index(drop=True)
    result["ATOM_ID"]   = result.index + 1
    result["ACRES_GIS"] = result.geometry.area * M2_TO_ACRES

    cols = [
        "ATOM_ID", "SOURCE_IDS", "EVENTS",
        "N_EVENTS", "TRT_ACTIVITIES",
        "TRT_EFF", "TRT_SET",
        "FIRST_TRT", "LAST_TRT",
        "FIRST_TRT_YEAR", "LAST_TRT_YEAR",
        "COMPLETE", "LAST_THIN",
        "CANOPY", "SURFACE",
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
    extra_event_fields: list[str] | None = None,
    event_dedupe=None,
) -> gpd.GeoDataFrame:
    """
    Full pipeline: clean CFT polygons → non-overlapping atomic treatment
    interactions database.

    Replaces the multi-tier geometry match and erase-and-stamp approach
    with a topologically clean atomic decomposition. Complete treatment
    identification (thin + broadcast burn, in any order) is derived from
    event attributes rather than spatial overlay.

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
    extra_event_fields : list[str], optional
        Additional source-GDF columns to carry into each per-event dict in
        the EVENTS JSON (e.g. ['PROJECT_NAME', 'MGT_REGION']).
    event_dedupe : callable, optional
        ``events -> events`` hook to reconcile cross-source duplicate events per
        atom (e.g. ``treatment_interactions.reconcile.make_event_deduper(cft_window=…)``
        for a CFT+TWIG build). Default None = no reconciliation.
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
    result = aggregate_atom_attributes(
        atomic_src, gdf, source_id_col, extra_event_fields, event_dedupe=event_dedupe,
    )

    # Step 4 — merge geometrically adjacent atoms with identical derived attributes
    # (before the final sliver filter, so a thin boundary sliver gets absorbed into
    # its identical-attribute neighbor instead of being independently dropped)
    n_before = len(result)
    result = merge_duplicate_atoms(result)
    print(f"Merged duplicate atoms: {n_before:,} -> {len(result):,}")

    result = drop_slivers(result, min_area_m2)
    print(f"Final features:  {len(result):,}")

    n_complete = result["COMPLETE"].sum()
    print(f"Complete (thin+burn): {n_complete:,}  ({n_complete/len(result)*100:.1f}%)")
    print(f"\nTRT_EFF distribution:\n{result['TRT_EFF'].value_counts().to_string()}")

    return result.to_crs(original_crs)
