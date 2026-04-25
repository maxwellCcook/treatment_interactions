"""
Helper functions for treatment interactions
author:maxwell.cook@colostate.edu
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import snap, unary_union
from shapely.validation import make_valid

# ------------------

m2_to_acres = 0.000247105 # conversion (multiplication) of m2 to acres

THIN_TYPES    = {'Mechanical', 'Manual'}
BURNING_TYPES = {'Broadcast Burn'}
MIN_AREA_M2 = (30*30)*m2_to_acres # approximate size of one 30m pixel

# --- Define the treatment priority for the effective treatment assignment
TRT_PRIORITY = [
    ('Mechanical', 'Broadcast Burn'),
    ('Manual', 'Broadcast Burn'),
    ('Broadcast Burn',),
    ('Mechanical',),
    ('Manual',),
    ('Mastication',),
    ('Removal',),
    ('Pile Burn',),
    ('Pile Fuels',),
    ('Chipping',),
    ('Mulching',),
    ('Lop and Scatter',),
]

# ------------------

def make_valid_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out['geometry'] = out.geometry.apply(make_valid)
    return out[~out.geometry.is_empty & out.geometry.notna()].copy()


def drop_slivers(gdf: gpd.GeoDataFrame, min_area_m2: float = MIN_AREA_M2) -> gpd.GeoDataFrame:
    return gdf[gdf.geometry.area > min_area_m2].copy()


def snap_to_network(
    gdf: gpd.GeoDataFrame,
    tolerance: float = 0.5,   # metres — tune to your data
) -> gpd.GeoDataFrame:
    """
    Snap all geometries to a common reference network (union of all boundaries).
    This suppresses near-identical boundary drift before the atomic decomposition,
    which is the main source of slivers at ~40k polygons.

    Note: snapping to the full union boundary is more stable than pairwise
    snapping because it forces all vertices toward the same shared edge set.
    """
    gdf = make_valid_gdf(gdf)
    reference = unary_union(gdf.geometry)
    gdf['geometry'] = gdf.geometry.apply(lambda g: snap(g, reference, tolerance))
    gdf['geometry'] = gdf.geometry.apply(make_valid)
    return gdf[~gdf.geometry.is_empty].copy()

def dissolve_duplicates(
    gdf: gpd.GeoDataFrame,
    dissolve_by: list[str],
    agg_rules: dict,
) -> gpd.GeoDataFrame:
    """
    Dissolve exact (year × treatment type) duplicates before decomposition.
    This is your existing geom_key grouping logic, generalized.
    Reduces ~40k → much smaller set before the expensive overlay step.
    """
    dissolved = (
        gdf.dissolve(by=dissolve_by, aggfunc=agg_rules, as_index=False)
           .explode(index_parts=False)
           .reset_index(drop=True)
    )
    return make_valid_gdf(dissolved)

def build_atomic_zones(
    gdf: gpd.GeoDataFrame,
    snap_tolerance: float = 0.5,
    min_area_m2: float = MIN_AREA_M2,
) -> gpd.GeoDataFrame:
    """
    Decompose all treatment polygons into non-overlapping atomic zones.

    Each output polygon is the intersection of a unique combination of
    input polygons — analogous to a full topological overlay.

    Strategy
    --------
    Use iterative symmetric difference + intersection to build the atomic
    partition. This is equivalent to shapely's coverage_union or a full
    overlay(how='union') self-join, but we go through unary_union of
    boundaries to get the partition geometry, then spatial-join attributes
    back. This avoids the O(n²) pairwise overlay for n=40k.

    Parameters
    ----------
    gdf : GeoDataFrame in projected CRS
    snap_tolerance : metres, pre-snapping tolerance
    min_area_m2 : sliver filter threshold
    """
    assert gdf.crs.is_projected, "Must be in projected CRS."

    gdf = snap_to_network(gdf, tolerance=snap_tolerance)
    gdf = drop_slivers(gdf, min_area_m2)

    # Build the atomic partition from the union of all boundaries
    # This gives us the set of unique non-overlapping zones
    all_boundaries = unary_union([g.boundary for g in gdf.geometry])
    all_polys      = unary_union(gdf.geometry)

    # polygonize the boundary network → atomic faces
    from shapely.ops import polygonize
    atomic_faces = list(polygonize(all_boundaries))

    if not atomic_faces:
        raise ValueError("polygonize returned no faces — check input geometry validity.")

    atomic = gpd.GeoDataFrame(geometry=atomic_faces, crs=gdf.crs)
    atomic = drop_slivers(atomic, min_area_m2)
    atomic = atomic.reset_index(drop=True)
    atomic['_atom_id'] = atomic.index

    return atomic

def assign_treatments_to_atoms(
    atomic: gpd.GeoDataFrame,
    gdf: gpd.GeoDataFrame,
    id_col: str = 'PROJ_ID',
) -> gpd.GeoDataFrame:
    """
    Spatial join: for each atomic zone, find all source treatment polygons
    that contain it (using centroid to avoid boundary edge cases).

    Returns atomic zones with a list of source PROJ_IDs per zone.
    """
    # Use representative point (guaranteed interior) for the join
    atomic_pts = atomic.copy()
    atomic_pts['geometry'] = atomic.geometry.representative_point()

    joined = gpd.sjoin(
        atomic_pts[['_atom_id', 'geometry']],
        gdf[[id_col, 'geometry']],
        how='left',
        predicate='within',
    )

    # Aggregate: which source polygons cover each atom?
    atom_sources = (
        joined.groupby('_atom_id')[id_col]
              .apply(lambda s: tuple(sorted(s.dropna().astype(int))))
              .reset_index()
              .rename(columns={id_col: 'SOURCE_IDS'})
    )

    return atomic.merge(atom_sources, on='_atom_id', how='left')


# ── Attribute aggregation (mirrors your existing helpers) ──────────────────────

def is_complete(trt_years: dict) -> bool:
    """
    True if a burning treatment followed (or co-occurred with) a thinning treatment.
    Thinning-before-burning temporal logic.
    """
    thin_yrs  = [y for t in THIN_TYPES    for y in trt_years.get(t, ())]
    burn_yrs  = [y for t in BURNING_TYPES for y in trt_years.get(t, ())]
    if not thin_yrs or not burn_yrs:
        return False
    return min(burn_yrs) >= min(thin_yrs)

def label_priority(trt_years: dict, priority: list) -> str:
    """
    Assign TRT_EFF based on the highest-priority treatment type present
    across ALL years --- not just the most recent year.
    For combos, all members must be present in trt_years.
    """
    present = set(trt_years.keys())
    if not present:
        return 'Unknown'

    for combo in priority:
        if all(t in present for t in combo):
            return ' + '.join(combo)

    # Fallback: join whatever is present (shouldn't hit this often)
    return ' + '.join(sorted(present))

def last_thin_year(trt_years: dict, analysis_start: int = 2014) -> int | None:
    """
    Return the most recent thinning year >= analysis_start, or None.
    None means either no thinning occurred, or it all predates the analysis window.
    """
    thin_yrs = [
        y for t in THIN_TYPES
        for y in trt_years.get(t, ())
        if y >= analysis_start
    ]
    return max(thin_yrs) if thin_yrs else None

def first_last(trt_years: dict, priority: list, which: str = "first") -> str | None:
    """
    Return the activity type associated with the earliest ('first') or
    most recent ('last') recorded year, broken by priority if multiple
    types share that year.
    """
    if not trt_years:
        return None

    all_years = [y for ys in trt_years.values() for y in ys]
    boundary_year = min(all_years) if which == "first" else max(all_years)
    boundary_types = {t for t, ys in trt_years.items() if boundary_year in ys}

    for combo in priority:
        if len(combo) == 1 and combo[0] in boundary_types:
            return combo[0]
    return sorted(boundary_types)[0]

def event_years(df, year_col="YEAR_COMP", type_col="ACTIVITY"):
    """Return {activity: (sorted unique years)} for a group."""
    sub = df[[year_col, type_col]].dropna(subset=[year_col, type_col]).copy()
    sub[year_col] = pd.to_numeric(sub[year_col], errors="coerce")
    sub = sub.dropna(subset=[year_col])
    sub[year_col] = sub[year_col].astype(int)
    return (
        sub.groupby(type_col)[year_col]
           .apply(lambda s: tuple(sorted(set(s.tolist()))))
           .to_dict()
    )

def first_last_year(trt_years: dict) -> tuple:
    """{type: (years...)} -> (min_year, max_year)"""
    yrs = [y for ys in trt_years.values() for y in ys]
    return (min(yrs), max(yrs)) if yrs else (None, None)

def trt_set(trt_years: dict, priority: list) -> str:
    """
    Return a '+'-delimited string of all treatment types present,
    ordered by priority (highest impact first).
    """
    if not trt_years:
        return 'Unknown'
    present = set(trt_years.keys())
    # pull out types in priority order, then append anything not in priority list
    ordered = [combo[0] for combo in priority if len(combo) == 1 and combo[0] in present]
    remainder = sorted(present - set(ordered))
    return ' + '.join(ordered + remainder)

def aggregate_atom_attributes(
    atomic_with_sources: gpd.GeoDataFrame,
    gdf: gpd.GeoDataFrame,
    id_col: str = 'PROJ_ID',
) -> gpd.GeoDataFrame:
    """
    For each atomic zone, aggregate treatment attributes from all source
    polygons that cover it. Reconstructs EVENTS, TRT_EFF, etc. using
    your existing helper functions.
    """
    # Build a lookup: PROJ_ID → row attributes
    src_lookup = gdf.set_index(id_col)

    records = []
    for _, atom_row in atomic_with_sources.iterrows():
        source_ids = atom_row['SOURCE_IDS']
        if not source_ids:
            continue

        src_rows = src_lookup.loc[src_lookup.index.isin(source_ids)]

        # Merge EVENTS dicts across all covering treatments
        merged_events = {}
        for _, src in src_rows.iterrows():
            for activity, years in src.get('EVENTS', {}).items():
                existing = set(merged_events.get(activity, ()))
                merged_events[activity] = tuple(sorted(existing | set(years)))

        agencies = tuple(sorted(set(
            a for _, src in src_rows.iterrows()
            for a in (src.get('AGENCIES') or ())
        )))

        proj_ids = tuple(sorted(set(
            pid for _, src in src_rows.iterrows()
            for pid in (src.get('PROJ_IDs') or ())
        )))

        records.append({
            '_atom_id':   atom_row['_atom_id'],
            'EVENTS':     merged_events,
            'AGENCIES':   agencies,
            'PROJ_IDs':   proj_ids,
            'geometry':   atom_row['geometry'],
        })

    result = gpd.GeoDataFrame(records, geometry='geometry', crs=atomic_with_sources.crs)

    # Apply your existing label functions
    result['TRT_SET']    = result['EVENTS'].apply(lambda d: trt_set(d, TRT_PRIORITY))
    result['TRT_EFF']    = result['EVENTS'].apply(lambda d: label_priority(d, TRT_PRIORITY))
    result['FIRST_TRT']  = result['EVENTS'].apply(lambda d: first_last(d, TRT_PRIORITY, 'first'))
    result['LAST_TRT']   = result['EVENTS'].apply(lambda d: first_last(d, TRT_PRIORITY, 'last'))
    result['COMPLETE']   = result['EVENTS'].apply(is_complete)
    result['LAST_THIN']  = result['EVENTS'].apply(
        lambda d: last_thin_year(d, analysis_start=2014)
    ).fillna(0).astype(int)
    result['FIRST_YEAR'], result['LAST_YEAR'] = zip(
        *result['EVENTS'].apply(first_last_year)
    )
    result['ACRES_GIS'] = result.geometry.area / 4046.86

    return result.reset_index(drop=True)

# ── Top-level pipeline ─────────────────────────────────────────────────────────

def build_treatment_interactions(
    cft_raw: gpd.GeoDataFrame,
    dissolve_by: list[str] = ['ACTIVITY', 'YEAR_COMP'],
    agg_rules: dict | None = None,
    target_crs: str = 'EPSG:26913',
    snap_tolerance: float = 0.5,
    min_acres: float = 0.1,
) -> gpd.GeoDataFrame:
    """
    Full pipeline: raw CFT polygons → model-ready treatment interaction layer.

    Replaces the geom_key grouping + erase-and-stamp approach with a
    topologically clean atomic decomposition.

    Parameters
    ----------
    cft_raw : GeoDataFrame
        Raw Colorado Forest Tracker polygons with at minimum ACTIVITY, YEAR_COMP,
        AGENCY_C columns.
    dissolve_by : list of str
        Columns to dissolve on before decomposition (removes exact duplicates).
    target_crs : str
        Projected CRS for distance operations. Default UTM 13N for CO.
    snap_tolerance : float
        Pre-snap tolerance in metres. 0.5m is conservative; increase if you
        see persistent slivers in output.
    min_acres : float
        Sliver filter threshold.
    """
    min_area_m2 = min_acres * 4046.86

    if agg_rules is None:
        agg_rules = {
            'AGENCY_C': lambda x: tuple(sorted(set(x.dropna()))),
        }

    # 0. Project
    gdf = cft_raw.to_crs(target_crs).copy()
    gdf = make_valid_gdf(gdf)
    gdf['PROJ_ID'] = range(len(gdf))

    # 1. Dissolve exact year × treatment duplicates
    print(f"Input polygons: {len(gdf)}")
    gdf_dissolved = dissolve_duplicates(gdf, dissolve_by, agg_rules)
    gdf_dissolved['PROJ_ID'] = range(len(gdf_dissolved))
    print(f"After dissolve: {len(gdf_dissolved)}")

    # 2. Atomic decomposition
    atomic = build_atomic_zones(gdf_dissolved, snap_tolerance, min_area_m2)
    print(f"Atomic zones:   {len(atomic)}")

    # 3. Attribute join
    atomic_src = assign_treatments_to_atoms(atomic, gdf_dissolved)

    # 4. Aggregate and label
    result = aggregate_atom_attributes(atomic_src, gdf_dissolved)
    result = drop_slivers(result, min_area_m2)

    print(f"Final features: {len(result)}")
    print(result['TRT_EFF'].value_counts())

    return result.to_crs(cft_raw.crs)