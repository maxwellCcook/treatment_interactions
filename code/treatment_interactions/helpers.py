"""
Utility functions for working with the CFT treatment interactions database.

The primary output of build_treatment_interactions() stores per-event
attributes as a JSON string in the EVENTS column. These helpers make it
easy to extract, filter, and explode that data without requiring callers
to parse JSON manually.

author: maxwell.cook@colostate.edu
"""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------

def events_to_rows(
    gdf: gpd.GeoDataFrame,
    keep_geometry: bool = False,
) -> pd.DataFrame:
    """
    Explode EVENTS JSON into one row per source treatment event.

    Useful for tabular analysis across the full set of treatments
    (e.g., acreage by agency, funding source summaries). ATOM_ID is
    preserved so results can be joined back to the spatial layer.

    Parameters
    ----------
    gdf : GeoDataFrame  (output of build_treatment_interactions)
    keep_geometry : bool
        If True, the atom geometry is repeated on every event row.

    Returns
    -------
    DataFrame — one row per event, columns = ATOM_ID + EVENT_FIELDS +
                (geometry if keep_geometry).
    """
    rows = []
    for _, atom in gdf.iterrows():
        events = json.loads(atom["EVENTS"]) if isinstance(atom["EVENTS"], str) else atom["EVENTS"]
        for event in events:
            row = {"ATOM_ID": atom["ATOM_ID"], "ACRES_GIS": atom["ACRES_GIS"]}
            row.update(event)
            if keep_geometry:
                row["geometry"] = atom["geometry"]
            rows.append(row)

    df = pd.DataFrame(rows)
    if keep_geometry:
        df = gpd.GeoDataFrame(df, geometry="geometry", crs=gdf.crs)
    return df


def extract_field(
    gdf: gpd.GeoDataFrame,
    field: str,
    col_name: str | None = None,
    delimiter: str = "|",
) -> gpd.GeoDataFrame:
    """
    Add a flat column containing all unique values of *field* across an
    atom's EVENTS records, pipe-delimited (or custom delimiter).

    Useful for adding a specific attribute column without a full explode.

    Parameters
    ----------
    gdf : GeoDataFrame
    field : str
        Field name inside each event dict (e.g., 'FUND_TYPE', 'LANDOWNER').
    col_name : str, optional
        Output column name. Defaults to *field*.
    delimiter : str
        Delimiter for multiple values (default '|').

    Returns
    -------
    GeoDataFrame — copy with new column added.
    """
    col_name = col_name or field

    def _extract(events_json):
        events = json.loads(events_json) if isinstance(events_json, str) else events_json
        vals = sorted({
            str(e[field]) for e in events
            if e.get(field) is not None and str(e.get(field, "")).strip()
        })
        return delimiter.join(vals) if vals else None

    out = gdf.copy()
    out[col_name] = out["EVENTS"].apply(_extract)
    return out


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_by_activity(
    gdf: gpd.GeoDataFrame,
    activities: list[str],
    match: str = "any",
) -> gpd.GeoDataFrame:
    """
    Keep atoms where TRT_ACTIVITIES overlaps with *activities*.

    Parameters
    ----------
    gdf : GeoDataFrame
    activities : list of str
        Activity names to match (e.g., ['Mechanical', 'Manual']).
    match : str
        'any' — atom has at least one of the given activities (default).
        'all' — atom has every one of the given activities.

    Returns
    -------
    GeoDataFrame — filtered copy.
    """
    activity_set = set(activities)

    def _match(trt_str):
        if not isinstance(trt_str, str) or not trt_str.strip():
            return False
        atom_acts = set(trt_str.split("|"))
        if match == "all":
            return activity_set <= atom_acts
        return bool(activity_set & atom_acts)

    mask = gdf["TRT_ACTIVITIES"].apply(_match)
    return gdf[mask].copy()


def filter_by_year_range(
    gdf: gpd.GeoDataFrame,
    start: int,
    end: int,
) -> gpd.GeoDataFrame:
    """
    Keep atoms where any treatment falls within [start, end] (inclusive).

    Uses FIRST_TRT_YEAR / LAST_TRT_YEAR — an atom is retained if its
    treatment window overlaps with the requested range.

    Parameters
    ----------
    gdf : GeoDataFrame
    start : int  — earliest year (inclusive)
    end : int    — latest year (inclusive)

    Returns
    -------
    GeoDataFrame — filtered copy.
    """
    mask = (
        gdf["FIRST_TRT_YEAR"].notna()
        & gdf["LAST_TRT_YEAR"].notna()
        & (gdf["FIRST_TRT_YEAR"] <= end)
        & (gdf["LAST_TRT_YEAR"] >= start)
    )
    return gdf[mask].copy()


def filter_complete(
    gdf: gpd.GeoDataFrame,
    trt_eff: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Keep only atoms with COMPLETE == True.

    Parameters
    ----------
    gdf : GeoDataFrame
    trt_eff : str, optional
        If provided, further filter to a specific TRT_EFF value
        (e.g., 'Mechanical + Broadcast Burn').

    Returns
    -------
    GeoDataFrame — filtered copy.
    """
    out = gdf[gdf["COMPLETE"]].copy()
    if trt_eff is not None:
        out = out[out["TRT_EFF"] == trt_eff].copy()
    return out
