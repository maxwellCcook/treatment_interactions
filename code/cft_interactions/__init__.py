"""
cft_interactions — Colorado Forest Tracker treatment interactions database.

A self-contained workflow for building a statewide, non-overlapping atomic
treatment layer from the Colorado Forest Tracker (CFT v2.0).

Typical usage
-------------
from cft_interactions.ingest import load_or_download, clean_geometry, flag_acre_mismatch
from cft_interactions.interactions import build_treatment_interactions
from cft_interactions.helpers import events_to_rows, filter_by_year_range

# 1. Ingest
cft = load_or_download("data/spatial/raw/CFTv2_CO_AllTreatments.gpkg")
cft = clean_geometry(cft)
cft = flag_acre_mismatch(cft)

# 2. Exclude flagged records
cft_clean = cft[~cft["ac_flag"] & ~cft["ac_flag_abs"]].copy()

# 3. Build interactions database
interactions = build_treatment_interactions(cft_clean)
interactions.to_file("data/spatial/mod/CFTv2_CO_Interactions.gpkg")

# 4. Downstream helpers
rows = events_to_rows(interactions)          # one row per event
subset = filter_by_year_range(interactions, 2014, 2024)
"""

from .ingest import (
    download_cft,
    load_or_download,
    clean_geometry,
    flag_acre_mismatch,
    CFT_URL,
    CFT_LAYER,
    MIN_AREA_ACRES,
)

from .interactions import (
    build_treatment_interactions,
    build_atomic_zones,
    assign_treatments_to_atoms,
    aggregate_atom_attributes,
    make_valid_gdf,
    drop_slivers,
    snap_to_network,
    THIN_TYPES,
    BURNING_TYPES,
    TRT_PRIORITY,
    MGT_GROUPS,
    EVENT_FIELDS,
)

from .helpers import (
    events_to_rows,
    extract_field,
    filter_by_activity,
    filter_by_year_range,
    filter_complete,
)

from .plotting import plot_treatment_heatmap

__all__ = [
    # ingest
    "download_cft", "load_or_download", "clean_geometry", "flag_acre_mismatch",
    "CFT_URL", "CFT_LAYER", "MIN_AREA_ACRES",
    # interactions
    "build_treatment_interactions",
    "build_atomic_zones", "assign_treatments_to_atoms", "aggregate_atom_attributes",
    "make_valid_gdf", "drop_slivers", "snap_to_network",
    "THIN_TYPES", "BURNING_TYPES", "TRT_PRIORITY", "MGT_GROUPS", "EVENT_FIELDS",
    # helpers
    "events_to_rows", "extract_field",
    "filter_by_activity", "filter_by_year_range", "filter_complete",
    # plotting
    "plot_treatment_heatmap",
]
