"""
treatment_interactions — multi-source treatment interactions database.

A self-contained workflow for building a statewide, non-overlapping atomic
treatment layer from the Colorado Forest Tracker (CFT v2.0) and TWIG.

Typical usage
-------------
from treatment_interactions.ingest import load_or_download, clean_geometry, flag_acre_mismatch
from treatment_interactions.twig import load_or_download_twig, twig_to_cft_schema
from treatment_interactions.reconcile import combine_sources
from treatment_interactions.interactions import build_treatment_interactions
from treatment_interactions.helpers import events_to_rows, filter_by_year_range

# 1. Ingest
cft = load_or_download("data/spatial/raw/CFTv2_CO_AllTreatments.gpkg")
cft = clean_geometry(cft)
cft = flag_acre_mismatch(cft)

# 2. Exclude flagged records
cft_clean = cft[~cft["ac_flag"] & ~cft["ac_flag_abs"]].copy()

# 3. Add TWIG and combine sources
twig = load_or_download_twig("data/spatial/raw/TWIG.gpkg", aoi=cft_clean)
twig = twig_to_cft_schema(twig)
combined = combine_sources({"CFT": cft_clean, "TWIG": twig})

# 4. Build interactions database
interactions = build_treatment_interactions(combined)
interactions.to_file("data/spatial/mod/CFTv2_CO_Interactions.gpkg")

# 5. Downstream helpers
rows = events_to_rows(interactions)          # one row per event
subset = filter_by_year_range(interactions, 2014, 2024)
"""

from .ingest import (
    query_feature_service,
    download_cft,
    load_or_download,
    clean_geometry,
    flag_acre_mismatch,
    CFT_URL,
    CFT_LAYER,
    MIN_AREA_ACRES,
)

from .twig import (
    download_twig,
    load_or_download_twig,
    harmonize_twig_activity,
    twig_to_cft_schema,
    load_crosswalk,
    classify_thin,
    classify_range,
    classify_release,
    TWIG_URL,
    TWIG_URL_LEGACY,
    TWIG_LAYER,
    FUND_SOURCE_CROSSWALK,
)

from .reconcile import (
    combine_sources,
    source_coverage,
    dedupe_events,
    make_event_deduper,
    reconcile_report,
    ALIAS_RULES,
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

from .plotting import plot_treatment_heatmap, plot_activity_acres, plot_activity_map

__all__ = [
    # ingest
    "query_feature_service",
    "download_cft", "load_or_download", "clean_geometry", "flag_acre_mismatch",
    "CFT_URL", "CFT_LAYER", "MIN_AREA_ACRES",
    # twig
    "download_twig", "load_or_download_twig", "harmonize_twig_activity",
    "twig_to_cft_schema", "load_crosswalk", "classify_thin", "classify_range", "classify_release",
    "TWIG_URL", "TWIG_URL_LEGACY", "TWIG_LAYER", "FUND_SOURCE_CROSSWALK",
    # reconcile
    "combine_sources", "source_coverage", "dedupe_events", "make_event_deduper",
    "reconcile_report", "ALIAS_RULES",
    # interactions
    "build_treatment_interactions",
    "build_atomic_zones", "assign_treatments_to_atoms", "aggregate_atom_attributes",
    "make_valid_gdf", "drop_slivers", "snap_to_network",
    "THIN_TYPES", "BURNING_TYPES", "TRT_PRIORITY", "MGT_GROUPS", "EVENT_FIELDS",
    # helpers
    "events_to_rows", "extract_field",
    "filter_by_activity", "filter_by_year_range", "filter_complete",
    # plotting
    "plot_treatment_heatmap", "plot_activity_acres", "plot_activity_map",
]
