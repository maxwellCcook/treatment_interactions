"""
Tests for TWIG harmonization (treatment_interactions.twig) and cross-source
reconciliation (treatment_interactions.reconcile).

Run: pytest treatment_interactions/tests/test_twig_reconcile.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

# Make the package importable whether or not it is pip-installed.
_CODE = Path(__file__).resolve().parents[1] / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from treatment_interactions import twig as twig_mod
from treatment_interactions.twig import (
    harmonize_twig_activity, twig_to_cft_schema, load_crosswalk, classify_thin,
    classify_range, classify_release, download_twig,
)
from treatment_interactions.reconcile import (
    dedupe_events, combine_sources, source_coverage, reconcile_report, ALIAS_RULES,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _twig(rows, crs=26913):
    """Build a tiny TWIG-like GeoDataFrame; each row gets a small square."""
    recs = []
    for i, r in enumerate(rows):
        r = dict(r)
        r.setdefault("acres", 10.0)
        r["geometry"] = box(i, 0, i + 1, 1)
        recs.append(r)
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=crs)


# ---------------------------------------------------------------------------
# Crosswalk
# ---------------------------------------------------------------------------

def test_crosswalk_loads_and_keys_unique():
    xw = load_crosswalk()
    assert xw.index.is_unique
    assert set(xw["effect"].unique()) <= {"canopy", "surface", "none"}
    assert set(xw["confidence"].unique()) <= {"high", "medium", "low"}


# ---------------------------------------------------------------------------
# Harmonization — the four documented failure modes
# ---------------------------------------------------------------------------

def test_canopy_signal_beats_surface():
    # type says surface ("Biomass Removal"->Removal), activity says a canopy cut.
    g = _twig([{"type": "Biomass Removal",
                "activity": "Stand Clearcut (EA/RH/FH)",
                "method": None, "twig_category": "Mechanical"}])
    out, rep = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] == "Mechanical"
    assert out["ACTIVITY_FIELD"].iloc[0] == "activity"


def test_null_type_recovered_from_activity():
    # type is NULL; activity carries the mechanical signal (Gunnison SitePrep/Dozer block).
    g = _twig([{"type": None,
                "activity": "Site Preparation for Planting - Mechanical",
                "method": "Mechanical", "equipment": "Dozer",
                "twig_category": "Mechanical"}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] == "Mechanical"
    assert out["ACTIVITY_STATUS"].iloc[0] == "resolved"


def test_pileburn_method_does_not_promote_to_broadcast():
    # type/activity both say a pile burn; the coarse method "Prescribed Burn" must NOT win.
    g = _twig([{"type": "Machine Pile Burn",
                "activity": "Burning of Piled Material",
                "method": "Prescribed Burn", "twig_category": "Planned Ignition"}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] == "Pile Burn"


def test_generic_thinning_uses_classify_thin_once():
    g = _twig([
        {"type": "Thinning", "activity": None, "equipment": "Feller Buncher", "method": None},
        {"type": "Thinning", "activity": None, "equipment": "Chain Saw", "method": None},
    ])
    out, rep = harmonize_twig_activity(g)
    assert list(out["ACTIVITY"]) == ["Mechanical", "Manual"]
    # classify_thin ran exactly on the two Thin-sentinel records
    assert int(rep["RAN_CLASSIFY_THIN"].sum()) == 2


# ---------------------------------------------------------------------------
# Harmonization — drop vs quarantine (never a silent NaN)
# ---------------------------------------------------------------------------

def test_chemical_is_dropped_not_quarantined():
    g = _twig([{"type": "Chemical", "activity": "Invasives - Pesticide Application",
                "method": "Chemical", "twig_category": "Chemical"}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] is None
    assert out["ACTIVITY_STATUS"].iloc[0] == "dropped"


def test_unknown_value_is_quarantined_not_dropped():
    g = _twig([{"type": "Totally New Treatment Type", "activity": None,
                "method": None, "twig_category": None}])
    out, rep = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] is None
    assert out["ACTIVITY_STATUS"].iloc[0] == "quarantined"
    assert (rep["ACTIVITY_STATUS"] == "quarantined").sum() == 1


def test_no_silent_nan_every_row_labeled():
    g = _twig([
        {"type": "Mastication", "activity": None},
        {"type": "Chemical", "activity": None},
        {"type": "Brand New Thing", "activity": None},
    ])
    out, _ = harmonize_twig_activity(g)
    assert set(out["ACTIVITY_STATUS"]) <= {"resolved", "dropped", "quarantined"}
    # resolved rows have an ACTIVITY; non-resolved are None (never NaN float sneaking through)
    assert out.loc[out["ACTIVITY_STATUS"] == "resolved", "ACTIVITY"].notna().all()


# ---------------------------------------------------------------------------
# Acreage reporting — geometric footprint vs TWIG's inflated reported field
# ---------------------------------------------------------------------------

def test_report_carries_geo_and_reported_acres():
    # TWIG's `acres` is per-project reported acreage (here 7795) stamped on a tiny polygon;
    # the report must carry BOTH, and the geometric footprint must be ~the box area (<<7795)
    # so downstream summaries can aggregate ACRES_GEO instead of the inflated field.
    g = _twig([{"type": "Mastication", "activity": None, "acres": 7795.0}])
    _, rep = harmonize_twig_activity(g)
    assert {"ACRES_GEO", "ACRES_REPORTED"} <= set(rep.columns)
    assert rep["ACRES_REPORTED"].iloc[0] == pytest.approx(7795.0)
    # a 1 m x 1 m box in EPSG:26913 is ~0.000247 ac — nowhere near the reported 7795
    assert rep["ACRES_GEO"].iloc[0] < 1.0


# ---------------------------------------------------------------------------
# Ingest — honor TWIG's DUPLICATE_DROP QA flag
# ---------------------------------------------------------------------------

def test_download_twig_drops_duplicate_drop(monkeypatch):
    raw = _twig([
        {"status": "Completed", "error": None,            "treatment_date": "2020-06-01"},
        {"status": "Completed", "error": "DUPLICATE_DROP", "treatment_date": "2020-06-01"},
    ])
    monkeypatch.setattr(twig_mod, "query_feature_service", lambda *a, **k: raw.copy())
    aoi = gpd.GeoDataFrame({"geometry": [box(-1, -1, 5, 5)]}, geometry="geometry", crs=26913)
    out = download_twig(aoi, year_start=2014, year_end=2024)
    assert len(out) == 1
    assert (out["error"] != "DUPLICATE_DROP").all()


def test_download_twig_keeps_duplicate_drop_when_disabled(monkeypatch):
    raw = _twig([
        {"status": "Completed", "error": "DUPLICATE_DROP", "treatment_date": "2020-06-01"},
    ])
    monkeypatch.setattr(twig_mod, "query_feature_service", lambda *a, **k: raw.copy())
    aoi = gpd.GeoDataFrame({"geometry": [box(-1, -1, 5, 5)]}, geometry="geometry", crs=26913)
    out = download_twig(aoi, year_start=2014, year_end=2024, drop_errors=())
    assert len(out) == 1  # opt-out keeps the flagged record


# ---------------------------------------------------------------------------
# classify_thin (ported)
# ---------------------------------------------------------------------------

def test_classify_thin_manual_mech_and_nonthin():
    assert classify_thin({"type": "Thinning", "equipment": "Chain Saw"}) == "Manual"
    assert classify_thin({"type": "Thinning", "equipment": "Feller Buncher"}) == "Mechanical"
    assert classify_thin({"type": "Thinning", "equipment": "unknown"}) == "Mechanical"  # default
    r = classify_thin({"type": "Broadcast Burn", "activity": "Burning of Piled Material"})
    assert isinstance(r, float) and np.isnan(r)


# ---------------------------------------------------------------------------
# Schema harmonization
# ---------------------------------------------------------------------------

def test_twig_to_cft_schema_columns_and_fund_crosswalk():
    g = _twig([{"type": "Mastication", "activity": None, "agency": "USFS",
                "fund_source": "BIL", "year_comp": 2025}])
    out, _ = harmonize_twig_activity(g)
    out = out[out["ACTIVITY"].notna()]
    mapped = twig_to_cft_schema(out, id_offset=100)
    assert {"OBJECTID", "ACTIVITY", "AGENCY_C", "FUND_SOURCE", "SOURCE"} <= set(mapped.columns)
    assert mapped["OBJECTID"].iloc[0] == 101
    assert mapped["SOURCE"].iloc[0] == "TWIG"
    assert "Bipartisan" in mapped["FUND_SOURCE"].iloc[0]  # BIL expanded via crosswalk


def test_twig_to_cft_schema_rejects_null_activity():
    g = _twig([{"type": "Chemical", "activity": None}])
    out, _ = harmonize_twig_activity(g)
    with pytest.raises(ValueError):
        twig_to_cft_schema(out)  # still has a null-ACTIVITY (dropped) row


# ---------------------------------------------------------------------------
# Event de-duplication
# ---------------------------------------------------------------------------

def test_dedupe_collapses_same_activity_cross_source():
    ev = [{"ACTIVITY": "Manual", "YEAR_COMP": 2024, "SOURCE": "CFT"},
          {"ACTIVITY": "Manual", "YEAR_COMP": 2024, "SOURCE": "TWIG"}]
    out = dedupe_events(ev, cft_window=(2014, 2024))
    assert len(out) == 1 and out[0]["SOURCE"] == "CFT+TWIG"


def test_dedupe_keeps_far_apart_years():
    ev = [{"ACTIVITY": "Broadcast Burn", "YEAR_COMP": 2017, "SOURCE": "CFT"},
          {"ACTIVITY": "Broadcast Burn", "YEAR_COMP": 2022, "SOURCE": "TWIG"}]
    assert len(dedupe_events(ev, year_tol=1)) == 2  # real re-treatment, not a duplicate


def test_dedupe_keeps_type_conflict_inclusive():
    ev = [{"ACTIVITY": "Manual", "YEAR_COMP": 2020, "SOURCE": "CFT"},
          {"ACTIVITY": "Broadcast Burn", "YEAR_COMP": 2020, "SOURCE": "TWIG"}]
    out = dedupe_events(ev)
    assert len(out) == 2
    assert {e["ACTIVITY"] for e in out} == {"Manual", "Broadcast Burn"}


def test_dedupe_twig_only_tail_survives():
    ev = [{"ACTIVITY": "Mechanical", "YEAR_COMP": 2025, "SOURCE": "TWIG"}]
    out = dedupe_events(ev, cft_window=(2014, 2024))
    assert out == ev  # unchanged; additive record


def test_dedupe_alias_collapses_to_canonical():
    ev = [{"ACTIVITY": "Pile Burn", "YEAR_COMP": 2020, "SOURCE": "CFT"},
          {"ACTIVITY": "Broadcast Burn", "YEAR_COMP": 2020, "SOURCE": "TWIG"}]
    out = dedupe_events(ev)
    assert len(out) == 1
    assert out[0]["ACTIVITY"] == "Pile Burn" and out[0]["SOURCE"] == "CFT+TWIG"


def test_dedupe_year_rule_cft_inside_twig_outside():
    inside = dedupe_events(
        [{"ACTIVITY": "Manual", "YEAR_COMP": 2024, "SOURCE": "CFT"},
         {"ACTIVITY": "Manual", "YEAR_COMP": 2025, "SOURCE": "TWIG"}],
        cft_window=(2014, 2024))
    assert inside[0]["YEAR_COMP"] == 2024  # CFT year wins inside window


# ---------------------------------------------------------------------------
# combine_sources / source_coverage
# ---------------------------------------------------------------------------

def test_combine_sources_unique_oid():
    a = _twig([{"ACTIVITY": "Manual", "YEAR_COMP": 2024}]); a["SOURCE"] = "CFT"
    b = _twig([{"ACTIVITY": "Mechanical", "YEAR_COMP": 2025}]); b["SOURCE"] = "TWIG"
    c = combine_sources({"CFT": a, "TWIG": b})
    assert c["OBJECTID"].is_unique and len(c) == 2
    assert set(c["SOURCE"]) == {"CFT", "TWIG"}


def test_source_coverage_from_atoms():
    atoms = gpd.GeoDataFrame({
        "ATOM_ID": [1, 2],
        "ACRES_GIS": [10.0, 20.0],
        "ATTR": [
            '[{"SOURCE":"CFT","YEAR_COMP":2024,"AGENCY_C":"USFS"}]',
            '[{"SOURCE":"TWIG","YEAR_COMP":2025,"AGENCY_C":"BLM"}]',
        ],
        "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
    }, geometry="geometry", crs=26913)
    cov = source_coverage(atoms)
    assert set(cov["SOURCE"]) == {"CFT", "TWIG"}
    assert cov.loc[cov["SOURCE"] == "TWIG", "YEAR_COMP"].iloc[0] == 2025


def test_reconcile_report_classes():
    atoms = gpd.GeoDataFrame({
        "ATOM_ID": [1, 2, 3],
        "ACRES_GIS": [10.0, 20.0, 30.0],
        "ATTR": [
            # agree (collapsed)
            '[{"SOURCE":"CFT+TWIG","ACTIVITY":"Manual","YEAR_COMP":2024}]',
            # type conflict
            '[{"SOURCE":"CFT","ACTIVITY":"Manual","YEAR_COMP":2020},'
            ' {"SOURCE":"TWIG","ACTIVITY":"Mastication","YEAR_COMP":2020}]',
            # twig only
            '[{"SOURCE":"TWIG","ACTIVITY":"Mechanical","YEAR_COMP":2025}]',
        ],
        "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
    }, geometry="geometry", crs=26913)
    rep = reconcile_report(atoms)
    cls = dict(zip(rep["ATOM_ID"], rep["CLASS"]))
    assert cls == {1: "AGREE", 2: "TYPE_CONFLICT", 3: "TWIG_ONLY"}


# ---------------------------------------------------------------------------
# Fire Use / wildfire veto / range / release  (2026-07 taxonomy fixes)
# ---------------------------------------------------------------------------

def test_fire_use_kept_as_own_class_not_broadcast():
    # Managed/beneficial fire must land in 'Fire Use', never Broadcast Burn.
    g = _twig([{"type": None, "activity": "Wildland Fire Use",
                "method": "Fire", "twig_category": "Unplanned Ignition"}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] == "Fire Use"
    assert out["ACTIVITY_STATUS"].iloc[0] == "resolved"


def test_fuels_benefit_fire_use_survives_wildfire_method():
    # activity resolves to Fire Use in the primary fields, so method='Wildfire' /
    # twig_category='Unplanned Ignition' do NOT drop it (veto only fires with empty primary).
    g = _twig([{"type": None, "activity": "Wildfire - Fuels Benefit",
                "method": "Wildfire", "twig_category": "Unplanned Ignition"}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] == "Fire Use"


def test_wildfire_veto_no_broadcast_fallback():
    # No explicit treatment in activity/type; method='Prescribed Burn' would fall back to
    # Broadcast, but the Unplanned-Ignition wildfire marker vetoes it -> dropped.
    g = _twig([{"type": None, "activity": None,
                "method": "Prescribed Burn", "twig_category": "Unplanned Ignition"}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] is None
    assert out["ACTIVITY_STATUS"].iloc[0] == "dropped"
    assert out["ACTIVITY_FIELD"].iloc[0] == "wildfire"


def test_legit_broadcast_fallback_still_works_when_planned():
    # Same shape but a PLANNED ignition -> the method fallback to Broadcast is retained.
    g = _twig([{"type": None, "activity": None,
                "method": "Prescribed Burn", "twig_category": "Planned Ignition"}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] == "Broadcast Burn"


def test_range_prescribed_burn_new_class():
    g = _twig([{"type": None, "activity": "Range Cover Manipulation",
                "method": "Prescribed Burn", "twig_category": "Planned Ignition"}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] == "Prescribed Burn (Range)"


def test_range_nonfire_is_dropped():
    g = _twig([{"type": None, "activity": "Range Cover Manipulation",
                "method": "Tractor Logging", "twig_category": None}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] is None
    assert out["ACTIVITY_STATUS"].iloc[0] == "dropped"


def test_tree_release_chemical_dropped():
    g = _twig([{"type": None, "activity": "Tree Release and Weed",
                "method": "Chemical", "twig_category": None}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] is None
    assert out["ACTIVITY_STATUS"].iloc[0] == "dropped"


def test_tree_release_mechanical_to_mastication():
    g = _twig([{"type": None, "activity": "Tree Release and Weed",
                "method": "Mechanical", "twig_category": None}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] == "Mastication"


def test_tree_release_coded_as_thinning_defers_to_thin():
    g = _twig([{"type": "Thinning", "activity": "Tree Release and Weed",
                "equipment": "Chain Saw", "method": None}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] == "Manual"


def test_tree_release_by_fire_is_quarantined():
    g = _twig([{"type": None, "activity": "Tree Release and Weed",
                "method": "Fire", "twig_category": "Planned Ignition"}])
    out, _ = harmonize_twig_activity(g)
    assert out["ACTIVITY"].iloc[0] is None
    assert out["ACTIVITY_STATUS"].iloc[0] == "quarantined"


def test_classify_range_and_release_units():
    assert classify_range({"method": "Prescribed Burn"}) == "Prescribed Burn (Range)"
    assert classify_range({"method": "Tractor Logging"}) == "DROP"
    assert classify_release({"method": "Chemical"}) == "DROP"
    assert classify_release({"method": "Manual"}) == "Lop and Scatter"
    assert classify_release({"method": "Masticator"}) == "Mastication"
    assert np.isnan(classify_release({"method": "Fire"}))
