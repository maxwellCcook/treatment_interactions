"""
TWIG (Treatment & Wildfire Interagency Geodatabase) — ingest and harmonization
to the Colorado Forest Tracker (CFT) ``ACTIVITY`` vocabulary.

TWIG is broader and more current than the CFT (it carries federal completions to
2026 where the CFT presently ends at 2024) but has no single clean activity field:
the "what was done" is spread across ``activity`` / ``type`` and can disagree, and
the "how" lives in ``method`` / ``equipment`` / ``twig_category``. This module maps
all of that onto the CFT vocabulary so both sources feed the **same** atomic
decomposition (:mod:`treatment_interactions.interactions`).

Harmonization is table-driven (``data/twig_crosswalk.csv``) rather than a wall of
inline dicts, and follows one documented rule with a single overriding priority:

    **be inclusive toward canopy-fuel-affecting treatments.**

TEALOM adjusts canopy/surface fuels and re-runs fire models *based on treatment
type*, so a canopy treatment that is misfiled as surface-only (and later dropped by
the modeled-scenario filter) silently removes real acres from the outcome. The
resolution therefore lets a canopy signal win over a surface-only one.

Resolution rule (see :func:`harmonize_twig_activity`)
-----------------------------------------------------
For each record, each populated descriptor field yields at most one candidate
``(target, effect, confidence)`` via the crosswalk. ``effect`` is ``canopy`` /
``surface`` / ``none``; ``target`` is a CFT activity, a *sentinel* resolved later by a
method-aware classifier (``"Thin"`` → :func:`classify_thin`, ``"Range"`` →
:func:`classify_range`, ``"Release"`` → :func:`classify_release`), or ``"DROP"``
(non-fuels: chemical, grazing, seeding, survey).

Two dedicated classes keep fire from being over-collapsed onto ``Broadcast Burn``:
``"Fire Use"`` (managed / beneficial wildland fire — kept, never broadcast) and
``"Prescribed Burn (Range)"`` (non-forest range/pollinator burns). A **wildfire veto**
(see :data:`WILDFIRE_MARKERS`) drops a record marked as unplanned/wildfire when no
primary field names a treatment, instead of letting the coarse
``method='Fire'/'Prescribed Burn'`` fallback misfile it as a broadcast burn.

1. **Decide from the two treatment-describing fields first** — ``activity`` and
   ``type``. Among their real-treatment candidates, a ``canopy`` candidate beats a
   ``surface`` one; ties break by field specificity (``activity`` > ``type``) then
   confidence. This is what turns ``type="Biomass Removal"`` +
   ``activity="Stand Clearcut"`` into ``Mechanical`` instead of ``Removal``.
2. **Fall back to ``method`` / ``twig_category``** only when neither ``activity``
   nor ``type`` names a treatment. These coarse fields never *override* a populated
   type/activity — that guard is deliberate: it stops ``method="Prescribed Burn"``
   from promoting a ``type="Machine Pile Burn"`` (a surface pile burn) into a
   canopy broadcast burn.
3. If the winner is ``"DROP"`` → excluded (logged). If nothing resolves →
   ``ACTIVITY = None`` and the record lands in the **quarantine report** with its
   acres and every descriptor field, so a reviewer can extend the crosswalk. A
   record is never silently turned into NaN and lost.
4. A ``"Thin"`` winner is split into ``Manual`` / ``Mechanical`` by
   :func:`classify_thin` (equipment/method), applied **once**.

author: maxwell.cook@colostate.edu
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .ingest import query_feature_service, M2_TO_ACRES
from .interactions import EVENT_FIELDS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Current TWIG (ReshapeWildfire) treatment index. The NCFC pilot's cached
# GeoPackage came from the older ``Treatment_Index_and_Intersections`` service;
# both are recorded so either can be reproduced.
TWIG_URL        = "https://gis.reshapewildfire.org/arcgis/rest/services/Hosted/Treatment_Index_View/FeatureServer"
TWIG_URL_LEGACY = "https://gis.reshapewildfire.org/arcgis/rest/services/Hosted/Treatment_Index_and_Intersections/FeatureServer"
TWIG_LAYER      = 0

# Descriptor fields consulted for the activity crosswalk, in specificity order.
# The first two ("what was done") drive the decision; the last two are fallbacks.
PRIMARY_FIELDS  = ["activity", "type"]
FALLBACK_FIELDS = ["method", "twig_category"]
DESCRIPTOR_FIELDS = PRIMARY_FIELDS + FALLBACK_FIELDS

_FIELD_RANK  = {"activity": 4, "type": 3, "method": 2, "twig_category": 1}
_EFFECT_RANK = {"canopy": 2, "surface": 1, "none": 0}
_CONF_RANK   = {"high": 3, "medium": 2, "low": 1}

THIN_SENTINEL    = "Thin"     # crosswalk target resolved by classify_thin()
RANGE_SENTINEL   = "Range"    # crosswalk target resolved by classify_range()
RELEASE_SENTINEL = "Release"  # crosswalk target resolved by classify_release()
SENTINELS = {THIN_SENTINEL, RANGE_SENTINEL, RELEASE_SENTINEL}

# Descriptors that mark a record as (unplanned) wildfire rather than a treatment. When a
# record carries one of these AND no primary field (activity/type) names a real treatment,
# it is dropped instead of falling back to method='Fire'/'Prescribed Burn' -> Broadcast Burn.
# That fallback previously misfiled wildfire (and fire-use lacking an explicit activity) as
# broadcast burning. Managed/beneficial fire is protected because it carries an explicit
# activity/type ('Wildland Fire Use', 'Fire Use', ...) that resolves to the Fire Use class
# in the primary fields, so the veto (which only fires when the primary is empty) never
# reaches it. Values are compared lower-cased.
WILDFIRE_MARKERS = {
    "twig_category": {"unplanned ignition"},
    "method":        {"wildfire"},
    "category":      {"wildfire non-treatment"},
}

# TWIG abbreviated funding-source codes → CFT's full-name vocabulary. Only the
# confirmed 1:1 matches; the NBSP in the BIL string matches CFT's exact spacing so
# the two labels merge instead of concatenating.
FUND_SOURCE_CROSSWALK = {
    "BIL":  "Bipartisan\xa0Infrastructure Law\xa0(BIL)",
    "CFLR": "Collaborative Forest Landscape Restoration (CFLR)",
}

_CROSSWALK_FP = Path(__file__).parent / "data" / "twig_crosswalk.csv"


# ---------------------------------------------------------------------------
# Crosswalk
# ---------------------------------------------------------------------------

def load_crosswalk(fp: str | Path | None = None) -> pd.DataFrame:
    """
    Load the versioned TWIG→CFT crosswalk table.

    Columns: ``field, value, activity, effect, confidence, note``. Returns it
    indexed by ``(field, value)`` for O(1) lookup, with string values stripped.
    """
    fp = Path(fp) if fp is not None else _CROSSWALK_FP
    xw = pd.read_csv(fp)
    for c in ("field", "value", "activity", "effect", "confidence"):
        xw[c] = xw[c].astype(str).str.strip()
    dups = xw.duplicated(["field", "value"])
    if dups.any():
        raise ValueError(f"Duplicate (field, value) keys in crosswalk:\n{xw[dups]}")
    return xw.set_index(["field", "value"])


# ---------------------------------------------------------------------------
# Thinning classification (Manual vs Mechanical)
# ---------------------------------------------------------------------------

def classify_thin(row) -> "str | float":
    """
    Classify a thinning record as ``'Manual'`` or ``'Mechanical'`` from its
    equipment / method, else ``np.nan`` if the row is not a thinning treatment.

    Ported verbatim from ``tealom.treatments.geometry.classify_thin`` so this
    package stays free of any ``tealom`` import (portability goal). Reads columns
    ``type, activity, equipment, method``. Defaults an unresolvable thin to
    ``'Mechanical'``.
    """
    ttype    = str(row.get("type")     or "").strip().lower()
    activity = str(row.get("activity") or "").strip().lower()

    thin_activities = [
        "sanitation cut", "improvement cut", "group selection cut",
        "overstory removal cut", "salvage cut", "recreation removal", "salvage",
    ]
    is_thin = (
        ttype == "thinning"
        or "thin" in activity
        or any(k in activity for k in thin_activities)
    )
    if not is_thin:
        return np.nan

    eq  = str(row.get("equipment") or "").strip().lower()
    mth = str(row.get("method")    or "").strip().lower()

    manual_eq = {"chain saw", "hand work", "hand saw", "manual logging"}
    mech_eq   = {
        "feller buncher", "tree shear", "dozer", "masticator",
        "rubber tired skidder logging", "helicopter logging -medium",
        "helicopter logging -small",
    }
    manual_m = {"manual", "power hand", "manual logging", "cut trees and brush"}
    mech_m   = {"mechanical", "tractor logging", "logging methods", "helicopter", "removal"}

    if eq in manual_eq:  return "Manual"
    if eq in mech_eq:    return "Mechanical"
    if mth in manual_m:  return "Manual"
    if mth in mech_m:    return "Mechanical"
    return "Mechanical"   # default


def classify_range(row) -> str:
    """
    Resolve a range/pollinator objective activity (RANGE_SENTINEL) to a treatment by its
    ``method``.

    Range activities name a management *objective*, not a treatment; the treatment lives in
    ``method``. Fire methods -> ``'Prescribed Burn (Range)'`` (a distinct non-forest burn
    class, kept separate from a forest broadcast burn). Any non-fire method -> ``'DROP'``
    (non-forest range management, excluded from the fuels analysis). Returns ``'DROP'`` or a
    label -- never NaN.
    """
    mth = str(row.get("method") or "").strip().lower()
    if any(k in mth for k in ("fire", "prescribed burn", "broadcast", "jackpot", "underburn")):
        return "Prescribed Burn (Range)"
    return "DROP"


def classify_release(row) -> "str | float":
    """
    Resolve a 'Tree Release and Weed' activity (RELEASE_SENTINEL) -- release / weed control,
    not a burn -- to a treatment by its ``method`` (and ``type``).

    ``chemical`` -> ``'DROP'``; a release coded as ``type='Thinning'`` is a (precommercial)
    thin, deferred to :func:`classify_thin`; ``manual`` / hand / cut -> ``'Lop and Scatter'``;
    mechanical equipment (masticator / tractor / dozer / mobile ground / logging / crushing)
    -> ``'Mastication'``; ``fire`` or anything else -> ``np.nan`` (quarantine for review).
    Returns a label, ``'DROP'``, or NaN.
    """
    mth = str(row.get("method") or "").strip().lower()
    if "chemical" in mth:
        return "DROP"
    # A release recorded as a thinning is a (precommercial) thin -> use the thin classifier.
    if str(row.get("type") or "").strip().lower() == "thinning":
        return classify_thin(row) or "Mechanical"
    if any(k in mth for k in ("manual", "hand", "cut trees", "power hand", "girdle")):
        return "Lop and Scatter"
    if any(k in mth for k in ("mechanical", "masticat", "tractor", "dozer", "grind", "mow",
                              "mobile ground", "logging", "crush", "push", "feller",
                              "shear", "skidder")):
        return "Mastication"
    return np.nan   # fire / not-applicable / unknown -> quarantine


def _is_wildfire(row_d) -> bool:
    """True if a descriptor marks the record as (unplanned) wildfire (see WILDFIRE_MARKERS)."""
    for field, vals in WILDFIRE_MARKERS.items():
        if str(row_d.get(field) or "").strip().lower() in vals:
            return True
    return False


# ---------------------------------------------------------------------------
# Activity harmonization
# ---------------------------------------------------------------------------

def _candidate(xw: pd.DataFrame, field: str, value) -> "tuple | None":
    """Look up one (field, value) → (target, effect, conf, field_rank) or None."""
    if value is None:
        return None
    val = str(value).strip()
    if not val or val.lower() in {"nan", "none", "n/a", "na"}:
        return None
    try:
        rec = xw.loc[(field, val)]
    except KeyError:
        return None
    return (rec["activity"], rec["effect"], rec["confidence"], _FIELD_RANK[field], field)


def _pick(cands: list) -> "tuple | None":
    """Best real-treatment candidate by (effect, field specificity, confidence)."""
    real = [c for c in cands if c[0] != "DROP" and c[1] in _EFFECT_RANK and _EFFECT_RANK[c[1]] > 0]
    if not real:
        return None
    return max(real, key=lambda c: (_EFFECT_RANK[c[1]], c[3], _CONF_RANK.get(c[2], 0)))


def harmonize_twig_activity(
    twig: gpd.GeoDataFrame,
    crosswalk: "pd.DataFrame | str | Path | None" = None,
    thin_classifier=classify_thin,
    acres_col: str = "acres",
) -> "tuple[gpd.GeoDataFrame, pd.DataFrame]":
    """
    Resolve each TWIG record to a single CFT ``ACTIVITY`` (or drop / quarantine it).

    Implements the canopy-inclusive resolution documented at the module head.
    Adds ``ACTIVITY`` (None for dropped/quarantined records), plus provenance
    columns ``ACTIVITY_FIELD`` (which descriptor field decided), ``ACTIVITY_CONF``
    (its confidence), and ``ACTIVITY_STATUS`` (``resolved`` / ``dropped`` /
    ``quarantined``). Callers keep ``twig[twig["ACTIVITY"].notna()]``.

    Parameters
    ----------
    twig : GeoDataFrame
        Raw TWIG records (columns ``activity, type, method, twig_category,
        equipment`` where present).
    crosswalk : DataFrame or path, optional
        Pre-loaded crosswalk (see :func:`load_crosswalk`) or a path to one.
        Default = the packaged ``data/twig_crosswalk.csv``.
    thin_classifier : callable
        Row → ``'Manual'`` / ``'Mechanical'`` / NaN. Applied once, only to
        records whose resolved target is the ``"Thin"`` sentinel.
    acres_col : str
        TWIG's reported-acreage column, carried into the report as ``ACRES_REPORTED``
        for provenance only (it is per-project accomplishment acreage, not footprint).

    Returns
    -------
    (GeoDataFrame, DataFrame)
        The annotated TWIG frame, and a per-record report (descriptor fields,
        ``ACRES_GEO`` (polygon footprint) + ``ACRES_REPORTED`` (TWIG's inflated field),
        ACTIVITY, ACTIVITY_FIELD/CONF/STATUS) for auditing quarantined and dropped
        records. The printed summary aggregates ``ACRES_GEO``.
    """
    if not isinstance(crosswalk, pd.DataFrame):
        crosswalk = load_crosswalk(crosswalk)

    out = twig.copy().reset_index(drop=True)

    # Two distinct acreage series — they mean different things and must not be conflated:
    #  * ACRES_REPORTED — TWIG's reported/accomplishment ``acres`` field. TWIG stamps the same
    #    project-level value on *every* sub-polygon of a project, so summing it over records
    #    over-counts the footprint ~20-30x. Kept for provenance; NEVER aggregate it as area.
    #  * ACRES_GEO — actual polygon area. This is the real footprint and the only series the
    #    summary/report should sum (still per-record, so it may include cross-record overlap).
    if acres_col in out.columns:
        rep_acres = pd.to_numeric(out[acres_col], errors="coerce")
    else:
        rep_acres = pd.Series(np.nan, index=out.index)
    if out.crs is not None and out.crs.is_projected:
        geo_acres = out.geometry.area * M2_TO_ACRES
    else:
        geo_acres = pd.Series(np.nan, index=out.index)

    activities, fields, confs, statuses, thin_flags = [], [], [], [], []

    for row in out.itertuples(index=False):
        row_d = row._asdict()

        prim = [c for f in PRIMARY_FIELDS
                if (c := _candidate(crosswalk, f, row_d.get(f))) is not None]
        prim_pick = _pick(prim)

        # Wildfire veto: a wildfire/unplanned-ignition record with no explicit primary
        # treatment is dropped, rather than falling back to method='Fire'/'Prescribed Burn'
        # -> Broadcast Burn. Managed/beneficial fire carries a Fire Use activity/type in the
        # primary fields, so prim_pick is non-None for it and the veto never applies.
        if prim_pick is None and _is_wildfire(row_d):
            activities.append(None); fields.append("wildfire")
            confs.append(None); statuses.append("dropped"); thin_flags.append(False)
            continue

        fall = [c for f in FALLBACK_FIELDS
                if (c := _candidate(crosswalk, f, row_d.get(f))) is not None]
        chosen = prim_pick or _pick(fall)
        ran_thin = False

        if chosen is not None:
            target, _effect, conf, _rank, field = chosen
            # Sentinels name an objective whose treatment depends on other fields; resolve them.
            if target == THIN_SENTINEL:
                t = thin_classifier(row_d)
                target = t if (t is not None and not (isinstance(t, float) and np.isnan(t))) else "Mechanical"
                ran_thin = True
            elif target == RANGE_SENTINEL:
                target = classify_range(row_d)      # 'Prescribed Burn (Range)' or 'DROP'
            elif target == RELEASE_SENTINEL:
                target = classify_release(row_d)    # label / 'DROP' / NaN

            # A sentinel classifier may resolve to DROP or NaN (quarantine).
            if target == "DROP":
                activities.append(None); fields.append(field)
                confs.append(None); statuses.append("dropped"); thin_flags.append(ran_thin)
                continue
            if target is None or (isinstance(target, float) and np.isnan(target)):
                activities.append(None); fields.append(field)
                confs.append(None); statuses.append("quarantined"); thin_flags.append(ran_thin)
                continue

            activities.append(target); fields.append(field)
            confs.append(conf); statuses.append("resolved"); thin_flags.append(ran_thin)
        else:
            # No real-treatment candidate anywhere → dropped (a DROP signal exists)
            # or quarantined (nothing in the crosswalk at all).
            has_drop = any(
                (c := _candidate(crosswalk, f, row_d.get(f))) is not None and c[0] == "DROP"
                for f in DESCRIPTOR_FIELDS
            )
            activities.append(None)
            fields.append(next((c[4] for f in DESCRIPTOR_FIELDS
                                if (c := _candidate(crosswalk, f, row_d.get(f))) is not None
                                and c[0] == "DROP"), None) if has_drop else None)
            confs.append(None)
            statuses.append("dropped" if has_drop else "quarantined")
            thin_flags.append(ran_thin)

    out["ACTIVITY"]        = activities
    out["ACTIVITY_FIELD"]  = fields
    out["ACTIVITY_CONF"]   = confs
    out["ACTIVITY_STATUS"] = statuses

    report = pd.DataFrame({
        **{f: out[f] if f in out.columns else None for f in DESCRIPTOR_FIELDS + ["equipment"]},
        "ACRES_GEO": geo_acres.values,
        "ACRES_REPORTED": rep_acres.values,
        "ACTIVITY": out["ACTIVITY"].values,
        "ACTIVITY_FIELD": out["ACTIVITY_FIELD"].values,
        "ACTIVITY_CONF": out["ACTIVITY_CONF"].values,
        "ACTIVITY_STATUS": out["ACTIVITY_STATUS"].values,
        "RAN_CLASSIFY_THIN": thin_flags,
    })

    # --- Summary (aggregate ACRES_GEO — the real footprint, not the inflated reported field)
    vc = out["ACTIVITY_STATUS"].value_counts()
    ac = report.groupby("ACTIVITY_STATUS")["ACRES_GEO"].sum()
    print("TWIG harmonization (footprint acres):")
    for st in ("resolved", "dropped", "quarantined"):
        print(f"  {st:<12} {int(vc.get(st, 0)):>5,} records  {ac.get(st, 0):>10,.0f} ac")
    n_thin = int(np.sum(thin_flags))
    print(f"  (classify_thin split {n_thin:,} 'Thin' records into Manual/Mechanical)")
    q = report[report["ACTIVITY_STATUS"] == "quarantined"]
    if len(q):
        print(f"\n[quarantine] {len(q):,} records ({q['ACRES_GEO'].sum():,.0f} ac) unresolved — "
              f"extend the crosswalk:")
        cols = [c for c in DESCRIPTOR_FIELDS if c in q.columns]
        print(q.groupby(cols, dropna=False)["ACRES_GEO"].agg(["size", "sum"]).to_string())

    return out, report


# ---------------------------------------------------------------------------
# Schema harmonization (TWIG → CFT event schema)
# ---------------------------------------------------------------------------

def twig_to_cft_schema(
    twig: gpd.GeoDataFrame,
    id_offset: int = 0,
    source_label: str = "TWIG",
) -> gpd.GeoDataFrame:
    """
    Rename / synthesize columns so a harmonized TWIG frame matches the CFT event
    schema expected by :func:`treatment_interactions.interactions.build_treatment_interactions`.

    - ``agency`` → ``AGENCY_C``, ``fund_source`` → ``FUND_SOURCE`` (through
      :data:`FUND_SOURCE_CROSSWALK`), ``YEAR_COMP`` from ``year_comp`` if needed.
    - Any missing ``EVENT_FIELDS`` (``FUND_TYPE, PARTNERS, LANDOWNER, MGT_TYPE``,
      …) are created as ``None``.
    - ``SOURCE = source_label`` for provenance.
    - A non-colliding ``OBJECTID = id_offset + 1 + row`` (pass the CFT max OID so
      source IDs stay unique after the concat).

    Requires an ``ACTIVITY`` column (run :func:`harmonize_twig_activity` first and
    drop null-ACTIVITY rows). Returns only the ``EVENT_FIELDS + SOURCE + geometry``
    columns plus ``OBJECTID``.
    """
    out = twig.copy().reset_index(drop=True)
    if "ACTIVITY" not in out.columns:
        raise ValueError("Run harmonize_twig_activity() first — no ACTIVITY column.")
    if out["ACTIVITY"].isna().any():
        raise ValueError("Drop null-ACTIVITY (dropped/quarantined) rows before schema mapping.")

    if "fund_source" in out.columns:
        out["FUND_SOURCE"] = out["fund_source"].map(FUND_SOURCE_CROSSWALK).fillna(out["fund_source"])
    if "agency" in out.columns and "AGENCY_C" not in out.columns:
        out = out.rename(columns={"agency": "AGENCY_C"})
    if "YEAR_COMP" not in out.columns and "year_comp" in out.columns:
        out = out.rename(columns={"year_comp": "YEAR_COMP"})

    for col in EVENT_FIELDS:
        if col not in out.columns:
            out[col] = None

    out["SOURCE"]   = source_label
    out["OBJECTID"] = id_offset + 1 + np.arange(len(out))

    keep = ["OBJECTID", *EVENT_FIELDS, "SOURCE", out.geometry.name]
    return out[keep]


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def download_twig(
    aoi: "gpd.GeoDataFrame | gpd.GeoSeries",
    url: str = TWIG_URL,
    layer: int = TWIG_LAYER,
    year_start: "int | None" = None,
    year_end: "int | None" = None,
    statuses: "tuple[str, ...]" = ("Completed",),
    drop_errors: "tuple[str, ...]" = ("DUPLICATE_DROP",),
    crs: int = 26913,
    out_fp: "str | Path | None" = None,
) -> gpd.GeoDataFrame:
    """
    Download TWIG treatment records for an AOI, derive ``YEAR_COMP``, and clip.

    Parameters
    ----------
    aoi : GeoDataFrame / GeoSeries
        Analysis area; used both for the server-side envelope pre-filter and the
        exact ``sjoin`` clip afterwards.
    url, layer : str, int
        TWIG FeatureServer root + layer (default the current ReshapeWildfire
        Treatment Index View, layer 0).
    year_start, year_end : int, optional
        Inclusive ``YEAR_COMP`` bounds (derived from ``treatment_date``).
    statuses : tuple[str]
        ``status`` values to keep. Default ``("Completed",)``; the NCFC pilot also
        allowed ``"Other"`` for NPS/DOI records that carry a treatment date.
    drop_errors : tuple[str]
        ``error`` flag values to exclude. Default ``("DUPLICATE_DROP",)`` — TWIG's own
        de-duplication marks the copy it discarded; keeping it would re-introduce the
        duplicate. Pass ``()`` to keep every record regardless of ``error``.
    crs : int
        EPSG code to work in / return.
    out_fp : path-like, optional
        Save the clipped result to GeoPackage before returning.

    Returns
    -------
    GeoDataFrame in ``EPSG:crs`` with ``YEAR_COMP`` and (if projected) ``GIS_ACRES``.
    """
    where = "1=1"
    if statuses:
        quoted = ", ".join(f"'{s}'" for s in statuses)
        where = f"status IN ({quoted})"

    gdf = query_feature_service(url, layer, where=where, crs=crs, label="TWIG records", aoi=aoi)
    if gdf.empty:
        print("No TWIG records returned.")
        return gdf

    # Honor TWIG's own QA flag: drop the records it already marked as discarded duplicates.
    if drop_errors and "error" in gdf.columns:
        n0 = len(gdf)
        gdf = gdf[~gdf["error"].isin(drop_errors)].reset_index(drop=True)
        if len(gdf) < n0:
            print(f"Dropped {n0 - len(gdf):,} TWIG records flagged {list(drop_errors)}")

    # treatment_date → YEAR_COMP (robust to epoch-ms vs datetime storage)
    td = gdf.get("treatment_date")
    if td is not None:
        if pd.api.types.is_numeric_dtype(td):
            gdf["treatment_date"] = pd.to_datetime(td, unit="ms", errors="coerce")
        else:
            gdf["treatment_date"] = pd.to_datetime(td, errors="coerce", utc=True)
        gdf["YEAR_COMP"] = gdf["treatment_date"].dt.year

    if "YEAR_COMP" in gdf.columns:
        if year_start is not None:
            gdf = gdf[gdf["YEAR_COMP"] >= year_start]
        if year_end is not None:
            gdf = gdf[gdf["YEAR_COMP"] <= year_end]

    # Exact AOI clip (server filter is a bbox only); explode multiparts first.
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = (
        gpd.sjoin(gdf, aoi.to_crs(gdf.crs)[["geometry"]], how="inner", predicate="intersects")
        .drop(columns="index_right")
        .reset_index(drop=True)
    )
    gdf["GIS_ACRES"] = gdf.geometry.area * M2_TO_ACRES
    print(f"TWIG after year/AOI filter: {len(gdf):,} records")

    if out_fp is not None:
        out_fp = Path(out_fp)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_fp)
        print(f"Saved → {out_fp}")

    return gdf


def load_or_download_twig(
    out_fp: "str | Path",
    aoi: "gpd.GeoDataFrame | gpd.GeoSeries",
    **kwargs,
) -> gpd.GeoDataFrame:
    """Load from disk if ``out_fp`` exists, else download and save (see :func:`download_twig`)."""
    out_fp = Path(out_fp)
    if out_fp.exists():
        gdf = gpd.read_file(out_fp)
        print(f"Loaded {len(gdf):,} TWIG records from {out_fp}")
        return gdf
    return download_twig(aoi, out_fp=out_fp, **kwargs)
