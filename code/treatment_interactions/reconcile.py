"""
Cross-source reconciliation for CFT + TWIG treatment records.

The CFT and TWIG describe overlapping ground: in a single Colorado county the two
sources can cover nearly the same treated acres for a given year. The atomic
decomposition (:mod:`treatment_interactions.interactions`) already prevents *area*
double-counting — every square metre belongs to exactly one atom regardless of how
many source polygons cover it. What it does **not** do on its own is reconcile the
*event list* an atom inherits: when both sources log the same treatment on the same
ground, the atom carries two events, and an annual accomplishment table
(``atom_treatment_table``) then counts that treatment twice.

This module supplies:

- :func:`combine_sources` — concatenate per-source frames onto one schema with a
  globally unique ``OBJECTID`` (replaces the ad-hoc ``max_oid + 1 + index``).
- :func:`source_coverage` — a machine-readable record/acre/agency table by
  ``SOURCE × YEAR_COMP``. This is how a coverage gap (e.g. CFT ending in 2024 while
  TWIG runs to 2026, federal-only) is surfaced rather than buried.
- :func:`dedupe_events` — the per-atom event de-duplicator wired into
  ``build_treatment_interactions(event_dedupe=…)``. It removes the TWIG *copy* of a
  CFT record while keeping TWIG's genuinely additive records.
- :func:`reconcile_report` — a per-atom audit classifying cross-source
  relationships (agree / year-conflict / type-conflict / alias / single-source).

Design choices (documented, and deliberately conservative)
----------------------------------------------------------
- **Only cross-source pairs collapse.** Two same-source events in adjacent years are
  left alone — they may be a real re-treatment, which is not this module's to judge.
- **CFT is authoritative for the year inside its coverage window.** CFT is the
  curated Colorado record the QWRA temporal filters were built against; TWIG's year
  is used outside that window (i.e. the recent TWIG-only tail).
- **Type conflicts that are not aliases keep BOTH activities.** That is the intended
  inclusive behaviour: the atom carries both in ``TRT_SET`` and
  ``classify_scenario`` resolves by priority (thinning dominates; a co-occurring
  broadcast burn makes it a complete treatment). This module makes the conflict
  *countable*, it does not suppress it.
- **Aliases are the one exception** — a small, explicit table of "same treatment,
  two labels" pairs (see :data:`ALIAS_RULES`) where blindly keeping both would
  wrongly *promote* the atom (e.g. a pile burn re-labelled as a broadcast burn would
  turn a thin into a complete thin+broadcast scenario and inflate modeled canopy
  change).

author: maxwell.cook@colostate.edu
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd

from .interactions import EVENT_FIELDS, THIN_TYPES, BURNING_TYPES, M2_TO_ACRES


# ---------------------------------------------------------------------------
# Alias rules
# ---------------------------------------------------------------------------
#
# Each rule: a set of activity labels that denote the same on-the-ground treatment
# recorded at different granularity, plus the ``canonical`` label a collapsed
# cross-source pair should carry. Extend as more one-treatment-two-labels cases are
# confirmed.
ALIAS_RULES: list[dict] = [
    {
        "members": {"Pile Burn", "Broadcast Burn"},
        "canonical": "Pile Burn",
        "note": ("Same-ground/same-year cross-source burn recorded at different "
                 "granularity. Keep the surface Pile Burn label so a co-located thin "
                 "is not falsely promoted to a complete thin+broadcast-burn scenario."),
    },
]


def _alias_members(a: str, b: str, rules: list[dict]) -> "dict | None":
    """Return the alias rule whose member-set contains both ``a`` and ``b``, else None."""
    for r in rules or []:
        if a in r["members"] and b in r["members"] and a != b:
            return r
    return None


# ---------------------------------------------------------------------------
# Source combination
# ---------------------------------------------------------------------------

def combine_sources(
    frames: "dict[str, gpd.GeoDataFrame]",
    keep: "list[str] | None" = None,
    crs: "int | None" = None,
) -> gpd.GeoDataFrame:
    """
    Concatenate per-source treatment frames onto one schema for atomic decomposition.

    Parameters
    ----------
    frames : dict[str, GeoDataFrame]
        ``{"CFT": cft_clean, "TWIG": twig_harmonized, …}``. Each frame should carry
        ``EVENT_FIELDS`` (missing ones are filled with None) and a geometry column;
        ``SOURCE`` is set from the dict key if absent.
    keep : list[str], optional
        Output columns. Default ``["OBJECTID"] + EVENT_FIELDS + ["SOURCE", "geometry"]``.
    crs : int, optional
        Target EPSG. Default = the CRS of the first frame.

    Returns
    -------
    GeoDataFrame with a globally unique ``OBJECTID`` (1-based) across all sources.
    """
    if not frames:
        raise ValueError("No source frames provided.")
    keep = keep or (["OBJECTID"] + list(EVENT_FIELDS) + ["SOURCE", "geometry"])
    target_crs = crs or next(iter(frames.values())).crs

    parts = []
    for name, gdf in frames.items():
        g = gdf.copy()
        if g.crs is not None and target_crs is not None:
            g = g.to_crs(target_crs)
        if "SOURCE" not in g.columns:
            g["SOURCE"] = name
        for col in EVENT_FIELDS:
            if col not in g.columns:
                g[col] = None
        geom_name = g.geometry.name
        cols = [c for c in keep if c != "OBJECTID"]
        cols = [c if c != "geometry" else geom_name for c in cols]
        parts.append(g[[c for c in cols if c in g.columns]])

    combined = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry=parts[0].geometry.name, crs=target_crs,
    )
    combined["OBJECTID"] = np.arange(1, len(combined) + 1)
    ordered = [c for c in keep if c in combined.columns or c == "geometry"]
    ordered = [c if c != "geometry" else combined.geometry.name for c in ordered]
    return combined[["OBJECTID"] + [c for c in ordered if c != "OBJECTID"]]


# ---------------------------------------------------------------------------
# Source coverage
# ---------------------------------------------------------------------------

def source_coverage(
    data: "dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame",
    attr_col: str = "ATTR",
    year_col: str = "YEAR_COMP",
    agency_col: str = "AGENCY_C",
    acres_col: str = "GIS_ACRES",
) -> pd.DataFrame:
    """
    Records / acres / agencies by ``SOURCE × YEAR``.

    Accepts either the pre-atomization per-source frames (a dict) or a built atomic
    layer (a GeoDataFrame with an ``attr_col`` event list). For atoms, each event is
    counted once and its acres are the atom's acres attributed to that event's year
    and source (an *accomplishment* view, so a re-treated atom contributes to each
    event year — matching ``atom_treatment_table``).

    Returns columns: ``SOURCE, YEAR_COMP, n_records, acres, agencies``.
    """
    rows = []
    if isinstance(data, dict):
        for name, gdf in data.items():
            g = gdf.copy()
            if acres_col not in g.columns:
                g[acres_col] = g.geometry.area * M2_TO_ACRES if g.crs and g.crs.is_projected else np.nan
            src = g["SOURCE"] if "SOURCE" in g.columns else pd.Series(name, index=g.index)
            grouped = g.assign(_S=src).groupby(["_S", year_col], dropna=False)
            for (s, yr), grp in grouped:
                rows.append({
                    "SOURCE": s, "YEAR_COMP": yr, "n_records": len(grp),
                    "acres": float(grp[acres_col].sum()),
                    "agencies": "|".join(sorted({str(a) for a in grp.get(agency_col, pd.Series(dtype=str)).dropna()})),
                })
    else:
        acres_name = acres_col if acres_col in data.columns else "ACRES_GIS"
        for r in data.itertuples():
            attrs = getattr(r, attr_col, None)
            attrs = json.loads(attrs) if isinstance(attrs, str) else (attrs or [])
            ac = getattr(r, acres_name, np.nan)
            for e in attrs:
                rows.append({
                    "SOURCE": e.get("SOURCE"), "YEAR_COMP": e.get(year_col),
                    "_agency": e.get(agency_col), "acres": ac, "_one": 1,
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["SOURCE", "YEAR_COMP", "n_records", "acres", "agencies"])
        out = (df.groupby(["SOURCE", "YEAR_COMP"], dropna=False)
                 .agg(n_records=("_one", "sum"), acres=("acres", "sum"),
                      agencies=("_agency", lambda s: "|".join(sorted({str(a) for a in s.dropna()}))))
                 .reset_index())
        return out.sort_values(["SOURCE", "YEAR_COMP"]).reset_index(drop=True)

    return pd.DataFrame(rows).sort_values(["SOURCE", "YEAR_COMP"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-atom event de-duplication  (the build hook)
# ---------------------------------------------------------------------------

def _reconcile_year(cft_e: dict, other_e: dict, cft_window):
    """CFT's year inside its coverage window, else the other source's year."""
    cy = cft_e.get("YEAR_COMP")
    oy = other_e.get("YEAR_COMP")
    if cy is None:
        return oy
    if cft_window is None:
        return cy
    lo, hi = cft_window
    return cy if (lo <= cy <= hi) else (oy if oy is not None else cy)


def dedupe_events(
    events: list[dict],
    year_tol: int = 1,
    cft_window: "tuple[int, int] | None" = None,
    alias_rules: "list[dict] | None" = ALIAS_RULES,
    cft_source: str = "CFT",
) -> list[dict]:
    """
    Collapse cross-source duplicate events within a single atom.

    Greedy pairing: each non-CFT (e.g. TWIG) event is matched to at most one unused
    CFT event of the **same activity** (or an :data:`ALIAS_RULES` partner) whose year
    is within ``year_tol``; the closest year wins the match. A matched pair collapses
    to the CFT event, re-tagged ``SOURCE="CFT+TWIG"``, its ``ACTIVITY`` set to the
    alias canonical where applicable, and its year reconciled by
    :func:`_reconcile_year`. Unmatched non-CFT events survive unchanged — those are
    the genuinely additive records (e.g. the recent TWIG-only tail, or a treatment
    CFT never captured).

    Pure and side-effect-free on the input (operates on copies). Intended as the
    ``event_dedupe`` hook of
    :func:`treatment_interactions.interactions.build_treatment_interactions`, e.g.::

        deduper = lambda ev: dedupe_events(ev, cft_window=(2014, 2024))

    Parameters
    ----------
    events : list[dict]
        One atom's source-event records (each with ``ACTIVITY``, ``YEAR_COMP``,
        ``SOURCE``, …).
    year_tol : int
        Max absolute year difference for a cross-source match (default 1).
    cft_window : (int, int), optional
        Inclusive (lo, hi) years where CFT's reported year is authoritative. Outside
        it (or if None → always prefer CFT when present).
    alias_rules : list[dict]
        See :data:`ALIAS_RULES`. Pass ``[]`` to disable alias collapsing.
    cft_source : str
        The authoritative source label (default ``"CFT"``).

    Returns
    -------
    list[dict] — the reconciled event list.
    """
    evs = [dict(e) for e in events]
    cft_idx   = [i for i, e in enumerate(evs) if e.get("SOURCE") == cft_source]
    other_idx = [i for i, e in enumerate(evs) if e.get("SOURCE") != cft_source]

    used_cft: set[int] = set()
    dropped: set[int] = set()

    for j in other_idx:
        ej = evs[j]
        aj, yj = ej.get("ACTIVITY"), ej.get("YEAR_COMP")
        best = None
        for i in cft_idx:
            if i in used_cft:
                continue
            ei = evs[i]
            ai, yi = ei.get("ACTIVITY"), ei.get("YEAR_COMP")
            same = (ai == aj) or (_alias_members(ai, aj, alias_rules) is not None)
            if not same:
                continue
            if yi is None or yj is None or abs(int(yi) - int(yj)) <= year_tol:
                d = abs(int(yi) - int(yj)) if (yi is not None and yj is not None) else 99
                if best is None or d < best[0]:
                    best = (d, i)
        if best is not None:
            i = best[1]
            used_cft.add(i)
            dropped.add(j)
            ei = evs[i]
            alias = _alias_members(ei.get("ACTIVITY"), aj, alias_rules)
            if alias is not None:
                ei["ACTIVITY"] = alias["canonical"]
            ei["YEAR_COMP"] = _reconcile_year(ei, ej, cft_window)
            ei["SOURCE"] = "CFT+TWIG"

    return [evs[k] for k in range(len(evs)) if k not in dropped]


def make_event_deduper(**kwargs):
    """Return a one-arg ``events -> events`` hook bound to :func:`dedupe_events` options."""
    return lambda events: dedupe_events(events, **kwargs)


# ---------------------------------------------------------------------------
# Reconciliation audit
# ---------------------------------------------------------------------------

def reconcile_report(
    atoms: gpd.GeoDataFrame,
    attr_col: str = "ATTR",
    acres_col: str = "ACRES_GIS",
    year_tol: int = 1,
    alias_rules: "list[dict] | None" = ALIAS_RULES,
) -> pd.DataFrame:
    """
    Classify each atom's cross-source relationship for auditing.

    Run this on the **un-deduped** interactions layer to see the raw agreements and
    conflicts; it also reads a deduped layer (collapsed events carry
    ``SOURCE="CFT+TWIG"`` → ``AGREE``). One row per atom, with ``CLASS`` one of:

    - ``AGREE``            — both sources, same activity, years within ``year_tol``
                             (or an already-collapsed ``CFT+TWIG`` event).
    - ``YEAR_CONFLICT``    — same activity from both sources, years differ by more.
    - ``ALIAS``            — cross-source alias pair present (e.g. Pile/Broadcast Burn).
    - ``TYPE_CONFLICT``    — both sources present but disagree on activity type.
    - ``CFT_ONLY`` / ``TWIG_ONLY`` — single-source atom.

    Columns: ``ATOM_ID, ACRES, CLASS, CFT_ACTS, TWIG_ACTS, CFT_YEARS, TWIG_YEARS``.
    A summary (atoms and acres per class) is printed.
    """
    def _acts_years(events, src):
        a, y = set(), set()
        for e in events:
            es = e.get("SOURCE")
            if es == src or es == "CFT+TWIG":
                if e.get("ACTIVITY") is not None:
                    a.add(e["ACTIVITY"])
                if e.get("YEAR_COMP") is not None:
                    y.add(int(e["YEAR_COMP"]))
        return a, y

    rows = []
    for r in atoms.itertuples():
        attrs = getattr(r, attr_col, None)
        attrs = json.loads(attrs) if isinstance(attrs, str) else (attrs or [])
        srcs = {e.get("SOURCE") for e in attrs}
        has_cft  = ("CFT" in srcs) or ("CFT+TWIG" in srcs)
        has_twig = ("TWIG" in srcs) or ("CFT+TWIG" in srcs)

        cft_a, cft_y   = _acts_years(attrs, "CFT")
        twig_a, twig_y = _acts_years(attrs, "TWIG")

        if not (has_cft and has_twig):
            cls = "CFT_ONLY" if has_cft else "TWIG_ONLY"
        elif "CFT+TWIG" in srcs and len(srcs) == 1:
            cls = "AGREE"                       # fully collapsed
        else:
            shared = cft_a & twig_a
            alias_hit = any(_alias_members(a, b, alias_rules)
                            for a in cft_a for b in twig_a)
            if shared:
                yrs_close = (not cft_y or not twig_y
                             or min(abs(a - b) for a in cft_y for b in twig_y) <= year_tol)
                cls = "AGREE" if yrs_close else "YEAR_CONFLICT"
            elif alias_hit:
                cls = "ALIAS"
            else:
                cls = "TYPE_CONFLICT"

        rows.append({
            "ATOM_ID": getattr(r, "ATOM_ID", None),
            "ACRES": getattr(r, acres_col, np.nan),
            "CLASS": cls,
            "CFT_ACTS": "|".join(sorted(cft_a)),
            "TWIG_ACTS": "|".join(sorted(twig_a)),
            "CFT_YEARS": "|".join(str(v) for v in sorted(cft_y)),
            "TWIG_YEARS": "|".join(str(v) for v in sorted(twig_y)),
        })

    rep = pd.DataFrame(rows)
    if len(rep):
        summ = rep.groupby("CLASS")["ACRES"].agg(["size", "sum"]).round(0)
        print("Cross-source reconciliation (atoms | acres):")
        print(summ.to_string())
    return rep
