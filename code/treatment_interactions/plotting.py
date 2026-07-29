"""
Plotting utilities for treatment spatial data.

author: maxwell.cook@colostate.edu
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LogNorm

# Canonical ACTIVITY vocabulary (matches interactions.py THIN_TYPES / BURNING_TYPES /
# TRT_PRIORITY) mapped onto tab10 in a fixed order, so colors never shift between
# calls with different data subsets. Unrecognized activities fall back to gray.
ACTIVITY_ORDER = [
    "Mechanical", "Manual", "Mastication", "Removal", "Pile Burn",
    "Pile Fuels", "Chipping", "Mulching", "Lop and Scatter", "Broadcast Burn",
    "Prescribed Burn (Range)", "Fire Use",
]
_ACTIVITY_FALLBACK_COLOR = "#7f7f7f"


def _activity_color_map(activities, cmap: str = "tab10") -> dict[str, str]:
    """Stable ACTIVITY -> hex color mapping, fixed order, independent of input subset."""
    palette = colormaps[cmap]
    base = {
        act: palette(i % palette.N) for i, act in enumerate(ACTIVITY_ORDER)
    }
    return {
        act: base.get(act, _ACTIVITY_FALLBACK_COLOR)
        for act in dict.fromkeys(activities)
    }


def plot_treatment_heatmap(
    points: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    grid_size: float = 5000,
    cmap: str = "YlOrRd",
    title: str = "Treatment Heatmap",
    cbar_label: str | None = None,
    boundary_color: str = "black",
    boundary_lw: float = 0.5,
    alpha: float = 0.85,
    figsize: tuple[float, float] = (8, 6),
    out_fp: str | Path | None = None,
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2-D histogram heatmap of treatment locations on a regular grid.

    Accepts polygon or point GeoDataFrames — polygons are converted to
    centroids automatically. The grid extent is driven by *boundary*.

    Parameters
    ----------
    points : GeoDataFrame
        Treatment polygons or points. Polygons are centroided before binning.
        Must share a CRS with *boundary* (or will be reprojected to match).
    boundary : GeoDataFrame
        Reference boundary layer plotted as an outline overtop the heatmap
        (e.g., state counties, watershed boundary). Drives the grid extent.
    grid_size : float
        Grid cell size in the CRS map units (default 5,000 m = 5 km).
    cmap : str
        Matplotlib colormap (default 'YlOrRd').
    title : str
        Figure title.
    cbar_label : str, optional
        Colorbar label. Defaults to 'Treatments per {grid_size/1000:.0f}km² cell (log scale)'.
    boundary_color : str
        Edge color for the boundary overlay (default 'black').
    boundary_lw : float
        Line width for the boundary overlay (default 0.5).
    alpha : float
        Heatmap transparency (default 0.85).
    figsize : tuple
        Figure size in inches (default (8, 6)).
    out_fp : path-like, optional
        If provided, save the figure to this path.
    dpi : int
        Resolution for saved figure (default 150).

    Returns
    -------
    (fig, ax)
    """
    crs = boundary.crs

    # --- Reproject points to match boundary CRS
    pts = points.to_crs(crs).copy()

    # --- Convert polygons to centroids if needed
    if not all(pts.geometry.geom_type == "Point"):
        pts["geometry"] = pts.geometry.centroid

    # --- Build the grid over the boundary extent
    bounds = boundary.total_bounds  # [xmin, ymin, xmax, ymax]
    xx = np.arange(bounds[0], bounds[2], grid_size)
    yy = np.arange(bounds[1], bounds[3], grid_size)

    heatmap, x_edges, y_edges = np.histogram2d(
        pts.geometry.x, pts.geometry.y, bins=[xx, yy]
    )

    # --- Mask zeros, log-norm
    heatmap_m = np.ma.masked_where(heatmap == 0, heatmap)
    pos = heatmap_m[heatmap_m > 0]
    if pos.size == 0:
        raise ValueError("No non-zero cells — check that points overlap the boundary extent.")
    vmin, vmax = pos.min(), heatmap_m.max()

    # --- Plot
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(
        heatmap_m.T,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin="lower",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        alpha=alpha,
        interpolation="bilinear",
    )
    boundary.plot(ax=ax, color="none", edgecolor=boundary_color, linewidth=boundary_lw)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.grid(False)

    if cbar_label is None:
        km = grid_size / 1000
        cbar_label = f"Treatments per {km:.0f}km² cell (log scale)"
    plt.colorbar(img, ax=ax, label=cbar_label, pad=0.03, shrink=0.75)

    plt.tight_layout()

    if out_fp is not None:
        out_fp = Path(out_fp)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_fp, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_activity_acres(
    events,
    activity_col: str = "ACTIVITY",
    acres_col: str = "ACRES_GIS",
    figsize: tuple[float, float] = (7, 5),
    out_fp: str | Path | None = None,
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes, dict[str, str]]:
    """
    Horizontal bar chart of total acreage by treatment ACTIVITY.

    Parameters
    ----------
    events : DataFrame or GeoDataFrame
        One row per treatment event, e.g. helpers.events_to_rows(interactions).
        Must contain *activity_col* and *acres_col*.
    activity_col : str
        Column of treatment type labels (default 'ACTIVITY').
    acres_col : str
        Column of acreage to sum per activity (default 'ACRES_GIS').
    figsize : tuple
        Figure size in inches (default (7, 5)).
    out_fp : path-like, optional
        If provided, save the figure to this path.
    dpi : int
        Resolution for saved figure (default 150).

    Returns
    -------
    (fig, ax, color_map) — color_map maps activity label -> color, reusable
    by plot_activity_map() to keep the two visualizations in sync.
    """
    summary = events.groupby(activity_col)[acres_col].sum().sort_values(ascending=True)
    color_map = _activity_color_map(summary.index)
    colors = [color_map[a] for a in summary.index]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(summary.index.astype(str), summary.values, color=colors)
    ax.set_xlabel("Acres")

    plt.tight_layout()

    if out_fp is not None:
        out_fp = Path(out_fp)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_fp, dpi=dpi, bbox_inches="tight")

    return fig, ax, color_map


def plot_activity_map(
    events: gpd.GeoDataFrame,
    activity_col: str = "ACTIVITY",
    color_map: dict[str, str] | None = None,
    markersize: float = 8,
    alpha: float = 0.8,
    figsize: tuple[float, float] = (8, 6),
    out_fp: str | Path | None = None,
    dpi: int = 150,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Simple point map of treatment events, colored by ACTIVITY.

    Polygon geometries are converted to centroids automatically.

    Parameters
    ----------
    events : GeoDataFrame
        One row per treatment event with geometry, e.g.
        helpers.events_to_rows(interactions, keep_geometry=True).
    activity_col : str
        Column of treatment type labels (default 'ACTIVITY').
    color_map : dict, optional
        activity label -> color. Pass the color_map returned by
        plot_activity_acres() to keep colors in sync; if omitted, one is
        built with the same fixed ordering.
    markersize : float
        Point marker size (default 8).
    alpha : float
        Point transparency (default 0.8).
    figsize : tuple
        Figure size in inches (default (8, 6)).
    out_fp : path-like, optional
        If provided, save the figure to this path.
    dpi : int
        Resolution for saved figure (default 150).

    Returns
    -------
    (fig, ax)
    """
    pts = events.copy()
    if not all(pts.geometry.geom_type == "Point"):
        pts["geometry"] = pts.geometry.centroid

    if color_map is None:
        color_map = _activity_color_map(pts[activity_col])
    colors = pts[activity_col].map(color_map).fillna(_ACTIVITY_FALLBACK_COLOR)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pts.geometry.x, pts.geometry.y, c=colors, s=markersize, alpha=alpha, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")

    plt.tight_layout()

    if out_fp is not None:
        out_fp = Path(out_fp)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_fp, dpi=dpi, bbox_inches="tight")

    return fig, ax
