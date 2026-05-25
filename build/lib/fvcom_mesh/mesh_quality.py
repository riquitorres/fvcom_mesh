"""
mesh_quality.py — Element quality metrics and visualisation utilities.

Implements the element quality measure from Eq. 38 of Kang & Kubatko (2024):

    q = 2r / R

where r = inradius and R = circumradius of a triangle.
A perfectly equilateral triangle has q = 1; degenerate elements approach 0.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Per-element quality
# ---------------------------------------------------------------------------

def element_quality(
    triangles: NDArray[np.int_],
    pts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the quality metric q = 2r/R for each triangle.

    Parameters
    ----------
    triangles : (T, 3) element connectivity (0-based)
    pts : (N, 2) node coordinates

    Returns
    -------
    q : (T,) quality values in [0, 1]
    """
    a = pts[triangles[:, 0]]
    b = pts[triangles[:, 1]]
    c = pts[triangles[:, 2]]

    # Side lengths
    la = np.hypot(b[:, 0] - c[:, 0], b[:, 1] - c[:, 1])  # opposite A
    lb = np.hypot(a[:, 0] - c[:, 0], a[:, 1] - c[:, 1])  # opposite B
    lc = np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])  # opposite C

    # Semi-perimeter
    s = 0.5 * (la + lb + lc)

    # Area (Heron's formula)
    area = np.sqrt(np.maximum(s * (s - la) * (s - lb) * (s - lc), 0.0))

    # Inradius r = area / s
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(s > 0, area / s, 0.0)

    # Circumradius R = (la * lb * lc) / (4 * area)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(area > 0, (la * lb * lc) / (4.0 * area), np.inf)

    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(R > 0, 2.0 * r / R, 0.0)

    return np.clip(q, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def quality_report(q: NDArray[np.float64]) -> dict[str, float]:
    """Return a dict of quality statistics.

    Keys: ``mean``, ``min``, ``median``, ``p5`` (5th percentile),
    ``p10`` (10th percentile), ``n_bad`` (elements with q < 0.3).
    """
    return {
        "mean": float(np.mean(q)),
        "min": float(np.min(q)),
        "median": float(np.median(q)),
        "p5": float(np.percentile(q, 5)),
        "p10": float(np.percentile(q, 10)),
        "n_bad": int(np.sum(q < 0.3)),
        "n_elements": len(q),
    }


def print_quality_report(q: NDArray[np.float64]) -> None:
    """Print a formatted quality summary to stdout."""
    r = quality_report(q)
    print(
        f"Mesh quality (q = 2r/R, ideal = 1):\n"
        f"  Elements : {r['n_elements']}\n"
        f"  Mean     : {r['mean']:.4f}\n"
        f"  Median   : {r['median']:.4f}\n"
        f"  Min      : {r['min']:.4f}\n"
        f"  5th pct  : {r['p5']:.4f}\n"
        f"  10th pct : {r['p10']:.4f}\n"
        f"  q < 0.3  : {r['n_bad']}\n"
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mesh(
    pts: NDArray[np.float64],
    triangles: NDArray[np.int_],
    *,
    quality: Optional[NDArray[np.float64]] = None,
    constraints: Optional[list] = None,
    ax=None,
    figsize: tuple[float, float] = (10, 10),
    cmap: str = "RdYlGn",
    title: str = "FVCOM mesh",
) -> "matplotlib.axes.Axes":
    """Plot the triangular mesh, optionally coloured by quality.

    Parameters
    ----------
    pts : (N, 2) node coordinates
    triangles : (T, 3) connectivity
    quality : (T,) quality array — if provided, colour each element
    constraints : list of Constraint objects — plotted as coloured lines
    ax : existing Axes (created if None)
    figsize : figure size
    cmap : colormap for quality plot

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles)

    if quality is not None:
        tpc = ax.tripcolor(triang, facecolors=quality, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(tpc, ax=ax, label="Element quality q = 2r/R")
    else:
        ax.triplot(triang, "k-", linewidth=0.3, alpha=0.5)

    # Constraints
    if constraints is not None:
        colours = {1: "blue", 2: "red", 3: "green"}
        labels = {1: "Channel", 2: "Internal bnd", 3: "Shoreline"}
        plotted_types: set[int] = set()
        for c in constraints:
            ctype = int(c.ctype)
            col = colours.get(ctype, "grey")
            pts_c = c.smoothed_pts if c.smoothed_pts is not None else c.pts
            lbl = labels.get(ctype, "Constraint") if ctype not in plotted_types else None
            ax.plot(pts_c[:, 0], pts_c[:, 1], "-", color=col, linewidth=1.0, label=lbl)
            plotted_types.add(ctype)
        ax.legend(loc="upper right", fontsize=8)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    return ax


def plot_quality_histogram(
    q: NDArray[np.float64],
    *,
    ax=None,
    bins: int = 50,
    figsize: tuple[float, float] = (7, 4),
) -> "matplotlib.axes.Axes":
    """Plot a histogram of element quality values."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.hist(q, bins=bins, range=(0, 1), edgecolor="white", linewidth=0.5)
    ax.axvline(np.mean(q), color="red", linestyle="--", label=f"Mean = {np.mean(q):.3f}")
    ax.set_xlabel("Element quality q = 2r/R")
    ax.set_ylabel("Count")
    ax.set_title("Element quality distribution")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Multi-panel mesh report
# ---------------------------------------------------------------------------

def plot_mesh_report(
    pts: NDArray[np.float64],
    triangles: NDArray[np.int_],
    depths: Optional[NDArray[np.float64]] = None,
    *,
    save_dir: Optional[str | "Path"] = None,  # type: ignore[name-defined]
    show: bool = True,
    figsize: tuple[float, float] = (10, 9),
) -> list:
    """Generate a set of quality and geometry plots for a mesh.

    Produces up to four figures saved as PNG files in *save_dir*:

    * ``mesh_quality.png``   — mesh coloured by element quality q = 2r/R
    * ``quality_hist.png``   — histogram of element quality
    * ``element_size.png``   — spatial map of element representative size (√area)
    * ``depth.png``          — spatial map of nodal depth (only when *depths* given)

    Parameters
    ----------
    pts : (N, 2) node coordinates
    triangles : (T, 3) element connectivity
    depths : (N,) nodal water depths (optional)
    save_dir : directory in which to save PNG files; created if absent.
               If None, figures are only shown (controlled by *show*).
    show : whether to call ``plt.show()`` for each figure
    figsize : figure size for the spatial maps

    Returns
    -------
    figs : list of matplotlib Figure objects
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from pathlib import Path

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    q = element_quality(triangles, pts)
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], triangles)

    # Element representative size: sqrt(area) for each triangle
    def _triangle_areas(pts, tris):
        v0 = pts[tris[:, 0]]
        v1 = pts[tris[:, 1]]
        v2 = pts[tris[:, 2]]
        cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
                (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        return np.abs(cross) * 0.5

    elem_size = np.sqrt(_triangle_areas(pts, triangles))

    figs = []

    # ---- Figure 1: mesh coloured by quality ----
    fig1, ax1 = plt.subplots(figsize=figsize)
    tpc = ax1.tripcolor(triang, facecolors=q, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(tpc, ax=ax1, label="Element quality  q = 2r/R  (ideal = 1)")
    ax1.set_aspect("equal")
    ax1.set_title("Mesh — element quality")
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    r = quality_report(q)
    ax1.set_xlabel(
        f"Easting (m)   |   {r['n_elements']} elements · mean q = {r['mean']:.3f}"
        f" · min q = {r['min']:.3f}"
    )
    if save_dir is not None:
        fig1.savefig(save_dir / "mesh_quality.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    figs.append(fig1)
    plt.close(fig1)

    # ---- Figure 2: quality histogram ----
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    plot_quality_histogram(q, ax=ax2)
    if save_dir is not None:
        fig2.savefig(save_dir / "quality_hist.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    figs.append(fig2)
    plt.close(fig2)

    # ---- Figure 3: element size map ----
    fig3, ax3 = plt.subplots(figsize=figsize)
    tpc3 = ax3.tripcolor(triang, facecolors=elem_size, cmap="viridis_r")
    cb3 = plt.colorbar(tpc3, ax=ax3, label="Element size  √area  [m]")
    ax3.set_aspect("equal")
    ax3.set_title("Mesh — element size")
    ax3.set_xlabel("Easting (m)")
    ax3.set_ylabel("Northing (m)")
    if save_dir is not None:
        fig3.savefig(save_dir / "element_size.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    figs.append(fig3)
    plt.close(fig3)

    # ---- Figure 4: depth map (nodal, if available) ----
    if depths is not None:
        fig4, ax4 = plt.subplots(figsize=figsize)
        tpc4 = ax4.tripcolor(triang, depths, cmap="Blues", shading="gouraud")
        plt.colorbar(tpc4, ax=ax4, label="Depth [m]")
        ax4.set_aspect("equal")
        ax4.set_title("Mesh — bathymetry")
        ax4.set_xlabel("Easting (m)")
        ax4.set_ylabel("Northing (m)")
        if save_dir is not None:
            fig4.savefig(save_dir / "depth.png", dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        figs.append(fig4)
        plt.close(fig4)

    return figs
