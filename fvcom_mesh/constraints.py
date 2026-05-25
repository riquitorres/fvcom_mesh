"""
constraints.py — Identify, smooth and characterise internal mesh constraints.

Implements Sect. 3.2 (constraint types) and Sect. 4.1–4.3 of
Kang & Kubatko (2024), GMD.

Constraint types
----------------
1  Channel centrelines — define the 1-D hydrodynamic domain.
2  Internal boundaries — centrelines of narrow land features (barrier
   islands, levees) acting as sub-grid-scale flow barriers.
3  Land–water boundary lines — shoreline edges that enforce a clear
   delineation between wet and dry cells in the 2-D mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt
from shapely.geometry import (
    LineString, MultiLineString, Polygon, MultiPolygon,
)
from shapely.ops import unary_union, polygonize

from .water_mask import MaskSet, extract_skeleton_polylines
from .utils import polyline_length, resample_polyline


class ConstraintType(IntEnum):
    CHANNEL = 1       # 1-D domain (open channel centrelines)
    INTERNAL_BND = 2  # sub-grid-scale barrier (narrow land features)
    LAND_WATER = 3    # land–water interface (shoreline)


@dataclass
class Constraint:
    """A single internal constraint polyline.

    Parameters
    ----------
    pts : (N, 2) array of (x, y) coordinates in projected metres
    ctype : ConstraintType
    is_junction : bool — True if pts[0] or pts[-1] is a branch junction
    """
    pts: NDArray[np.float64]
    ctype: ConstraintType
    smoothed_pts: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    curvature: Optional[NDArray[np.float64]] = field(default=None, repr=False)

    @property
    def length(self) -> float:
        return polyline_length(self.pts)

    @property
    def n_pts(self) -> int:
        return len(self.pts)

    def start(self) -> NDArray[np.float64]:
        return self.pts[0]

    def end(self) -> NDArray[np.float64]:
        return self.pts[-1]


# ---------------------------------------------------------------------------
# Constraint identification
# ---------------------------------------------------------------------------

def identify_constraints(
    masks: MaskSet,
    delta_w: float,
    min_length: float,
) -> list[Constraint]:
    """Extract all three types of constraints from a processed MaskSet.

    Parameters
    ----------
    masks : fully processed MaskSet (after process_water_mask)
    delta_w : minimum channel width (metres) — used to confirm level-1
              regions are sufficiently narrow
    min_length : discard constraint polylines shorter than this (metres);
                 recommended ≥ h_min / 2

    Returns
    -------
    list of Constraint objects sorted by type
    """
    constraints: list[Constraint] = []
    grid = masks.grid

    # ---- Type 1: channel centrelines (skeletons of water_l1_final) ----
    if masks.water_l1_final is not None:
        channels = extract_skeleton_polylines(
            masks.water_l1_final, grid, min_length=min_length
        )
        for pts in channels:
            constraints.append(Constraint(pts=pts, ctype=ConstraintType.CHANNEL))

    # ---- Type 2: internal boundary centrelines ----
    # These are centrelines of the narrow-land regions that were transferred
    # to the water mask during complex-geometry processing.
    # We obtain them by skeletonising (water_updated \ water) i.e. the newly
    # added water pixels from narrow land.
    if (masks.water_updated is not None and masks.water is not None):
        newly_wet = masks.water_updated & ~masks.water
        if newly_wet.any():
            ib_lines = extract_skeleton_polylines(
                newly_wet, grid, min_length=min_length
            )
            for pts in ib_lines:
                constraints.append(Constraint(pts=pts, ctype=ConstraintType.INTERNAL_BND))

    # ---- Type 3: land–water boundary (shoreline) ----
    # Obtained from the boundary of the updated water mask.
    if masks.water_updated is not None:
        shore = _mask_boundary_polylines(masks.water_updated, grid, min_length=min_length)
    else:
        shore = _mask_boundary_polylines(masks.water, grid, min_length=min_length)
    for pts in shore:
        constraints.append(Constraint(pts=pts, ctype=ConstraintType.LAND_WATER))

    return constraints


def _mask_boundary_polylines(
    mask: NDArray[np.bool_],
    grid: MaskSet,  # actually BackgroundGrid
    min_length: float,
) -> list[NDArray[np.float64]]:
    """Trace the boundary of a binary mask as ordered polylines.

    Uses morphological edge detection: boundary = mask & ~eroded_mask.
    """
    from skimage.morphology import binary_erosion, square
    boundary_pixels = mask & ~binary_erosion(mask, square(3))
    # Use the skeleton of the boundary for ordered traversal
    from skimage.morphology import skeletonize
    skel = skeletonize(boundary_pixels)
    return extract_skeleton_polylines(skel, grid, min_length=min_length)


# ---------------------------------------------------------------------------
# Mainstream construction (Sect. 3.2.2, Step 6)
# ---------------------------------------------------------------------------

def build_mainstreams(
    constraints: list[Constraint],
    junction_tol: float,
) -> list[Constraint]:
    """Merge constraint branches at junctions to form channel mainstreams.

    At each junction (points within *junction_tol* metres of each other),
    the pair of branches with minimum absolute curvature change is merged
    into a single mainstream.

    Parameters
    ----------
    constraints : list of Constraint (all types)
    junction_tol : distance threshold (metres) for identifying junctions

    Returns
    -------
    updated list where channels are merged where possible
    """
    channels = [c for c in constraints if c.ctype == ConstraintType.CHANNEL]
    others = [c for c in constraints if c.ctype != ConstraintType.CHANNEL]

    if len(channels) <= 1:
        return constraints

    # Build endpoint map
    endpoints = []  # (constraint_index, end_index [0 or 1], xy)
    for ci, c in enumerate(channels):
        endpoints.append((ci, 0, c.pts[0]))
        endpoints.append((ci, 1, c.pts[-1]))

    # Find junctions: groups of endpoints within junction_tol
    eps_arr = np.array([ep[2] for ep in endpoints])
    visited = [False] * len(endpoints)
    junction_groups: list[list[int]] = []
    for i in range(len(endpoints)):
        if visited[i]:
            continue
        group = [i]
        for j in range(i + 1, len(endpoints)):
            if not visited[j]:
                d = np.hypot(eps_arr[i, 0] - eps_arr[j, 0],
                              eps_arr[i, 1] - eps_arr[j, 1])
                if d <= junction_tol:
                    group.append(j)
        if len(group) > 1:
            for idx in group:
                visited[idx] = True
            junction_groups.append(group)

    # For each junction, merge the pair with smallest curvature change
    merged_flags = [False] * len(channels)
    merged_channels: list[Constraint] = []

    for group in junction_groups:
        if len(group) < 2:
            continue
        # Find pair with minimum directional change
        best_pair = None
        best_angle = np.inf
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                ci_a, end_a, _ = endpoints[group[a]]
                ci_b, end_b, _ = endpoints[group[b]]
                if ci_a == ci_b:
                    continue
                # Direction vectors at the junction end
                pts_a = channels[ci_a].pts
                pts_b = channels[ci_b].pts
                v_a = _endpoint_direction(pts_a, end_a)
                v_b = _endpoint_direction(pts_b, end_b)
                cos_angle = np.clip(np.dot(v_a, v_b), -1, 1)
                angle = np.arccos(cos_angle)
                if angle < best_angle:
                    best_angle = angle
                    best_pair = (ci_a, end_a, ci_b, end_b)

        if best_pair is None:
            continue

        ci_a, end_a, ci_b, end_b = best_pair
        if merged_flags[ci_a] or merged_flags[ci_b]:
            continue

        pts_a = channels[ci_a].pts
        pts_b = channels[ci_b].pts
        # Orient so they connect head-to-tail
        if end_a == 1 and end_b == 0:
            merged_pts = np.vstack([pts_a, pts_b[1:]])
        elif end_a == 0 and end_b == 1:
            merged_pts = np.vstack([pts_b, pts_a[1:]])
        elif end_a == 1 and end_b == 1:
            merged_pts = np.vstack([pts_a, pts_b[-2::-1]])
        else:
            merged_pts = np.vstack([pts_a[::-1], pts_b[1:]])

        merged_channels.append(Constraint(pts=merged_pts, ctype=ConstraintType.CHANNEL))
        merged_flags[ci_a] = True
        merged_flags[ci_b] = True

    # Add unmerged channels
    for ci, c in enumerate(channels):
        if not merged_flags[ci]:
            merged_channels.append(c)

    return merged_channels + others


def _endpoint_direction(pts: NDArray[np.float64], end: int) -> NDArray[np.float64]:
    """Unit vector pointing away from endpoint *end* (0=start, 1=end)."""
    if end == 0:
        v = pts[0] - pts[min(1, len(pts) - 1)]
    else:
        v = pts[-1] - pts[max(-2, -len(pts))]
    norm = np.hypot(v[0], v[1])
    return v / (norm + 1e-12)


# ---------------------------------------------------------------------------
# Smoothing and curvature (Sect. 4.1)
# ---------------------------------------------------------------------------

def smooth_constraint(
    constraint: Constraint,
    rmse_desired: float,
    *,
    n_eval: int = 500,
) -> Constraint:
    """Fit a cubic spline to *constraint* with target smoothing RMSE.

    Implements Sect. 4.1 (Eqs. 31–32).  The smoothed coordinates are stored
    in ``constraint.smoothed_pts`` and curvature in ``constraint.curvature``.

    Parameters
    ----------
    rmse_desired : target root-mean-square deviation between original and
                   smoothed points (metres).  Typically 1–10 m.
    n_eval : number of evaluation points for the smoothed curve.
    """
    pts = constraint.pts
    if len(pts) < 4:
        constraint.smoothed_pts = pts
        constraint.curvature = np.zeros(len(pts))
        return constraint

    x, y = pts[:, 0], pts[:, 1]
    # Parametric arc-length parameter
    dists = np.hypot(np.diff(x), np.diff(y))
    t = np.concatenate([[0.0], np.cumsum(dists)])
    t /= t[-1]  # normalise to [0, 1]

    # Binary search for smoothing parameter p that gives rmse ≈ rmse_desired
    def _rmse(p: float) -> float:
        try:
            tck, _ = splprep([x, y], u=t, s=p * len(x), k=3, quiet=True)
        except Exception:
            return np.inf
        xs, ys = splev(t, tck)
        return float(np.sqrt(np.mean((xs - x)**2 + (ys - y)**2)))

    # Bracket: p=0 → no smoothing (rmse≈0); p=1 → heavy smoothing
    p_lo, p_hi = 0.0, 1.0
    for _ in range(40):
        p_mid = (p_lo + p_hi) / 2
        if _rmse(p_mid) < rmse_desired:
            p_lo = p_mid
        else:
            p_hi = p_mid
        if p_hi - p_lo < 1e-6:
            break
    p_best = (p_lo + p_hi) / 2

    try:
        tck, _ = splprep([x, y], u=t, s=p_best * len(x), k=3, quiet=True)
    except Exception:
        constraint.smoothed_pts = pts
        constraint.curvature = np.zeros(len(pts))
        return constraint

    t_eval = np.linspace(0, 1, n_eval)
    xs, ys = splev(t_eval, tck)
    constraint.smoothed_pts = np.column_stack([xs, ys])

    # Curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    dxdt, dydt = splev(t_eval, tck, der=1)
    d2xdt2, d2ydt2 = splev(t_eval, tck, der=2)
    denom = (dxdt**2 + dydt**2) ** 1.5
    kappa = np.abs(dxdt * d2ydt2 - dydt * d2xdt2) / (denom + 1e-12)
    constraint.curvature = kappa

    return constraint


def smooth_all_constraints(
    constraints: list[Constraint],
    rmse_desired: float,
) -> list[Constraint]:
    """Apply smooth_constraint to every constraint in the list."""
    return [smooth_constraint(c, rmse_desired) for c in constraints]
