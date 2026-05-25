"""
mesh_gen_1d.py — 1-D force-equilibrium node placement along constraint polylines.

Implements Sect. 5.1 of Kang & Kubatko (2024), GMD.

The 1-D generation places nodes along each constraint polyline so that the
distance between adjacent nodes matches the local desired mesh size h(s).
This is done by a spring-force iteration analogous to DistMesh but in 1-D.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .constraints import Constraint, ConstraintType
from .utils import polyline_length, resample_polyline


# ---------------------------------------------------------------------------
# Core 1-D force equilibrium
# ---------------------------------------------------------------------------

def _arc_length_param(pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cumulative arc-length parameter (0 … total_length)."""
    dists = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    return np.concatenate([[0.0], np.cumsum(dists)])


def _project_on_polyline(
    s_nodes: NDArray[np.float64],
    s_poly: NDArray[np.float64],
    pts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate (x, y) on polyline at arc-length positions *s_nodes*."""
    x = np.interp(s_nodes, s_poly, pts[:, 0])
    y = np.interp(s_nodes, s_poly, pts[:, 1])
    return np.column_stack([x, y])


def force_equilibrium_1d(
    constraint: Constraint,
    h_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    h_min: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-4,
    dt: float = 0.2,
) -> NDArray[np.float64]:
    """Place nodes along *constraint* using 1-D spring-force equilibrium.

    Starting from an initial uniform placement, each node is displaced
    towards a position where adjacent spring lengths equal the local desired
    mesh size.  Nodes at the two endpoints are fixed.

    Parameters
    ----------
    constraint : Constraint — uses smoothed_pts if available, else pts
    h_func : callable accepting (N, 2) array → (N,) desired mesh size
    h_min : minimum mesh spacing (metres) — prevents zero-length segments
    max_iter : maximum spring iterations
    tol : convergence criterion — max movement (metres) per iteration
    dt : step size (relaxation factor, 0 < dt ≤ 1)

    Returns
    -------
    (N, 2) array of node positions along the constraint
    """
    pts = constraint.smoothed_pts if constraint.smoothed_pts is not None else constraint.pts
    if len(pts) < 2:
        return pts

    s_poly = _arc_length_param(pts)
    L_total = s_poly[-1]

    if L_total < h_min:
        return pts[[0, -1]]  # too short — just endpoints

    # Initial uniform spacing using h at midpoint
    mid = _project_on_polyline(np.array([L_total / 2]), s_poly, pts)
    h_mid = float(h_func(mid)[0])
    n_init = max(2, int(np.round(L_total / h_mid)))
    s_nodes = np.linspace(0.0, L_total, n_init)

    for _ in range(max_iter):
        xy = _project_on_polyline(s_nodes, s_poly, pts)
        h_vals = np.maximum(h_func(xy), h_min)

        # Spring lengths = desired size at midpoint between adjacent nodes
        h_mid_spring = 0.5 * (h_vals[:-1] + h_vals[1:])
        current_gaps = np.diff(s_nodes)  # always positive (sorted)
        # Force on each internal node: push from neighbours
        # F_i = (h_mid_spring[i-1] - current_gaps[i-1])
        #      - (h_mid_spring[i]   - current_gaps[i])
        forces = np.zeros_like(s_nodes)
        forces[1:-1] = (
            (h_mid_spring[:-1] - current_gaps[:-1])
            - (h_mid_spring[1:] - current_gaps[1:])
        )
        # Fix endpoints
        forces[0] = 0.0
        forces[-1] = 0.0

        delta = dt * forces
        s_nodes += delta

        # Re-sort and clamp (nodes must stay in [0, L_total])
        s_nodes = np.clip(np.sort(s_nodes), 0.0, L_total)
        s_nodes[0] = 0.0
        s_nodes[-1] = L_total

        if np.max(np.abs(delta)) < tol:
            break

    # Resample to match converged spacing
    xy_final = _project_on_polyline(s_nodes, s_poly, pts)
    return xy_final


# ---------------------------------------------------------------------------
# Post-processing (Sect. 5.1.2)
# ---------------------------------------------------------------------------

def _merge_clusters(
    pts_all: NDArray[np.float64],
    h_min: float,
) -> NDArray[np.float64]:
    """Merge clusters of nodes that are closer than h_min/2."""
    if len(pts_all) == 0:
        return pts_all
    keep = [pts_all[0]]
    for p in pts_all[1:]:
        if np.hypot(p[0] - keep[-1][0], p[1] - keep[-1][1]) >= h_min / 2:
            keep.append(p)
    return np.array(keep)


def _remove_short_elements(
    nodes: NDArray[np.float64],
    h_min: float,
    threshold: float = 0.5,
) -> NDArray[np.float64]:
    """Remove nodes that create element lengths shorter than threshold*h_min."""
    if len(nodes) <= 2:
        return nodes
    keep = [True] * len(nodes)
    for i in range(1, len(nodes) - 1):
        if not keep[i]:
            continue
        d_prev = np.hypot(nodes[i, 0] - nodes[i - 1, 0], nodes[i, 1] - nodes[i - 1, 1])
        d_next = np.hypot(nodes[i, 0] - nodes[i + 1, 0], nodes[i, 1] - nodes[i + 1, 1])
        if min(d_prev, d_next) < threshold * h_min:
            keep[i] = False
    return nodes[keep]


def _remove_near_type3(
    nodes: NDArray[np.float64],
    type3_pts: NDArray[np.float64],
    h_min: float,
    proximity_factor: float = 0.5,
) -> NDArray[np.float64]:
    """Remove internal nodes that are too close to Type-3 (shoreline) nodes."""
    if len(type3_pts) == 0 or len(nodes) <= 2:
        return nodes
    thresh = proximity_factor * h_min
    keep = np.ones(len(nodes), dtype=bool)
    # Keep endpoints
    keep[0] = True
    keep[-1] = True
    for i in range(1, len(nodes) - 1):
        dists = np.hypot(
            nodes[i, 0] - type3_pts[:, 0],
            nodes[i, 1] - type3_pts[:, 1],
        )
        if dists.min() < thresh:
            keep[i] = False
    return nodes[keep]


def postprocess_1d(
    nodes_per_constraint: list[NDArray[np.float64]],
    h_min: float,
    type3_nodes: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    """Post-process all 1-D node sets.

    Applies cluster merging, short-element removal and proximity-to-Type3
    removal in sequence.

    Parameters
    ----------
    nodes_per_constraint : list of (N_i, 2) node arrays (one per constraint)
    h_min : minimum mesh size (metres)
    type3_nodes : (M, 2) array of all Type-3 (shoreline) fixed nodes

    Returns
    -------
    cleaned list of node arrays
    """
    result = []
    for nodes in nodes_per_constraint:
        nodes = _merge_clusters(nodes, h_min)
        nodes = _remove_short_elements(nodes, h_min)
        nodes = _remove_near_type3(nodes, type3_nodes, h_min)
        result.append(nodes)
    return result


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def generate_1d_nodes(
    constraints: list[Constraint],
    h_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    h_min: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-4,
    dt: float = 0.2,
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Generate 1-D nodes along all constraints.

    Parameters
    ----------
    constraints : list of Constraint (all types)
    h_func : mesh-size function callable
    h_min : minimum mesh size

    Returns
    -------
    nodes_per_constraint : list of (N_i, 2) arrays
    all_fixed_nodes : (M, 2) array of all unique node positions
    """
    nodes_per_constraint: list[NDArray[np.float64]] = []
    for c in constraints:
        nodes = force_equilibrium_1d(c, h_func, h_min, max_iter=max_iter, tol=tol, dt=dt)
        nodes_per_constraint.append(nodes)

    # Extract Type-3 (shoreline) nodes for proximity check
    type3_nodes_list = [
        nodes
        for c, nodes in zip(constraints, nodes_per_constraint)
        if c.ctype == ConstraintType.LAND_WATER
    ]
    type3_nodes = (
        np.vstack(type3_nodes_list)
        if type3_nodes_list
        else np.empty((0, 2), dtype=float)
    )

    nodes_per_constraint = postprocess_1d(nodes_per_constraint, h_min, type3_nodes)

    # All fixed nodes (unique within h_min/4)
    if any(len(n) > 0 for n in nodes_per_constraint):
        all_nodes = np.vstack([n for n in nodes_per_constraint if len(n) > 0])
        all_fixed_nodes = _deduplicate_nodes(all_nodes, h_min / 4)
    else:
        all_fixed_nodes = np.empty((0, 2), dtype=float)

    return nodes_per_constraint, all_fixed_nodes


def _deduplicate_nodes(
    pts: NDArray[np.float64],
    tol: float,
) -> NDArray[np.float64]:
    """Remove duplicate points closer than *tol*."""
    if len(pts) == 0:
        return pts
    from scipy.spatial import KDTree
    tree = KDTree(pts)
    pairs = tree.query_pairs(tol)
    # Keep the first of each pair
    remove = set()
    for i, j in pairs:
        if j not in remove:
            remove.add(j)
    keep = [i for i in range(len(pts)) if i not in remove]
    return pts[keep]
