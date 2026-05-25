"""
mesh_gen_2d.py — 2-D DistMesh force-equilibrium mesh generation.

Implements Sect. 5.2 of Kang & Kubatko (2024), GMD, based on the
DistMesh algorithm (Persson & Strang 2004).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, KDTree

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from .utils import (
    ensure_ccw,
    boundary_edges,
    triangles_in_domain,
    triangle_edge_lengths,
)


# ---------------------------------------------------------------------------
# Rejection-method initial node placement (Sect. 5.2.1)
# ---------------------------------------------------------------------------

def initial_nodes(
    domain: Polygon | MultiPolygon,
    h_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    h_min: float,
    fixed_pts: NDArray[np.float64],
    *,
    rejection_oversample: float = 1.3,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate an initial node set inside *domain* using the rejection method.

    Nodes are drawn uniformly from the bounding box and accepted with
    probability proportional to (h_min / h(x))^2, following Persson & Strang
    (2004) Sect. 2.

    Parameters
    ----------
    domain : domain polygon (projected CRS)
    h_func : mesh-size function
    h_min : minimum mesh size (metres)
    fixed_pts : (M, 2) constraint nodes that are always included
    rejection_oversample : oversampling factor to compensate for rejections
    seed : random seed for reproducibility

    Returns
    -------
    (N, 2) array of initial node positions (includes fixed_pts)
    """
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = domain.bounds
    bbox_area = (xmax - xmin) * (ymax - ymin)

    # Expected number of nodes from h_min (uniform lower bound)
    n_target = int(rejection_oversample * bbox_area / (h_min ** 2 * np.sqrt(3) / 2))
    n_target = max(n_target, 10)

    candidates = rng.uniform(
        [xmin, ymin], [xmax, ymax], (n_target, 2)
    )

    # Reject nodes outside domain
    from fvcom_mesh.utils import _contains_xy
    in_domain = _contains_xy(domain, candidates[:, 0], candidates[:, 1])
    candidates = candidates[in_domain]

    if len(candidates) == 0:
        return fixed_pts.copy() if len(fixed_pts) > 0 else np.empty((0, 2))

    # Rejection probability proportional to (h_min / h(x))^2
    h_vals = np.maximum(h_func(candidates), h_min)
    prob = (h_min / h_vals) ** 2
    r = rng.uniform(0, 1, len(candidates))
    accepted = candidates[r < prob]

    # Combine with fixed nodes (deduplicate)
    if len(fixed_pts) > 0:
        all_pts = np.vstack([fixed_pts, accepted])
    else:
        all_pts = accepted

    return all_pts


# ---------------------------------------------------------------------------
# Signed distance from polygon for DistMesh movement clamping
# ---------------------------------------------------------------------------

def _signed_distance(
    pts: NDArray[np.float64],
    domain: Polygon | MultiPolygon,
) -> NDArray[np.float64]:
    """Signed distance: negative inside, positive outside."""
    from shapely.geometry import Point
    from fvcom_mesh.utils import _contains_xy

    in_dom = _contains_xy(domain, pts[:, 0], pts[:, 1])
    dists = np.array([domain.boundary.distance(Point(p)) for p in pts])
    d = np.where(in_dom, -dists, dists)
    return d


# ---------------------------------------------------------------------------
# DistMesh 2-D
# ---------------------------------------------------------------------------

_FSCALE_START = 1.2
_FSCALE_END = 1.0


def distmesh_2d(
    domain: Polygon | MultiPolygon,
    h_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    fixed_pts: NDArray[np.float64],
    h_min: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-4,
    dt: float = 0.2,
    density_control_start: float = 0.8,
    seed: int = 42,
    report_every: int = 25,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """2-D DistMesh node-placement and triangulation.

    Implements the DistMesh algorithm adapted for ADMESH+ (Sect. 5.2).

    Parameters
    ----------
    domain : Shapely polygon (projected CRS)
    h_func : mesh-size function (callable accepting (N,2) → (N,))
    fixed_pts : (M, 2) fixed boundary / constraint nodes
    h_min : minimum mesh size (metres)
    max_iter : total iterations
    tol : movement convergence threshold (metres)
    dt : relaxation step
    density_control_start : fraction of max_iter at which to start
                             density control (adding/removing nodes)
    seed : random seed

    Returns
    -------
    pts : (N, 2) final node positions
    triangles : (T, 3) element connectivity (CCW, 0-based)
    """
    import sys as _sys
    n_fixed = len(fixed_pts)

    # Initial node placement
    pts = initial_nodes(domain, h_func, h_min, fixed_pts, seed=seed)
    if len(pts) == 0:
        raise RuntimeError("No initial nodes generated — check domain/h_min.")
    print(f"  [DistMesh]  initial nodes: {len(pts)}  (fixed: {n_fixed})",
          file=_sys.stderr, flush=True)

    density_iter = int(density_control_start * max_iter)

    for it in range(max_iter):
        # --- Triangulate ---
        if len(pts) < 3:
            break
        tri = Delaunay(pts)
        triangles = tri.simplices.copy()

        # Keep only triangles whose centroid is inside the domain
        triangles = triangles_in_domain(triangles, pts, domain)
        triangles = np.asarray(triangles)
        if triangles.ndim != 2 or len(triangles) == 0:
            break

        # --- Force scale: decrease linearly from FSCALE_START to FSCALE_END ---
        fscale = _FSCALE_START - (_FSCALE_START - _FSCALE_END) * it / max_iter

        # --- Build edge list from triangles ---
        edges = np.vstack([
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ])
        # Keep unique undirected edges
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        # --- Midpoints and desired lengths ---
        mid = 0.5 * (pts[edges[:, 0]] + pts[edges[:, 1]])
        h_mid = np.maximum(h_func(mid), h_min)
        current_len = np.hypot(
            pts[edges[:, 0], 0] - pts[edges[:, 1], 0],
            pts[edges[:, 0], 1] - pts[edges[:, 1], 1],
        )

        # Scale desired length so average matches h_mid (density control)
        L0 = fscale * h_mid
        F = np.maximum(L0 - current_len, 0.0) / (current_len + 1e-12)

        # Direction vectors (from node 1 → node 0)
        diff = pts[edges[:, 0]] - pts[edges[:, 1]]
        unit = diff / (current_len[:, np.newaxis] + 1e-12)
        force_vecs = F[:, np.newaxis] * unit

        # Accumulate forces on all nodes
        dp = np.zeros_like(pts)
        np.add.at(dp, edges[:, 0], force_vecs)
        np.add.at(dp, edges[:, 1], -force_vecs)

        # Fix constraint nodes
        dp[:n_fixed] = 0.0

        # --- Move nodes ---
        pts = pts + dt * dp

        # --- Project nodes that moved outside domain back to boundary ---
        d = _signed_distance(pts, domain)
        outside = d > 0
        if outside[n_fixed:].any():
            out_idx = np.where(outside)[0]
            for idx in out_idx:
                if idx < n_fixed:
                    continue
                # Project onto boundary
                from shapely.geometry import Point
                near_pt = domain.boundary.interpolate(
                    domain.boundary.project(Point(pts[idx]))
                )
                pts[idx] = [near_pt.x, near_pt.y]

        # --- Density control: remove nodes in over-dense regions ---
        if it >= density_iter and it % 10 == 0:
            pts, n_fixed_new = _density_control(
                pts, n_fixed, h_func, h_min, domain
            )
            if n_fixed_new != n_fixed:
                n_fixed = n_fixed_new

        # --- Convergence check ---
        max_move = np.max(np.abs(dp[n_fixed:])) if len(dp) > n_fixed else 0.0
        if report_every > 0 and (it + 1) % report_every == 0:
            print(f"  [DistMesh]  iter {it+1:4d}/{max_iter}  nodes={len(pts)}  "
                  f"max_move={max_move:.2f} m",
                  file=_sys.stderr, flush=True)
        if max_move < tol:
            print(f"  [DistMesh]  converged at iter {it+1}  max_move={max_move:.4f} m",
                  file=_sys.stderr, flush=True)
            break

    # --- Final triangulation ---
    tri = Delaunay(pts)
    triangles = tri.simplices.copy()
    triangles = triangles_in_domain(triangles, pts, domain)
    triangles = np.asarray(triangles)
    if triangles.ndim != 2 or len(triangles) == 0:
        raise RuntimeError("Final triangulation produced no elements inside domain.")
    triangles = ensure_ccw(triangles, pts)

    return pts, triangles


def _density_control(
    pts: NDArray[np.float64],
    n_fixed: int,
    h_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    h_min: float,
    domain: Polygon | MultiPolygon,
) -> tuple[NDArray[np.float64], int]:
    """Remove nodes where the local density is too high.

    For each pair of moveable nodes closer than h_min / 2, remove one.
    """
    if len(pts) <= n_fixed:
        return pts, n_fixed

    moveable = pts[n_fixed:]
    if len(moveable) < 2:
        return pts, n_fixed

    tree = KDTree(moveable)
    pairs = tree.query_pairs(h_min / 2)
    to_remove: set[int] = set()
    for i, j in sorted(pairs):
        if j not in to_remove:
            to_remove.add(j)

    if to_remove:
        keep_moveable = np.array(
            [i for i in range(len(moveable)) if i not in to_remove]
        )
        if len(keep_moveable) > 0:
            moveable = moveable[keep_moveable]
        else:
            moveable = np.empty((0, 2))
        pts = np.vstack([pts[:n_fixed], moveable]) if len(moveable) > 0 else pts[:n_fixed]

    return pts, n_fixed
