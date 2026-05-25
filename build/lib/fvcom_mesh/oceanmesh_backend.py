"""
oceanmesh_backend.py — OceanMesh-based triangulation for fvcom_mesh.

Provides a drop-in replacement for distmesh_2d that uses the OceanMesh
library's C++ Delaunay triangulator (lower memory than DistMesh at fine
resolutions with h_min ≤ 200 m).

Key function
------------
generate_mesh_oceanmesh(water_polygon, h_func, h_min, h_max, fixed_pts,
                         max_iter) -> (pts, triangles)
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon, MultiPolygon

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signed distance function from Shapely polygon
# ---------------------------------------------------------------------------

def _make_sdf(water_polygon: Polygon | MultiPolygon):
    """Return a signed-distance callable from *water_polygon*.

    Convention (OceanMesh standard):
      * **negative** → inside the ocean domain (to be meshed)
      * **positive** → outside (land / beyond boundary)

    Evaluation is vectorised via Shapely 2.x bulk operations.
    """
    import shapely

    # Pre-extract the combined boundary geometry once (exterior + all holes).
    boundary = water_polygon.boundary  # LinearRing or MultiLineString
    prepared = shapely.prepare(water_polygon)  # for fast containment tests

    def sdf(pts: NDArray[np.float64]) -> NDArray[np.float64]:
        pts = np.asarray(pts, dtype=float)
        geom_pts = shapely.points(pts[:, 0], pts[:, 1])

        # Distance to the nearest boundary ring (0 if on boundary).
        dist = shapely.distance(geom_pts, boundary)

        # Containment: True where points are inside the water area.
        inside = shapely.contains_xy(water_polygon, pts[:, 0], pts[:, 1])

        return np.where(inside, -dist, dist)

    return sdf


# ---------------------------------------------------------------------------
# OceanMesh Domain wrapper
# ---------------------------------------------------------------------------

def _build_domain(water_polygon: Polygon | MultiPolygon):
    """Build an ``oceanmesh.signed_distance_function.Domain`` from a Shapely polygon."""
    from oceanmesh.signed_distance_function import Domain

    bounds = water_polygon.bounds  # (minx, miny, maxx, maxy)
    # OceanMesh bbox convention: (xmin, xmax, ymin, ymax)
    bbox = (bounds[0], bounds[2], bounds[1], bounds[3])
    sdf = _make_sdf(water_polygon)
    return Domain(bbox, sdf)


# ---------------------------------------------------------------------------
# OceanMesh Grid wrapper
# ---------------------------------------------------------------------------

def _build_sizing_grid(
    water_polygon: Polygon | MultiPolygon,
    h_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    h_min: float,
    h_max: float,
    grid_resolution: Optional[float] = None,
):
    """Build an ``oceanmesh.grid.Grid`` sizing function from *h_func*.

    The sizing function is sampled on a regular grid and stored as a
    ``RegularGridInterpolator`` so OceanMesh can query it efficiently
    during mesh generation.

    Parameters
    ----------
    water_polygon : domain polygon (projected metres)
    h_func : callable – takes (N, 2) projected pts → (N,) sizes in metres
    h_min, h_max : clamp values
    grid_resolution : spacing for the background evaluation grid; defaults to
                      ``max(h_min / 2, (domain_span) / 1000)`` to keep memory
                      manageable for large domains with fine resolution.
    """
    from oceanmesh.grid import Grid

    bounds = water_polygon.bounds  # (minx, miny, maxx, maxy)
    xmin, ymin, xmax, ymax = bounds

    # Choose grid resolution to cap the grid at ~1 M cells
    domain_span = max(xmax - xmin, ymax - ymin)
    if grid_resolution is None:
        grid_resolution = max(h_min / 2.0, domain_span / 1000.0)

    dx = dy = grid_resolution

    # OceanMesh bbox: (xmin, xmax, ymin, ymax)
    bbox = (xmin, xmax, ymin, ymax)

    nx = int((xmax - xmin) / dx) + 2
    ny = int((ymax - ymin) / dy) + 2

    log.info("Building OceanMesh sizing grid: %d × %d = %d cells", nx, ny, nx * ny)

    # Create coordinate vectors (x increases along first axis, y along second)
    x_vec = xmin + np.arange(nx) * dx
    y_vec = ymin + np.arange(ny) * dy

    # Evaluate h_func on the full grid
    xx, yy = np.meshgrid(x_vec, y_vec, indexing="ij")  # (nx, ny)
    pts_flat = np.column_stack([xx.ravel(), yy.ravel()])
    h_flat = h_func(pts_flat)
    h_flat = np.clip(h_flat, h_min, h_max)
    h_values = h_flat.reshape(nx, ny)

    grid = Grid(bbox=bbox, dx=dx, dy=dy, hmin=float(h_min), values=h_values)
    grid.build_interpolant()
    return grid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_mesh_oceanmesh(
    water_polygon: Polygon | MultiPolygon,
    h_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    h_min: float,
    h_max: float,
    fixed_pts: Optional[NDArray[np.float64]] = None,
    max_iter: int = 100,
    grid_resolution: Optional[float] = None,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Generate a triangulated mesh using the OceanMesh Delaunay backend.

    This is a memory-efficient replacement for ``distmesh_2d`` intended for
    fine-resolution meshes (h_min ≤ 200 m) where the DistMesh force-iteration
    approach runs out of memory.

    Parameters
    ----------
    water_polygon : Shapely Polygon (or MultiPolygon)
        Ocean domain to mesh; holes represent islands/land masses.
    h_func : callable
        Mesh-size function ``h_func(pts) → sizes``.  *pts* is (N, 2) in
        projected metres; return value is (N,) in metres.
    h_min : float
        Minimum element size [m].
    h_max : float
        Maximum element size [m].
    fixed_pts : array-like, optional
        (M, 2) array of constraint nodes to lock in the mesh.
    max_iter : int
        Maximum number of Delaunay force iterations (default 100).
    grid_resolution : float, optional
        Background grid cell size for the sizing function.  Auto-selected if
        not specified.

    Returns
    -------
    pts : (N, 2) ndarray – node coordinates in projected metres
    triangles : (T, 3) ndarray – element connectivity (0-based, CCW)
    """
    try:
        from oceanmesh import generate_mesh
    except ImportError as e:
        raise ImportError(
            "OceanMesh is required for the 'oceanmesh' backend.  "
            "Install it with: OCEANMESH_PREFIX=<prefix> pip install oceanmesh"
        ) from e

    log.info("OceanMesh backend: building domain SDF")
    domain = _build_domain(water_polygon)

    log.info("OceanMesh backend: building sizing grid (h_min=%.1f, h_max=%.1f)", h_min, h_max)
    sizing_grid = _build_sizing_grid(
        water_polygon, h_func, h_min, h_max, grid_resolution=grid_resolution
    )

    kwargs: dict = {
        "max_iter": int(max_iter),
        "min_edge_length": float(h_min),
        # Lock boundary nodes so island/coast edges don't drift during the
        # force iteration — critical for domains with many island holes.
        "lock_boundary": True,
    }
    if fixed_pts is not None and len(fixed_pts) > 0:
        kwargs["pfix"] = np.asarray(fixed_pts, dtype=float)

    # Estimate initial node count so the user can see the problem scale
    import sys as _sys
    xmin, ymin, xmax, ymax = water_polygon.bounds
    _est_grid = int((xmax - xmin) / h_min) * int((ymax - ymin) / h_min)
    _water_frac = water_polygon.area / ((xmax - xmin) * (ymax - ymin))
    _est_nodes = int(_est_grid * _water_frac)
    print(f"  [OceanMesh] bbox {(xmax-xmin)/1e3:.0f} km × {(ymax-ymin)/1e3:.0f} km"
          f"  h_min={h_min:.0f} m  est. nodes≈{_est_nodes:,}  max_iter={max_iter}",
          file=_sys.stderr, flush=True)

    log.info("OceanMesh backend: starting mesh generation (max_iter=%d)", max_iter)
    pts, triangles = generate_mesh(domain, sizing_grid, **kwargs)
    print(f"  [OceanMesh] finished: {len(pts):,} nodes  {len(triangles):,} elements",
          file=_sys.stderr, flush=True)

    pts = np.asarray(pts, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.intp)

    log.info(
        "OceanMesh backend: finished — %d nodes, %d elements",
        len(pts),
        len(triangles),
    )
    return pts, triangles
