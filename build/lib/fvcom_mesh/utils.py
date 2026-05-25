"""
utils.py — Shared geometric and numerical utilities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, generic_gradient_magnitude
from shapely.geometry import MultiPolygon, Polygon, LineString, MultiLineString
from shapely.ops import unary_union
import shapely

try:
    from shapely import contains_xy as _contains_xy
except ImportError:
    from shapely.vectorized import contains as _contains_xy_v
    def _contains_xy(geom, x, y):  # type: ignore[misc]
        return _contains_xy_v(geom, x, y)



# ---------------------------------------------------------------------------
# Distance / signed-distance helpers
# ---------------------------------------------------------------------------

def signed_distance_raster(
    mask: NDArray[np.bool_],
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    """Signed distance function on a 2-D boolean mask.

    Returns positive inside the mask (water/domain) and negative outside,
    with units consistent with *dx* and *dy* (metres if the mask is in
    projected coordinates).

    Parameters
    ----------
    mask : 2-D bool array, True = inside region
    dx, dy : grid cell size in x and y directions
    """
    dist_in = distance_transform_edt(mask, sampling=(dy, dx))
    dist_out = distance_transform_edt(~mask, sampling=(dy, dx))
    return dist_in - dist_out


def distance_to_boundary(mask: NDArray[np.bool_], dx: float, dy: float) -> NDArray[np.float64]:
    """Euclidean distance from every cell to the nearest boundary of *mask*.

    Boundary cells are those where the mask changes value.  Returns positive
    distances everywhere (both inside and outside the mask).
    """
    dist_in = distance_transform_edt(mask, sampling=(dy, dx))
    dist_out = distance_transform_edt(~mask, sampling=(dy, dx))
    return dist_in + dist_out - 0.5 * (dx + dy)  # offset so boundary ≈ 0


def vector_distance_transform(
    mask: NDArray[np.bool_],
    dx: float,
    dy: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Vector distance transform (VDT) of a binary mask.

    Returns (dist, vx, vy) where *dist* is the Euclidean distance to the
    nearest boundary pixel, and (*vx*, *vy*) is the unit vector pointing
    from each pixel toward its nearest boundary point.

    This implements the VDT approach used to compute the medial axis
    divergence in Kang & Kubatko (2024), Sect. 3.2.1 and Appendix A.
    """
    # scipy gives nearest-feature indices via return_indices
    dist, (iy, ix) = distance_transform_edt(
        mask, sampling=(dy, dx), return_indices=True
    )
    rows, cols = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    # Vector from pixel (row,col) to its nearest boundary pixel
    vy_raw = (iy - rows).astype(np.float64) * dy
    vx_raw = (ix - cols).astype(np.float64) * dx
    # Normalise (distance may be zero on the boundary itself)
    eps = 1e-12
    norm = np.sqrt(vx_raw**2 + vy_raw**2) + eps
    vx = vx_raw / norm
    vy = vy_raw / norm
    return dist, vx, vy


def divergence_of_vdt(
    vx: NDArray[np.float64],
    vy: NDArray[np.float64],
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    """Numerical divergence of the vector field (*vx*, *vy*) using
    central differences.

    Positive values on the medial axis; -1 to -2 elsewhere (see Appendix A
    of Kang & Kubatko 2024).
    """
    dvx_dx = np.gradient(vx, dx, axis=1)
    dvy_dy = np.gradient(vy, dy, axis=0)
    return dvx_dx + dvy_dy


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rasterize_polygon(
    polygon: Polygon | MultiPolygon,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Rasterize a Shapely polygon onto a regular grid.

    Parameters
    ----------
    polygon : shapely Polygon or MultiPolygon
    x : 1-D array of x coordinates (columns), length M
    y : 1-D array of y coordinates (rows), length N, *increasing* order

    Returns
    -------
    mask : (N, M) bool array, True = inside polygon
    """
    xx, yy = np.meshgrid(x, y)
    return _contains_xy(polygon, xx.ravel(), yy.ravel()).reshape(xx.shape)


def polygon_boundary_polylines(polygon: Polygon | MultiPolygon) -> list[NDArray[np.float64]]:
    """Return exterior and interior rings of *polygon* as arrays of shape (N, 2)."""
    polys = polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]
    lines = []
    for poly in polys:
        lines.append(np.array(poly.exterior.coords))
        for interior in poly.interiors:
            lines.append(np.array(interior.coords))
    return lines


def polyline_length(pts: NDArray[np.float64]) -> float:
    """Total arc length of an ordered (N, 2) point array."""
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def resample_polyline(
    pts: NDArray[np.float64],
    target_spacing: float | NDArray[np.float64],
) -> NDArray[np.float64]:
    """Uniformly resample a polyline to approximately *target_spacing* between points.

    If *target_spacing* is a scalar, uniform spacing is used.  A 1-D array
    of the same length as the parameterisation can also be supplied (used
    for curvature-based variable spacing).
    """
    diffs = np.diff(pts, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total == 0:
        return pts[:1]
    if np.isscalar(target_spacing):
        n = max(2, int(np.ceil(total / target_spacing)) + 1)
        t_new = np.linspace(0.0, total, n)
    else:
        t_new = np.concatenate([[0.0], np.cumsum(np.asarray(target_spacing))])
        t_new = t_new[t_new <= total]
        if t_new[-1] < total:
            t_new = np.append(t_new, total)
    x_new = np.interp(t_new, cum, pts[:, 0])
    y_new = np.interp(t_new, cum, pts[:, 1])
    return np.column_stack([x_new, y_new])


def points_in_polygon(
    pts: NDArray[np.float64],
    polygon: Polygon | MultiPolygon,
) -> NDArray[np.bool_]:
    """Boolean mask: which rows of *pts* (shape N×2) lie inside *polygon*."""
    from shapely.vectorized import contains
    return contains(polygon, pts[:, 0], pts[:, 1])


def project_to_polyline(
    pts: NDArray[np.float64],
    polyline: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project each point in *pts* (N×2) to the nearest point on *polyline* (M×2).

    Returns projected coordinates, shape (N, 2).
    """
    ls = LineString(polyline)
    result = np.empty_like(pts)
    for i, p in enumerate(pts):
        proj = ls.interpolate(ls.project(shapely.geometry.Point(p)))
        result[i] = [proj.x, proj.y]
    return result


# ---------------------------------------------------------------------------
# Connectivity / mesh helpers
# ---------------------------------------------------------------------------

def delaunay_triangulation(pts: NDArray[np.float64]):
    """Return scipy Delaunay triangulation of (N, 2) points."""
    from scipy.spatial import Delaunay
    return Delaunay(pts)


def triangles_in_domain(
    triangles: NDArray[np.int_],
    pts: NDArray[np.float64],
    domain: Polygon | MultiPolygon,
) -> NDArray[np.bool_]:
    """Boolean mask selecting triangles whose centroid lies inside *domain*."""
    centroids = pts[triangles].mean(axis=1)  # (M, 2)
    inside = _contains_xy(domain, centroids[:, 0], centroids[:, 1])
    return triangles[inside]


def triangle_edge_lengths(
    triangles: NDArray[np.int_],
    pts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the three edge lengths for each triangle, shape (M, 3)."""
    p0 = pts[triangles[:, 0]]
    p1 = pts[triangles[:, 1]]
    p2 = pts[triangles[:, 2]]
    l01 = np.hypot(p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1])
    l12 = np.hypot(p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1])
    l20 = np.hypot(p0[:, 0] - p2[:, 0], p0[:, 1] - p2[:, 1])
    return np.column_stack([l01, l12, l20])


def ensure_ccw(
    triangles: NDArray[np.int_],
    pts: NDArray[np.float64],
) -> NDArray[np.int_]:
    """Return *triangles* (M×3) with all elements oriented counter-clockwise."""
    p0 = pts[triangles[:, 0]]
    p1 = pts[triangles[:, 1]]
    p2 = pts[triangles[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - \
            (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    cw = cross < 0
    triangles = triangles.copy()
    triangles[cw, 1], triangles[cw, 2] = triangles[cw, 2].copy(), triangles[cw, 1].copy()
    return triangles


def boundary_edges(
    triangles: NDArray[np.int_],
) -> NDArray[np.int_]:
    """Return boundary edge pairs (N, 2) — edges shared by exactly one triangle."""
    edges = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(a, b), max(a, b))
            edges[key] = edges.get(key, 0) + 1
    return np.array([list(k) for k, v in edges.items() if v == 1], dtype=np.int_)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def make_background_grid(
    bounds: tuple[float, float, float, float],
    resolution: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return 1-D x and y coordinate arrays for a uniform background grid.

    Parameters
    ----------
    bounds : (xmin, ymin, xmax, ymax)
    resolution : grid cell size in metres
    """
    xmin, ymin, xmax, ymax = bounds
    x = np.arange(xmin, xmax + resolution, resolution)
    y = np.arange(ymin, ymax + resolution, resolution)
    return x, y
