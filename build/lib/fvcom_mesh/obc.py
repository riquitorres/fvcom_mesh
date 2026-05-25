"""
obc.py — Open Boundary Condition (OBC) node detection for FVCOM.

Auto-detects boundary edges on the domain polygon perimeter and
optionally accepts a user-supplied shapefile of OBC segments to override
or supplement the auto-detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from shapely.geometry import Polygon, MultiPolygon, LineString, Point
import geopandas as gpd

from .utils import boundary_edges


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _project_point_to_polyline(
    pt: NDArray[np.float64],
    polyline: NDArray[np.float64],
) -> float:
    """Return distance of *pt* from the nearest point on *polyline* (metres)."""
    ls = LineString(polyline)
    p = Point(pt)
    return float(ls.distance(p))


def _nodes_on_domain_boundary(
    pts: NDArray[np.float64],
    domain: Polygon | MultiPolygon,
    tol: float,
) -> NDArray[np.bool_]:
    """Return boolean mask of pts within *tol* of the domain boundary."""
    from shapely.geometry import Point
    bnd = domain.boundary
    on_bnd = np.array([bnd.distance(Point(p)) <= tol for p in pts])
    return on_bnd


# ---------------------------------------------------------------------------
# OBC detection
# ---------------------------------------------------------------------------

def detect_obc_nodes(
    mesh_pts: NDArray[np.float64],
    mesh_tris: NDArray[np.int_],
    domain: Polygon | MultiPolygon,
    tol: float,
    obc_shp: Optional[str | Path] = None,
    obc_geom=None,
    target_crs: Optional[str] = None,
    obc_type: int = 1,
) -> list[tuple[int, int, int]]:
    """Identify OBC nodes.

    Strategy
    --------
    1. Collect all mesh boundary edges (edges belonging to exactly one triangle).
    2. Collect all mesh boundary nodes (nodes that appear in any boundary edge).
    3. From boundary nodes, keep only those within *tol* of the domain boundary.
    4. If *obc_geom* is supplied (Shapely geometry), filter to nodes near it.
       Otherwise if *obc_shp* is supplied, load shapefile and filter similarly.
       If neither is given, all domain-boundary nodes become OBC nodes.

    Parameters
    ----------
    mesh_pts : (N, 2) array of node coordinates
    mesh_tris : (T, 3) connectivity array (0-based)
    domain : outer domain polygon (projected CRS)
    tol : distance threshold (metres) — nodes within *tol* of the domain
          boundary are candidates
    obc_shp : optional path to a shapefile of OBC segment polylines
    obc_geom : optional Shapely geometry (e.g. MultiLineString) of OBC arcs;
               takes precedence over *obc_shp* when provided
    target_crs : CRS for reprojecting *obc_shp* (required if obc_shp is given)
    obc_type : FVCOM OBC type code (default 1 — prescribed elevation)

    Returns
    -------
    obc_list : list of (obc_idx, node_idx_1based, obc_type) tuples sorted by
               obc_idx (1-based counter), ready for write_obc()
    """
    # --- Step 1: boundary edges ---
    bnd_edges = boundary_edges(mesh_tris)  # (M, 2), 0-based
    if len(bnd_edges) == 0:
        return []

    bnd_node_set = set(bnd_edges.ravel().tolist())
    bnd_node_idx = np.array(sorted(bnd_node_set), dtype=int)
    bnd_pts = mesh_pts[bnd_node_idx]

    # --- Step 2: filter by proximity to domain boundary ---
    on_dom_bnd = _nodes_on_domain_boundary(bnd_pts, domain, tol)
    candidate_idx = bnd_node_idx[on_dom_bnd]

    if len(candidate_idx) == 0:
        return []

    # --- Step 3: optional filter ---
    if obc_geom is not None:
        candidate_idx = _filter_by_obc_geometry(
            mesh_pts, candidate_idx, obc_geom, tol
        )
    elif obc_shp is not None:
        candidate_idx = _filter_by_obc_shapefile(
            mesh_pts, candidate_idx, obc_shp, target_crs, tol
        )

    if len(candidate_idx) == 0:
        return []

    # --- Step 4: order nodes along the domain boundary ---
    candidate_idx = _order_obc_nodes_along_boundary(
        mesh_pts, candidate_idx, domain
    )

    obc_list = [
        (i + 1, int(node_idx + 1), obc_type)  # 1-based
        for i, node_idx in enumerate(candidate_idx)
    ]
    return obc_list


def _filter_by_obc_geometry(
    mesh_pts: NDArray[np.float64],
    candidate_idx: NDArray[np.int_],
    obc_geom,
    tol: float,
) -> NDArray[np.int_]:
    """Keep only candidates within *tol* of a Shapely geometry."""
    keep = [
        idx for idx in candidate_idx
        if obc_geom.distance(Point(mesh_pts[idx])) <= tol
    ]
    return np.array(keep, dtype=int)


def _filter_by_obc_shapefile(
    mesh_pts: NDArray[np.float64],
    candidate_idx: NDArray[np.int_],
    obc_shp: str | Path,
    target_crs: Optional[str],
    tol: float,
) -> NDArray[np.int_]:
    """Keep only candidates within *tol* of an OBC shapefile polyline."""
    gdf = gpd.read_file(obc_shp)
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    from shapely.ops import unary_union
    obc_lines = unary_union(gdf.geometry.values)

    keep = []
    for idx in candidate_idx:
        p = Point(mesh_pts[idx])
        if obc_lines.distance(p) <= tol:
            keep.append(idx)
    return np.array(keep, dtype=int)


def _order_obc_nodes_along_boundary(
    pts: NDArray[np.float64],
    candidate_idx: NDArray[np.int_],
    domain: Polygon | MultiPolygon,
) -> NDArray[np.int_]:
    """Sort OBC node indices by their arc-length position along the domain boundary."""
    boundary = domain.boundary if isinstance(domain, Polygon) else domain.exterior
    positions = []
    for idx in candidate_idx:
        p = Point(pts[idx])
        pos = boundary.project(p)
        positions.append(pos)
    order = np.argsort(positions)
    return candidate_idx[order]


# ---------------------------------------------------------------------------
# Sponge layer nodes (optional)
# ---------------------------------------------------------------------------

def detect_sponge_nodes(
    obc_list: list[tuple[int, int, int]],
    mesh_pts: NDArray[np.float64],
    mesh_tris: NDArray[np.int_],
    sponge_radius: float,
    sponge_coeff: float = 0.01,
) -> list[tuple[int, int, float]]:
    """Build a sponge layer by selecting mesh nodes within *sponge_radius* of
    any OBC node.

    Parameters
    ----------
    obc_list : output of detect_obc_nodes
    mesh_pts : (N, 2) node coordinates
    mesh_tris : (T, 3) connectivity
    sponge_radius : radius of the sponge layer (metres)
    sponge_coeff : damping coefficient (written to _spg.dat)

    Returns
    -------
    spg_list : list of (spg_idx, node_idx_1based, coeff) tuples (1-based)
    """
    if not obc_list:
        return []

    obc_nodes_1based = np.array([row[1] for row in obc_list], dtype=int)
    obc_nodes_0based = obc_nodes_1based - 1
    obc_pts = mesh_pts[obc_nodes_0based]

    from scipy.spatial import KDTree
    tree = KDTree(obc_pts)
    dists, _ = tree.query(mesh_pts, workers=-1)

    in_sponge = np.where(dists <= sponge_radius)[0]
    # Exclude OBC nodes themselves
    obc_set = set(obc_nodes_0based.tolist())
    sponge_nodes = [idx for idx in in_sponge if idx not in obc_set]

    spg_list = [
        (i + 1, int(node_idx + 1), sponge_coeff)
        for i, node_idx in enumerate(sponge_nodes)
    ]
    return spg_list
