"""
mesh_modify.py — Generate a modified mesh from an existing FVCOM grid.

Extracts domain geometry and open-boundary information directly from
FVCOM ``_grd.dat`` / ``_dep.dat`` / ``_obc.dat`` files so no shapefiles
are required.  The extracted geometry is passed to the normal ADMESH+
pipeline to produce a mesh with different resolution (typically coarser).

Typical usage
-------------
>>> from fvcom_mesh.mesh_modify import coarsen_fvcom_mesh
>>> mesh = coarsen_fvcom_mesh(
...     grd_path="maldives_v0_grd.dat",
...     dep_path="maldives_v0_dep.dat",
...     obc_path="maldives_v0_obc.dat",
...     output_prefix="maldives_v0_coarse",
...     h_min=500.0,
...     h_max=10000.0,
...     delta_w=1000.0,
...     projection="EPSG:32643",
... )

Or use the CLI::

    fvcom-mesh coarsen maldives_v0_grd.dat       \\
        --dep  maldives_v0_dep.dat               \\
        --obc  maldives_v0_obc.dat               \\
        --projection EPSG:32643                  \\
        --h-min 500 --h-max 10000 --delta-w 1000 \\
        --output-prefix maldives_v0_coarse
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import (
    LinearRing, LineString, MultiLineString, MultiPolygon, Point, Polygon,
)


# ---------------------------------------------------------------------------
# Boundary-loop extraction
# ---------------------------------------------------------------------------

def extract_boundary_loops(
    pts: NDArray[np.float64],
    triangles: NDArray[np.int_],
) -> list[NDArray[np.int_]]:
    """Extract ordered boundary node loops from a triangular mesh.

    Uses the undirected boundary-edge adjacency so that it works even
    when the triangle orientation is not globally consistent.

    Parameters
    ----------
    pts : (N, 2) node coordinates
    triangles : (T, 3) element connectivity (0-based)

    Returns
    -------
    loops : list of 1-D integer arrays (node indices, 0-based), each
            forming one closed boundary loop.  Sorted by enclosed area,
            **largest first** (the outermost loop is always ``loops[0]``).
    """
    # Step 1: count how many triangles share each undirected edge
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for tri in triangles:
        for i in range(3):
            e = (int(tri[i]), int(tri[(i + 1) % 3]))
            key = (min(e), max(e))
            edge_count[key] += 1

    # Step 2: build undirected adjacency for boundary nodes
    adj: dict[int, list[int]] = defaultdict(list)
    for (a, b), cnt in edge_count.items():
        if cnt == 1:
            adj[a].append(b)
            adj[b].append(a)

    if not adj:
        return []

    # Step 3: trace closed loops
    visited: set[int] = set()
    loops: list[NDArray[np.int_]] = []

    for start in sorted(adj.keys()):
        if start in visited:
            continue

        loop: list[int] = [start]
        visited.add(start)
        prev = start   # prevents immediately stepping back to start
        cur = adj[start][0]  # arbitrary first step

        while cur != start:
            if cur in visited:
                break  # mesh topology error — abort this loop
            visited.add(cur)
            loop.append(cur)
            nexts = [n for n in adj[cur] if n != prev]
            if not nexts:
                break
            prev = cur
            cur = nexts[0]

        if len(loop) >= 3:
            loops.append(np.array(loop, dtype=int))

    # Step 4: sort by enclosed area, largest first
    def _loop_area(nodes: NDArray[np.int_]) -> float:
        coords = pts[nodes]
        try:
            return float(abs(LinearRing(np.vstack([coords, coords[:1]])).area))
        except Exception:
            return 0.0

    loops.sort(key=_loop_area, reverse=True)
    return loops


# ---------------------------------------------------------------------------
# OBC file reader
# ---------------------------------------------------------------------------

def read_obc_file(
    obc_path: str | Path,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Read an FVCOM ``_obc.dat`` open-boundary file.

    File format::

        OBC Node Number = N
        <obc_idx>  <node_idx>  <type>    ← 1-based node indices

    Parameters
    ----------
    obc_path : path to ``_obc.dat``

    Returns
    -------
    node_indices : (N,) 0-based node indices
    types : (N,) OBC type codes
    """
    node_indices: list[int] = []
    types: list[int] = []
    with open(obc_path) as fh:
        for line in fh.readlines()[1:]:  # skip header
            parts = line.strip().split()
            if len(parts) >= 3:
                node_indices.append(int(parts[1]) - 1)  # 0-based
                types.append(int(parts[2]))
    return np.array(node_indices, dtype=int), np.array(types, dtype=int)


# ---------------------------------------------------------------------------
# Domain and water polygon builders
# ---------------------------------------------------------------------------

def build_domain_polygon(
    pts: NDArray[np.float64],
    triangles: NDArray[np.int_],
    simplify_tol: float = 0.0,
) -> Polygon:
    """Build a Shapely Polygon covering the full model domain.

    The domain polygon is the outermost boundary loop of the mesh
    (the loop with the largest enclosed area).  All islands are excluded.

    Parameters
    ----------
    pts : (N, 2) node coordinates
    triangles : (T, 3) element connectivity
    simplify_tol : Douglas–Peucker simplification tolerance [m].
                   Set to 0 (default) to keep all boundary vertices.

    Returns
    -------
    domain : Shapely Polygon
    """
    loops = extract_boundary_loops(pts, triangles)
    if not loops:
        raise ValueError("No boundary loops found in the mesh.")

    outer_coords = pts[loops[0]]
    domain = Polygon(outer_coords)
    if not domain.is_valid:
        from shapely.validation import make_valid
        domain = make_valid(domain)
        if isinstance(domain, MultiPolygon):
            domain = max(domain.geoms, key=lambda g: g.area)

    if simplify_tol > 0:
        domain = domain.simplify(simplify_tol, preserve_topology=True)

    return domain


def build_water_polygon(
    pts: NDArray[np.float64],
    triangles: NDArray[np.int_],
    obc_node_indices: Optional[NDArray[np.int_]] = None,
    min_island_area: float = 0.0,
    simplify_tol: float = 0.0,
) -> Polygon:
    """Build a Shapely Polygon representing the water body.

    The water polygon is the outer boundary loop with inner loops (islands)
    cut out as holes.  Small islands below *min_island_area* are dropped.

    Parameters
    ----------
    pts : (N, 2) node coordinates
    triangles : (T, 3) element connectivity
    obc_node_indices : 0-based indices of OBC nodes.  If provided, inner loops
                       whose nodes are all OBC nodes are treated as open
                       boundaries rather than islands and are omitted.
                       (This handles rare meshes where the OBC is closed.)
    min_island_area : islands smaller than this [m²] are silently removed.
    simplify_tol : Douglas–Peucker simplification tolerance for each ring [m].

    Returns
    -------
    water : Shapely Polygon (may have holes for islands)
    """
    loops = extract_boundary_loops(pts, triangles)
    if not loops:
        raise ValueError("No boundary loops found in the mesh.")

    obc_set: set[int] = set() if obc_node_indices is None else set(
        obc_node_indices.tolist()
    )

    # --- Outer ring ---
    outer_coords = pts[loops[0]]
    outer_ring = LinearRing(outer_coords)

    # --- Island holes ---
    holes: list[LinearRing] = []
    for loop_nodes in loops[1:]:
        # Skip loops that are entirely OBC nodes (open boundary arc, not island)
        if obc_set and all(int(n) in obc_set for n in loop_nodes):
            continue

        island_coords = pts[loop_nodes]
        island_poly = Polygon(island_coords)
        if not island_poly.is_valid:
            from shapely.validation import make_valid
            island_poly = make_valid(island_poly)
            if not isinstance(island_poly, Polygon):
                continue  # skip degenerate islands

        if island_poly.area < min_island_area:
            continue

        if simplify_tol > 0:
            island_poly = island_poly.simplify(simplify_tol, preserve_topology=True)
            if island_poly.is_empty or not isinstance(island_poly, Polygon):
                continue

        holes.append(island_poly.exterior)

    water = Polygon(outer_ring, holes)
    if not water.is_valid:
        from shapely.validation import make_valid
        water = make_valid(water)
        if isinstance(water, MultiPolygon):
            water = max(water.geoms, key=lambda g: g.area)

    if simplify_tol > 0:
        water = water.simplify(simplify_tol, preserve_topology=True)
        if isinstance(water, MultiPolygon):
            water = max(water.geoms, key=lambda g: g.area)

    return water


# ---------------------------------------------------------------------------
# OBC arc extraction
# ---------------------------------------------------------------------------

def extract_obc_linestrings(
    pts: NDArray[np.float64],
    outer_loop_nodes: NDArray[np.int_],
    obc_node_set: set[int],
) -> list[LineString]:
    """Extract ordered OBC arcs as Shapely LineStrings.

    The outer boundary loop is traversed to find contiguous runs of OBC
    nodes.  Each run becomes a separate ``LineString``.

    Parameters
    ----------
    pts : (N, 2) node coordinates
    outer_loop_nodes : ordered node indices for the outer boundary loop
    obc_node_set : 0-based indices of OBC nodes

    Returns
    -------
    arcs : list of LineString objects (one per contiguous OBC arc)
    """
    n = len(outer_loop_nodes)
    is_obc = np.array([int(outer_loop_nodes[i]) in obc_node_set
                       for i in range(n)], dtype=bool)

    arcs: list[LineString] = []
    i = 0
    while i < n:
        if is_obc[i]:
            arc_nodes: list[int] = []
            j = i
            while j < n and is_obc[j]:
                arc_nodes.append(int(outer_loop_nodes[j]))
                j += 1
            if len(arc_nodes) >= 2:
                arcs.append(LineString(pts[arc_nodes]))
            i = j
        else:
            i += 1

    # Handle wrap-around: check if last and first runs are both OBC and should merge
    if len(arcs) >= 2 and is_obc[0] and is_obc[n - 1]:
        last = np.array(arcs[-1].coords)
        first = np.array(arcs[0].coords)
        merged = np.vstack([last, first[1:]])  # avoid duplicating junction
        arcs = [LineString(merged)] + arcs[1:-1]

    return arcs


# ---------------------------------------------------------------------------
# Point-cloud DEM (interpolated from existing mesh depths)
# ---------------------------------------------------------------------------

class PointCloudDEM:
    """DEM constructed from scattered node depths of an existing FVCOM mesh.

    Uses ``scipy.interpolate.LinearNDInterpolator`` for interior points
    and ``NearestNDInterpolator`` as a fallback for points outside the
    original mesh's convex hull.

    Parameters
    ----------
    pts : (N, 2) node coordinates (projected, metres)
    depths : (N,) nodal water depths (positive = below sea level)
    min_depth : clipping minimum [m] applied to all samples
    """

    def __init__(
        self,
        pts: NDArray[np.float64],
        depths: NDArray[np.float64],
        min_depth: float = 0.1,
    ) -> None:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        self._linear = LinearNDInterpolator(pts, depths)
        self._nearest = NearestNDInterpolator(pts, depths)
        self._min_depth = min_depth

    def sample(self, query_pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate depths at *query_pts*.

        Parameters
        ----------
        query_pts : (M, 2) query coordinates

        Returns
        -------
        depths : (M,) interpolated depths ≥ ``min_depth``
        """
        depths = self._linear(query_pts)
        nan_mask = np.isnan(depths)
        if np.any(nan_mask):
            depths[nan_mask] = self._nearest(query_pts[nan_mask])
        return np.maximum(depths, self._min_depth)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def coarsen_fvcom_mesh(
    grd_path: str | Path,
    h_min: float,
    h_max: float,
    delta_w: float,
    projection: str,
    output_prefix: str | Path,
    dep_path: Optional[str | Path] = None,
    obc_path: Optional[str | Path] = None,
    boundary_simplify_tol: Optional[float] = None,
    min_island_area: Optional[float] = None,
    **kwargs,
) -> "Mesh":  # type: ignore[name-defined]
    """Generate a coarser mesh from an existing FVCOM grid.

    Reads the existing mesh files to extract domain geometry, coastline
    and OBC locations.  No shapefiles are required.  Depths at new node
    positions are interpolated from the original nodal depths.

    Parameters
    ----------
    grd_path : path to the existing FVCOM ``_grd.dat`` file
    h_min : minimum element size for the new mesh [m]
    h_max : maximum element size for the new mesh [m]
    delta_w : minimum channel width threshold [m]
    projection : target CRS string (e.g. ``"EPSG:32643"``)
    output_prefix : file-name prefix for the four output ``*.dat`` files
    dep_path : path to ``_dep.dat`` (auto-detected from *grd_path* if None)
    obc_path : path to ``_obc.dat`` (auto-detected from *grd_path* if None)
    boundary_simplify_tol : Douglas–Peucker tolerance for boundary
                            simplification [m].  Defaults to ``h_min / 4``.
    min_island_area : islands smaller than this [m²] are removed from the
                      water mask.  Defaults to ``h_min ** 2``.
    **kwargs : extra keys passed directly into the MeshGenerator config
               (e.g. ``max_iter_2d=300``, ``gradient_limit=0.25``).

    Returns
    -------
    mesh : :class:`fvcom_mesh.core.Mesh` dataclass
    """
    from .core import MeshGenerator

    mg = MeshGenerator.from_fvcom_mesh(
        grd_path=grd_path,
        dep_path=dep_path,
        obc_path=obc_path,
        output_prefix=output_prefix,
        h_min=h_min,
        h_max=h_max,
        delta_w=delta_w,
        projection=projection,
        boundary_simplify_tol=boundary_simplify_tol,
        min_island_area=min_island_area,
        **kwargs,
    )
    return mg.run()
