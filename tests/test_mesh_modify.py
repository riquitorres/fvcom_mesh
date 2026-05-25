"""
tests/test_mesh_modify.py — Unit tests for fvcom_mesh.mesh_modify.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, MultiLineString

from fvcom_mesh.mesh_modify import (
    PointCloudDEM,
    build_domain_polygon,
    build_water_polygon,
    extract_boundary_loops,
    extract_obc_linestrings,
    read_obc_file,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic meshes
# ---------------------------------------------------------------------------

def _unit_square_mesh():
    r"""Simple unit-square mesh: 4 nodes, 2 right triangles (CCW).

         3 -------- 2
         |  \       |
         |    \     |
         |      \   |
         0 -------- 1
    """
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    return pts, triangles


def _square_with_island():
    """5 × 5 outer domain with a 1 × 1 island hole in the centre.

    Outer boundary: 4 corner nodes (0–3)
    Island boundary: 4 nodes (4–7)
    12 triangles to fill the annular region.
    """
    pts = np.array([
        # Outer corners
        [0.0, 0.0],  # 0
        [5.0, 0.0],  # 1
        [5.0, 5.0],  # 2
        [0.0, 5.0],  # 3
        # Island corners
        [2.0, 2.0],  # 4
        [3.0, 2.0],  # 5
        [3.0, 3.0],  # 6
        [2.0, 3.0],  # 7
    ], dtype=float)
    # Fill the gap with triangles (manually constructed, CCW)
    triangles = np.array([
        # Bottom strip
        [0, 1, 5],
        [0, 5, 4],
        # Right strip
        [1, 2, 6],
        [1, 6, 5],
        # Top strip
        [2, 3, 7],
        [2, 7, 6],
        # Left strip
        [3, 0, 4],
        [3, 4, 7],
        # Diagonal connectors
        [0, 5, 4],  # duplicate — handled by edge count
        [4, 5, 6],
        [4, 6, 7],
    ])
    return pts, triangles


# ---------------------------------------------------------------------------
# extract_boundary_loops
# ---------------------------------------------------------------------------

class TestExtractBoundaryLoops:
    def test_unit_square_single_loop(self):
        pts, tris = _unit_square_mesh()
        loops = extract_boundary_loops(pts, tris)
        assert len(loops) == 1
        assert len(loops[0]) == 4

    def test_all_boundary_nodes_present(self):
        pts, tris = _unit_square_mesh()
        loops = extract_boundary_loops(pts, tris)
        assert set(loops[0].tolist()) == {0, 1, 2, 3}

    def test_loop_is_ordered(self):
        """Adjacent loop nodes must share a boundary edge."""
        pts, tris = _unit_square_mesh()
        loops = extract_boundary_loops(pts, tris)
        loop = loops[0].tolist()
        n = len(loop)
        for i in range(n):
            a, b = loop[i], loop[(i + 1) % n]
            # Either (a,b) or (b,a) must be an edge of a triangle
            edge = {a, b}
            found = any(
                edge == {int(tris[t, j]), int(tris[t, (j + 1) % 3])}
                for t in range(len(tris))
                for j in range(3)
            )
            assert found, f"Edge ({a},{b}) is not a triangle edge"

    def test_largest_loop_first(self):
        """With multiple loops the outer (larger) comes first."""
        pts = np.array([
            [0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0],  # outer
            [1.5, 1.5], [2.5, 1.5], [2.5, 2.5], [1.5, 2.5],  # island
        ], dtype=float)
        # Outer ring triangles only (no island connections — fake mesh)
        tris = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
        ])
        loops = extract_boundary_loops(pts, tris)
        areas = []
        from shapely.geometry import LinearRing
        for lp in loops:
            c = pts[lp]
            areas.append(abs(LinearRing(np.vstack([c, c[:1]])).area))
        # First loop must have the largest area
        assert areas[0] == max(areas)


# ---------------------------------------------------------------------------
# read_obc_file
# ---------------------------------------------------------------------------

class TestReadObcFile:
    def test_basic_parse(self, tmp_path):
        content = (
            "OBC Node Number = 3\n"
            "1  10  1\n"
            "2  20  1\n"
            "3  30  1\n"
        )
        f = tmp_path / "test_obc.dat"
        f.write_text(content)
        indices, types = read_obc_file(f)
        assert list(indices) == [9, 19, 29]   # 0-based
        assert list(types) == [1, 1, 1]

    def test_type_variety(self, tmp_path):
        content = "OBC Node Number = 2\n1  5  2\n2  6  3\n"
        f = tmp_path / "obc2.dat"
        f.write_text(content)
        indices, types = read_obc_file(f)
        assert list(types) == [2, 3]

    def test_empty_file(self, tmp_path):
        content = "OBC Node Number = 0\n"
        f = tmp_path / "obc_empty.dat"
        f.write_text(content)
        indices, types = read_obc_file(f)
        assert len(indices) == 0


# ---------------------------------------------------------------------------
# build_domain_polygon
# ---------------------------------------------------------------------------

class TestBuildDomainPolygon:
    def test_returns_polygon(self):
        pts, tris = _unit_square_mesh()
        poly = build_domain_polygon(pts, tris)
        assert isinstance(poly, Polygon)
        assert poly.is_valid

    def test_approximate_area(self):
        pts, tris = _unit_square_mesh()
        poly = build_domain_polygon(pts, tris)
        assert abs(poly.area - 1.0) < 0.01

    def test_simplify_tol_does_not_break(self):
        pts, tris = _unit_square_mesh()
        poly = build_domain_polygon(pts, tris, simplify_tol=0.01)
        assert isinstance(poly, Polygon)
        assert poly.is_valid


# ---------------------------------------------------------------------------
# build_water_polygon
# ---------------------------------------------------------------------------

class TestBuildWaterPolygon:
    def test_no_islands(self):
        pts, tris = _unit_square_mesh()
        water = build_water_polygon(pts, tris)
        assert isinstance(water, Polygon)
        assert water.is_valid

    def test_area_equals_domain(self):
        pts, tris = _unit_square_mesh()
        domain = build_domain_polygon(pts, tris)
        water = build_water_polygon(pts, tris)
        # Without real islands the water should equal the domain
        assert abs(water.area - domain.area) < 1e-6

    def test_contains_point_inside(self):
        pts, tris = _unit_square_mesh()
        water = build_water_polygon(pts, tris)
        from shapely.geometry import Point
        assert water.contains(Point(0.5, 0.5))


# ---------------------------------------------------------------------------
# extract_obc_linestrings
# ---------------------------------------------------------------------------

class TestExtractObcLinestrings:
    def test_all_obc_nodes_form_one_arc(self):
        pts, tris = _unit_square_mesh()
        loops = extract_boundary_loops(pts, tris)
        # Mark nodes 0 and 1 (bottom edge) as OBC
        obc_set = {0, 1}
        arcs = extract_obc_linestrings(pts, loops[0], obc_set)
        assert len(arcs) >= 1
        # All arcs together should contain exactly the OBC coordinates
        all_coords = set()
        for arc in arcs:
            for coord in arc.coords:
                all_coords.add(coord)
        obc_coords = {tuple(pts[n]) for n in obc_set}
        assert obc_coords.issubset(all_coords)

    def test_returns_linestrings(self):
        from shapely.geometry import LineString
        pts, tris = _unit_square_mesh()
        loops = extract_boundary_loops(pts, tris)
        arcs = extract_obc_linestrings(pts, loops[0], {0, 1})
        for arc in arcs:
            assert isinstance(arc, LineString)

    def test_no_obc_nodes_returns_empty(self):
        pts, tris = _unit_square_mesh()
        loops = extract_boundary_loops(pts, tris)
        arcs = extract_obc_linestrings(pts, loops[0], set())
        assert arcs == []


# ---------------------------------------------------------------------------
# PointCloudDEM
# ---------------------------------------------------------------------------

class TestPointCloudDEM:
    def test_exact_nodes_interpolated(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        depths = np.array([10.0, 20.0, 30.0, 40.0])
        dem = PointCloudDEM(pts, depths)
        result = dem.sample(pts)
        np.testing.assert_allclose(result, depths, atol=1e-6)

    def test_interior_interpolation(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        depths = np.array([10.0, 10.0, 10.0, 10.0])
        dem = PointCloudDEM(pts, depths)
        result = dem.sample(np.array([[0.5, 0.5]]))
        assert abs(result[0] - 10.0) < 1e-6

    def test_min_depth_clipping(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        depths = np.array([-5.0, -5.0, -5.0])  # negative depths
        dem = PointCloudDEM(pts, depths, min_depth=0.1)
        result = dem.sample(np.array([[0.1, 0.1]]))
        assert result[0] >= 0.1

    def test_extrapolation_nearest(self):
        """Points outside convex hull fall back to nearest-neighbour."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        depths = np.array([5.0, 5.0, 5.0, 5.0])
        dem = PointCloudDEM(pts, depths)
        # Far outside the domain
        result = dem.sample(np.array([[10.0, 10.0]]))
        assert result[0] >= 0.1
