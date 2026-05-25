"""
tests/test_oceanmesh_backend.py — Tests for the OceanMesh triangulation backend.

Tests are skipped if OceanMesh is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

om = pytest.importorskip(
    "oceanmesh",
    reason="OceanMesh not installed — skipping backend tests",
)

from fvcom_mesh.oceanmesh_backend import (
    _make_sdf,
    _build_domain,
    _build_sizing_grid,
    generate_mesh_oceanmesh,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def square_km():
    """1 km × 1 km square centred at origin (projected metres)."""
    return Polygon([(-500, -500), (500, -500), (500, 500), (-500, 500)])


@pytest.fixture()
def square_km_with_island():
    """1 km square with a small square island hole in the middle."""
    outer = [(-500, -500), (500, -500), (500, 500), (-500, 500)]
    inner = [(-50, -50), (50, -50), (50, 50), (-50, 50)]  # 100 m hole
    return Polygon(outer, [inner])


# ---------------------------------------------------------------------------
# SDF tests
# ---------------------------------------------------------------------------

class TestMakeSDF:
    def test_interior_negative(self, square_km):
        sdf = _make_sdf(square_km)
        pts = np.array([[0.0, 0.0], [100.0, 100.0]])
        vals = sdf(pts)
        assert np.all(vals < 0), "Interior points must give negative SDF"

    def test_exterior_positive(self, square_km):
        sdf = _make_sdf(square_km)
        pts = np.array([[1000.0, 0.0], [0.0, 1000.0]])
        vals = sdf(pts)
        assert np.all(vals > 0), "Exterior points must give positive SDF"

    def test_boundary_near_zero(self, square_km):
        sdf = _make_sdf(square_km)
        # Points on the boundary
        pts = np.array([[500.0, 0.0], [0.0, 500.0], [-500.0, 0.0]])
        vals = sdf(pts)
        assert np.all(np.abs(vals) < 1.0), "Boundary points must be ~0"

    def test_island_excluded(self, square_km_with_island):
        """Points inside the island hole must be positive (outside water polygon)."""
        sdf = _make_sdf(square_km_with_island)
        pts = np.array([[0.0, 0.0]])  # inside the island
        vals = sdf(pts)
        assert vals[0] > 0, "Island interior must be positive (land)"

    def test_returns_1d_array(self, square_km):
        sdf = _make_sdf(square_km)
        pts = np.random.default_rng(0).uniform(-400, 400, (100, 2))
        vals = sdf(pts)
        assert vals.shape == (100,)


# ---------------------------------------------------------------------------
# Domain tests
# ---------------------------------------------------------------------------

class TestBuildDomain:
    def test_bbox_tuple(self, square_km):
        domain = _build_domain(square_km)
        bbox = domain.bbox
        assert len(bbox) == 4, "Domain bbox must have 4 elements"
        # OceanMesh convention: (xmin, xmax, ymin, ymax)
        assert bbox[0] <= bbox[1]
        assert bbox[2] <= bbox[3]

    def test_eval_callable(self, square_km):
        domain = _build_domain(square_km)
        pts = np.array([[0.0, 0.0], [1000.0, 0.0]])
        result = domain.eval(pts)
        assert result.shape == (2,)
        assert result[0] < 0  # inside
        assert result[1] > 0  # outside


# ---------------------------------------------------------------------------
# Sizing grid tests
# ---------------------------------------------------------------------------

class TestBuildSizingGrid:
    def test_grid_has_eval(self, square_km):
        h_func = lambda pts: np.full(len(pts), 200.0)
        grid = _build_sizing_grid(square_km, h_func, h_min=100.0, h_max=500.0,
                                  grid_resolution=100.0)
        assert callable(grid.eval)

    def test_hmin_set(self, square_km):
        h_func = lambda pts: np.full(len(pts), 200.0)
        grid = _build_sizing_grid(square_km, h_func, h_min=100.0, h_max=500.0,
                                  grid_resolution=100.0)
        assert grid.hmin == pytest.approx(100.0)

    def test_eval_in_range(self, square_km):
        h_func = lambda pts: np.full(len(pts), 200.0)
        grid = _build_sizing_grid(square_km, h_func, h_min=100.0, h_max=500.0,
                                  grid_resolution=100.0)
        pts = np.array([[0.0, 0.0]])
        val = grid.eval(pts)
        assert 100.0 <= float(val[0]) <= 500.0


# ---------------------------------------------------------------------------
# End-to-end meshing test
# ---------------------------------------------------------------------------

class TestGenerateMeshOceanmesh:
    def test_returns_pts_and_triangles(self, square_km):
        h_func = lambda pts: np.full(len(pts), 100.0)
        pts, triangles = generate_mesh_oceanmesh(
            square_km,
            h_func,
            h_min=80.0,
            h_max=200.0,
            max_iter=30,
            grid_resolution=100.0,
        )
        assert pts.ndim == 2 and pts.shape[1] == 2
        assert triangles.ndim == 2 and triangles.shape[1] == 3
        assert len(pts) > 0
        assert len(triangles) > 0

    def test_pts_within_domain(self, square_km):
        """All mesh nodes should lie within (or on the boundary of) the domain."""
        import shapely
        h_func = lambda pts: np.full(len(pts), 100.0)
        pts, _ = generate_mesh_oceanmesh(
            square_km, h_func, h_min=80.0, h_max=200.0,
            max_iter=30, grid_resolution=100.0,
        )
        pts_geom = shapely.points(pts[:, 0], pts[:, 1])
        # Allow a small tolerance for boundary snapping
        outside = ~(
            shapely.contains_xy(square_km.buffer(10.0), pts[:, 0], pts[:, 1])
        )
        assert not np.any(outside), "Nodes should be inside the domain (±10 m)"

    def test_triangle_indices_valid(self, square_km):
        h_func = lambda pts: np.full(len(pts), 100.0)
        pts, triangles = generate_mesh_oceanmesh(
            square_km, h_func, h_min=80.0, h_max=200.0,
            max_iter=30, grid_resolution=100.0,
        )
        assert triangles.min() >= 0
        assert triangles.max() < len(pts)

    def test_with_fixed_pts(self, square_km):
        """Providing fixed boundary nodes should not crash."""
        h_func = lambda pts: np.full(len(pts), 100.0)
        fixed = np.array([[0.0, 0.0], [100.0, 100.0]])
        pts, triangles = generate_mesh_oceanmesh(
            square_km, h_func, h_min=80.0, h_max=200.0,
            max_iter=30, grid_resolution=100.0,
            fixed_pts=fixed,
        )
        assert len(pts) > 0
