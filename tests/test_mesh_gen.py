"""
Tests for mesh_gen_2d.py and mesh_quality.py — basic mesh generation and quality.
"""

import numpy as np
import pytest
from shapely.geometry import box

from fvcom_mesh.mesh_quality import element_quality, quality_report


# ---------------------------------------------------------------------------
# element_quality
# ---------------------------------------------------------------------------

class TestElementQuality:
    def test_equilateral_triangle(self):
        """Perfect equilateral triangle → q = 1."""
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
        ])
        tris = np.array([[0, 1, 2]])
        q = element_quality(tris, pts)
        assert q[0] == pytest.approx(1.0, abs=1e-6)

    def test_degenerate_triangle(self):
        """Collinear points → q ≈ 0."""
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ])
        tris = np.array([[0, 1, 2]])
        q = element_quality(tris, pts)
        assert q[0] < 0.01

    def test_quality_range(self):
        """Quality should always be in [0, 1]."""
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 100, (20, 2))
        from scipy.spatial import Delaunay
        tri = Delaunay(pts)
        q = element_quality(tri.simplices, pts)
        assert np.all(q >= 0.0)
        assert np.all(q <= 1.0)


# ---------------------------------------------------------------------------
# quality_report
# ---------------------------------------------------------------------------

class TestQualityReport:
    def test_keys(self):
        q = np.array([0.3, 0.5, 0.7, 0.9])
        report = quality_report(q)
        for key in ("mean", "min", "median", "p5", "p10", "n_bad", "n_elements"):
            assert key in report

    def test_n_bad_counts(self):
        q = np.array([0.1, 0.2, 0.5, 0.8])
        report = quality_report(q)
        assert report["n_bad"] == 2  # 0.1 and 0.2 are < 0.3


# ---------------------------------------------------------------------------
# Simple DistMesh smoke test (fast, low-iteration)
# ---------------------------------------------------------------------------

class TestDistmeshSmoke:
    def test_generates_triangles(self):
        """DistMesh on a small box should produce a valid mesh."""
        from fvcom_mesh.mesh_size import MeshSizeFunction, BackgroundGrid
        from fvcom_mesh.water_mask import BackgroundGrid as BG
        from fvcom_mesh.mesh_gen_2d import distmesh_2d

        domain = box(0, 0, 200, 200)
        h_min = 30.0
        h_max = 50.0

        # Constant size function
        def h_func(pts):
            return np.full(len(pts), 40.0)

        fixed_pts = np.array([
            [0.0, 0.0], [200.0, 0.0],
            [200.0, 200.0], [0.0, 200.0],
        ])
        pts, tris = distmesh_2d(
            domain, h_func, fixed_pts, h_min,
            max_iter=20, tol=1.0, seed=0,
        )
        assert len(pts) >= 4
        assert len(tris) >= 1
        q = element_quality(tris, pts)
        assert np.mean(q) > 0.2  # very relaxed for low-iteration run
