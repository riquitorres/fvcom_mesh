"""
Tests for output_fvcom.py — verify FVCOM file format compliance.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fvcom_mesh.output_fvcom import write_grd, write_dep, write_obc, write_spg, write_all


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_mesh():
    """A trivial 4-node, 2-triangle mesh."""
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0],
        [1.5, 1.0],
    ])
    # Two CCW triangles
    triangles = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    depths = np.array([5.0, 6.0, 4.0, 7.0])
    return pts, triangles, depths


# ---------------------------------------------------------------------------
# _grd.dat
# ---------------------------------------------------------------------------

class TestWriteGrd:
    def test_header(self, simple_mesh, tmp_path):
        pts, triangles, _ = simple_mesh
        path = tmp_path / "test_grd.dat"
        write_grd(pts, triangles, path)
        lines = path.read_text().splitlines()
        assert lines[0].strip() == "Node Number = 4"
        assert lines[1].strip() == "Cell Number = 2"

    def test_connectivity_1based(self, simple_mesh, tmp_path):
        pts, triangles, _ = simple_mesh
        path = tmp_path / "test_grd.dat"
        write_grd(pts, triangles, path)
        lines = path.read_text().splitlines()
        # First cell row is lines[2]
        parts = lines[2].split()
        assert len(parts) == 5  # cell_id n1 n3 n2 cell_id
        # All indices should be >= 1
        for p in parts:
            assert int(p) >= 1

    def test_node_coords(self, simple_mesh, tmp_path):
        pts, triangles, _ = simple_mesh
        path = tmp_path / "test_grd.dat"
        write_grd(pts, triangles, path)
        lines = path.read_text().splitlines()
        # Node lines start at line 2 + n_cells
        node_start = 2 + len(triangles)
        assert len(lines) == node_start + len(pts)
        x0, y0 = map(float, lines[node_start].split())
        assert x0 == pytest.approx(pts[0, 0])
        assert y0 == pytest.approx(pts[0, 1])


# ---------------------------------------------------------------------------
# _dep.dat
# ---------------------------------------------------------------------------

class TestWriteDep:
    def test_header_and_values(self, simple_mesh, tmp_path):
        pts, _, depths = simple_mesh
        path = tmp_path / "test_dep.dat"
        write_dep(pts, depths, path)
        lines = path.read_text().splitlines()
        assert lines[0].strip() == "Node Number = 4"
        assert len(lines) == 1 + len(pts)
        # Check first depth
        parts = lines[1].split()
        assert float(parts[2]) == pytest.approx(depths[0])


# ---------------------------------------------------------------------------
# _obc.dat
# ---------------------------------------------------------------------------

class TestWriteObc:
    def test_empty(self, tmp_path):
        path = tmp_path / "test_obc.dat"
        write_obc([], path)
        lines = path.read_text().splitlines()
        assert lines[0].strip() == "OBC Node Number = 0"

    def test_entries(self, tmp_path):
        obc_list = [(1, 5, 1), (2, 6, 1), (3, 7, 1)]
        path = tmp_path / "test_obc.dat"
        write_obc(obc_list, path)
        lines = path.read_text().splitlines()
        assert lines[0].strip() == "OBC Node Number = 3"
        parts = lines[1].split()
        assert int(parts[0]) == 1
        assert int(parts[1]) == 5
        assert int(parts[2]) == 1


# ---------------------------------------------------------------------------
# _spg.dat
# ---------------------------------------------------------------------------

class TestWriteSpg:
    def test_entries(self, tmp_path):
        spg_list = [(1, 10, 0.01), (2, 11, 0.01)]
        path = tmp_path / "test_spg.dat"
        write_spg(spg_list, path)
        lines = path.read_text().splitlines()
        assert lines[0].strip() == "Sponge Node Number = 2"
        parts = lines[1].split()
        assert float(parts[2]) == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# write_all
# ---------------------------------------------------------------------------

class TestWriteAll:
    def test_creates_four_files(self, simple_mesh, tmp_path):
        pts, triangles, depths = simple_mesh
        obc_list = [(1, 1, 1)]
        spg_list = [(1, 2, 0.01)]
        prefix = tmp_path / "test"
        paths = write_all(str(prefix), pts, triangles, depths, obc_list, spg_list)
        assert set(paths.keys()) == {"grd", "dep", "obc", "spg"}
        for p in paths.values():
            assert p.exists()
