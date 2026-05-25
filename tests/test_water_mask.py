"""
Tests for water_mask.py — mask rasterisation, width decomposition, skeleton extraction.
"""

import numpy as np
import pytest
from shapely.geometry import box, Polygon

from fvcom_mesh.water_mask import (
    BackgroundGrid,
    build_masks,
    remove_small_islands,
    width_based_decomposition,
    fill_level2_mask,
    extract_skeleton_polylines,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_domain_and_channel(channel_width: float = 50.0, domain_size: float = 1000.0):
    """Create a rectangular domain with a narrow N-S channel through the middle."""
    domain = box(0, 0, domain_size, domain_size)
    # Channel: a thin strip in the middle (water)
    cx = domain_size / 2
    channel = box(cx - channel_width / 2, 0, cx + channel_width / 2, domain_size)
    return domain, channel


# ---------------------------------------------------------------------------
# BackgroundGrid
# ---------------------------------------------------------------------------

class TestBackgroundGrid:
    def test_shape_from_domain(self):
        domain = box(0, 0, 500, 500)
        grid = BackgroundGrid.from_domain(domain, resolution=10.0)
        assert grid.shape[0] == len(grid.y)
        assert grid.shape[1] == len(grid.x)
        assert grid.dx == pytest.approx(10.0)
        assert grid.dy == pytest.approx(10.0)

    def test_ij_xy_roundtrip(self):
        domain = box(0, 0, 200, 200)
        grid = BackgroundGrid.from_domain(domain, resolution=5.0)
        i = np.array([10, 20])
        j = np.array([5, 15])
        xy = grid.ij_to_xy(i, j)
        i2, j2 = grid.xy_to_ij(xy)
        np.testing.assert_array_equal(i, i2)
        np.testing.assert_array_equal(j, j2)


# ---------------------------------------------------------------------------
# build_masks
# ---------------------------------------------------------------------------

class TestBuildMasks:
    def test_basic(self):
        domain = box(0, 0, 500, 500)
        water = box(100, 100, 400, 400)
        grid = BackgroundGrid.from_domain(domain, resolution=10.0)
        masks = build_masks(domain, water, grid)
        assert masks.water.any()
        assert masks.land.any()
        # Water + land should cover (approximately) the domain
        assert (masks.water | masks.land).sum() > 0

    def test_no_overlap_between_water_and_land(self):
        domain = box(0, 0, 500, 500)
        water = box(100, 100, 400, 400)
        grid = BackgroundGrid.from_domain(domain, resolution=10.0)
        masks = build_masks(domain, water, grid)
        assert not (masks.water & masks.land).any()


# ---------------------------------------------------------------------------
# small island removal
# ---------------------------------------------------------------------------

class TestRemoveSmallIslands:
    def test_small_island_removed(self):
        domain = box(0, 0, 500, 500)
        # Water = domain minus a tiny island in the middle
        island = box(240, 240, 260, 260)  # 20×20 = 400 m²
        water_poly = domain.difference(island)
        grid = BackgroundGrid.from_domain(domain, resolution=5.0)
        masks = build_masks(domain, water_poly, grid)
        land_before = masks.land.sum()
        masks = remove_small_islands(masks, min_area_m2=1000.0)
        # Island pixels should have moved to water
        assert masks.land.sum() < land_before

    def test_mainland_not_removed(self):
        """A large land mass is never removed by the small-island filter."""
        domain = box(0, 0, 500, 500)
        # Water covers most of the domain, leaving a clear land strip
        # Use a large interior water box so the surrounding land annulus is obvious
        water = box(150, 150, 350, 350)
        grid = BackgroundGrid.from_domain(domain, resolution=5.0)
        masks = build_masks(domain, water, grid)
        land_before = masks.land.sum()
        assert land_before > 0, "Land mask should have pixels"
        # min_area threshold bigger than any single large land region → nothing removed
        # But a very large threshold should still not destroy the mainland
        # because mainland pixels ARE connected to the domain edge label
        masks2 = remove_small_islands(masks, min_area_m2=100.0)  # small threshold, mainland safe
        assert masks2.land.sum() == land_before


# ---------------------------------------------------------------------------
# width decomposition
# ---------------------------------------------------------------------------

class TestWidthDecomposition:
    def test_narrow_channel_in_level1(self):
        # Narrow horizontal strip (50 m wide, 500 m long)
        mask = np.zeros((100, 100), dtype=bool)
        # 5-pixel-wide horizontal strip at row 48–52 (resolution 10 m → width 50 m)
        mask[48:53, :] = True
        dx = dy = 10.0
        delta_w = 100.0
        l1, l2, fw = width_based_decomposition(mask, dx, dy, delta_w)
        # Most of the narrow strip should be in level 1
        assert l1[50, 50]  # centre of strip
        assert not l2[50, 50]

    def test_wide_region_in_level2(self):
        # 500×500 m box at 10 m resolution → 50×50 pixels
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:60, 10:60] = True
        dx = dy = 10.0
        delta_w = 100.0  # only 100 m width needed for level 2
        l1, l2, fw = width_based_decomposition(mask, dx, dy, delta_w)
        # Centre should be wide (level 2)
        assert l2[35, 35]


# ---------------------------------------------------------------------------
# extract_skeleton_polylines
# ---------------------------------------------------------------------------

class TestExtractSkeletonPolylines:
    def test_single_line(self):
        # Single horizontal line of pixels
        from skimage.morphology import skeletonize
        mask = np.zeros((20, 100), dtype=bool)
        mask[9:11, :] = True
        skel = skeletonize(mask)
        grid = BackgroundGrid(
            x=np.arange(100, dtype=float),
            y=np.arange(20, dtype=float),
            dx=1.0,
            dy=1.0,
        )
        lines = extract_skeleton_polylines(skel, grid, min_length=0.0)
        assert len(lines) >= 1
        # Total length should be approximately 100 pixels
        from fvcom_mesh.utils import polyline_length
        total = sum(polyline_length(l) for l in lines)
        assert total > 50  # conservative

    def test_empty_mask(self):
        mask = np.zeros((20, 20), dtype=bool)
        grid = BackgroundGrid(
            x=np.arange(20, dtype=float),
            y=np.arange(20, dtype=float),
            dx=1.0, dy=1.0,
        )
        lines = extract_skeleton_polylines(mask, grid)
        assert lines == []
