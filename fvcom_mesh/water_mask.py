"""
water_mask.py — Rasterize land/water masks, compute Vector Distance Transform,
and extract medial axis for narrow-channel identification.

Implements Sect. 3.2 of Kang & Kubatko (2024), GMD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from skimage.morphology import (
    skeletonize,
    binary_dilation,
    binary_erosion,
    label,
    remove_small_objects,
    disk,
)
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from .utils import (
    rasterize_polygon,
    make_background_grid,
    vector_distance_transform,
    divergence_of_vdt,
    polygon_boundary_polylines,
)


# ---------------------------------------------------------------------------
# Raster background grid
# ---------------------------------------------------------------------------

@dataclass
class BackgroundGrid:
    """Uniform Cartesian grid used for all raster operations.

    Attributes
    ----------
    x, y : 1-D coordinate arrays (metres, increasing)
    dx, dy : cell size
    """
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    dx: float
    dy: float

    @classmethod
    def from_domain(cls, domain: Polygon, resolution: float) -> "BackgroundGrid":
        xmin, ymin, xmax, ymax = domain.bounds
        # Extend half a cell on each side so boundary pixels are inside
        x = np.arange(xmin - resolution, xmax + 2 * resolution, resolution)
        y = np.arange(ymin - resolution, ymax + 2 * resolution, resolution)
        return cls(x=x, y=y, dx=resolution, dy=resolution)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.y), len(self.x))

    def ij_to_xy(
        self, i: NDArray[np.int_], j: NDArray[np.int_]
    ) -> NDArray[np.float64]:
        """Convert (row i, col j) indices to (x, y) coordinates."""
        return np.column_stack([self.x[j], self.y[i]])

    def xy_to_ij(
        self, xy: NDArray[np.float64]
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Convert (x, y) coordinates to nearest (row, col) indices."""
        j = np.clip(
            np.round((xy[:, 0] - self.x[0]) / self.dx).astype(int),
            0, len(self.x) - 1,
        )
        i = np.clip(
            np.round((xy[:, 1] - self.y[0]) / self.dy).astype(int),
            0, len(self.y) - 1,
        )
        return i, j

    def meshgrid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.meshgrid(self.x, self.y)


# ---------------------------------------------------------------------------
# Mask processing
# ---------------------------------------------------------------------------

@dataclass
class MaskSet:
    """Holds the land and water binary masks (and their decomposed versions).

    Row 0 = y[0] (southernmost), columns increase eastward.  True = inside
    the respective region.
    """
    grid: BackgroundGrid
    land: NDArray[np.bool_]
    water: NDArray[np.bool_]
    # Level 1 = narrow, Level 2 = wide
    land_l1: NDArray[np.bool_] = field(default=None)
    land_l2: NDArray[np.bool_] = field(default=None)
    water_l1: NDArray[np.bool_] = field(default=None)
    water_l2: NDArray[np.bool_] = field(default=None)
    # Updated masks after complex-geometry processing
    land_updated: NDArray[np.bool_] = field(default=None)
    water_updated: NDArray[np.bool_] = field(default=None)
    water_l1_final: NDArray[np.bool_] = field(default=None)
    water_l2_final: NDArray[np.bool_] = field(default=None)


def build_masks(
    domain: Polygon,
    water: Polygon | MultiPolygon,
    grid: BackgroundGrid,
) -> MaskSet:
    """Rasterize domain and water polygons onto *grid*.

    The land mask is domain minus water.  Both masks are binary arrays
    on the background grid.
    """
    water_mask = rasterize_polygon(water, grid.x, grid.y)
    domain_mask = rasterize_polygon(domain, grid.x, grid.y)
    land_mask = domain_mask & ~water_mask
    return MaskSet(grid=grid, land=land_mask, water=water_mask)


def remove_small_islands(
    masks: MaskSet,
    min_area_m2: float,
) -> MaskSet:
    """Remove water-mask holes (islands) smaller than *min_area_m2*.

    An 'island' is a connected component of the land mask that is fully
    surrounded by the water mask (i.e. not connected to the domain edge).
    Such small islands are transferred to the water mask.

    Parameters
    ----------
    min_area_m2 : minimum island area in square metres.
    """
    dx, dy = masks.grid.dx, masks.grid.dy
    cell_area = dx * dy
    min_pixels = int(np.ceil(min_area_m2 / cell_area))

    land = masks.land.copy()
    water = masks.water.copy()

    # Label connected components of the land mask
    labeled_land, n_land = label(land, return_num=True)  # type: ignore[call-overload]

    # Find domain-edge-connected component indices (background label = 0)
    # Edge pixels: first/last row/col
    edge_labels = set(np.unique(
        np.concatenate([
            labeled_land[0, :], labeled_land[-1, :],
            labeled_land[:, 0], labeled_land[:, -1],
        ])
    )) - {0}

    for region in regionprops(labeled_land):
        lbl = region.label
        if lbl in edge_labels:
            continue  # part of mainland, keep
        if region.area < min_pixels:
            # Transfer to water
            pixels = labeled_land == lbl
            land[pixels] = False
            water[pixels] = True

    masks.land = land
    masks.water = water
    return masks


def width_based_decomposition(
    mask: NDArray[np.bool_],
    dx: float,
    dy: float,
    delta_w: float,
    use_vdt_medial: bool = False,
    *,
    prune_angle_thresh: float = 0.9 * np.pi,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
    """Decompose a binary mask into level-1 (narrow) and level-2 (wide) regions.

    Implements Steps 1–4 of Sect. 3.2.1 of Kang & Kubatko (2024).

    Parameters
    ----------
    mask : 2-D bool array (True = inside region)
    dx, dy : cell size in metres
    delta_w : user-specified minimum width threshold (metres)
    use_vdt_medial : if True use the VDT divergence approach (more accurate);
                     otherwise use scikit-image skeletonize.
    prune_angle_thresh : angle threshold for medial axis pruning.

    Returns
    -------
    level1 : narrow-region mask
    level2 : wide-region mask
    width_fn : the width function f_w on the grid
    """
    # --- Distance to boundary ---
    d_boundary = distance_transform_edt(mask, sampling=(dy, dx))

    # --- Medial axis ---
    if use_vdt_medial:
        _, vx, vy = vector_distance_transform(mask, dx, dy)
        div = divergence_of_vdt(vx, vy, dx, dy)
        # Positive divergence → medial axis candidate
        medial_raw = (div > 0) & mask
    else:
        medial_raw = skeletonize(mask)

    # --- Prune medial axis near corners (Step 2) ---
    medial_pruned = _prune_medial_axis(medial_raw, d_boundary, dx, dy, prune_angle_thresh)

    # --- Distance to medial axis (Step 3) ---
    d_medial = distance_transform_edt(~medial_pruned, sampling=(dy, dx))
    # Only meaningful inside the mask
    d_medial[~mask] = 0.0

    # --- Width function f_w = 2*(d_boundary + d_medial) (Eq. 14) ---
    width_fn = 2.0 * (d_boundary + d_medial)
    width_fn[~mask] = 0.0

    # --- Decompose (Eqs. 15–16) ---
    level1 = mask & (width_fn < delta_w)
    level2 = mask & (width_fn >= delta_w)

    return level1, level2, width_fn


def _prune_medial_axis(
    medial: NDArray[np.bool_],
    d_boundary: NDArray[np.float64],
    dx: float,
    dy: float,
    angle_thresh: float,
) -> NDArray[np.bool_]:
    """Prune medial axis branches near corners.

    Identifies order-1 (free-end) branches and removes points where the
    medial axis angle is greater than *angle_thresh* (indicating corners),
    following Steps 2 of Sect. 3.2.1.
    """
    from skimage.morphology import skeletonize
    from skimage.graph import pixel_graph

    if not medial.any():
        return medial

    # Build pixel connectivity graph
    g, nodes = pixel_graph(medial, connectivity=2)  # 8-connectivity

    # Degree of each medial pixel
    degrees = np.array(g.sum(axis=1)).ravel()

    # End points: degree == 1 (or 0 if isolated)
    # Branch points: degree >= 3
    is_end = degrees <= 1
    is_branch = degrees >= 3

    pruned = medial.copy()

    # Walk from each end point; remove points where angle > thresh
    # Simplified: remove short order-1 branches where the VDT width
    # is below delta_w/2 (conservative corner pruning)
    end_pixels = np.array(nodes)[is_end]
    rows, cols = np.unravel_index(np.where(medial.ravel())[0], medial.shape)
    idx_map = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}

    for ep in end_pixels:
        r, c = np.unravel_index(ep, medial.shape)
        w = 2 * d_boundary[r, c]
        # Remove end-point if its local width is small relative to cell size
        if w < (dx + dy):  # heuristic: narrower than one cell diagonal
            pruned[r, c] = False

    return pruned


def fill_level2_mask(
    level1: NDArray[np.bool_],
    level2: NDArray[np.bool_],
    dx: float,
    dy: float,
    d_boundary: NDArray[np.float64],
    delta_w: float = 0.0,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Apply the maximal-disk filling to level-2 masks.

    Implements Eqs. 17–20 and the filling method of Sect. 3.2.2, Step 2.
    Uses morphological dilation with a disk structuring element whose radius
    is the local VDT value at each level-2 point.

    For efficiency we approximate with a single dilation using the median
    VDT of level-2 points as the disk radius.  The refinement from
    Eq. (20) (removing level-1 regions without medial axis) is also applied.
    """
    if not level2.any():
        return level1, level2

    # Median maximal disk radius from the distance function on level-2 pixels
    l2_dist = d_boundary[level2]
    if l2_dist.size == 0:
        return level1, level2
    pixel_size = (dx + dy) / 2.0
    radius_px = int(np.round(np.median(l2_dist) / pixel_size))
    radius_px = max(1, radius_px)

    # Cap at delta_w scale: the fill is meant to bridge level-1 gaps, not
    # dilate across the entire domain.  Large open-water regions would
    # otherwise produce a disk with millions of pixels and exhaust memory.
    if delta_w > 0:
        max_radius_px = max(1, int(np.ceil(delta_w / pixel_size)))
        radius_px = min(radius_px, max_radius_px)

    selem = disk(radius_px)
    level2_filled = binary_dilation(level2, selem)
    # Only keep cells within the original mask
    original_mask = level1 | level2
    level2_filled = level2_filled & original_mask

    # Transfer filled region from level1 to level2
    newly_wide = level2_filled & ~level2
    level1_updated = level1 & ~newly_wide
    level2_updated = level2 | newly_wide

    # Eq. 20: level-1 regions not containing any medial axis → transfer to level2
    # (these are isolated regions with no skeleton, rounded islands)
    medial_in_l1 = skeletonize(level1_updated)
    l1_labeled, _ = label(level1_updated, return_num=True)
    for region in regionprops(l1_labeled):
        if not medial_in_l1[l1_labeled == region.label].any():
            pixels = l1_labeled == region.label
            level1_updated[pixels] = False
            level2_updated[pixels] = True

    return level1_updated, level2_updated


def process_water_mask(
    masks: MaskSet,
    delta_w: float,
    min_island_area: float,
    use_vdt_medial: bool = False,
) -> MaskSet:
    """Full water-mask preprocessing pipeline (Sect. 3.2.1–3.2.2).

    Steps performed:
    1. Remove small islands (< min_island_area)
    2. Width-based decomposition of land and water masks
    3. Filling of level-2 masks
    4. Find level-1 land surrounded by level-2 water → internal boundaries
    5. Transfer those regions to water mask
    6. Re-apply water-mask decomposition on updated water mask

    Parameters
    ----------
    masks : MaskSet (from build_masks)
    delta_w : minimum channel width (metres)
    min_island_area : minimum island area to keep (m²)
    use_vdt_medial : use VDT divergence for medial axis (slower, more accurate)
    """
    dx, dy = masks.grid.dx, masks.grid.dy

    # Step 1: remove small islands
    masks = remove_small_islands(masks, min_island_area)

    # Step 2a: decompose land mask
    land_l1, land_l2, _ = width_based_decomposition(
        masks.land, dx, dy, delta_w, use_vdt_medial
    )
    d_boundary_land = distance_transform_edt(masks.land, sampling=(dy, dx))
    land_l1, land_l2 = fill_level2_mask(land_l1, land_l2, dx, dy, d_boundary_land, delta_w)
    masks.land_l1 = land_l1
    masks.land_l2 = land_l2

    # Step 2b: decompose water mask
    water_l1, water_l2, _ = width_based_decomposition(
        masks.water, dx, dy, delta_w, use_vdt_medial
    )
    d_boundary_water = distance_transform_edt(masks.water, sampling=(dy, dx))
    water_l1, water_l2 = fill_level2_mask(water_l1, water_l2, dx, dy, d_boundary_water, delta_w)
    masks.water_l1 = water_l1
    masks.water_l2 = water_l2

    # Step 3: find level-1 land regions surrounded by level-2 water
    narrow_land_for_ib = _find_narrow_land_in_wide_water(
        land_l1, water_l2, dx, dy, ipr_thresh=30.0
    )

    # Step 4: transfer those regions to water mask
    land_updated = masks.land.copy()
    water_updated = masks.water.copy()
    land_updated[narrow_land_for_ib] = False
    water_updated[narrow_land_for_ib] = True
    masks.land_updated = land_updated
    masks.water_updated = water_updated

    # Step 5: re-apply decomposition to updated water mask
    water_l1_f, water_l2_f, _ = width_based_decomposition(
        water_updated, dx, dy, delta_w, use_vdt_medial
    )
    d_bnd_upd = distance_transform_edt(water_updated, sampling=(dy, dx))
    water_l1_f, water_l2_f = fill_level2_mask(water_l1_f, water_l2_f, dx, dy, d_bnd_upd, delta_w)
    masks.water_l1_final = water_l1_f
    masks.water_l2_final = water_l2_f

    return masks


def _find_narrow_land_in_wide_water(
    land_l1: NDArray[np.bool_],
    water_l2: NDArray[np.bool_],
    dx: float,
    dy: float,
    ipr_thresh: float = 30.0,
) -> NDArray[np.bool_]:
    """Identify level-1 land regions surrounded by level-2 water.

    Implements Step 3 of Sect. 3.2.2.  Returns a boolean mask of the
    selected land pixels that should become internal boundary constraints.

    Parameters
    ----------
    ipr_thresh : isoperimetric ratio threshold (dimensionless).  Regions
                 with IPR > ipr_thresh (thinner shapes) are candidates.
    """
    result = np.zeros_like(land_l1)
    labeled_l1, _ = label(land_l1, return_num=True)

    cell_area = dx * dy

    for region in regionprops(labeled_l1):
        area = region.area * cell_area  # m²
        perimeter = region.perimeter * ((dx + dy) / 2)  # m (approx)
        if perimeter == 0:
            continue
        ipr = (perimeter ** 2) / (4 * np.pi * area)
        if ipr < ipr_thresh:
            continue  # not thin enough

        # Buffer by half the minor axis length
        minor_axis_px = region.minor_axis_length / 2.0  # in pixels
        buffer_px = max(1, int(np.round(minor_axis_px / 2.0)))

        # Region pixels
        region_pixels = labeled_l1 == region.label
        dilated = binary_dilation(region_pixels, disk(buffer_px))
        buffer_zone = dilated & ~region_pixels

        area_l2_in_buf = np.sum(buffer_zone & water_l2) * cell_area
        area_l1_in_buf = np.sum(buffer_zone & land_l1) * cell_area  # approx

        # Select if level-2 water area in buffer > 2× level-1 land area in buffer
        if area_l1_in_buf > 0 and area_l2_in_buf > 2 * area_l1_in_buf:
            result |= region_pixels

    return result.astype(bool)


def extract_skeleton_polylines(
    mask: NDArray[np.bool_],
    grid: BackgroundGrid,
    min_length: float = 0.0,
) -> list[NDArray[np.float64]]:
    """Extract ordered polylines from a binary skeleton mask.

    Uses 8-connectivity pixel graph traversal to trace branches from
    endpoints/branch-points.  Returns a list of (N, 2) coordinate arrays
    in the grid's projected CRS.

    Parameters
    ----------
    mask : 2-D bool array (True = skeleton pixels)
    grid : BackgroundGrid for converting pixel to coordinates
    min_length : discard polylines shorter than this length (metres)
    """
    from skimage.graph import pixel_graph
    from .utils import polyline_length

    if not mask.any():
        return []

    g, nodes = pixel_graph(mask, connectivity=2)
    n = len(nodes)
    if n == 0:
        return []

    degrees = np.array(g.sum(axis=1)).ravel()
    nodes_arr = np.array(nodes)  # flat pixel indices
    rows, cols = np.unravel_index(nodes_arr, mask.shape)

    is_end = degrees <= 1
    is_branch = degrees >= 3
    start_indices = np.where(is_end | is_branch)[0]
    if len(start_indices) == 0:
        # Single circular chain — use any pixel as start
        start_indices = np.array([0])

    visited_edges: set[tuple[int, int]] = set()
    polylines = []

    adj = {i: list(g[i].nonzero()[1]) for i in range(n)}

    for si in start_indices:
        for ni in adj[si]:
            edge_key = (min(si, ni), max(si, ni))
            if edge_key in visited_edges:
                continue
            # Walk the branch until hitting another endpoint/branch/dead-end
            chain = [si, ni]
            visited_edges.add(edge_key)
            prev, cur = si, ni
            while True:
                nbrs = [nb for nb in adj[cur] if nb != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                ek = (min(cur, nxt), max(cur, nxt))
                if ek in visited_edges:
                    break
                if degrees[nxt] != 2:
                    chain.append(nxt)
                    visited_edges.add(ek)
                    break
                chain.append(nxt)
                visited_edges.add(ek)
                prev, cur = cur, nxt

            chain_rows = rows[chain]
            chain_cols = cols[chain]
            xy = grid.ij_to_xy(chain_rows, chain_cols)
            if polyline_length(xy) >= min_length:
                polylines.append(xy)

    return polylines
