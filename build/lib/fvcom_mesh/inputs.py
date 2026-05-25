"""
inputs.py — Load domain boundary, water mask and DEM from disk.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import fiona
import geopandas as gpd
import rasterio
from rasterio.crs import CRS as RasterioCRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS, Transformer
from shapely.geometry import box, shape, mapping, MultiPolygon, Polygon
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Domain boundary
# ---------------------------------------------------------------------------

def load_domain_boundary(
    shp_path: str | Path,
    target_crs: str | CRS,
) -> Polygon:
    """Load the outer domain boundary from a shapefile.

    Parameters
    ----------
    shp_path : path to a shapefile containing a single closed polygon
    target_crs : target projected CRS (e.g. ``"EPSG:32643"`` for UTM zone 43N)

    Returns
    -------
    shapely Polygon in *target_crs* coordinates (metres)
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        raise ValueError(f"Shapefile {shp_path} has no CRS. Set it first.")
    gdf = gdf.to_crs(target_crs)
    union = unary_union(gdf.geometry.values)
    if isinstance(union, MultiPolygon):
        # Take the largest polygon as the domain
        union = max(union.geoms, key=lambda g: g.area)
    if not isinstance(union, Polygon):
        raise ValueError("Domain boundary shapefile must contain polygon geometry.")
    return union


# ---------------------------------------------------------------------------
# Water mask (coastline / shoreline polygons)
# ---------------------------------------------------------------------------

def load_water_mask(
    shp_path: str | Path,
    domain: Polygon,
    target_crs: str | CRS,
) -> MultiPolygon | Polygon:
    """Load the water mask (wet area) from a shoreline shapefile.

    Shoreline data may be provided as:
    - closed polygon(s) representing water bodies, or
    - polylines that together form closed rings (automatically polygonised).

    The result is clipped to *domain* and returned as a single geometry in
    *target_crs* (projected, metres).

    Parameters
    ----------
    shp_path : path to shoreline shapefile
    domain : outer domain boundary polygon (in *target_crs*)
    target_crs : target projected CRS
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        raise ValueError(f"Shapefile {shp_path} has no CRS.")
    gdf = gdf.to_crs(target_crs)

    geom_types = set(gdf.geometry.geom_type.unique())
    if geom_types <= {"Polygon", "MultiPolygon"}:
        water = unary_union(gdf.geometry.values)
    elif geom_types <= {"LineString", "MultiLineString"}:
        from shapely.ops import polygonize
        all_lines = unary_union(gdf.geometry.values)
        water = unary_union(list(polygonize(all_lines)))
        if water.is_empty:
            raise ValueError(
                "Could not polygonise the shoreline linestrings. "
                "Ensure they form closed rings."
            )
    else:
        # Mixed or Point — try to extract polygonal part
        polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        if polys.empty:
            raise ValueError(
                f"Unsupported geometry types in water mask: {geom_types}. "
                "Provide Polygon or LineString data."
            )
        water = unary_union(polys.geometry.values)

    # Clip to domain
    water = water.intersection(domain)
    if water.is_empty:
        raise ValueError("Water mask does not overlap the domain boundary.")
    return water


# ---------------------------------------------------------------------------
# Digital Elevation Model (DEM / bathymetry)
# ---------------------------------------------------------------------------

class DEM:
    """Wrapper around a clipped, reprojected raster DEM.

    Attributes
    ----------
    data : 2-D float array (row=south→north, col=west→east), NaN for nodata
    x : 1-D array of column centre x-coordinates
    y : 1-D array of row centre y-coordinates (increasing order)
    transform : rasterio Affine transform
    crs : pyproj CRS of the data
    nodata : original nodata sentinel (or NaN)
    """

    def __init__(
        self,
        data: NDArray[np.float64],
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        crs: CRS,
    ):
        self.data = data
        self.x = x
        self.y = y
        self.crs = crs

    # ------------------------------------------------------------------
    def sample(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Bilinear interpolation of DEM values at arbitrary (x, y) points.

        Parameters
        ----------
        pts : (N, 2) array of (x, y) coordinates in the DEM's CRS.

        Returns
        -------
        depths : (N,) array; NaN where pts fall outside the DEM extent or on
                 nodata cells.
        """
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (self.y, self.x),
            self.data,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        # RegularGridInterpolator expects (y, x) ordering
        return interp(pts[:, ::-1])

    def __repr__(self) -> str:
        return (
            f"DEM(shape={self.data.shape}, "
            f"x=[{self.x[0]:.1f}, {self.x[-1]:.1f}], "
            f"y=[{self.y[0]:.1f}, {self.y[-1]:.1f}])"
        )


def load_dem(
    dem_path: str | Path,
    domain: Polygon,
    target_crs: str | CRS,
    *,
    positive_depth: bool = True,
) -> DEM:
    """Load and reproject a DEM/bathymetry raster to *target_crs*.

    The raster is clipped to the bounding box of *domain* with a small
    buffer (5 % of domain extent) to ensure complete coverage at the edges.

    Parameters
    ----------
    dem_path : path to a GeoTIFF or any GDAL-supported raster
    domain : outer domain boundary polygon (in *target_crs*)
    target_crs : target projected CRS
    positive_depth : if True, negate elevation so that below-sea-level
                     values become positive depths (FVCOM convention).
    """
    target_crs_obj = CRS.from_user_input(target_crs)
    dst_crs_str = target_crs_obj.to_wkt()

    bounds = domain.bounds  # (xmin, ymin, xmax, ymax)
    # 5 % buffer
    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]
    buf = 0.05 * max(dx, dy)
    window_bounds = (
        bounds[0] - buf, bounds[1] - buf,
        bounds[2] + buf, bounds[3] + buf,
    )

    with rasterio.open(dem_path) as src:
        src_crs = src.crs
        # Compute transform for reprojected output
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs_str,
            src.width, src.height,
            *src.bounds,
        )
        reprojected = np.full((height, width), np.nan, dtype=np.float64)
        reproject(
            source=rasterio.band(src, 1),
            destination=reprojected,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs_str,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        nodata = src.nodata

    # Replace original nodata with NaN
    if nodata is not None:
        reprojected[reprojected == nodata] = np.nan

    # Build coordinate arrays from affine transform
    # transform * (col + 0.5, row + 0.5) = centre of cell
    cols = np.arange(width)
    rows = np.arange(height)
    x_coords = transform.c + (cols + 0.5) * transform.a
    y_coords = transform.f + (rows + 0.5) * transform.e  # e is negative

    # rasterio stores rows top-to-bottom; flip so y is increasing
    if y_coords[0] > y_coords[-1]:
        reprojected = reprojected[::-1, :]
        y_coords = y_coords[::-1]

    # Clip to window
    xi = np.searchsorted(x_coords, [window_bounds[0], window_bounds[2]])
    yi = np.searchsorted(y_coords, [window_bounds[1], window_bounds[3]])
    x0, x1 = max(0, xi[0] - 1), min(width, xi[1] + 1)
    y0, y1 = max(0, yi[0] - 1), min(height, yi[1] + 1)
    data_clip = reprojected[y0:y1, x0:x1]
    x_clip = x_coords[x0:x1]
    y_clip = y_coords[y0:y1]

    if positive_depth:
        data_clip = -data_clip  # elevation → depth (positive = below sea level)

    return DEM(data_clip, x_clip, y_clip, target_crs_obj)
