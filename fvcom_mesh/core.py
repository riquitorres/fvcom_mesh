"""
core.py — MeshGenerator orchestrator and Mesh result dataclass.

Loads a YAML config, runs all pipeline phases in order, and writes output.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import MultiLineString

log = logging.getLogger(__name__)


def _rss_mb() -> str:
    """Return current RSS memory usage as a short string, or empty string."""
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return f"  [{kb / 1024:.0f} MB RSS]"
    except Exception:
        return ""


def _step(t0: float, phase: str, detail: str = "") -> None:
    """Print an always-visible phase progress line to stderr."""
    elapsed = time.perf_counter() - t0
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{elapsed:6.1f}s]{_rss_mb()}  {phase}{suffix}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Config loading / validation
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        cfg = {}
    return cfg


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Return a list of human-readable issues with the config (empty = OK)."""
    issues: list[str] = []
    required = [
        "domain_shapefile",
        "water_mask_shapefile",
        "projection",
        "h_min",
        "h_max",
        "delta_w",
        "output_prefix",
    ]
    for key in required:
        if key not in cfg:
            issues.append(f"Missing required key: '{key}'")
    if "h_min" in cfg and "h_max" in cfg:
        if cfg["h_min"] >= cfg["h_max"]:
            issues.append("h_min must be less than h_max")
    if "delta_w" in cfg and "h_min" in cfg:
        if cfg["delta_w"] < cfg["h_min"]:
            issues.append("delta_w should be >= h_min")
    return issues


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Mesh:
    """Container for the generated mesh and associated data.

    Attributes
    ----------
    pts : (N, 2) node coordinates (projected metres)
    triangles : (T, 3) element connectivity (0-based, CCW)
    depths : (N,) water depths (positive = below sea level)
    obc_list : open boundary condition list [(obc_idx, node_1based, type)]
    spg_list : sponge layer list [(spg_idx, node_1based, coeff)]
    constraints : list of Constraint objects (for quality inspection)
    """
    pts: NDArray[np.float64]
    triangles: NDArray[np.int_]
    depths: NDArray[np.float64]
    obc_list: list[tuple[int, int, int]] = field(default_factory=list)
    spg_list: list[tuple[int, int, float]] = field(default_factory=list)
    constraints: list = field(default_factory=list)

    def write_fvcom(self, prefix: str | Path) -> dict[str, Path]:
        """Write all four FVCOM input files with *prefix*."""
        from .output_fvcom import write_all
        return write_all(
            prefix,
            self.pts,
            self.triangles,
            self.depths,
            self.obc_list,
            self.spg_list,
        )

    @property
    def n_nodes(self) -> int:
        return len(self.pts)

    @property
    def n_elements(self) -> int:
        return len(self.triangles)


# ---------------------------------------------------------------------------
# MeshGenerator
# ---------------------------------------------------------------------------

class MeshGenerator:
    """Orchestrates all phases of ADMESH+ mesh generation."""

    def __init__(self, config: dict[str, Any], output_dir_override: Optional[Path] = None):
        self.config = config
        self._output_dir_override = output_dir_override
        # In-memory geometry overrides — set by from_fvcom_mesh()
        self._domain_geom = None   # Shapely Polygon
        self._water_geom = None    # Shapely Polygon (may have holes)
        self._dem_obj = None       # PointCloudDEM or DEM object
        self._obc_geom = None      # Shapely MultiLineString of OBC arcs

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        output_dir_override: Optional[Path] = None,
    ) -> "MeshGenerator":
        cfg = load_config(path)
        issues = validate_config(cfg)
        if issues:
            raise ValueError("Invalid config:\n" + "\n".join(f"  {i}" for i in issues))
        return cls(cfg, output_dir_override=output_dir_override)

    @classmethod
    def from_fvcom_mesh(
        cls,
        grd_path: str | Path,
        output_prefix: str | Path,
        h_min: float,
        h_max: float,
        delta_w: float,
        projection: str,
        dep_path: Optional[Path] = None,
        obc_path: Optional[Path] = None,
        output_dir_override: Optional[Path] = None,
        boundary_simplify_tol: Optional[float] = None,
        min_island_area: Optional[float] = None,
        **kwargs,
    ) -> "MeshGenerator":
        """Create a MeshGenerator from an existing FVCOM mesh.

        Extracts domain geometry, coastline and OBC arcs directly from the
        provided FVCOM files.  No shapefiles are required.

        Parameters
        ----------
        grd_path : path to existing ``_grd.dat`` file
        output_prefix : file-name prefix for the new mesh output files
        h_min : minimum element size [m]
        h_max : maximum element size [m]
        delta_w : minimum channel width [m]
        projection : target CRS string (e.g. ``"EPSG:32643"``)
        dep_path : path to ``_dep.dat`` (auto-detected from *grd_path* if None)
        obc_path : path to ``_obc.dat`` (auto-detected from *grd_path* if None)
        output_dir_override : optional directory override for output files
        boundary_simplify_tol : Douglas–Peucker tolerance for boundary
                                simplification [m].  Defaults to ``h_min / 4``.
        min_island_area : islands smaller than this [m²] are removed.
                          Defaults to ``h_min ** 2``.
        **kwargs : extra config keys (e.g. ``max_iter_2d``, ``gradient_limit``)
        """
        from .dynamic_quality import read_fvcom_mesh as _read_mesh
        from .mesh_modify import (
            extract_boundary_loops,
            read_obc_file,
            build_domain_polygon,
            build_water_polygon,
            extract_obc_linestrings,
            PointCloudDEM,
        )

        grd_path = Path(grd_path)

        # Auto-detect dep_path and obc_path
        if dep_path is None:
            candidate = Path(str(grd_path).replace("_grd.dat", "_dep.dat"))
            if candidate.exists():
                dep_path = candidate
        if obc_path is None:
            candidate = Path(str(grd_path).replace("_grd.dat", "_obc.dat"))
            if candidate.exists():
                obc_path = candidate

        # Read existing mesh
        pts, triangles, depths = _read_mesh(str(grd_path), str(dep_path) if dep_path else None)

        # Resolve simplification / island parameters
        if boundary_simplify_tol is None:
            boundary_simplify_tol = h_min / 4
        if min_island_area is None:
            min_island_area = h_min ** 2

        # Read OBC node indices
        obc_node_indices = None
        if obc_path is not None:
            obc_node_indices, _ = read_obc_file(obc_path)

        # Build domain and water polygons
        domain_geom = build_domain_polygon(
            pts, triangles, simplify_tol=boundary_simplify_tol
        )
        water_geom = build_water_polygon(
            pts, triangles,
            obc_node_indices=obc_node_indices,
            min_island_area=min_island_area,
            simplify_tol=boundary_simplify_tol,
        )

        # Build OBC arc geometry (Shapely MultiLineString)
        obc_geom = None
        if obc_node_indices is not None and len(obc_node_indices) > 0:
            loops = extract_boundary_loops(pts, triangles)
            if loops:
                arcs = extract_obc_linestrings(
                    pts, loops[0], set(obc_node_indices.tolist())
                )
                if arcs:
                    obc_geom = MultiLineString([list(a.coords) for a in arcs])

        # Point-cloud DEM from existing depths
        dem_obj = PointCloudDEM(pts, depths) if depths is not None else None

        # Build config dict (no shapefile keys needed)
        cfg: dict = {
            "projection": projection,
            "h_min": float(h_min),
            "h_max": float(h_max),
            "delta_w": float(delta_w),
            "output_prefix": str(output_prefix),
            "min_island_area": float(min_island_area),
            **kwargs,
        }

        obj = cls(cfg, output_dir_override=output_dir_override)
        obj._domain_geom = domain_geom
        obj._water_geom = water_geom
        obj._dem_obj = dem_obj
        obj._obc_geom = obc_geom
        return obj

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _get(self, key: str, default=None):
        return self.config.get(key, default)

    def _output_prefix(self) -> Path:
        prefix = Path(self._get("output_prefix", "mesh"))
        if self._output_dir_override is not None:
            prefix = self._output_dir_override / prefix.name
        return prefix

    # ------------------------------------------------------------------ #
    #  Pipeline
    # ------------------------------------------------------------------ #

    def run(self) -> Mesh:
        """Execute the full mesh-generation pipeline.

        Returns
        -------
        Mesh dataclass with pts, triangles, depths, obc_list, spg_list.
        Also writes the four FVCOM files to disk.
        """
        cfg = self.config
        t0 = time.perf_counter()

        projection = cfg["projection"]
        h_min = float(cfg["h_min"])
        h_max = float(cfg["h_max"])
        delta_w = float(cfg["delta_w"])

        # ---- Phase 1: load inputs ----
        log.info("Phase 1: loading inputs")
        _step(t0, "Phase 1/9  loading inputs")

        if self._domain_geom is not None:
            # In-memory geometry provided by from_fvcom_mesh()
            domain = self._domain_geom
            water = self._water_geom
            dem = self._dem_obj
        else:
            from .inputs import load_domain_boundary, load_water_mask, load_dem

            domain = load_domain_boundary(cfg["domain_shapefile"], projection)
            water = load_water_mask(cfg["water_mask_shapefile"], domain, projection)

            dem = None
            if cfg.get("dem"):
                dem = load_dem(cfg["dem"], domain, projection, positive_depth=True)

        # ---- Phase 2: build background grid + rasterise ----
        # Default resolution: coarser of h_min/4 and delta_w/4.
        # The background grid only needs to resolve features at the delta_w
        # scale; using h_min/4 on large domains with small h_min produces
        # grids with tens of millions of cells and causes OOM.
        _default_bg_res = max(h_min / 4, delta_w / 4)
        bg_resolution = float(cfg.get("bg_resolution", _default_bg_res))
        _xmin, _ymin, _xmax, _ymax = domain.bounds
        _nx = int((_xmax - _xmin) / bg_resolution) + 1
        _ny = int((_ymax - _ymin) / bg_resolution) + 1
        log.info("Phase 2: building background grid")
        _step(t0, "Phase 2/9  building background grid",
              f"resolution={bg_resolution:.0f} m  grid={_nx}×{_ny}={_nx*_ny:,} cells")
        from .water_mask import BackgroundGrid, build_masks, process_water_mask

        grid = BackgroundGrid.from_domain(domain, bg_resolution)
        masks = build_masks(domain, water, grid)

        # ---- Phase 3: water-mask preprocessing ----
        log.info("Phase 3: water mask preprocessing")
        _step(t0, "Phase 3/9  water mask preprocessing", f"delta_w={delta_w:.0f} m")
        min_island_area = float(cfg.get("min_island_area", h_min ** 2))
        use_vdt = bool(cfg.get("use_vdt_medial", False))
        masks = process_water_mask(masks, delta_w, min_island_area, use_vdt_medial=use_vdt)

        # ---- Phase 4: identify constraints ----
        log.info("Phase 4: identifying constraints")
        _step(t0, "Phase 4/9  identifying constraints")
        from .constraints import identify_constraints, build_mainstreams, smooth_all_constraints

        min_constraint_length = float(cfg.get("min_constraint_length", h_min))
        constraints = identify_constraints(masks, delta_w, min_constraint_length)

        junction_tol = float(cfg.get("junction_tol", h_min * 2))
        constraints = build_mainstreams(constraints, junction_tol)

        rmse_smooth = float(cfg.get("smoothing_rmse", max(1.0, h_min / 20)))
        constraints = smooth_all_constraints(constraints, rmse_smooth)
        log.info("  %d constraint polylines identified", len(constraints))

        # ---- Phase 5: mesh-size function ----
        log.info("Phase 5: building mesh-size function")
        _step(t0, "Phase 5/9  mesh-size function", f"h_min={h_min:.0f} h_max={h_max:.0f}")
        from .mesh_size import build_mesh_size_function

        K = float(cfg.get("curvature_constant", 2.0))
        g = float(cfg.get("gradient_limit", 0.3))
        h_func = build_mesh_size_function(constraints, grid, K, h_min, h_max, g)

        # ---- Phase 6: 1-D node placement ----
        log.info("Phase 6: 1-D node placement")
        _step(t0, "Phase 6/9  1-D node placement")
        from .mesh_gen_1d import generate_1d_nodes

        _, fixed_pts = generate_1d_nodes(constraints, h_func, h_min)
        log.info("  %d fixed nodes from 1-D placement", len(fixed_pts))

        # ---- Phase 7: triangulation ----
        backend = cfg.get("backend", "distmesh").lower()
        _step(t0, f"Phase 7/9  2-D triangulation", f"backend={backend}  fixed_pts={len(fixed_pts)}")
        if backend == "oceanmesh":
            log.info("Phase 7: OceanMesh Delaunay triangulation")
            from .oceanmesh_backend import generate_mesh_oceanmesh

            max_iter_2d = int(cfg.get("max_iter_2d", 200))
            grid_resolution = cfg.get("grid_resolution", None)
            if grid_resolution is not None:
                grid_resolution = float(grid_resolution)
            pts, triangles = generate_mesh_oceanmesh(
                water,
                h_func,
                h_min,
                h_max,
                fixed_pts=fixed_pts if len(fixed_pts) > 0 else None,
                max_iter=max_iter_2d,
                grid_resolution=grid_resolution,
            )
        else:
            log.info("Phase 7: 2-D DistMesh")
            from .mesh_gen_2d import distmesh_2d

            max_iter_2d = int(cfg.get("max_iter_2d", 200))
            tol_2d = float(cfg.get("tol_2d", 1e-4))
            # Use the water polygon (with island holes) so that DistMesh
            # neither generates initial nodes inside islands nor keeps
            # island-interior triangles in the final mesh.
            pts, triangles = distmesh_2d(
                water,
                h_func,
                fixed_pts,
                h_min,
                max_iter=max_iter_2d,
                tol=tol_2d,
            )
        log.info("  %d nodes, %d elements", len(pts), len(triangles))
        _step(t0, f"Phase 7/9  done", f"{len(pts)} nodes  {len(triangles)} elements")

        # ---- Phase 8: OBC and sponge layers ----
        log.info("Phase 8: OBC detection")
        _step(t0, "Phase 8/9  OBC detection")
        from .obc import detect_obc_nodes, detect_sponge_nodes

        obc_tol = float(cfg.get("obc_tol", h_min * 2))
        obc_shp = cfg.get("obc_shapefile")
        obc_type = int(cfg.get("obc_type", 1))
        obc_list = detect_obc_nodes(
            pts, triangles, domain, obc_tol,
            obc_shp=obc_shp,
            obc_geom=self._obc_geom,
            target_crs=projection,
            obc_type=obc_type,
        )

        spg_list: list[tuple[int, int, float]] = []
        sponge_radius = cfg.get("sponge_radius")
        if sponge_radius is not None:
            spg_list = detect_sponge_nodes(
                obc_list, pts, triangles,
                float(sponge_radius),
                float(cfg.get("sponge_coeff", 0.01)),
            )

        # ---- Phase 9: depths ----
        log.info("Phase 9: interpolating depths")
        _step(t0, "Phase 9/9  interpolating depths")
        if dem is not None:
            from .output_fvcom import interpolate_depths
            depths = interpolate_depths(pts, dem, default_depth=float(cfg.get("default_depth", 1.0)))
        else:
            depths = np.full(len(pts), float(cfg.get("default_depth", 1.0)))

        # ---- Write output ----
        prefix = self._output_prefix()
        log.info("Writing FVCOM files to %s_*.dat", prefix)
        mesh = Mesh(
            pts=pts,
            triangles=triangles,
            depths=depths,
            obc_list=obc_list,
            spg_list=spg_list,
            constraints=constraints,
        )
        mesh.write_fvcom(prefix)
        log.info("Done.")
        _step(t0, "Done", f"wrote {len(triangles)} elements  {len(pts)} nodes")
        return mesh
