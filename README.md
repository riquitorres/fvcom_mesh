# fvcom_mesh

An automatic, unstructured triangular mesh generator for the
[FVCOM](http://fvcom.smast.umassd.edu/) hydrodynamic model.
The algorithm follows the **ADMESH+** methodology described in:

> Kang & Kubatko (2024). *ADMESH+: An efficient algorithm for the automated
> generation of unstructured meshes for coastal ocean models.*
> Geoscientific Model Development **17**, 1603–1627.
> <https://doi.org/10.5194/gmd-17-1603-2024>

The tool takes geospatial inputs (domain boundary, shoreline, optional
bathymetry raster) and produces the four FVCOM mesh input files:

| File | Contents |
|---|---|
| `<prefix>_grd.dat` | Node coordinates and element connectivity |
| `<prefix>_dep.dat` | Water depth at each node |
| `<prefix>_obc.dat` | Open boundary condition node list |
| `<prefix>_spg.dat` | Sponge layer node list |

---

## Installation

### ARCHER2 (recommended)

```bash
module load PrgEnv-gnu cray-python
python -m venv --system-site-packages /work/<project>/<project>/<user>/fvcom_mesh_venv
source /work/<project>/<project>/<user>/fvcom_mesh_venv/bin/activate
pip install -e /path/to/fvcom_mesh      # editable install
```

The `--system-site-packages` flag inherits NumPy 1.23 and SciPy 1.10 from
`cray-python`, avoiding a slow recompile.  All other dependencies (Shapely,
GeoPandas, rasterio, scikit-image, pyproj, matplotlib, click, PyYAML) are
installed from PyPI automatically.

### Conda (recommended for local development)

Create and activate the environment using the provided `environment.yml`,
which pulls all dependencies (including `oceanmesh`) from `conda-forge`:

```bash
conda env create -f environment.yml
conda activate reefMaldives
```

The package itself is installed in editable mode automatically.

### General

GDAL must be installed before the Python dependencies.

**Ubuntu/Debian** — install the system library, then match the pip package to its version:

```bash
sudo apt install libgdal-dev
gdal-config --version   # e.g. 3.8.4
pip install gdal==3.8.4
```

Then install the package:

```bash
pip install -e .
```

Requires Python ≥ 3.10.  See `requirements.txt` for the full dependency list.

---

## Quick start

### 1. Prepare input files

| Input | Format | Notes |
|---|---|---|
| Domain boundary | Shapefile (polygon) | Outer ocean boundary of your model domain |
| Shoreline / water mask | Shapefile (polygon or linestring) | Defines the land–water interface |
| Bathymetry (optional) | GeoTIFF raster | Any CRS; reprojected automatically |
| OBC segments (optional) | Shapefile (linestring) | Override auto-detected open boundaries |

All inputs can be in any CRS; the tool reprojects them to the `projection`
you specify in the config file.

### 2. Write a configuration file

Copy `examples/config_example.yml` and edit the required fields:

```yaml
domain_shapefile: "input/domain_boundary.shp"
water_mask_shapefile: "input/shoreline.shp"
dem: "input/bathymetry.tif"          # optional
projection: "EPSG:32643"             # UTM zone matching your region
h_min: 50.0                          # metres
h_max: 2000.0                        # metres
delta_w: 100.0                       # minimum channel width (metres)
output_prefix: "output/my_domain"
```

### 3. Run from the command line

```bash
fvcom-mesh run config.yml
fvcom-mesh run config.yml --output-dir ./results --verbose
```

Validate the config without running:

```bash
fvcom-mesh check config.yml
```

### 4. Use the Python API

```python
from fvcom_mesh import MeshGenerator

gen = MeshGenerator.from_config("config.yml")
mesh = gen.run()

# mesh.pts          — (N, 2) node coordinates (metres)
# mesh.triangles    — (T, 3) element connectivity (0-based, CCW)
# mesh.depths       — (N,)   water depths (positive = below sea level)
# mesh.obc_list     — open boundary node list
# mesh.spg_list     — sponge layer node list

# FVCOM files are written automatically; you can also call:
mesh.write_fvcom("output/my_domain")

# Inspect quality
from fvcom_mesh.mesh_quality import element_quality, print_quality_report
q = element_quality(mesh.triangles, mesh.pts)
print_quality_report(q)
```

---

## Algorithm overview

The pipeline implements the nine-phase ADMESH+ workflow:

```
Phase 1  Load inputs          — domain polygon, shoreline, DEM
Phase 2  Background grid      — uniform raster at resolution h_min/4
Phase 3  Water-mask prep      — rasterise, small-island removal,
                                width decomposition, complex-geometry
Phase 4  Constraints          — identify 3 types, merge mainstreams,
                                cubic spline smoothing, curvature
Phase 5  Mesh-size function   — curvature → h₀, gradient limiting
Phase 6  1-D node placement   — spring-force equilibrium on constraints
Phase 7  2-D DistMesh         — rejection-method init, force equilibrium
Phase 8  OBC / sponge         — auto-detect + optional override
Phase 9  Depths & output      — DEM interpolation, write FVCOM files
```

### Phase 3 — Water-mask preprocessing

The shoreline polygon is rasterised onto the background grid.  Connected
components of the land mask that are entirely surrounded by water and
smaller than `min_island_area` (default `h_min²`) are removed.

The remaining water mask is decomposed into two levels using the
**width function**:

$$f_w(\mathbf{x}) = 2\bigl(d_\text{boundary}(\mathbf{x}) + d_\text{medial}(\mathbf{x})\bigr)$$

- **Level 1** — narrow channels: $f_w < \delta_w$
- **Level 2** — wide water bodies: $f_w \ge \delta_w$

The medial axis is computed either via `skimage.morphology.skeletonize`
(default, fast) or via the divergence of the Vector Distance Transform
(`use_vdt_medial: true`, more accurate).  The level-2 mask is then
expanded by maximal-disk filling (morphological dilation).

Complex geometry (barrier islands, levees) is identified using the
isoperimetric ratio IPR > 30 combined with a buffer/area test; these
regions are transferred to the water mask and their centrelines become
Type-2 (internal boundary) constraints.

### Phase 4 — Internal constraints

Three types of constraint polylines are extracted:

| Type | Source | Purpose |
|---|---|---|
| 1 — Channel centrelines | Skeleton of level-1 water | 1-D hydrodynamic network |
| 2 — Internal boundaries | Centrelines of transferred narrow land | Sub-grid-scale barriers |
| 3 — Shoreline | Boundary of the updated water mask | Land–water interface |

All constraint polylines are smoothed with a cubic parametric spline
fitted to a target RMSE (`smoothing_rmse`, default `max(1, h_min/20)` m).
Curvature $\kappa(s)$ is computed analytically from the spline derivatives.

### Phase 5 — Mesh-size function

The desired mesh size at each constraint node is:

$$h_0(s) = \operatorname{clamp}\!\left(\frac{1}{K\,\kappa(s)},\; h_\min,\; h_\max\right)$$

where $K$ (`curvature_constant`, default 2) controls the ratio between
curvature and element size.  These values are scattered onto the background
grid and the spatial gradient is limited to $g$ (`gradient_limit`, default
0.3) by solving the static Hamilton–Jacobi equation via a fast vectorised
sweeping scheme.  The resulting gridded field is wrapped in a bilinear
interpolator (`MeshSizeFunction`).

### Phase 6 — 1-D node placement

Nodes are placed along each constraint polyline using a 1-D spring-force
iteration: adjacent springs have rest length equal to the local desired
mesh size $h(s)$.  Endpoints are fixed.  Post-processing removes clusters
closer than $h_\min/2$, short elements shorter than $0.5\,h_\min$, and
nodes too close to Type-3 shoreline nodes.

### Phase 7 — 2-D DistMesh

The 2-D mesh is generated following the DistMesh algorithm
(Persson & Strang 2004) adapted for ADMESH+:

1. **Initial nodes** — drawn uniformly inside the bounding box and
   accepted with probability $(h_\min / h(\mathbf{x}))^2$ (rejection method).
2. **Force iteration** — spring forces push nodes to fill the domain
   uniformly according to the mesh-size function.  The force scale
   decreases linearly from 1.2 to 1.0 over the iterations.
3. **Boundary projection** — nodes that drift outside the domain are
   projected back onto the boundary.
4. **Density control** — nodes closer than $h_\min/2$ are removed
   every 10 iterations after 80 % of `max_iter_2d`.
5. **Final triangulation** — Delaunay triangulation with centroids inside
   the domain; all elements are reoriented counter-clockwise (FVCOM convention).

### Phase 8 — Open boundary conditions

OBC nodes are mesh nodes that lie within `obc_tol` metres of the domain
boundary polygon.  If `obc_shapefile` is provided, only nodes near those
polylines are kept.  OBC nodes are ordered by arc-length along the domain
boundary.

An optional sponge layer can be generated by selecting all mesh nodes
within `sponge_radius` metres of any OBC node.

---

## Configuration reference

All distances in **metres**; all tolerances relative to the projected CRS.

### Required keys

| Key | Type | Description |
|---|---|---|
| `domain_shapefile` | path | Outer domain boundary polygon |
| `water_mask_shapefile` | path | Shoreline / water-mask polygons or polylines |
| `projection` | string | Target projected CRS (EPSG code or WKT) |
| `h_min` | float | Minimum element size (m) |
| `h_max` | float | Maximum element size (m) |
| `delta_w` | float | Minimum channel-width threshold (m) |
| `output_prefix` | path | Prefix for the four output `.dat` files |

### Optional keys

| Key | Default | Description |
|---|---|---|
| `dem` | — | Path to bathymetry/DEM raster |
| `obc_shapefile` | — | Path to OBC segment shapefile |
| `bg_resolution` | `h_min / 4` | Background grid cell size (m) |
| `use_vdt_medial` | `false` | Use VDT divergence for medial axis |
| `min_island_area` | `h_min²` | Minimum island area to preserve (m²) |
| `min_constraint_length` | `h_min` | Shortest constraint polyline kept (m) |
| `junction_tol` | `2 * h_min` | Distance threshold for merging junctions (m) |
| `smoothing_rmse` | `max(1, h_min/20)` | Target spline smoothing RMSE (m) |
| `curvature_constant` | `2.0` | Curvature-to-size constant $K$ |
| `gradient_limit` | `0.3` | Maximum mesh-size gradient (dimensionless) |
| `max_iter_2d` | `200` | DistMesh iteration limit |
| `tol_2d` | `0.001` | DistMesh convergence tolerance (m) |
| `obc_tol` | `2 * h_min` | Distance threshold for OBC node detection (m) |
| `obc_type` | `1` | FVCOM OBC type code |
| `sponge_radius` | — | Sponge layer radius (m); disabled if absent |
| `sponge_coeff` | `0.01` | Sponge damping coefficient |
| `default_depth` | `1.0` | Fallback depth for land / no-DEM nodes (m) |

---

## FVCOM output file formats

### `_grd.dat`

```
Node Number = N
Cell Number = M
<cell_id>  <n1>  <n3>  <n2>  <cell_id>   ← M rows, 1-based, column order n1 n3 n2
<x>  <y>                                  ← N rows of node coordinates
```

Note: FVCOM stores triangle nodes in the order N1, N3, N2 (not N1, N2, N3).

### `_dep.dat`

```
Node Number = N
<x>  <y>  <depth>     ← N rows; depth positive = below sea level
```

### `_obc.dat`

```
OBC Node Number = N
<obc_idx>  <node_idx>  <type>    ← N rows, all 1-based
```

### `_spg.dat`

```
Sponge Node Number = N
<spg_idx>  <node_idx>  <coeff>   ← N rows, indices 1-based
```

### Element quality metric

The quality of each triangular element is reported as:

$$q = \frac{2r}{R}$$

where $r$ is the inradius and $R$ is the circumradius.  An equilateral
triangle has $q = 1$; degenerate (sliver) elements approach $q = 0$.
The `print_quality_report` function prints mean, median, minimum, 5th and
10th percentile, and the count of elements with $q < 0.3$.

---

## Generating a mesh from an existing FVCOM mesh

The `coarsen` command re-meshes an existing FVCOM grid at a different
resolution **without requiring any shapefiles**.  The domain boundary,
coastline, islands, and open boundary arcs are extracted directly from the
provided `_grd.dat` / `_dep.dat` / `_obc.dat` files.

```bash
fvcom-mesh coarsen maldives_v0_grd.dat \
    --projection EPSG:32643            \
    --h-min 500 --h-max 5000           \
    --delta-w 1000                     \
    --output-prefix maldives_v0_coarse
```

Matching `_dep.dat` and `_obc.dat` are auto-detected from the same
directory; pass `--dep` / `--obc` to override.

### Coarsen options

| Flag | Default | Description |
|---|---|---|
| `--dep PATH` | auto-detect | Path to `_dep.dat` |
| `--obc PATH` | auto-detect | Path to `_obc.dat` |
| `--output-prefix TEXT` | *(required)* | File-name prefix for the new mesh |
| `--projection TEXT` | *(required)* | Target CRS (e.g. `EPSG:32643`) |
| `--h-min FLOAT` | *(required)* | Minimum element size (m) |
| `--h-max FLOAT` | *(required)* | Maximum element size (m) |
| `--delta-w FLOAT` | *(required)* | Minimum channel width (m) |
| `--output-dir PATH` | same as `_grd.dat` | Directory for output files |
| `--boundary-simplify-tol FLOAT` | `h_min / 4` | Douglas–Peucker tolerance for the extracted boundary (m) |
| `--min-island-area FLOAT` | `h_min²` | Islands smaller than this are removed (m²) |
| `--max-iter-2d INT` | 200 (distmesh) / 100 (oceanmesh) | Maximum meshing iterations |
| `--gradient-limit FLOAT` | `0.3` | Mesh-size gradient limit |
| `--backend [distmesh\|oceanmesh]` | `distmesh` | Triangulation backend (see below) |
| `--grid-resolution FLOAT` | auto | OceanMesh sizing-grid cell size (m) |
| `--plot-dir PATH` | — | Save mesh quality plots to this directory |
| `--verbose` | off | Print per-phase progress |

### Python API

```python
from fvcom_mesh.core import MeshGenerator

mg = MeshGenerator.from_fvcom_mesh(
    grd_path="maldives_v0_grd.dat",
    output_prefix="maldives_v0_coarse",
    h_min=500.0,
    h_max=5000.0,
    delta_w=1000.0,
    projection="EPSG:32643",
)
mesh = mg.run()
```

### How geometry extraction works

The following functions in `fvcom_mesh.mesh_modify` do the extraction:

| Function | Purpose |
|---|---|
| `extract_boundary_loops(pts, triangles)` | Walk boundary edges to find ordered node loops; the longest loop is the outer coast |
| `read_obc_file(obc_path)` | Return 0-based OBC node indices and type codes |
| `build_domain_polygon(pts, triangles)` | Shapely Polygon from the outermost boundary loop |
| `build_water_polygon(pts, triangles, obc_node_indices, …)` | Polygon with island holes; OBC segments are left open |
| `extract_obc_linestrings(pts, outer_loop, obc_set)` | Ordered LineString arcs for each contiguous OBC segment |
| `PointCloudDEM(pts, depths)` | Depth interpolator using `LinearNDInterpolator` + nearest-neighbour fallback |

---

## OceanMesh backend

For fine-resolution meshes (h_min ≤ 200 m) the default DistMesh
force-iteration solver can exhaust RAM because it stores a full dense
point-displacement matrix.  The `--backend oceanmesh` option routes
Phase 7 through the [OceanMesh](https://github.com/CHLNDDEV/oceanmesh)
C++ Delaunay triangulator, which has much lower peak memory usage.

```bash
fvcom-mesh coarsen maldives_v0_grd.dat \
    --projection EPSG:32643            \
    --h-min 200 --h-max 5000           \
    --delta-w 500                      \
    --backend oceanmesh                \
    --output-prefix maldives_v0_fine
```

### How the backend works

The adapter (`fvcom_mesh/oceanmesh_backend.py`) bridges the two APIs:

1. **Signed distance function** — a vectorised SDF is built from the
   Shapely water polygon using Shapely 2.x bulk operations
   (`shapely.distance`, `shapely.contains_xy`).  The OceanMesh
   convention is negative inside the ocean domain, positive outside.

2. **Sizing grid** — the ADMESH+ mesh-size function is evaluated on a
   regular grid and stored in an `oceanmesh.Grid` object backed by a
   `RegularGridInterpolator`.  Grid resolution defaults to
   `max(h_min/2, domain_span/1000)` to keep memory manageable.

3. **Mesh generation** — `oceanmesh.generate_mesh(domain, sizing_grid,
   max_iter=…, pfix=…)` is called; the returned `(pts, triangles)` are
   passed back to the normal ADMESH+ Phase 8–9 pipeline.

### Installing OceanMesh on ARCHER2

OceanMesh requires CGAL ≥ 5.0 headers (header-only since CGAL 5.0),
Boost, and `libgmp`/`libmpfr`.  On ARCHER2 these are not available as
system packages, so a local prefix is used:

```bash
# --- build once ---
# 1. CGAL headers (header-only, ~5 MB)
wget https://github.com/CGAL/cgal/releases/download/v5.6.2/CGAL-5.6.2-library.tar.xz
tar -xf CGAL-5.6.2-library.tar.xz
mkdir -p /work/<proj>/<proj>/<user>/local_prefix/include
cp -r CGAL-5.6.2/include/CGAL /work/<proj>/<proj>/<user>/local_prefix/include/

# 2. Boost headers (via module)
module load boost
ln -sf $BOOST_ROOT/include/boost /work/<proj>/<proj>/<user>/local_prefix/include/boost

# 3. gmp / mpfr symlinks
mkdir -p /work/<proj>/<proj>/<user>/local_prefix/lib
ln -sf /usr/lib64/libgmp.so.10  /work/<proj>/<proj>/<user>/local_prefix/lib/libgmp.so
ln -sf /usr/lib64/libmpfr.so.6  /work/<proj>/<proj>/<user>/local_prefix/lib/libmpfr.so

# 4. Build OceanMesh
cd /work/<proj>/<proj>/<user>/Code/oceanmesh
module load PrgEnv-gnu cray-python boost
source /work/<proj>/<proj>/<user>/fvcom_mesh_venv/bin/activate
OCEANMESH_PREFIX=/work/<proj>/<proj>/<user>/local_prefix pip install -U .
```

---

## Mesh quality plots

Passing `--plot-dir PATH` to either `coarsen` or `run` saves four PNG
files to that directory after mesh generation:

| File | Contents |
|---|---|
| `mesh_quality.png` | Tripcolor map of element quality $q = 2r/R$ |
| `quality_hist.png` | Histogram of quality values |
| `element_size.png` | Spatial distribution of $\sqrt{\text{area}}$ (element size proxy) |
| `depth.png` | Bathymetry map (only when depth data is available) |

These can also be generated from the Python API:

```python
from fvcom_mesh.mesh_quality import plot_mesh_report
import matplotlib
matplotlib.use("Agg")   # non-interactive; omit when using a display

plot_mesh_report(
    mesh.pts, mesh.triangles, mesh.depths,
    save_dir="./plots",
    show=False,
)
```

---

## Dynamic mesh quality assessment

The `quality` command reads an existing set of FVCOM output NetCDF files
and computes time-varying mesh quality metrics (Courant number, wetting/drying
stability, etc.):

```bash
fvcom-mesh quality output_0001.nc output_0002.nc \
    --dep maldives_v0_dep.dat                     \
    --plot-dir ./dynamic_plots
```

See `fvcom_mesh.dynamic_quality` for the full API.

---

## Tests

```bash
cd /path/to/fvcom_mesh
python -m pytest tests/ -v
```

The test suite (92 tests) covers:

| Test file | What is tested |
|---|---|
| `test_water_mask.py` | `BackgroundGrid`, `build_masks`, island removal, width decomposition, skeleton extraction |
| `test_output.py` | FVCOM file format compliance for all four output files |
| `test_mesh_gen.py` | Element quality metric, quality statistics, DistMesh smoke test |
| `test_mesh_modify.py` | Boundary-loop extraction, OBC file reading, polygon building, `PointCloudDEM` |
| `test_oceanmesh_backend.py` | SDF correctness, `Grid` construction, end-to-end mesh generation via OceanMesh |
| `test_dynamic_quality.py` | Dynamic quality metric computations |

---

## Module reference

| Module | Key public symbols |
|---|---|
| `fvcom_mesh.inputs` | `load_domain_boundary`, `load_water_mask`, `load_dem`, `DEM` |
| `fvcom_mesh.water_mask` | `BackgroundGrid`, `build_masks`, `remove_small_islands`, `process_water_mask`, `extract_skeleton_polylines` |
| `fvcom_mesh.constraints` | `Constraint`, `ConstraintType`, `identify_constraints`, `build_mainstreams`, `smooth_all_constraints` |
| `fvcom_mesh.mesh_size` | `build_mesh_size_function`, `MeshSizeFunction`, `gradient_limiting_2d_vectorised` |
| `fvcom_mesh.mesh_gen_1d` | `generate_1d_nodes`, `force_equilibrium_1d`, `postprocess_1d` |
| `fvcom_mesh.mesh_gen_2d` | `distmesh_2d`, `initial_nodes` |
| `fvcom_mesh.mesh_modify` | `extract_boundary_loops`, `read_obc_file`, `build_domain_polygon`, `build_water_polygon`, `extract_obc_linestrings`, `PointCloudDEM`, `coarsen_fvcom_mesh` |
| `fvcom_mesh.oceanmesh_backend` | `generate_mesh_oceanmesh` |
| `fvcom_mesh.obc` | `detect_obc_nodes`, `detect_sponge_nodes` |
| `fvcom_mesh.output_fvcom` | `write_all`, `write_grd`, `write_dep`, `write_obc`, `write_spg`, `interpolate_depths` |
| `fvcom_mesh.mesh_quality` | `element_quality`, `quality_report`, `print_quality_report`, `plot_mesh_report`, `plot_mesh`, `plot_quality_histogram` |
| `fvcom_mesh.dynamic_quality` | Dynamic Courant / stability metrics for FVCOM NetCDF output |
| `fvcom_mesh.core` | `MeshGenerator`, `Mesh`, `load_config`, `validate_config` |
| `fvcom_mesh.utils` | `signed_distance_raster`, `vector_distance_transform`, `rasterize_polygon`, `ensure_ccw`, `boundary_edges`, … |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'shapely'`**
→ Activate the virtual environment: `source /path/to/fvcom_mesh_venv/bin/activate`

**No nodes generated (empty initial point set)**
→ `h_min` is too large relative to the domain, or the domain polygon is in a
geographic CRS (degrees).  Check that `projection` is a projected CRS in metres.

**Out of memory with small h_min**
→ Use `--backend oceanmesh`.  The OceanMesh C++ Delaunay triangulator uses
significantly less memory than the default DistMesh solver at fine resolutions
(h_min ≤ 200 m).  See the *OceanMesh backend* section above for installation
instructions on ARCHER2.

**`ImportError: OceanMesh is required for the 'oceanmesh' backend`**
→ OceanMesh has not been installed.  Follow the ARCHER2 installation steps in
the *OceanMesh backend* section, then re-run with `OCEANMESH_PREFIX=…`.

**All mesh quality very low (q < 0.3)**
→ Increase `max_iter_2d` (e.g. 500) and decrease `tol_2d` (e.g. 0.0001).

**OBC node list is empty**
→ Increase `obc_tol` so more boundary nodes are captured, or supply an
explicit `obc_shapefile`.

**Memory error during gradient limiting**
→ Increase `bg_resolution` (coarser background grid) or decrease the domain
extent.  The background grid is `(domain_width / bg_resolution) × (domain_height / bg_resolution)` cells.

---

## Reference

Kang, D. and Kubatko, E.: ADMESH+: An efficient algorithm for the automated generation of unstructured meshes for coastal ocean models, Geoscientific Model Development, 17, 1603–1627, https://doi.org/10.5194/gmd-17-1603-2024, 2024.
