"""
cli.py — Command-line interface for fvcom-mesh.

Usage:
    fvcom-mesh run config.yml [--output-dir OUTPUT_DIR]
    fvcom-mesh check config.yml
"""

from __future__ import annotations

import sys
from pathlib import Path

import click


@click.group()
def main() -> None:
    """fvcom-mesh: ADMESH+-based unstructured mesh generator for FVCOM."""


@main.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output directory from config.",
)
@click.option("--verbose", "-v", is_flag=True, help="Print detailed progress.")
@click.option(
    "--backend",
    type=click.Choice(["distmesh", "oceanmesh"], case_sensitive=False),
    default=None,
    help="Triangulation backend (overrides 'backend' in config).",
)
@click.option(
    "--plot-dir", "plot_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Save mesh and quality plots as PNG files in this directory.",
)
def run(config: Path, output_dir: Path | None, verbose: bool, backend: str | None, plot_dir: Path | None) -> None:
    """Generate a mesh from a YAML configuration file and write FVCOM output."""
    from .core import MeshGenerator

    mg = MeshGenerator.from_config(config, output_dir_override=output_dir)
    if backend is not None:
        mg.config["backend"] = backend.lower()
    if verbose:
        click.echo(f"Loaded config: {config}")
        click.echo(f"Output prefix: {mg.config.get('output_prefix', 'mesh')}")
        click.echo(f"Triangulation backend: {mg.config.get('backend', 'distmesh')}")

    click.echo("Running mesh generation pipeline …")
    mesh = mg.run()

    if verbose:
        from .mesh_quality import print_quality_report, element_quality
        q = element_quality(mesh.triangles, mesh.pts)
        print_quality_report(q)

    click.echo(f"Done.  Wrote {len(mesh.triangles)} elements, {len(mesh.pts)} nodes.")

    if plot_dir is not None:
        from .mesh_quality import plot_mesh_report
        import matplotlib
        matplotlib.use("Agg")
        depths = mesh.depths if mesh.depths is not None and len(mesh.depths) > 0 else None
        plot_mesh_report(
            mesh.pts, mesh.triangles, depths,
            save_dir=plot_dir, show=False,
        )
        click.echo(f"Saved plots: {plot_dir}")


@main.command()
@click.argument("config", type=click.Path(exists=True, path_type=Path))
def check(config: Path) -> None:
    """Validate the configuration file and report any issues."""
    from .core import load_config, validate_config

    cfg = load_config(config)
    issues = validate_config(cfg)
    if issues:
        click.echo("Configuration issues found:", err=True)
        for issue in issues:
            click.echo(f"  - {issue}", err=True)
        sys.exit(1)
    else:
        click.echo("Configuration OK.")


@main.command()
@click.argument("grd", type=click.Path(exists=True, path_type=Path),
               metavar="GRD_FILE")
@click.option("--dep", "dep_path", type=click.Path(path_type=Path),
              default=None,
              help="FVCOM _dep.dat depth file (auto-detected if omitted).")
@click.option("--output", "-o", "nc_paths",
              type=click.Path(exists=True, path_type=Path),
              multiple=True,
              help="FVCOM NetCDF output file(s).  Can be repeated.")
@click.option("--dt", type=float, default=None,
              help="Model external time step [s].  Required for Courant numbers.")
@click.option("--period", "period_specs", multiple=True, default=("M2",),
              metavar="NAME[:SECONDS]",
              show_default=True,
              help="Tidal constituent(s) to evaluate.  Use a built-in name "
                   "(M2, S2, N2, K1, O1) or NAME:PERIOD_IN_SECONDS.")
@click.option("--max-timesteps", type=int, default=None,
              help="Limit NetCDF reading to the first N timesteps.")
@click.option("--plot-dir", type=click.Path(path_type=Path), default=None,
              help="Directory to save per-metric PNG plots.")
@click.option("--save-csv", type=click.Path(path_type=Path), default=None,
              help="Write per-element metric table to a CSV file.")
def dynamics(
    grd: Path,
    dep_path: Path | None,
    nc_paths: tuple[Path, ...],
    dt: float | None,
    period_specs: tuple[str, ...],
    max_timesteps: int | None,
    plot_dir: Path | None,
    save_csv: Path | None,
) -> None:
    """Compute dynamic mesh quality metrics (Vitousek & Fringer 2011).

    GRD_FILE is the FVCOM _grd.dat grid file.  If a matching _dep.dat
    file is present in the same directory it will be used automatically.

    Examples
    --------
    Mesh-only (no simulation output, no Courant numbers):

        fvcom-mesh dynamics maldives_v0_grd.dat

    With simulation output and explicit time step:

        fvcom-mesh dynamics maldives_v0_grd.dat \\
            --dep maldives_v0_dep.dat          \\
            --output maldives_v0_0001.nc       \\
            --dt 300                           \\
            --period M2 --period K1            \\
            --plot-dir ./dynamic_plots
    """
    from .dynamic_quality import DynamicQuality, TIDAL_PERIODS

    # Auto-detect dep file alongside grd file if not given
    if dep_path is None:
        candidate = grd.with_name(grd.name.replace("_grd.dat", "_dep.dat"))
        if candidate.exists():
            dep_path = candidate

    # Parse tidal period specifications
    periods: dict[str, float] = {}
    for spec in period_specs:
        if ":" in spec:
            name, val = spec.split(":", 1)
            periods[name.upper()] = float(val)
        else:
            name = spec.upper()
            if name not in TIDAL_PERIODS:
                raise click.BadParameter(
                    f"Unknown constituent '{name}'. "
                    f"Known: {', '.join(TIDAL_PERIODS)}. "
                    "Use NAME:PERIOD_IN_SECONDS for custom values."
                )
            periods[name] = TIDAL_PERIODS[name]

    click.echo(f"Reading mesh: {grd}")
    dq = DynamicQuality.from_files(grd, dep_path)

    for nc in nc_paths:
        click.echo(f"Loading output: {nc}")
        dq.load_output(nc, max_timesteps=max_timesteps)

    dq.report(dt=dt, periods=periods)

    if save_csv is not None:
        import csv
        m = dq._metrics
        keys = sorted(m.keys())
        with open(save_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["element_id"] + keys)
            for i in range(len(m["element_size"])):
                writer.writerow([i] + [float(m[k][i]) for k in keys])
        click.echo(f"Saved metrics CSV: {save_csv}")

    if plot_dir is not None:
        dq.compute(dt=dt, periods=periods)  # ensure computed
        dq.plot_all(save_dir=plot_dir, show=False)
        click.echo(f"Saved plots: {plot_dir}")


@main.command()
@click.argument("grd", type=click.Path(exists=True, path_type=Path),
               metavar="GRD_FILE")
@click.option("--dep", "dep_path", type=click.Path(path_type=Path), default=None,
              help="FVCOM _dep.dat depth file (auto-detected if omitted).")
@click.option("--obc", "obc_path", type=click.Path(path_type=Path), default=None,
              help="FVCOM _obc.dat open-boundary file (auto-detected if omitted).")
@click.option("--output-prefix", "-p", required=True,
              help="File-name prefix for new mesh output files.")
@click.option("--projection", required=True,
              help="Projected CRS for the mesh (e.g. EPSG:32643).")
@click.option("--h-min", type=float, required=True,
              help="Minimum element size [m].")
@click.option("--h-max", type=float, required=True,
              help="Maximum element size [m].")
@click.option("--delta-w", type=float, required=True,
              help="Minimum channel width threshold [m].")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None,
              help="Override output directory.")
@click.option("--boundary-simplify-tol", type=float, default=None,
              help="Douglas–Peucker tolerance for boundary simplification [m]. "
                   "Defaults to h_min/4.")
@click.option("--min-island-area", type=float, default=None,
              help="Islands smaller than this [m²] are removed. "
                   "Defaults to h_min².")
@click.option("--max-iter-2d", type=int, default=None,
              help="Maximum meshing iterations (default: 200 for distmesh, 100 for oceanmesh).")
@click.option("--gradient-limit", type=float, default=None,
              help="Size-function gradient limit (default 0.3).")
@click.option("--backend", type=click.Choice(["distmesh", "oceanmesh"], case_sensitive=False),
              default="distmesh", show_default=True,
              help=(
                  "Triangulation backend. 'distmesh' (default) is the built-in "
                  "DistMesh force-iteration solver.  'oceanmesh' uses the OceanMesh "
                  "C++ Delaunay triangulator which uses significantly less memory and "
                  "is recommended when h_min ≤ 200 m."
              ))
@click.option("--grid-resolution", type=float, default=None,
              help="Background sizing-grid cell size [m] for the OceanMesh backend "
                   "(auto-selected if not specified).")
@click.option("--bg-resolution", type=float, default=None,
              help="Water-mask background grid cell size [m].  "
                   "Defaults to max(h_min/4, delta_w/4).  "
                   "Reduce to capture finer channel detail; increase to save memory.")
@click.option("--plot-dir", type=click.Path(path_type=Path), default=None,
              help="Save quality/geometry plots (PNG) to this directory.")
@click.option("--verbose", "-v", is_flag=True, help="Print detailed progress.")
def coarsen(
    grd: Path,
    dep_path: Path | None,
    obc_path: Path | None,
    output_prefix: str,
    projection: str,
    h_min: float,
    h_max: float,
    delta_w: float,
    output_dir: Path | None,
    boundary_simplify_tol: float | None,
    min_island_area: float | None,
    max_iter_2d: int | None,
    gradient_limit: float | None,
    backend: str,
    grid_resolution: float | None,
    bg_resolution: float | None,
    plot_dir: Path | None,
    verbose: bool,
) -> None:
    """Generate a modified (e.g. coarser) mesh from an existing FVCOM mesh.

    GRD_FILE is an existing FVCOM _grd.dat file.  Matching _dep.dat and
    _obc.dat files are auto-detected from the same directory.  No shapefiles
    are required — coastline and OBC geometry are extracted directly from the
    existing mesh.

    Examples
    --------
    Basic coarsening (h_min 500 m, h_max 10 km):

        fvcom-mesh coarsen maldives_v0_grd.dat       \\
            --projection EPSG:32643                  \\
            --h-min 500 --h-max 10000 --delta-w 1000 \\
            --output-prefix maldives_v0_coarse

    With explicit dep/obc paths and custom simplification tolerance:

        fvcom-mesh coarsen maldives_v0_grd.dat       \\
            --dep maldives_v0_dep.dat                \\
            --obc maldives_v0_obc.dat                \\
            --projection EPSG:32643                  \\
            --h-min 500 --h-max 10000 --delta-w 1000 \\
            --boundary-simplify-tol 200              \\
            --output-prefix maldives_v0_coarse
    """
    import logging

    if verbose:
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s: %(message)s")

    from .core import MeshGenerator

    extra: dict = {}
    if max_iter_2d is not None:
        extra["max_iter_2d"] = max_iter_2d
    if gradient_limit is not None:
        extra["gradient_limit"] = gradient_limit
    if backend != "distmesh":
        extra["backend"] = backend.lower()
    if grid_resolution is not None:
        extra["grid_resolution"] = grid_resolution
    if bg_resolution is not None:
        extra["bg_resolution"] = bg_resolution
    extra["backend"] = backend.lower()

    click.echo(f"Reading existing mesh: {grd}")
    mg = MeshGenerator.from_fvcom_mesh(
        grd_path=grd,
        dep_path=dep_path,
        obc_path=obc_path,
        output_prefix=output_prefix,
        h_min=h_min,
        h_max=h_max,
        delta_w=delta_w,
        projection=projection,
        output_dir_override=output_dir,
        boundary_simplify_tol=boundary_simplify_tol,
        min_island_area=min_island_area,
        **extra,
    )

    click.echo("Running mesh generation pipeline …")
    click.echo(f"Triangulation backend: {backend}")
    mesh = mg.run()
    click.echo(
        f"Done.  Wrote {len(mesh.triangles)} elements, {len(mesh.pts)} nodes."
    )

    if plot_dir is not None:
        from .mesh_quality import plot_mesh_report
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for HPC/batch
        depths = mesh.depths if mesh.depths is not None and len(mesh.depths) > 0 else None
        plot_mesh_report(
            mesh.pts, mesh.triangles, depths,
            save_dir=plot_dir, show=False,
        )
        click.echo(f"Saved plots: {plot_dir}")


if __name__ == "__main__":
    main()
