"""
output_fvcom.py — Write FVCOM mesh files from a triangulated mesh.

Produces the four standard FVCOM mesh input files:
    <prefix>_grd.dat  — node/element connectivity and coordinates
    <prefix>_dep.dat  — nodal water depths
    <prefix>_obc.dat  — open boundary condition nodes
    <prefix>_spg.dat  — sponge layer nodes
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# _grd.dat
# ---------------------------------------------------------------------------

def write_grd(
    pts: NDArray[np.float64],
    triangles: NDArray[np.int_],
    path: str | Path,
) -> None:
    """Write FVCOM _grd.dat connectivity + coordinates file.

    Format (1-based indices, CCW orientation):

        Node Number = N
        Cell Number = M
        <cell_id> <n1> <n3> <n2> <cell_id>     ← note column order
        ...  (M rows)
        <x> <y>                                  ← N rows of node coords

    Parameters
    ----------
    pts : (N, 2) node coordinates in projected metres
    triangles : (M, 3) element connectivity (0-based, CCW)
    path : output file path
    """
    path = Path(path)
    n_nodes = len(pts)
    n_cells = len(triangles)

    with open(path, "w") as fh:
        fh.write(f"Node Number = {n_nodes}\n")
        fh.write(f"Cell Number = {n_cells}\n")
        # Connectivity: CELL# N1 N3 N2 CELL# (1-based, note N1 N3 N2 column order)
        for i, tri in enumerate(triangles):
            n1, n2, n3 = tri[0] + 1, tri[1] + 1, tri[2] + 1
            # FVCOM format stores nodes as N1, N3, N2 (see FVCOM manual)
            fh.write(f"{i + 1:8d} {n1:8d} {n3:8d} {n2:8d} {i + 1:8d}\n")
        # Node coordinates
        for x, y in pts:
            fh.write(f"{x:18.6f} {y:18.6f}\n")


# ---------------------------------------------------------------------------
# _dep.dat
# ---------------------------------------------------------------------------

def write_dep(
    pts: NDArray[np.float64],
    depths: NDArray[np.float64],
    path: str | Path,
) -> None:
    """Write FVCOM _dep.dat depth file.

    Format:
        Node Number = N
        <x> <y> <depth>   ← N rows (depth positive = below sea level)

    Parameters
    ----------
    pts : (N, 2) node coordinates
    depths : (N,) water depth values (positive = below sea level)
    path : output file path
    """
    path = Path(path)
    n_nodes = len(pts)

    with open(path, "w") as fh:
        fh.write(f"Node Number = {n_nodes}\n")
        for (x, y), d in zip(pts, depths):
            fh.write(f"{x:18.6f} {y:18.6f} {d:12.4f}\n")


# ---------------------------------------------------------------------------
# _obc.dat
# ---------------------------------------------------------------------------

def write_obc(
    obc_list: list[tuple[int, int, int]],
    path: str | Path,
) -> None:
    """Write FVCOM _obc.dat open boundary condition file.

    Format:
        OBC Node Number = N
        <obc_idx> <node_idx> <type>   ← N rows (all 1-based)

    Parameters
    ----------
    obc_list : list of (obc_idx, node_idx_1based, obc_type)
    path : output file path
    """
    path = Path(path)
    with open(path, "w") as fh:
        fh.write(f"OBC Node Number = {len(obc_list)}\n")
        for obc_idx, node_idx, obc_type in obc_list:
            fh.write(f"{obc_idx:8d} {node_idx:8d} {obc_type:8d}\n")


# ---------------------------------------------------------------------------
# _spg.dat
# ---------------------------------------------------------------------------

def write_spg(
    spg_list: list[tuple[int, int, float]],
    path: str | Path,
) -> None:
    """Write FVCOM _spg.dat sponge layer file.

    Format mirrors _obc.dat but with a floating-point coefficient column:
        Sponge Node Number = N
        <spg_idx> <node_idx> <coeff>   ← N rows (indices 1-based)

    Parameters
    ----------
    spg_list : list of (spg_idx, node_idx_1based, coeff)
    path : output file path
    """
    path = Path(path)
    with open(path, "w") as fh:
        fh.write(f"Sponge Node Number = {len(spg_list)}\n")
        for spg_idx, node_idx, coeff in spg_list:
            fh.write(f"{spg_idx:8d} {node_idx:8d} {coeff:12.6f}\n")


# ---------------------------------------------------------------------------
# Convenience: interpolate depths from DEM
# ---------------------------------------------------------------------------

def interpolate_depths(
    pts: NDArray[np.float64],
    dem,
    default_depth: float = 1.0,
) -> NDArray[np.float64]:
    """Sample the DEM at mesh node positions.

    Parameters
    ----------
    pts : (N, 2) node coordinates
    dem : ``inputs.DEM`` object
    default_depth : depth assigned to nodes where the DEM is NaN (e.g. land)

    Returns
    -------
    depths : (N,) array (positive = below sea level)
    """
    depths = dem.sample(pts)
    # Replace NaN (no-data / land) with default
    nan_mask = ~np.isfinite(depths)
    depths[nan_mask] = default_depth
    return depths


# ---------------------------------------------------------------------------
# Write all four files at once
# ---------------------------------------------------------------------------

def write_all(
    prefix: str | Path,
    pts: NDArray[np.float64],
    triangles: NDArray[np.int_],
    depths: NDArray[np.float64],
    obc_list: list[tuple[int, int, int]],
    spg_list: Optional[list[tuple[int, int, float]]] = None,
) -> dict[str, Path]:
    """Write all four FVCOM mesh files with the given *prefix*.

    Parameters
    ----------
    prefix : filename prefix (e.g. ``"my_domain"`` or ``Path("output/my_domain")``)
    pts, triangles, depths, obc_list, spg_list : mesh data

    Returns
    -------
    dict mapping file role to Path for each written file
    """
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    grd_path = Path(str(prefix) + "_grd.dat")
    write_grd(pts, triangles, grd_path)
    paths["grd"] = grd_path

    dep_path = Path(str(prefix) + "_dep.dat")
    write_dep(pts, depths, dep_path)
    paths["dep"] = dep_path

    obc_path = Path(str(prefix) + "_obc.dat")
    write_obc(obc_list, obc_path)
    paths["obc"] = obc_path

    spg_path = Path(str(prefix) + "_spg.dat")
    if spg_list is None:
        spg_list = []
    write_spg(spg_list, spg_path)
    paths["spg"] = spg_path

    return paths
