"""
mesh_size.py — Mesh-size function construction and 2-D gradient limiting.

Implements Sect. 4.2–4.3 of Kang & Kubatko (2024), GMD.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .constraints import Constraint, ConstraintType
from .water_mask import BackgroundGrid
from .utils import resample_polyline


# ---------------------------------------------------------------------------
# Size function from curvature on constraints (Eq. 34 of Kang & Kubatko 2024)
# ---------------------------------------------------------------------------

def initial_size_on_constraints(
    constraints: list[Constraint],
    K: float,
    h_min: float,
    h_max: float,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Compute the desired mesh size at every constraint node from curvature.

    h_0(s) = clamp(1 / (K * κ(s)), h_min, h_max)   (Eq. 34)

    Parameters
    ----------
    K : constant controlling curvature-to-size ratio (typically 1–5)
    h_min, h_max : minimum and maximum mesh size (metres)

    Returns
    -------
    pts_list : list of (N_i, 2) coordinate arrays (smoothed constraint pts)
    size_list : list of (N_i,) mesh-size arrays
    """
    pts_list: list[NDArray[np.float64]] = []
    size_list: list[NDArray[np.float64]] = []

    for c in constraints:
        pts = c.smoothed_pts if c.smoothed_pts is not None else c.pts
        kappa = c.curvature if c.curvature is not None else np.zeros(len(pts))

        # Ensure curvature array matches pts
        if len(kappa) != len(pts):
            kappa = np.interp(
                np.linspace(0, 1, len(pts)),
                np.linspace(0, 1, len(kappa)),
                kappa,
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            h0 = np.where(kappa > 1e-12, 1.0 / (K * kappa), h_max)
        h0 = np.clip(h0, h_min, h_max)

        pts_list.append(pts)
        size_list.append(h0)

    return pts_list, size_list


# ---------------------------------------------------------------------------
# Background grid size initialisation
# ---------------------------------------------------------------------------

def build_size_grid(
    grid: BackgroundGrid,
    pts_list: list[NDArray[np.float64]],
    size_list: list[NDArray[np.float64]],
    h_max: float,
) -> NDArray[np.float64]:
    """Initialise the mesh-size field on the background grid.

    Assigns h_max everywhere, then sets cells near constraint nodes to the
    minimum of their current value and the constraint node's desired size.

    Parameters
    ----------
    grid : BackgroundGrid
    pts_list, size_list : output of initial_size_on_constraints
    h_max : fallback size away from constraints

    Returns
    -------
    h_grid : (ny, nx) float array of initial mesh sizes
    """
    h_grid = np.full(grid.shape, h_max, dtype=np.float64)

    for pts, sizes in zip(pts_list, size_list):
        if len(pts) == 0:
            continue
        i_idx, j_idx = grid.xy_to_ij(pts)
        for ii, jj, hv in zip(i_idx, j_idx, sizes):
            if h_grid[ii, jj] > hv:
                h_grid[ii, jj] = hv

    return h_grid


# ---------------------------------------------------------------------------
# 2-D gradient limiting (Eikonal / upwind scheme, Sect. 4.3)
# ---------------------------------------------------------------------------

def gradient_limiting_2d(
    h_grid: NDArray[np.float64],
    g: float,
    dx: float,
    dy: float,
    *,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> NDArray[np.float64]:
    """Limit the spatial gradient of the mesh-size field.

    Solves the static Hamilton–Jacobi equation

        |∇h| ≤ g     everywhere

    via a fast-marching / upwind-sweep scheme starting from the constraint
    seeds with the smallest size values.  Implements Sect. 4.3 of
    Kang & Kubatko (2024), which uses the fast-sweeping approach of
    Sethian (1999).

    Parameters
    ----------
    h_grid : initial mesh-size grid (output of build_size_grid)
    g : maximum allowable gradient (typically 0.2–0.4)
    dx, dy : cell spacing in metres
    max_iter : maximum number of Gauss–Seidel sweeps
    tol : convergence tolerance (maximum change per sweep)

    Returns
    -------
    h_limited : gradient-limited mesh-size grid, shape = h_grid.shape
    """
    h = h_grid.copy()
    ny, nx = h.shape
    dx2, dy2 = dx * g, dy * g

    for _ in range(max_iter):
        max_change = 0.0

        # Forward sweep (south-west to north-east)
        for i in range(1, ny):
            for j in range(1, nx):
                h_trial = _upwind_update(h, i, j, dx2, dy2)
                if h_trial < h[i, j]:
                    max_change = max(max_change, h[i, j] - h_trial)
                    h[i, j] = h_trial

        # Backward sweep (north-east to south-west)
        for i in range(ny - 2, -1, -1):
            for j in range(nx - 2, -1, -1):
                h_trial = _upwind_update(h, i, j, dx2, dy2)
                if h_trial < h[i, j]:
                    max_change = max(max_change, h[i, j] - h_trial)
                    h[i, j] = h_trial

        if max_change < tol:
            break

    return h


def _upwind_update(
    h: NDArray[np.float64],
    i: int,
    j: int,
    dx2: float,
    dy2: float,
) -> float:
    """Single upwind update step for the gradient-limiting PDE."""
    ny, nx = h.shape
    # Take the minimum neighbour in each direction
    hx = min(
        h[i, j - 1] if j > 0 else h[i, j],
        h[i, j + 1] if j < nx - 1 else h[i, j],
    )
    hy = min(
        h[i - 1, j] if i > 0 else h[i, j],
        h[i + 1, j] if i < ny - 1 else h[i, j],
    )
    # Solve (h - hx)^2/dx2^2 + (h - hy)^2/dy2^2 = 1 with h > hx, h > hy
    a = 1.0 / dx2**2 + 1.0 / dy2**2
    b = -2.0 * (hx / dx2**2 + hy / dy2**2)
    c = hx**2 / dx2**2 + hy**2 / dy2**2 - 1.0
    disc = b**2 - 4 * a * c
    if disc < 0:
        # Use the direction with smaller h only
        return min(hx + dx2, hy + dy2)
    h_new = (-b + np.sqrt(disc)) / (2 * a)
    return float(h_new)


def gradient_limiting_2d_vectorised(
    h_grid: NDArray[np.float64],
    g: float,
    dx: float,
    dy: float,
    *,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> NDArray[np.float64]:
    """Vectorised (NumPy) gradient limiting — faster for large grids.

    Uses alternating forward/backward passes over rows and columns.
    """
    h = h_grid.copy()
    dx_g = dx * g
    dy_g = dy * g
    ny, nx = h.shape

    for it in range(max_iter):
        h_old = h.copy()

        # Propagate from small values: take rowwise min with neighbour + dy_g
        for i in range(1, ny):
            h[i, :] = np.minimum(h[i, :], h[i - 1, :] + dy_g)
        for i in range(ny - 2, -1, -1):
            h[i, :] = np.minimum(h[i, :], h[i + 1, :] + dy_g)
        for j in range(1, nx):
            h[:, j] = np.minimum(h[:, j], h[:, j - 1] + dx_g)
        for j in range(nx - 2, -1, -1):
            h[:, j] = np.minimum(h[:, j], h[:, j + 1] + dx_g)

        if np.max(np.abs(h - h_old)) < tol:
            break

    return h


# ---------------------------------------------------------------------------
# MeshSizeFunction — evaluator
# ---------------------------------------------------------------------------

class MeshSizeFunction:
    """Bilinear interpolator of a gridded mesh-size field.

    Parameters
    ----------
    h_grid : (ny, nx) array of mesh sizes
    grid : BackgroundGrid corresponding to h_grid
    h_min, h_max : hard clamps applied after interpolation
    """

    def __init__(
        self,
        h_grid: NDArray[np.float64],
        grid: BackgroundGrid,
        h_min: float,
        h_max: float,
    ):
        from scipy.interpolate import RegularGridInterpolator
        self.h_min = h_min
        self.h_max = h_max
        self._interp = RegularGridInterpolator(
            (grid.y, grid.x),
            h_grid,
            method="linear",
            bounds_error=False,
            fill_value=h_max,
        )

    def evaluate(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return mesh size at (x, y) points.

        Parameters
        ----------
        pts : (N, 2) array of (x, y) coordinates

        Returns
        -------
        h : (N,) array of mesh sizes
        """
        # RegularGridInterpolator expects (y, x) ordering
        h = self._interp(pts[:, ::-1])
        return np.clip(h, self.h_min, self.h_max)

    def __call__(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.evaluate(pts)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_mesh_size_function(
    constraints: list[Constraint],
    grid: BackgroundGrid,
    K: float,
    h_min: float,
    h_max: float,
    g: float = 0.3,
    *,
    max_iter: int = 500,
    use_fast: bool = True,
) -> MeshSizeFunction:
    """One-shot builder for the mesh-size function.

    Parameters
    ----------
    constraints : list of Constraint with smoothed_pts / curvature set
    grid : BackgroundGrid on which to build the size field
    K : curvature constant (Eq. 34)
    h_min, h_max : size clamps (metres)
    g : maximum gradient (typically 0.2–0.4)
    max_iter : gradient-limiting iterations
    use_fast : use vectorised gradient limiting (recommended; much faster)

    Returns
    -------
    MeshSizeFunction ready to evaluate at arbitrary coordinates
    """
    pts_list, size_list = initial_size_on_constraints(constraints, K, h_min, h_max)
    h_grid = build_size_grid(grid, pts_list, size_list, h_max)

    if use_fast:
        h_limited = gradient_limiting_2d_vectorised(
            h_grid, g, grid.dx, grid.dy, max_iter=max_iter
        )
    else:
        h_limited = gradient_limiting_2d(
            h_grid, g, grid.dx, grid.dy, max_iter=max_iter
        )

    return MeshSizeFunction(h_limited, grid, h_min, h_max)
