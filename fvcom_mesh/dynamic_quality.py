"""
dynamic_quality.py — Dynamic mesh quality metrics for FVCOM meshes.

Implements the wavelength-resolution and numerical-dispersion metrics
from Vitousek & Fringer (2011):

    "Physical vs. numerical dispersion in nonhydrostatic ocean modeling"
    Ocean Modelling, 40(1-2), 72–86.
    https://doi.org/10.1016/j.ocemod.2011.07.002

The central idea is that a second-order accurate spatial discretisation
produces a numerical phase-speed ratio

    c_num / c  =  sin(π / N_λ) / (π / N_λ)

where N_λ = λ / L_e is the number of element widths per wavelength.
As N_λ decreases (coarser mesh), c_num / c drops below 1 and the mesh
introduces spurious numerical dispersion of the same functional form as
physical nonhydrostatic dispersion.

Metrics computed per element (all dimensionless unless noted):

    element_size          sqrt of element area [m]
    element_depth         mean nodal water depth [m]
    n_lambda_<name>       elements per tidal wavelength
    dispersion_ratio_<name>  numerical phase-speed ratio c_num / c ∈ (0, 1]
    cr_wave_<name>        barotropic wave Courant number c·dt / L_e
    cr_flow               advective Courant number |U|·dt / L_e
    froude                depth-averaged Froude number |U| / sqrt(gH)
    relative_amplitude    tidal amplitude relative to depth |ζ|_max / H
    tidal_excursion_<name>  tidal excursion distance / L_e  (U·T / L_e)

Mesh + depth are read from the FVCOM _grd.dat / _dep.dat files using the
same format conventions as PyFVCOM (pmlmodelling/pyfvcom).  Simulation
statistics (zeta, ua, va) are read from a FVCOM NetCDF output file.

Grid file format (as defined by PyFVCOM / FVCOM):
    Line 1  : Node Number = N
    Line 2  : Cell Number = M
    5-field lines: cell_id  N1  N2  N3  cell_id   (1-based indices, CW)
    4-field lines: node_id  X   Y   Z              (Z = bathymetric depth)

If the GRD file uses the simplified 2-field node format (X Y, no depth),
``dep_path`` must be supplied.  Dep-file format:
    Line 1  : Node Number = N
    3-field lines: X  Y  DEPTH
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Physical constants and tidal constituent periods
# ---------------------------------------------------------------------------

G: float = 9.81  # m s-2

#: Common tidal constituent periods in seconds.
TIDAL_PERIODS: dict[str, float] = {
    "M2": 44712.0,   # principal lunar semi-diurnal
    "S2": 43200.0,   # principal solar semi-diurnal
    "N2": 45570.0,   # larger lunar elliptic semi-diurnal
    "K1": 86164.1,   # lunar-solar declinational diurnal
    "O1": 92949.6,   # principal lunar diurnal
}

# ---------------------------------------------------------------------------
# Grid / mesh file readers
# ---------------------------------------------------------------------------

def read_fvcom_mesh(
    grd_path: str | Path,
    dep_path: Optional[str | Path] = None,
) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.float64]]:
    """Read an FVCOM unstructured grid file.

    Follows the PyFVCOM (pmlmodelling/pyfvcom) file-format conventions:

    * **5-field lines** → element connectivity:
      ``cell_id  N1  N2  N3  cell_id``  (1-based, stored CW in FVCOM)
    * **4-field lines** → node coordinates + depth:
      ``node_id  X  Y  Z``
    * **2-field lines** → node coordinates only:
      ``X  Y``  (depth read from ``dep_path``)

    Parameters
    ----------
    grd_path : str or Path
        Path to the FVCOM ``_grd.dat`` file.
    dep_path : str or Path, optional
        Path to the FVCOM ``_dep.dat`` file.  Required only when the GRD
        file uses 2-field node rows (no embedded depth).

    Returns
    -------
    pts : (N, 2) node coordinates [m]
    triangles : (T, 3) element connectivity (0-based)
    depths : (N,) water depth at nodes (positive = below sea level)
    """
    with open(grd_path) as fh:
        lines = fh.readlines()[2:]  # skip "Node Number = N" / "Cell Number = M"

    triangles: list[list[int]] = []
    x_list: list[float] = []
    y_list: list[float] = []
    z_list: list[float] = []

    for line in lines:
        parts = line.strip().split()
        n = len(parts)
        if n == 5:
            # Element row: cell_id N1 N2 N3 cell_id
            triangles.append([int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1])
        elif n == 4:
            # Node row (standard FVCOM): node_id X Y Z
            x_list.append(float(parts[1]))
            y_list.append(float(parts[2]))
            z_list.append(float(parts[3]))
        elif n == 2:
            # Node row (simplified): X Y
            x_list.append(float(parts[0]))
            y_list.append(float(parts[1]))

    pts = np.column_stack([x_list, y_list])
    tri_arr = np.asarray(triangles, dtype=int)

    if z_list:
        depths = np.asarray(z_list)
    elif dep_path is not None:
        depths = _read_dep(dep_path)
    else:
        warnings.warn(
            "No depth information found in GRD file and dep_path not supplied. "
            "Setting depths to 1.0 m everywhere.  Pass dep_path for correct results.",
            stacklevel=2,
        )
        depths = np.ones(len(pts))

    return pts, tri_arr, depths


def _read_dep(dep_path: str | Path) -> NDArray[np.float64]:
    """Read nodal depths from a FVCOM ``_dep.dat`` file.

    Handles both the standard format (3-field: ``X Y DEPTH``) and the
    format with a leading node index (4-field: ``node_id X Y DEPTH``).
    """
    depths: list[float] = []
    with open(dep_path) as fh:
        for line in fh.readlines()[1:]:  # skip "Node Number = N"
            parts = line.strip().split()
            if len(parts) == 3:
                depths.append(float(parts[2]))
            elif len(parts) == 4:
                depths.append(float(parts[3]))
    return np.asarray(depths)


def read_fvcom_output(
    nc_path: str | Path,
    max_timesteps: Optional[int] = None,
) -> dict:
    """Read simulation statistics from a FVCOM NetCDF output file.

    Loads only the variables needed for dynamic quality metrics.  All
    time-varying fields are reduced to per-node / per-element statistics
    (max absolute value and temporal mean) to keep memory usage bounded.

    Parameters
    ----------
    nc_path : str or Path
        Path to a FVCOM ``_NNNN.nc`` output file.
    max_timesteps : int, optional
        If given, only the first *max_timesteps* time indices are read.

    Returns
    -------
    dict with keys:

    ``x``, ``y`` : (N,) node coordinates (projected, metres or lon/lat)
    ``h``        : (N,) static water depth
    ``nv``       : (T, 3) element connectivity (0-based)
    ``zeta_max`` : (N,) max |ζ| over time at each node
    ``zeta_mean``: (N,) temporal mean ζ at each node
    ``ua_max``   : (T,) max |u_a| over time at each element
    ``va_max``   : (T,) max |v_a| over time at each element
    ``dt``       : output interval in seconds (inferred from Itime2)
    """
    try:
        import netCDF4 as nc4  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "netCDF4 is required to read FVCOM output files. "
            "Install it with: pip install netCDF4"
        ) from exc

    ds = nc4.Dataset(nc_path, "r")
    try:
        sl = slice(None) if max_timesteps is None else slice(0, max_timesteps)

        x = np.asarray(ds.variables["x"][:], dtype=float)
        y = np.asarray(ds.variables["y"][:], dtype=float)
        h = np.asarray(ds.variables["h"][:], dtype=float)
        # nv: (3, nele) 1-based → (nele, 3) 0-based
        nv = np.asarray(ds.variables["nv"][:], dtype=int).T - 1

        zeta = np.asarray(ds.variables["zeta"][sl], dtype=float)  # (nt, N)
        ua = np.asarray(ds.variables["ua"][sl], dtype=float)       # (nt, nele)
        va = np.asarray(ds.variables["va"][sl], dtype=float)       # (nt, nele)

        # Infer output interval from Itime2 (milliseconds)
        itime2 = np.asarray(ds.variables["Itime2"][:], dtype=float)
        if len(itime2) > 1:
            dt_out = float(np.median(np.diff(itime2))) / 1000.0  # → seconds
        else:
            dt_out = 3600.0  # default 1 h

    finally:
        ds.close()

    return {
        "x": x,
        "y": y,
        "h": h,
        "nv": nv,
        "zeta_max": np.max(np.abs(zeta), axis=0),
        "zeta_mean": np.mean(zeta, axis=0),
        "ua_max": np.max(np.abs(ua), axis=0),
        "va_max": np.max(np.abs(va), axis=0),
        "dt": dt_out,
    }


# ---------------------------------------------------------------------------
# Element-level geometric helpers
# ---------------------------------------------------------------------------

def element_area(
    pts: NDArray[np.float64],
    triangles: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Signed area of each triangle (always returned as positive).

    Parameters
    ----------
    pts : (N, 2)
    triangles : (T, 3) 0-based

    Returns
    -------
    area : (T,) in the same units as ``pts`` squared
    """
    a = pts[triangles[:, 0]]
    b = pts[triangles[:, 1]]
    c = pts[triangles[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) \
          - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.abs(cross)


def element_representative_size(
    pts: NDArray[np.float64],
    triangles: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Representative element width  L_e = sqrt(A_e).

    Returns
    -------
    le : (T,) [m]
    """
    return np.sqrt(element_area(pts, triangles))


def element_depth(
    triangles: NDArray[np.int_],
    depths: NDArray[np.float64],
    zeta_mean: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """Mean water depth per element, optionally corrected for mean sea level.

    Parameters
    ----------
    triangles : (T, 3) 0-based
    depths : (N,) static bathymetric depth (positive = below sea level)
    zeta_mean : (N,) temporal mean surface elevation, optional

    Returns
    -------
    he : (T,) effective mean depth [m], clipped to a minimum of 0.1 m
    """
    he = np.mean(depths[triangles], axis=1)
    if zeta_mean is not None:
        he = he + np.mean(zeta_mean[triangles], axis=1)
    return np.maximum(he, 0.1)


# ---------------------------------------------------------------------------
# Dynamic quality metrics
# ---------------------------------------------------------------------------

def wavelength_resolution(
    le: NDArray[np.float64],
    he: NDArray[np.float64],
    period: float = TIDAL_PERIODS["M2"],
) -> NDArray[np.float64]:
    """Number of element widths per tidal wavelength  N_λ = λ / L_e.

    The long-wave phase speed  c = sqrt(g · H_e)  gives wavelength
    λ = c · T  for a constituent with period T.

    Well-resolved dynamics require N_λ ≳ 20–25 (Walters 1992;
    Vitousek & Fringer 2011).

    Parameters
    ----------
    le : (T,) element representative sizes [m]
    he : (T,) element depths [m]
    period : float
        Tidal constituent period [s].  Defaults to M2 (44712 s).

    Returns
    -------
    n_lambda : (T,)
    """
    c = np.sqrt(G * he)
    lam = c * period            # wavelength [m]
    return lam / np.maximum(le, 1e-6)


def numerical_dispersion_ratio(
    n_lambda: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Numerical phase-speed ratio from Vitousek & Fringer (2011).

    For a second-order accurate spatial discretisation the numerical
    phase speed of a wave resolved by N_λ grid points per wavelength is

        c_num / c  =  sin(π / N_λ) / (π / N_λ)

    Values close to 1 indicate well-resolved dynamics.  The dispersion
    error  ε_d = 1 − c_num / c  exceeds 1 % at N_λ < 18 and 10 % at
    N_λ < 6.

    Parameters
    ----------
    n_lambda : (T,) elements per wavelength (≥ 2)

    Returns
    -------
    ratio : (T,) in (0, 1]
    """
    n_safe = np.maximum(n_lambda, 2.0)
    theta = np.pi / n_safe
    return np.where(n_lambda > 1e3, 1.0 - theta ** 2 / 6.0, np.sin(theta) / theta)


def barotropic_courant(
    le: NDArray[np.float64],
    he: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Barotropic wave Courant number  Cr = sqrt(g·H) · dt / L_e.

    Parameters
    ----------
    le  : (T,) element sizes [m]
    he  : (T,) element depths [m]
    dt  : model time step [s]

    Returns
    -------
    cr : (T,)
    """
    c = np.sqrt(G * he)
    return c * dt / np.maximum(le, 1e-6)


def velocity_courant(
    le: NDArray[np.float64],
    ua_max: NDArray[np.float64],
    va_max: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Advective Courant number  Cr_u = |U|_max · dt / L_e.

    Parameters
    ----------
    le     : (T,) element sizes [m]
    ua_max : (T,) peak depth-averaged eastward velocity [m/s]
    va_max : (T,) peak depth-averaged northward velocity [m/s]
    dt     : time step [s]

    Returns
    -------
    cr_u : (T,)
    """
    u_mag = np.hypot(ua_max, va_max)
    return u_mag * dt / np.maximum(le, 1e-6)


def froude_number(
    he: NDArray[np.float64],
    ua_max: NDArray[np.float64],
    va_max: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Depth-averaged Froude number  Fr = |U|_max / sqrt(g·H).

    Parameters
    ----------
    he     : (T,) element depths [m]
    ua_max : (T,) peak eastward velocity [m/s]
    va_max : (T,) peak northward velocity [m/s]

    Returns
    -------
    fr : (T,)
    """
    u_mag = np.hypot(ua_max, va_max)
    c = np.sqrt(G * he)
    return u_mag / np.maximum(c, 1e-6)


def relative_amplitude(
    triangles: NDArray[np.int_],
    he: NDArray[np.float64],
    zeta_max: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Relative tidal amplitude  A/H = max|ζ| / H_e.

    Values ≪ 1 justify the linearised shallow-water equations.

    Parameters
    ----------
    triangles : (T, 3) 0-based
    he        : (T,) element depths [m]
    zeta_max  : (N,) max |ζ| at each node [m]

    Returns
    -------
    a_over_h : (T,)
    """
    zeta_elem = np.mean(zeta_max[triangles], axis=1)
    return zeta_elem / np.maximum(he, 0.1)


def tidal_excursion(
    le: NDArray[np.float64],
    ua_max: NDArray[np.float64],
    va_max: NDArray[np.float64],
    period: float = TIDAL_PERIODS["M2"],
) -> NDArray[np.float64]:
    """Tidal excursion distance normalised by element size  U·T / L_e.

    The tidal excursion E = U · T / π (for a sinusoidal current) measures
    how far a fluid parcel travels per tidal cycle.  When E / L_e ≫ 1,
    advective nonlinearity is important and the mesh may need refinement.

    Parameters
    ----------
    le     : (T,) element sizes [m]
    ua_max : (T,) peak eastward velocity [m/s]
    va_max : (T,) peak northward velocity [m/s]
    period : float
        Tidal period [s].  Default: M2.

    Returns
    -------
    excursion_ratio : (T,)
    """
    u_mag = np.hypot(ua_max, va_max)
    excursion = u_mag * period / np.pi
    return excursion / np.maximum(le, 1e-6)


# ---------------------------------------------------------------------------
# Summary statistics helpers
# ---------------------------------------------------------------------------

def _metric_stats(arr: NDArray[np.float64], name: str) -> dict:
    """Return a dict of summary statistics for a per-element metric array."""
    return {
        f"{name}_mean": float(np.mean(arr)),
        f"{name}_median": float(np.median(arr)),
        f"{name}_p5": float(np.percentile(arr, 5)),
        f"{name}_p10": float(np.percentile(arr, 10)),
        f"{name}_p90": float(np.percentile(arr, 90)),
        f"{name}_p95": float(np.percentile(arr, 95)),
        f"{name}_min": float(np.min(arr)),
        f"{name}_max": float(np.max(arr)),
    }


def _print_metric(label: str, arr: NDArray[np.float64], unit: str = "") -> None:
    unit_str = f" {unit}" if unit else ""
    print(
        f"  {label:<32s}: "
        f"mean={np.mean(arr):.3g}{unit_str}  "
        f"p5={np.percentile(arr, 5):.3g}{unit_str}  "
        f"p50={np.median(arr):.3g}{unit_str}  "
        f"p95={np.percentile(arr, 95):.3g}{unit_str}"
    )


# ---------------------------------------------------------------------------
# DynamicQuality class
# ---------------------------------------------------------------------------

class DynamicQuality:
    """Dynamic mesh quality analysis for an FVCOM mesh.

    Wraps element-level computation of all dynamic quality metrics defined
    in Vitousek & Fringer (2011).  Mesh files are read with
    :func:`read_fvcom_mesh`; optional simulation output is read with
    :func:`read_fvcom_output`.

    Parameters
    ----------
    pts : (N, 2)
    triangles : (T, 3) 0-based
    depths : (N,) bathymetric depth [m]

    Examples
    --------
    Mesh-only analysis (no simulation output):

    >>> dq = DynamicQuality.from_files("maldives_v0_grd.dat", "maldives_v0_dep.dat")
    >>> metrics = dq.compute(periods={"M2": 44712, "K1": 86164})
    >>> dq.report()

    With simulation output:

    >>> dq = DynamicQuality.from_files(
    ...     "maldives_v0_grd.dat",
    ...     "maldives_v0_dep.dat",
    ...     nc_path="maldives_v0_0001.nc",
    ...     dt=300.0,
    ... )
    >>> dq.report()
    """

    def __init__(
        self,
        pts: NDArray[np.float64],
        triangles: NDArray[np.int_],
        depths: NDArray[np.float64],
    ) -> None:
        self.pts = pts
        self.triangles = triangles
        self.depths = depths

        # Precompute element geometry
        self._le: NDArray[np.float64] = element_representative_size(pts, triangles)

        # Simulation output statistics — set by load_output()
        self._zeta_max: Optional[NDArray[np.float64]] = None    # (N,)
        self._zeta_mean: Optional[NDArray[np.float64]] = None   # (N,)
        self._ua_max: Optional[NDArray[np.float64]] = None      # (nele,)
        self._va_max: Optional[NDArray[np.float64]] = None      # (nele,)
        self._dt_from_output: Optional[float] = None

        self.dt: Optional[float] = None        # explicitly supplied time step
        self._metrics: dict[str, NDArray] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        grd_path: str | Path,
        dep_path: Optional[str | Path] = None,
        nc_path: Optional[str | Path] = None,
        dt: Optional[float] = None,
        max_timesteps: Optional[int] = None,
    ) -> "DynamicQuality":
        """Create from FVCOM mesh files (and optionally a NetCDF output).

        Parameters
        ----------
        grd_path : path to ``_grd.dat``
        dep_path : path to ``_dep.dat`` (optional if depth embedded in GRD)
        nc_path  : path to a FVCOM NetCDF output file (optional)
        dt       : model external time step in seconds.  If *nc_path* is
                   provided and *dt* is ``None``, the output interval is
                   used as a proxy (conservative upper bound for CFL).
        max_timesteps : restrict reading to the first N timesteps
        """
        pts, tri, dep = read_fvcom_mesh(grd_path, dep_path)
        obj = cls(pts, tri, dep)
        if nc_path is not None:
            obj.load_output(nc_path, max_timesteps=max_timesteps)
        if dt is not None:
            obj.dt = dt
        return obj

    # ------------------------------------------------------------------
    # Loading simulation output
    # ------------------------------------------------------------------

    def load_output(
        self,
        nc_path: str | Path,
        max_timesteps: Optional[int] = None,
    ) -> None:
        """Load simulation statistics from a FVCOM NetCDF output file.

        Can be called multiple times (e.g., to accumulate statistics over
        several output files); the per-node maxima are updated in-place.

        Parameters
        ----------
        nc_path : path to FVCOM ``_NNNN.nc`` output file
        max_timesteps : if given, only the first *max_timesteps* are read
        """
        out = read_fvcom_output(nc_path, max_timesteps=max_timesteps)

        # Accumulate running maxima across multiple files
        if self._zeta_max is None:
            self._zeta_max = out["zeta_max"]
            self._zeta_mean = out["zeta_mean"]
            self._ua_max = out["ua_max"]
            self._va_max = out["va_max"]
        else:
            self._zeta_max = np.maximum(self._zeta_max, out["zeta_max"])
            self._zeta_mean = 0.5 * (self._zeta_mean + out["zeta_mean"])
            self._ua_max = np.maximum(self._ua_max, out["ua_max"])
            self._va_max = np.maximum(self._va_max, out["va_max"])

        self._dt_from_output = out["dt"]

    # ------------------------------------------------------------------
    # Effective depth and time step
    # ------------------------------------------------------------------

    @property
    def _effective_depth(self) -> NDArray[np.float64]:
        return element_depth(self.triangles, self.depths, self._zeta_mean)

    @property
    def _effective_dt(self) -> Optional[float]:
        return self.dt if self.dt is not None else self._dt_from_output

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute(
        self,
        dt: Optional[float] = None,
        periods: Optional[dict[str, float]] = None,
    ) -> dict[str, NDArray]:
        """Compute all dynamic quality metrics and return them.

        Parameters
        ----------
        dt : model external time step [s].  Overrides any previously set
             value.  Required for Courant number metrics; if not provided
             those metrics are skipped.
        periods : dict mapping constituent name to period in seconds.
                  Defaults to ``{"M2": 44712}``.

        Returns
        -------
        metrics : dict of (T,) arrays.  Stored in ``self._metrics`` too.
        """
        if dt is not None:
            self.dt = dt
        effective_dt = self._effective_dt

        if periods is None:
            periods = {"M2": TIDAL_PERIODS["M2"]}

        he = self._effective_depth
        le = self._le

        out: dict[str, NDArray] = {
            "element_size": le,
            "element_depth": he,
        }

        # Wavelength resolution and dispersion ratio (no dt needed)
        for name, T in periods.items():
            n_lam = wavelength_resolution(le, he, period=T)
            out[f"n_lambda_{name}"] = n_lam
            out[f"dispersion_ratio_{name}"] = numerical_dispersion_ratio(n_lam)

        # Courant numbers (require dt)
        if effective_dt is not None:
            for name, T in periods.items():
                out[f"cr_wave_{name}"] = barotropic_courant(le, he, effective_dt)
            if self._ua_max is not None:
                out["cr_flow"] = velocity_courant(
                    le, self._ua_max, self._va_max, effective_dt
                )
        else:
            warnings.warn(
                "dt is not set; Courant number metrics will be skipped. "
                "Pass dt= to compute() or set DynamicQuality.dt.",
                stacklevel=2,
            )

        # Output-based metrics
        if self._ua_max is not None:
            out["froude"] = froude_number(he, self._ua_max, self._va_max)
            out["relative_amplitude"] = relative_amplitude(
                self.triangles, he, self._zeta_max
            )
            for name, T in periods.items():
                out[f"tidal_excursion_{name}"] = tidal_excursion(
                    le, self._ua_max, self._va_max, period=T
                )

        self._metrics = out
        return out

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(
        self,
        dt: Optional[float] = None,
        periods: Optional[dict[str, float]] = None,
    ) -> None:
        """Compute metrics and print a summary report to stdout.

        Parameters
        ----------
        dt, periods : passed to :meth:`compute` if metrics not yet computed
        """
        if not self._metrics or dt is not None or periods is not None:
            self.compute(dt=dt, periods=periods)

        m = self._metrics
        n_elem = len(m["element_size"])
        has_output = "froude" in m

        print(f"\n{'='*62}")
        print("  FVCOM Dynamic Mesh Quality Report")
        print(f"  Elements: {n_elem:,}   Nodes: {len(self.pts):,}")
        print(f"{'='*62}")

        print("\n  [Geometry]")
        _print_metric("Element size L_e [m]", m["element_size"], "m")
        _print_metric("Water depth H_e [m]", m["element_depth"], "m")

        # Wavelength / dispersion groups (one per constituent)
        period_keys = [k for k in m if k.startswith("n_lambda_")]
        for pk in period_keys:
            name = pk[len("n_lambda_"):]
            print(f"\n  [Tidal resolution — {name}]")
            _print_metric("  Elements per wavelength N_λ", m[pk])
            _print_metric("  Dispersion ratio c_num/c",
                          m[f"dispersion_ratio_{name}"])
            frac_under = np.mean(m[pk] < 20) * 100
            print(f"  {'  N_λ < 20 (poorly resolved)':<34s}: "
                  f"{frac_under:.1f} % of elements")

            cr_key = f"cr_wave_{name}"
            if cr_key in m:
                _print_metric("  Wave Courant number Cr_w",
                              m[cr_key])

        if has_output:
            print("\n  [Dynamics from simulation output]")
            _print_metric("Froude number Fr", m["froude"])
            _print_metric("Relative amplitude |ζ|/H", m["relative_amplitude"])
            if "cr_flow" in m:
                _print_metric("Flow Courant number Cr_u", m["cr_flow"])
            for pk in [k for k in m if k.startswith("tidal_excursion_")]:
                name = pk[len("tidal_excursion_"):]
                _print_metric(f"Tidal excursion E/L_e ({name})", m[pk])

        print(f"\n{'='*62}\n")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_metric(
        self,
        metric_name: str,
        *,
        ax=None,
        figsize: tuple[float, float] = (12, 10),
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: Optional[str] = None,
        log_scale: bool = False,
    ) -> "matplotlib.axes.Axes":  # type: ignore[name-defined]
        """Plot a spatial map of a dynamic quality metric.

        Parameters
        ----------
        metric_name : key in ``self._metrics`` (e.g. ``"n_lambda_M2"``)
        ax          : existing Axes (created if None)
        figsize     : figure size
        cmap        : matplotlib colormap name
        vmin, vmax  : colour-scale limits (auto-scaled if None)
        title       : axes title (defaults to *metric_name*)
        log_scale   : if True, apply log10 to the field before plotting

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        if metric_name not in self._metrics:
            raise KeyError(
                f"Metric '{metric_name}' not found. "
                f"Available: {sorted(self._metrics.keys())}"
            )

        values = self._metrics[metric_name].copy()
        if log_scale:
            values = np.log10(np.maximum(values, 1e-10))

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        triang = mtri.Triangulation(
            self.pts[:, 0], self.pts[:, 1], self.triangles
        )
        tpc = ax.tripcolor(
            triang, facecolors=values, cmap=cmap,
            vmin=vmin, vmax=vmax, shading="flat"
        )
        cb_label = f"log10({metric_name})" if log_scale else metric_name
        plt.colorbar(tpc, ax=ax, label=cb_label, fraction=0.03)

        ax.set_aspect("equal")
        ax.set_title(title or metric_name)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        return ax

    def plot_all(
        self,
        save_dir: Optional[str | Path] = None,
        show: bool = True,
    ) -> list["matplotlib.axes.Axes"]:  # type: ignore[name-defined]
        """Plot all available metrics in separate figures.

        Parameters
        ----------
        save_dir : directory to save PNG files (optional)
        show     : whether to call ``plt.show()``

        Returns
        -------
        list of Axes objects
        """
        import matplotlib.pyplot as plt

        if not self._metrics:
            raise RuntimeError("Call compute() before plot_all().")

        save_path = Path(save_dir) if save_dir else None
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)

        log_metrics = {"n_lambda_M2", "n_lambda_S2", "n_lambda_K1",
                       "n_lambda_O1", "n_lambda_N2"}
        axes = []
        for name in sorted(self._metrics.keys()):
            log = name in log_metrics or name.startswith("n_lambda_")
            ax = self.plot_metric(name, log_scale=log)
            axes.append(ax)
            if save_path is not None:
                fname = save_path / f"dynamic_{name}.png"
                ax.get_figure().savefig(fname, dpi=150, bbox_inches="tight")
                plt.close(ax.get_figure())

        if show and save_path is None:
            plt.show()

        return axes
