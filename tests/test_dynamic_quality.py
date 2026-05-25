"""
tests/test_dynamic_quality.py — Unit tests for the dynamic quality metrics.

All tests use small synthetic meshes so there is no dependency on FVCOM
files or NetCDF output.
"""

from __future__ import annotations

import io
import textwrap

import numpy as np
import pytest

from fvcom_mesh.dynamic_quality import (
    TIDAL_PERIODS,
    DynamicQuality,
    _read_dep,
    barotropic_courant,
    element_area,
    element_depth,
    element_representative_size,
    froude_number,
    numerical_dispersion_ratio,
    read_fvcom_mesh,
    relative_amplitude,
    tidal_excursion,
    velocity_courant,
    wavelength_resolution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _square_mesh() -> tuple:
    """4 nodes at unit-square corners, 2 right triangles.

    pts (0,0) (1,0) (1,1) (0,1)
    tri0 = [0,1,2]   area = 0.5
    tri1 = [0,2,3]   area = 0.5
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    depths = np.full(4, 10.0)
    return pts, tris, depths


# ---------------------------------------------------------------------------
# element_area
# ---------------------------------------------------------------------------

class TestElementArea:
    def test_right_triangle(self):
        pts, tris, _ = _square_mesh()
        areas = element_area(pts, tris)
        assert areas.shape == (2,)
        np.testing.assert_allclose(areas, [0.5, 0.5], atol=1e-12)

    def test_equilateral(self):
        h = np.sqrt(3) / 2
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
        tris = np.array([[0, 1, 2]])
        area = element_area(pts, tris)
        np.testing.assert_allclose(area, [np.sqrt(3) / 4], rtol=1e-10)

    def test_cw_same_as_ccw(self):
        """element_area always returns positive regardless of orientation."""
        pts, tris, _ = _square_mesh()
        tris_cw = tris[:, ::-1]
        np.testing.assert_allclose(element_area(pts, tris),
                                   element_area(pts, tris_cw))


# ---------------------------------------------------------------------------
# element_representative_size
# ---------------------------------------------------------------------------

class TestElementRepresentativeSize:
    def test_square_mesh(self):
        pts, tris, _ = _square_mesh()
        le = element_representative_size(pts, tris)
        np.testing.assert_allclose(le, np.sqrt(0.5), rtol=1e-10)


# ---------------------------------------------------------------------------
# element_depth
# ---------------------------------------------------------------------------

class TestElementDepth:
    def test_uniform_depth(self):
        _, tris, depths = _square_mesh()
        he = element_depth(tris, depths)
        np.testing.assert_allclose(he, [10.0, 10.0])

    def test_with_zeta_mean(self):
        _, tris, depths = _square_mesh()
        zeta = np.full(4, 0.5)
        he = element_depth(tris, depths, zeta_mean=zeta)
        np.testing.assert_allclose(he, [10.5, 10.5])

    def test_minimum_clip(self):
        _, tris, _ = _square_mesh()
        depths = np.zeros(4)
        he = element_depth(tris, depths)
        assert np.all(he >= 0.1)


# ---------------------------------------------------------------------------
# wavelength_resolution
# ---------------------------------------------------------------------------

class TestWavelengthResolution:
    def test_positive(self):
        pts, tris, depths = _square_mesh()
        le = element_representative_size(pts, tris)
        he = element_depth(tris, depths)
        n = wavelength_resolution(le, he, period=44712.0)
        assert n.shape == (2,)
        assert np.all(n > 0)

    def test_deeper_gives_larger_n(self):
        le = np.array([100.0, 100.0])
        he_shallow = np.array([10.0, 10.0])
        he_deep = np.array([100.0, 100.0])
        n_shallow = wavelength_resolution(le, he_shallow)
        n_deep = wavelength_resolution(le, he_deep)
        assert np.all(n_deep > n_shallow)

    def test_larger_le_gives_smaller_n(self):
        le_fine = np.array([50.0])
        le_coarse = np.array([200.0])
        he = np.array([10.0])
        n_fine = wavelength_resolution(le_fine, he)
        n_coarse = wavelength_resolution(le_coarse, he)
        assert n_fine[0] > n_coarse[0]


# ---------------------------------------------------------------------------
# numerical_dispersion_ratio
# ---------------------------------------------------------------------------

class TestNumericalDispersionRatio:
    def test_nyquist(self):
        """N_λ=2 → ratio = 2/π (Nyquist limit)."""
        ratio = numerical_dispersion_ratio(np.array([2.0]))
        np.testing.assert_allclose(ratio[0], 2.0 / np.pi, rtol=1e-6)

    def test_well_resolved(self):
        """N_λ ≫ 1 → ratio ≈ 1."""
        ratio = numerical_dispersion_ratio(np.array([1000.0]))
        assert ratio[0] > 0.999

    def test_monotone_increasing(self):
        n = np.array([2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
        ratio = numerical_dispersion_ratio(n)
        assert np.all(np.diff(ratio) > 0)

    def test_range(self):
        n = np.linspace(2, 1000, 100)
        ratio = numerical_dispersion_ratio(n)
        assert np.all(ratio > 0)
        assert np.all(ratio <= 1.0)

    def test_n20_accuracy(self):
        """At N_λ=20 the error should be less than 0.5 %."""
        ratio = numerical_dispersion_ratio(np.array([20.0]))
        assert ratio[0] > 0.9950


# ---------------------------------------------------------------------------
# barotropic_courant
# ---------------------------------------------------------------------------

class TestBarotropicCourant:
    def test_formula(self):
        le = np.array([100.0])
        he = np.array([10.0])
        dt = 30.0
        cr = barotropic_courant(le, he, dt)
        expected = np.sqrt(9.81 * 10.0) * 30.0 / 100.0
        np.testing.assert_allclose(cr, [expected], rtol=1e-10)

    def test_deeper_gives_higher_cr(self):
        le = np.array([100.0, 100.0])
        he_shallow = np.array([5.0, 5.0])
        he_deep = np.array([50.0, 50.0])
        cr_s = barotropic_courant(le, he_shallow, dt=10.0)
        cr_d = barotropic_courant(le, he_deep, dt=10.0)
        assert np.all(cr_d > cr_s)


# ---------------------------------------------------------------------------
# velocity_courant
# ---------------------------------------------------------------------------

class TestVelocityCourant:
    def test_formula(self):
        le = np.array([100.0])
        ua = np.array([1.5])
        va = np.array([0.0])
        cr = velocity_courant(le, ua, va, dt=10.0)
        np.testing.assert_allclose(cr, [0.15], rtol=1e-10)

    def test_vector_speed(self):
        le = np.array([100.0])
        ua = np.array([3.0])
        va = np.array([4.0])
        cr = velocity_courant(le, ua, va, dt=10.0)
        # |U| = 5 m/s → Cr = 5*10/100 = 0.5
        np.testing.assert_allclose(cr, [0.5], rtol=1e-10)


# ---------------------------------------------------------------------------
# froude_number
# ---------------------------------------------------------------------------

class TestFroudeNumber:
    def test_subcritical(self):
        he = np.array([10.0])
        ua = np.array([0.5])
        va = np.array([0.0])
        fr = froude_number(he, ua, va)
        expected = 0.5 / np.sqrt(9.81 * 10.0)
        np.testing.assert_allclose(fr, [expected], rtol=1e-10)

    def test_critical_approx_one(self):
        he = np.array([1.0])
        c = np.sqrt(9.81 * 1.0)
        ua = np.array([c])
        va = np.array([0.0])
        fr = froude_number(he, ua, va)
        np.testing.assert_allclose(fr, [1.0], rtol=1e-6)


# ---------------------------------------------------------------------------
# relative_amplitude
# ---------------------------------------------------------------------------

class TestRelativeAmplitude:
    def test_small_amplitude(self):
        _, tris, depths = _square_mesh()
        he = element_depth(tris, depths)
        zeta_max = np.full(4, 0.5)
        a_h = relative_amplitude(tris, he, zeta_max)
        np.testing.assert_allclose(a_h, [0.05, 0.05], rtol=1e-6)


# ---------------------------------------------------------------------------
# tidal_excursion
# ---------------------------------------------------------------------------

class TestTidalExcursion:
    def test_formula(self):
        le = np.array([1000.0])
        ua = np.array([1.0])
        va = np.array([0.0])
        T = TIDAL_PERIODS["M2"]
        exc = tidal_excursion(le, ua, va, period=T)
        expected = (1.0 * T / np.pi) / 1000.0
        np.testing.assert_allclose(exc, [expected], rtol=1e-10)


# ---------------------------------------------------------------------------
# read_fvcom_mesh (4-field and 2-field node rows)
# ---------------------------------------------------------------------------

class TestReadFvcomMesh:
    def _write_grd_4field(self, tmp_path) -> str:
        """Write a minimal GRD file with 4-field node rows (standard FVCOM)."""
        content = textwrap.dedent("""\
            Node Number = 4
            Cell Number = 2
            1 1 2 3 1
            2 1 3 4 2
            1 0.0 0.0 10.0
            2 1.0 0.0 10.0
            3 1.0 1.0 10.0
            4 0.0 1.0 10.0
        """)
        p = tmp_path / "test_grd.dat"
        p.write_text(content)
        return str(p)

    def _write_grd_2field(self, tmp_path):
        grd_content = textwrap.dedent("""\
            Node Number = 4
            Cell Number = 2
            1 1 2 3 1
            2 1 3 4 2
            0.0 0.0
            1.0 0.0
            1.0 1.0
            0.0 1.0
        """)
        dep_content = textwrap.dedent("""\
            Node Number = 4
            0.0 0.0 10.0
            1.0 0.0 10.0
            1.0 1.0 10.0
            0.0 1.0 10.0
        """)
        grd = tmp_path / "test2_grd.dat"
        dep = tmp_path / "test2_dep.dat"
        grd.write_text(grd_content)
        dep.write_text(dep_content)
        return str(grd), str(dep)

    def test_4field_nodes(self, tmp_path):
        grd = self._write_grd_4field(tmp_path)
        pts, tris, depths = read_fvcom_mesh(grd)
        assert pts.shape == (4, 2)
        assert tris.shape == (2, 3)
        np.testing.assert_allclose(depths, [10.0, 10.0, 10.0, 10.0])

    def test_triangles_zero_based(self, tmp_path):
        grd = self._write_grd_4field(tmp_path)
        _, tris, _ = read_fvcom_mesh(grd)
        assert tris.min() == 0

    def test_2field_nodes_with_dep(self, tmp_path):
        grd, dep = self._write_grd_2field(tmp_path)
        pts, tris, depths = read_fvcom_mesh(grd, dep)
        assert pts.shape == (4, 2)
        np.testing.assert_allclose(depths, 10.0)

    def test_missing_dep_warns(self, tmp_path):
        grd, _ = self._write_grd_2field(tmp_path)
        with pytest.warns(UserWarning):
            pts, tris, depths = read_fvcom_mesh(grd)
        np.testing.assert_allclose(depths, 1.0)


# ---------------------------------------------------------------------------
# DynamicQuality class (using synthetic arrays, no file I/O)
# ---------------------------------------------------------------------------

class TestDynamicQuality:
    def _make_dq(self):
        pts, tris, depths = _square_mesh()
        return DynamicQuality(pts, tris, depths)

    def test_compute_returns_required_keys(self):
        dq = self._make_dq()
        m = dq.compute(dt=10.0, periods={"M2": 44712.0})
        for key in ("element_size", "element_depth",
                    "n_lambda_M2", "dispersion_ratio_M2", "cr_wave_M2"):
            assert key in m, f"Missing key: {key}"

    def test_compute_shapes(self):
        dq = self._make_dq()
        m = dq.compute(dt=10.0, periods={"M2": 44712.0})
        for v in m.values():
            assert v.shape == (2,)

    def test_no_dt_skips_courant(self):
        dq = self._make_dq()
        with pytest.warns(UserWarning):
            m = dq.compute(periods={"M2": 44712.0})
        assert "cr_wave_M2" not in m
        assert "n_lambda_M2" in m

    def test_multiple_periods(self):
        dq = self._make_dq()
        periods = {"M2": 44712.0, "K1": 86164.1}
        m = dq.compute(dt=10.0, periods=periods)
        for name in ("M2", "K1"):
            assert f"n_lambda_{name}" in m
            assert f"dispersion_ratio_{name}" in m
            assert f"cr_wave_{name}" in m

    def test_output_metrics_with_synthetic_data(self):
        pts, tris, depths = _square_mesh()
        dq = DynamicQuality(pts, tris, depths)
        # Inject synthetic output statistics
        dq._zeta_max = np.full(4, 0.2)
        dq._zeta_mean = np.full(4, 0.1)
        dq._ua_max = np.full(2, 0.5)
        dq._va_max = np.full(2, 0.3)
        m = dq.compute(dt=10.0, periods={"M2": 44712.0})
        for key in ("froude", "relative_amplitude",
                    "cr_flow", "tidal_excursion_M2"):
            assert key in m, f"Missing key: {key}"

    def test_load_output_accumulates_maxima(self):
        pts, tris, depths = _square_mesh()
        dq = DynamicQuality(pts, tris, depths)
        dq._zeta_max = np.array([0.1, 0.2, 0.3, 0.4])
        dq._zeta_mean = np.zeros(4)
        dq._ua_max = np.array([0.5, 0.6])
        dq._va_max = np.array([0.1, 0.2])
        # Simulate a second load with higher values
        from fvcom_mesh.dynamic_quality import read_fvcom_output as _rfo
        import unittest.mock as mock
        second = {
            "x": np.zeros(4), "y": np.zeros(4), "h": depths,
            "nv": tris,
            "zeta_max": np.array([0.5, 0.1, 0.1, 0.1]),
            "zeta_mean": np.zeros(4),
            "ua_max": np.array([0.3, 0.9]),
            "va_max": np.array([0.0, 0.0]),
            "dt": 3600.0,
        }
        with mock.patch("fvcom_mesh.dynamic_quality.read_fvcom_output",
                        return_value=second):
            dq.load_output("fake.nc")
        np.testing.assert_allclose(dq._zeta_max,
                                   [0.5, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(dq._ua_max, [0.5, 0.9])

    def test_report_runs_without_error(self, capsys):
        dq = self._make_dq()
        dq.report(dt=10.0, periods={"M2": 44712.0})
        captured = capsys.readouterr()
        assert "Element" in captured.out
        assert "M2" in captured.out
