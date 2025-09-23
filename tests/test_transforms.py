# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import pytest
from unittest import mock
import numpy as np

from astroforge.coordinates import (
    CIRSToMEMED,
    CIRSToTETED,
    ITRSToMEMED,
    ITRSToTIRS,
    ITRSToTETED,
    ITRSToLatLonAlt,
    ITRSToSEZ,
    LatLonAltToITRS,
    MEMEDToCIRS,
    MEMEDToITRS,
    PosVelConversion,
    PosVelToFPState,
    SEZToAzElRange,
    TETEDToCIRS,
    TETEDToITRS,
    TIRSToITRS,
    cartesian_to_keplerian,
    keplerian_to_cartesian,
    true_anomaly_from_eccentric_anomaly,
    true_anomaly_from_mean_anomaly,
    eccentric_anomaly_from_mean_anomaly,
    eccentric_anomaly_from_true_anomaly,
    mean_anomaly_from_true_anomaly,
    ConvergenceException,
)

# disable numba's jit
# os.environ.set("NUMBA_DISABLE_JIT", "1")
# os.environ["NUMBA_DIABLE_JIT"] = "1"

_mjd = 59025.0
_x = np.array([-5413.11749016, 4960.20830485, 3177.18312508])  # just a random vector
# random site ITRS vector
_xsite = np.array([-6027.05094847323, 1614.94343438464, 1317.4025312296])
_lat = 12.0  # site latitude, degrees
_lon = 165.0  # site longitude, degrees
_alt = 0.0


def test_CIRSToMEMED():
    truth = [-5435.77773183492, 4935.36501586931, 3177.18312508]
    xf = CIRSToMEMED(_mjd, _x)
    np.testing.assert_allclose(xf, truth)


def test_CIRSToTETED():
    truth = [-5435.40930839752, 4935.77076452752, 3177.18312508]
    xf = CIRSToTETED(_mjd, _x)
    np.testing.assert_allclose(xf, truth)


def test_ITRSToMEMED():
    truth = [4610.95218676542, 5713.54770008462, 3177.18312508]
    xf = ITRSToMEMED(_mjd, _x)
    np.testing.assert_allclose(xf, truth)


def test_ITRSToTIRS():
    truth = [-5413.11988399585, 4960.21499703899, 3177.16859872282]
    xf = ITRSToTIRS(_mjd, _x)
    np.testing.assert_allclose(xf, truth)


def test_ITRSToTETED():
    truth = [4611.38520000686, 5713.20630053908, 3177.16859872282]
    xf = ITRSToTETED(_mjd, _x)
    np.testing.assert_allclose(xf, truth)


def test_ITRSToTETED_vel():
    true_x = [4611.38520000686, 5713.20630053908, 3177.16859872282]
    true_v = [-0.416613597817601, 0.336267531568997, -2.92480443156954e-10]
    xf, vf = ITRSToTETED(_mjd, _x, V=np.zeros(3))
    np.testing.assert_allclose(xf, true_x)
    np.testing.assert_allclose(vf, true_v, atol=1e-5)


def test_ITRSToLatLonAlt():
    truth = [23.5119269345377, 137.500000000015, 1625.24637425662]
    xf = ITRSToLatLonAlt(_x)
    np.testing.assert_allclose(xf, truth)


def test_ITRSToSEZ():
    truth = [-1762.42076898368, -3390.17540559717, 653.512780600111]
    xf = ITRSToSEZ(_x, _xsite, _lat, _lon)
    np.testing.assert_allclose(xf, truth)


def test_LatLonAltToITRS():
    xf = LatLonAltToITRS(_lat, _lon, _alt)
    np.testing.assert_allclose(xf, _xsite)


def test_MEMEDToCIRS():
    truth = [-5390.34370758831, 4984.94755276745, 3177.18312508]
    xf = MEMEDToCIRS(_mjd, _x)
    np.testing.assert_allclose(xf, truth)


def test_MEMEDToITRS():
    truth = [-5289.93496350449, -5091.37461513802, 3177.18312508]
    xf = MEMEDToITRS(_mjd, _x)
    np.testing.assert_allclose(xf, truth)


def test_PosVelConversion():
    true_x = [4611.38520000686, 5713.20630053908, 3177.16859872282]
    true_v = [-0.416613597817601, 0.336267531568997, -2.92480443156954e-10]
    (xf, vf) = PosVelConversion(ITRSToTETED, _mjd, _x, np.zeros(3))
    np.testing.assert_allclose(xf, true_x)
    np.testing.assert_allclose(vf, true_v, atol=1e-3)


def test_PosVelToFPState():
    true_angles = [
        -2.96938576386307,
        0.268961952535225,
        2.4469271508524e-05,
        1.60983048375293e-05,
    ]
    true_r = 6998.76779054521
    true_rdot = -0.408750287213866

    xsite = np.array([1234.22841224, 6116.37855152, 1317.39458857])
    vsite = np.array([-4.46013384e-01, 9.00013605e-02, -4.56786872e-10])
    angles, r, rdot = PosVelToFPState(_x, np.zeros(3), xsite, vsite)
    angles = angles[0]  # AstroForge makes this an N-dimensional array, but N=1

    np.testing.assert_allclose(angles, true_angles)
    np.testing.assert_allclose(r, true_r)
    np.testing.assert_allclose(rdot, true_rdot)


def test_PosVelToFPState_multi():
    true_angles = [
        -2.96938576386307,
        0.268961952535225,
        2.4469271508524e-05,
        1.60983048375293e-05,
    ]
    true_r = 6998.76779054521
    true_rdot = -0.408750287213866

    xsite = np.array([1234.22841224, 6116.37855152, 1317.39458857])
    vsite = np.array([-4.46013384e-01, 9.00013605e-02, -4.56786872e-10])

    angles, r, rdot = PosVelToFPState(
        np.tile(_x, (2, 1)),
        np.zeros((2, 3)),
        np.tile(xsite, (2, 1)),
        np.tile(vsite, (2, 1)),
    )

    np.testing.assert_allclose(angles, np.tile(true_angles, (2, 1)))
    np.testing.assert_allclose(r, np.repeat(true_r, 2))
    np.testing.assert_allclose(rdot, np.repeat(true_rdot, 2))


def test_SEZToAzElRange():
    truth = [42.5, 23.4, 8000]
    xf = SEZToAzElRange(_x)
    np.testing.assert_allclose(xf, truth)


def test_TETEDToCIRS():
    truth = [-5390.71580203204, 4984.54516796679, 3177.18312508]
    xf = TETEDToCIRS(_mjd, _x)
    np.testing.assert_allclose(xf, truth)


def test_TETEDToITRS():
    truth = [-5289.55250103814, -5091.77616888967, 3177.1763855286]
    xf = TETEDToITRS(_mjd, _x)
    np.testing.assert_allclose(xf, truth)


def test_TIRSToITRS():
    truth = [-5413.1150963132, 4960.20161263041, 3177.19765142128]
    xf = TIRSToITRS(_mjd, _x)
    np.testing.assert_allclose(xf, truth)


def test_cartesian_keplerian():
    pos = np.array([-371.69607966, 7116.79717907, 707.23699658])
    vel = np.array([-8.19634168, 2.03599094, 0.36358961])

    truth = {
        "inclination_rad": 0.1,
        "raan_rad": 0.2,
        "argp_rad": 0.3,
        "eccentricity": 0.4,
        "semi_major_axis_km": 10000.0,
        "mean_anomaly_rad": 0.5,
        "eccentric_anomaly_rad": 0.7818320279930676,
        "true_anomaly_rad": 1.1237047648504743,
    }

    keplerian_elements = cartesian_to_keplerian(pos, vel)

    x_measured, v_measured = keplerian_to_cartesian(
        inclination_rad=truth["inclination_rad"],
        raan_rad=truth["raan_rad"],
        argp_rad=truth["argp_rad"],
        ecc=truth["eccentricity"],
        semi_major_axis_km=truth["semi_major_axis_km"],
        mean_anomaly_rad=truth["mean_anomaly_rad"],
    )

    for key, true_val in truth.items():
        measured_val = keplerian_elements[key]
        np.testing.assert_almost_equal(true_val, measured_val, decimal=3)

    np.testing.assert_allclose(x_measured, pos)
    np.testing.assert_allclose(v_measured, vel)


def test_cartesian_keplerian_zero():
    pos = np.array([-371.69607966, 7116.79717907, 0])
    vel = np.array([-8.19634168, 2.03599094, 0.36358961])

    truth = {
        "inclination_rad": 0.04497,
        "raan_rad": 1.62297,
        "argp_rad": 5.15655,
        "eccentricity": 0.39402,
        "semi_major_axis_km": 9864.67107,
        "mean_anomaly_rad": 0.50948,
        "eccentric_anomaly_rad": 0.789138,
        "true_anomaly_rad": 1.126627,
    }

    keplerian_elements = cartesian_to_keplerian(pos, vel)

    for key, true_val in truth.items():
        measured_val = keplerian_elements[key]
        np.testing.assert_almost_equal(true_val, measured_val, decimal=3)


def test_mean_true_anomaly():
    eccentricity = 0.5
    mean_anomaly = 0.2

    true_anomaly = true_anomaly_from_mean_anomaly(eccentricity, mean_anomaly)
    mean_anomaly_2 = mean_anomaly_from_true_anomaly(eccentricity, true_anomaly)

    eccentric_anomaly = eccentric_anomaly_from_mean_anomaly(eccentricity, mean_anomaly)

    np.testing.assert_almost_equal(mean_anomaly, mean_anomaly_2)
    np.testing.assert_almost_equal(true_anomaly, 0.659516385)
    np.testing.assert_almost_equal(eccentric_anomaly, 0.3901752)


def test_eccentric_true_anomaly():
    eccentricity = 0.5
    eccentric_anomaly = 0.2

    true_anomaly = true_anomaly_from_eccentric_anomaly(
        e=eccentricity, E=eccentric_anomaly
    )
    eccentric_anomaly_2 = eccentric_anomaly_from_true_anomaly(
        e=eccentricity, v=true_anomaly
    )

    np.testing.assert_almost_equal(eccentric_anomaly, eccentric_anomaly_2)
    np.testing.assert_almost_equal(true_anomaly, 0.34413257)


def test_polarmotion_failures():
    def mock_polarmotion(arg):
        return (0, 1)

    with mock.patch(
        "astroforge.coordinates._transformations.polarmotion", mock_polarmotion
    ):
        with pytest.raises(TypeError):
            fail = ITRSToTIRS(_mjd, _x)
        with pytest.raises(TypeError):
            fail = TIRSToITRS(_mjd, _x)


def test_ea_from_ma_divergence():
    with pytest.raises(ConvergenceException):
        ea = eccentric_anomaly_from_mean_anomaly(-1.0, 1e9)


class TestKeplerianEdgeCases:
    def test_zero_raan(self):
        pos = np.array([1, 0, 0])
        vel = np.array([1, 0, 0])
        kep = cartesian_to_keplerian(pos, vel)

        assert kep["raan_rad"] < 1e-6

    def test_wrapped_raan(self):
        pos = np.array([0, -1, 0])
        vel = np.array([0, 0, 1])
        kep = cartesian_to_keplerian(pos, vel)

        assert kep["raan_rad"] >= 0
        assert kep["raan_rad"] <= 2 * np.pi

    def test_true_anom_sign(self):
        pos = np.array([1, 0, 0])
        vel = np.array([-1, 0, 0])
        kep = cartesian_to_keplerian(pos, vel)

        assert kep["true_anomaly_rad"] >= 0
        assert kep["true_anomaly_rad"] <= 2 * np.pi

    def test_dot_overcalc(self):
        """Make sure everything works properly even if dot products
        overshoot calculations due to floating point issues."""
        pos = np.array([-371.69607966, 7116.79717907, 0])
        vel = np.array([-8.19634168, 2.03599094, 0.36358961])

        def mock_npdot(arg1, arg2):
            mag1 = np.linalg.norm(arg1)
            mag2 = np.linalg.norm(arg2)

            return mag1 * mag2 * (1 + 1e-6)

        with mock.patch("numpy.dot", mock_npdot):
            kep = cartesian_to_keplerian(pos, vel)

            assert kep["argp_rad"] >= 0
            assert kep["argp_rad"] <= 2 * np.pi
            assert kep["true_anomaly_rad"] >= 0
            assert kep["true_anomaly_rad"] <= 2 * np.pi
