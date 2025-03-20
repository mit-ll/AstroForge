# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from astroforge.coordinates import (
    dut1utc,
    nutate,
    obliquity,
    polarmotion,
    rmat,
    rmfu,
    shadow,
)

_mjd = 59025.0


def test_polarmotion():
    (x, y) = polarmotion(_mjd, bounds_check=False)
    truth = [0.15541, 0.434462]
    np.testing.assert_allclose(np.array([x, y]), truth, atol=1e-4)


def test_polarmotion_bounds():
    with pytest.raises(Exception):
        (x, y) = polarmotion(1.0e7, bounds_check=True)


def test_polarmotion_multi():
    # just testing that using an array works
    (x, y) = polarmotion(np.array([_mjd, _mjd]))
    assert x[0] == x[1] and y[0] == y[1]  # type: ignore


def test_dut1utc():
    dt = dut1utc(_mjd, bounds_check=False)
    truth = -0.2426007
    np.testing.assert_allclose(dt, truth, atol=1e-4)


def test_dut1utc_bounds():
    with pytest.raises(Exception):
        dt = dut1utc(1.0e7, bounds_check=True)


def test_dut1utc_multi():
    # just testing that using an array works
    dt = dut1utc(np.array([_mjd, _mjd]))
    assert dt[0] == dt[1]  # type: ignore


def test_rmfu():
    R = rmfu(0, np.cos(np.pi / 4), np.sin(np.pi / 4))
    truth = [[1.0, 0.0, 0.0], [0.0, 0.7071, 0.7071], [0.0, -0.7071, 0.7071]]
    np.testing.assert_allclose(R, truth, atol=1e-3)

    R = rmfu(1, np.cos(np.pi / 4), np.sin(np.pi / 4))
    truth = [[0.7071, 0.0, -0.7071], [0.0, 1.0, 0.0], [0.7071, 0.0, 0.7071]]
    np.testing.assert_allclose(R, truth, atol=1e-3)

    R = rmfu(2, np.cos(np.pi / 4), np.sin(np.pi / 4))
    truth = [[0.7071, 0.7071, 0.0], [-0.7071, 0.7071, 0.0], [0.0, 0.0, 1.0]]
    np.testing.assert_allclose(R, truth, atol=1e-3)

    with pytest.raises(ValueError):
        R = rmfu(3, np.cos(np.pi / 4), np.sin(np.pi / 4))


def test_obliquity():
    oblq = obliquity(_mjd)
    truth = 0.409046095373858
    np.testing.assert_allclose(oblq, truth)


def test_rmat():
    R = rmat(0, np.pi / 4)
    truth = [[1.0, 0.0, 0.0], [0.0, 0.7071, 0.7071], [0.0, -0.7071, 0.7071]]
    np.testing.assert_allclose(R, truth, atol=1e-3)


def test_nutate():
    out = nutate(_mjd)
    truth = (
        -8.13726247539319e-05,
        -1.47863329845096e-06,
        0.40904461674056,
        [
            [0.999999996689248, 7.46594352713905e-05, 3.23646839886705e-05],
            [-7.46594831268083e-05, 0.999999997211889, 1.4774251332339e-06],
            [-3.2364573594708e-05, -1.47984145881797e-06, 0.999999999475172],
        ],
    )

    for a, b in zip(out, truth):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_shadow():
    rs = np.array([-0.0633169941801626, 0.915641103156801, 0.396979002540166])

    u1 = -rs * 1e4
    u2 = np.cross(rs, np.array([0.0, 0.0, 1.0]))
    u2 = u2 / np.sqrt(u2 @ u2) * 1e4

    out1 = shadow(rs, u1)
    out2 = shadow(rs, u2)
    assert out1 == 0
    assert out2 == 1
