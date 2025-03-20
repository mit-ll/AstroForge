# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import numpy as np

from astroforge.coordinates import Rx, Ry, Rz

ang0 = np.pi / 2
ang1 = 2 * np.pi / 3
ang2 = 7 * np.pi / 4

truth = [
    {
        "x": [[1.0, 0, 0], [0, 0.0, 1.0], [0, -1.0, 0.0]],
        "y": [[0.0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0.0]],
        "z": [[0.0, 1.0, 0], [-1.0, 0.0, 0], [0, 0, 1.0]],
    },
    {
        "x": [[1.0, 0, 0], [0, -0.5, 0.8660], [0, -0.8660, -0.5]],
        "y": [[-0.5, 0, -0.8660], [0, 1.0, 0], [0.8660, 0, -0.5]],
        "z": [[-0.5, 0.8660, 0], [-0.8660, -0.5, 0], [0, 0, 1.0]],
    },
    {
        "x": [[1.0, 0, 0], [0, 0.7071, -0.7071], [0, 0.7071, 0.7071]],
        "y": [[0.7071, 0, 0.7071], [0, 1.0, 0], [-0.7071, 0, 0.7071]],
        "z": [[0.7071, -0.7071, 0], [0.7071, 0.7071, 0], [0, 0, 1.0]],
    },
]


def test_rx():
    np.testing.assert_allclose(Rx(ang0), truth[0]["x"], atol=1e-3)
    np.testing.assert_allclose(Rx(ang1), truth[1]["x"], atol=1e-3)
    np.testing.assert_allclose(Rx(ang2), truth[2]["x"], atol=1e-3)


def test_ry():
    np.testing.assert_allclose(Ry(ang0), truth[0]["y"], atol=1e-3)
    np.testing.assert_allclose(Ry(ang1), truth[1]["y"], atol=1e-3)
    np.testing.assert_allclose(Ry(ang2), truth[2]["y"], atol=1e-3)


def test_rz():
    np.testing.assert_allclose(Rz(ang0), truth[0]["z"], atol=1e-3)
    np.testing.assert_allclose(Rz(ang1), truth[1]["z"], atol=1e-3)
    np.testing.assert_allclose(Rz(ang2), truth[2]["z"], atol=1e-3)
