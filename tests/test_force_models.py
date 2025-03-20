# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import numpy as np

from astroforge.force_models import (
    F_mp,
    kepler,
)


_mjd = 59025.0
# just a random vector
_x = np.array([-5413.11749016, 4960.20830485, 3177.18312508])
# circular velocity in random direction
_v = np.array([4.94383690820171, 0.807568275491992, 4.97306397812187])
# random, small acceleration
_a = np.array([7.14979548359578e-11, 3.62563357150639e-11, 5.97789308602281e-11])
_X = np.hstack([_x, _v, _a])


def test_f_mp():
    out = F_mp(_mjd, np.hstack([_x, _v]))
    truth = [
        4.94383690820171,
        0.807568275491992,
        4.97306397812187,
        0.00421512043273244,
        -0.00386244710897183,
        -0.00247913601506188,
    ]
    np.testing.assert_allclose(out, truth)


def test_kepler():
    out = kepler(_mjd, np.hstack([_x, _v]))
    truth = [
        4.94383690820171,
        0.807568275491992,
        4.97306397812187,
        0.00421420121697155,
        -0.0038616039486912,
        -0.00247348944791846,
    ]
    np.testing.assert_allclose(out, truth)
