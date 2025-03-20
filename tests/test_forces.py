# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import numpy as np

from astroforge.force_models import F_J2, F_third, F_geo_MEMED

_mjd = 59025.0
_x = np.array([-5413.11749016, 4960.20830485, 3177.18312508])  # just a random vector
_r_third = np.array([-9628141.93697005, 139234697.077259, 60365629.0375675])
_GM_sun = 1.32712440018e11


def test_F_J2():
    out = F_J2(_x)
    truth = [0.00421512067459646, -0.0038624464763761, -0.00247913555008177]
    np.testing.assert_allclose(out, truth)


def test_F_third():
    out = F_third(_x, _r_third, _GM_sun)
    truth = [1.6027119005637e-10, 4.49993299146484e-10, 1.56340910921885e-10]
    np.testing.assert_allclose(out, truth)


def test_F_geo_MEMED():
    out = F_geo_MEMED(_mjd, _x)
    truth = [0.0042150911472052, -0.00386236848228582, -0.00247912543118148]
    np.testing.assert_allclose(out, truth)
