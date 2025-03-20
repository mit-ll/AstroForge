# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import numpy as np
from unittest import mock
import pytest

from scipy.integrate._ivp.ivp import OdeResult
from scipy.integrate import solve_ivp

# from astroforge.propagators import mp_model
from astroforge.propagators import mp, mp_srp

# some random times
_mjd = np.array([59025.0, 59025.5, 59026.0])
# just a random vector
_x = np.array([-5413.11749016, 4960.20830485, 3177.18312508])
# circular velocity in random direction
_v = np.array([4.94383690820171, 0.807568275491992, 4.97306397812187])
# random, small acceleration
_a = np.array([7.14979548359578e-11, 3.62563357150639e-11, 5.97789308602281e-11])

_true_x_mp = [
    [-5413.11749016, 4960.20830485, 3177.18312508],
    [-2558.48939326, 4795.40804938, 5269.77932578],
    [881.97408604, 3671.51391115, 6144.11917656],
]
_true_v_mp = [
    [4.94383691, 0.80756828, 4.97306398],
    [6.65679191, -1.18355118, 3.11905897],
    [7.10585302, -3.18000087, 0.19866456],
]

_true_x_mp_srp = [
    [-5413.11749016, 4960.20830485, 3177.18312508],
    [-2560.07919425, 4795.80645022, 5268.99190795],
    [879.44076531, 3672.6848089, 6144.06101254],
]
_true_v_mp_srp = [
    [4.94383691, 0.80756828, 4.97306398],
    [6.6561955, -1.182484, 3.12027702],
    [7.10610694, -3.17875813, 0.20130919],
]


def test_mp():
    xf, vf = mp(_x, _v, _mjd)
    np.testing.assert_allclose(xf, _true_x_mp)
    np.testing.assert_allclose(vf, _true_v_mp)


def test_mp_srp():
    xf, vf = mp_srp(_x, _v, _a, _mjd)
    np.testing.assert_allclose(xf, _true_x_mp_srp)
    np.testing.assert_allclose(vf, _true_v_mp_srp)


def test_tolerances():
    xf, vf = mp(_x, _v, _mjd, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(xf, _true_x_mp)
    np.testing.assert_allclose(vf, _true_v_mp)


def test_propagator_fail():

    original_solve_ivp = solve_ivp

    def mock_solve_ivp(arg1, arg2, arg3, t_eval=None, **kwargs):
        result = original_solve_ivp(arg1, arg2, arg3, t_eval=t_eval, **kwargs)
        result.__setitem__("success", False)
        return result

    solve_ivp_path = "astroforge.propagators._propagator.solve_ivp"

    with mock.patch(solve_ivp_path, mock_solve_ivp):
        with pytest.raises(RuntimeError):
            xf, vf = mp(_x, _v, _mjd, atol=1e-9, rtol=1e-9)
