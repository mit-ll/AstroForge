# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Models
"""

from numba import njit
from numpy.typing import NDArray
import numpy as np

from ._forces import F_J2, F_geo_MEMED, F_third
from ..constants import ss_GM, GM, GM_moon, GM_sun
from ..coordinates import shadow
from ..common import R_moon, R_moon_MEMED, R_sun, R_sun_MEMED

__all__ = [
    "F_mp",
    "F_mp_srp",
    "kepler",
]


@njit
def F_mp(
    t: float,
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """A built-in medium-fidelity force model for propagating a satellite. This model
    includes:

    * Two body + J2 perturbation Earth gravity
    * Third body gravitational perturbation from the Sun
    * Third body gravitational perturbation from the Moon

    Parameters
    ----------
    t : float
        Time at which the force model is to be evaluated, expressed as a Modified
        Julian Date (MJD) in the UT1 time system.
    y : NDArray[np.float64]
        State vector; should have shape (6, ) with the first three parameters
        corresponding the 3D position, and the last three parameters corresponding to
        the 3D velocity of the satellite. Units are km and km/s.

    Returns
    -------
    ydot : NDArray[np.float64]
        Derivate of the state vector.
    """
    x = y[:3]
    v = y[3:]
    mjd = t / 86400.0

    rm = R_moon(mjd)
    rs = R_sun(mjd)

    acc = F_J2(x) + F_third(x, rm, GM_moon) + F_third(x, rs, GM_sun)

    ydot = np.empty(6)
    ydot[:3] = v
    ydot[3:] = acc


    return ydot


@njit
def F_mp_srp(
    time: float,
    xxdot: NDArray[np.float64],
) -> NDArray[np.float64]:
    """A built-in, medium-fidelity force model for propagating an orbital state. This
    model includes:

    * 8 x 8 spherical harmonic Earth gravity model
    * Third body gravitational perturbation from the Sun
    * Third body gravitational perturbation from the Moon
    * Solar radiation pressure (SRP) model

    This model uses a 9-element state vector. The first six are the typical position and
    velocity vectors. The final three elements are time-constant coefficients of the SRP
    model: :math:`\\alpha_i`.

    The SRP model is:

    .. math::

        a_{SRP} = \\alpha_1 n_1 + \\alpha_2 n_2 + \\alpha_3 n_3

    where

    * :math:`n_1` is in the direction away from the Sun
    * :math:`n_2` is along the direction of the satellite's velocity vector
    * :math:`n_3` is along the direction of the satellite's position vector.

    Parameters
    ----------
    time : float
        Time at which the force model is to be evaluated, expressed as a Modified
        Julian Date (MJD) in the UT1 time system.
    xxdot : NDArray[np.float64]
        State vector; should have shape (9, ). The total state vector is:

        .. math::

            X = \\begin{bmatrix}
                    x \\\\
                    v \\\\
                    \\alpha
                \\end{bmatrix}

    Returns
    -------
    ydot : NDArray[np.float64]
        Derivative of the state vector
    """
    mjd = time / 86400.0

    x = xxdot[:3]
    v = xxdot[3:6]
    alpha = xxdot[6:]
    
    Rm = R_moon_MEMED(mjd)
    Rs = R_sun_MEMED(mjd)

    n1 = -Rs
    n1 = n1 / np.sqrt(n1 @ n1)
    n3 = x / np.sqrt(x @ x)
    n2 = v / np.sqrt(v @ v)
    a = shadow(-n1, x) * (alpha[0] * n1 + alpha[1] * n2 + alpha[2] * n3)

    acc = F_geo_MEMED(mjd, x) + F_third(x, Rm, GM_moon) + F_third(x, Rs, GM_sun) + a
    jerk = np.zeros(3)

    ydot = np.empty(9)
    ydot[:3] = v
    ydot[3:6] = acc
    ydot[6:] = jerk

    return ydot


@njit
def kepler(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
    """A built-in force model that computes only the two-body acceleration term. The
    two body equations of motions are:

    .. math::

        \\ddot{r} = - \\frac{GM}{||r||^3} r

    Parameters
    ----------
    t : float
        Time, unused. Time must be the first input to this function in order to satisfy
        the ODE solver API.
    y : NDArray[np.float64]
        State vector; should have shape (6, ) with the first three parameters
        corresponding the 3D position, and the last three parameters corresponding to
        the 3D velocity of the satellite. Units are km and km/s.

    Returns
    -------
    ydot : NDArray[np.float64]
        Derivative of the state vector.
    """

    x = y[:3]
    v = y[3:]

    r3 = np.sqrt(np.sum(x**2, axis=0)) ** 3
    acc = -GM * x / r3

    ydot = np.empty(6)
    ydot[:3] = v
    ydot[3:] = acc

    return ydot
