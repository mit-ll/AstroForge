# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Forces
"""

from numba import njit
from numpy.typing import NDArray
import numpy as np

from ..constants import GM, R_earth, J2
from ..coordinates import MEMEDToITRS, ITRSToMEMED

__all__ = [
    "F_J2",
    "F_third",
    "F_geo_ITRS",
    "F_geo_MEMED",
]


@njit
def F_J2(
    z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Computes the monopole (twobody) acceleration and the J2 perturbing acceleration
    for Earth.

    Parameters
    ----------
    z : NDArray[np.float64]
        Position in space to compute the monopole and J2 forces, should have shape
        (3, ) and units of km.

    Returns
    -------
    acc : NDArray[np.float64]
        Total vector acceleration due to monopole force and J2 perturbing force. The
        output is a vector with shape (3, ) and is in units of km / s\\ :sup:`2`.

    Examples
    --------

    >>> ra, dec = 11 * np.pi / 6, np.pi / 4
    >>> x = af.R_earth + 450.0 * np.array(
            [
                    np.cos(dec) * np.cos(ra),
                    np.cos(dec) * np.sin(ra),
                    np.sin(dec)
            ]
        )
    >>> x
    array([6653.70459606, 6219.03797423, 6696.33505153])
    >>> F_J2(x)
    array([-0.00183523, -0.00171534, -0.0018489 ])

    """

    M2 = J2 * np.diag(np.array([0.5, 0.5, -1.0]))
    r = np.sqrt(z @ z)

    # compute monopole force
    F0 = -GM * z / r**3

    # compute the quadropole force in ITRS
    F2 = (GM * R_earth**2 / r**5) * (-5 * z * (z @ M2 @ z) / r**2 + 2 * M2 @ z)

    return F0 + F2


@njit
def F_third(
    r_s: NDArray[np.float64],
    r_p: NDArray[np.float64],
    GM: float,
) -> NDArray[np.float64]:
    """Compute the perturbing gravitational acceleration due to a third body.

    .. note::
        There is not a required set of units for this function, but the input units must
        be consistent. That is, if r\\ :sub:`s` is in units of km, then
        r\\ :sub:`p` must also be in units of km and `GM` must be in units
        of km\\ :sup:`3` / s\\ :sup:`2`.

    Parameters
    ----------
    r_s : NDArray[np.float64]
        Satellite position vector, should have shape (3, )
    r_p : NDArray[np.float64]
        Perturbing body position vector, should have shape (3, )
    GM : float
        Gravitational parameter of perturbing body

    Returns
    -------
    a_p : NDArray[np.float64]
        Perturbing gravitational acceleration due to the third body

    Examples
    --------
    Compute the perturbing gravitational acceleration due to the Sun for a satellite in
    a GEO orbit around Earth:

    >>> import astroforge as af
    >>> x = np.array([af.Rgeo, 0.0, 0.0])
    array([42164.17236443,     0.        ,     0.        ])
    >>> rs = af.R_sun(51720.0)
    >>> rs
    array([-9.94416191e+06,  1.39217228e+08,  6.03580553e+07])
    >>> GMsun = af.ss_GM["sun"]
    >>> af.force_models.F_third(x, rs, GMsun)
    array([-1.57084634e-09, -2.86422805e-10, -1.24179484e-10])

    """
    r_ps = r_p - r_s
    a_p = GM * (r_ps / (sum(r_ps**2) ** (3 / 2)) - r_p / (sum(r_p**2) ** (3 / 2)))
    return a_p


@njit
def F_geo_ITRS(
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Earth's gravitational potential for a position in ITRS coordinates using
    an 8x8 spherical harmonics model.

    Parameters
    ----------
    x : NDArray[np.float64]
        ITRS position vector at which geo potential should computed, should have shape
        (3, ) and be in units of km

    Returns
    -------
    acc : NDArray[np.float64]
        Acceleration due to Earth's gravitational potential, has shape (3, ) and is in
        units of km / s\\ :sup:`2`\\

    Examples
    --------

    We know the acceleration due to gravity at the surface of the Earth is approximately
    9.81 :math:`m/s^2`. We can use this function to compute an even more accurate vector
    of acceleration using an 8x8 spherical harmonic model of Earth's gravitational
    potential.

    >>> x = np.array([af.R_earth, 0.0, 0.0])
    array([6378.137,    0.   ,    0.   ])
    >>> F_geo_ITRS(x)
    array([-9.81427717e-03, -6.14784939e-08,  3.15547395e-08])

    Indeed, the strongest component of Earth's gravity points toward the middle of the
    Earth with an approximate magnitude of 9.81 x 10\\ :sup:`-3` km / s\\ :sup:`2`. The
    other terms are significantly smaller, but may be consequential for precisely
    propagating an orbit.



    """

    GM = 3.986004415e5
    R_earth = 6.37813630e3

    # number of harmonics to use
    n_max = 8
    m_max = 8

    CS = np.array(
        [
            [
                1.000000e00,
                0.000000e00,
                1.543100e-09,
                2.680119e-07,
                -4.494599e-07,
                -8.066346e-08,
                2.116466e-08,
                6.936989e-08,
                4.019978e-08,
            ],
            [
                0.000000e00,
                0.000000e00,
                -9.038681e-07,
                -2.114024e-07,
                1.481555e-07,
                -5.232672e-08,
                -4.650395e-08,
                9.282314e-09,
                5.381316e-09,
            ],
            [
                -1.082627e-03,
                -2.414000e-10,
                1.574536e-06,
                1.972013e-07,
                -1.201129e-08,
                -7.100877e-09,
                1.843134e-10,
                -3.061150e-09,
                -8.723520e-10,
            ],
            [
                2.532435e-06,
                2.192799e-06,
                3.090160e-07,
                1.005589e-07,
                6.525606e-09,
                3.873005e-10,
                -1.784491e-09,
                -2.636182e-10,
                9.117736e-11,
            ],
            [
                1.619331e-06,
                -5.087253e-07,
                7.841223e-08,
                5.921574e-08,
                -3.982396e-09,
                -1.648204e-09,
                -4.329182e-10,
                6.397253e-12,
                1.612521e-11,
            ],
            [
                2.277161e-07,
                -5.371651e-08,
                1.055905e-07,
                -1.492615e-08,
                -2.297912e-09,
                4.304768e-10,
                -5.527712e-11,
                1.053488e-11,
                8.627743e-12,
            ],
            [
                -5.396485e-07,
                -5.987798e-08,
                6.012099e-09,
                1.182266e-09,
                -3.264139e-10,
                -2.155771e-10,
                2.213693e-12,
                4.475983e-13,
                3.814766e-13,
            ],
            [
                3.513684e-07,
                2.051487e-07,
                3.284490e-08,
                3.528541e-09,
                -5.851195e-10,
                5.818486e-13,
                -2.490718e-11,
                2.559078e-14,
                1.535338e-13,
            ],
            [
                2.025187e-07,
                1.603459e-08,
                6.576542e-09,
                -1.946358e-10,
                -3.189358e-10,
                -4.615173e-12,
                -1.839364e-12,
                3.429762e-13,
                -1.580332e-13,
            ],
        ]
    )

    # auxiliary quantities
    r2 = x @ x
    rho = R_earth * R_earth / r2
    n0 = R_earth * x / r2

    # evaluate harmonic functions
    # V_nm = (R_ref/r)^(n+1) * P_nm(sin(phi)) * cos(m*lambda)
    # and
    # W_nm = (R_ref/r)^(n+1) * P_nm(sin(phi)) * sin(m*lambda)
    # up to degree and order n_max+1

    # calculate zonal terms V(n,0); set W(n,0) = 0.0
    V = np.zeros((n_max + 2, n_max + 2))
    W = np.zeros((n_max + 2, n_max + 2))
    V[0, 0] = R_earth / np.sqrt(r2)
    W[0, 0] = 0.0

    V[1, 0] = n0[2] * V[0, 0]
    W[1, 0] = 0.0

    for n in range(2, n_max + 2):
        V[n, 0] = ((2 * n - 1) * n0[2] * V[n - 1, 0] - (n - 1) * rho * V[n - 2, 0]) / n
        W[n, 0] = 0.0

    # calculate tesseral and sectorial terms
    for m in range(1, m_max + 2):
        # calculate V(m,m) .. V(n_max+1,m)
        V[m, m] = (2 * m - 1) * (n0[0] * V[m - 1, m - 1] - n0[1] * W[m - 1, m - 1])
        W[m, m] = (2 * m - 1) * (n0[0] * W[m - 1, m - 1] + n0[1] * V[m - 1, m - 1])

        if m <= n_max:
            V[m + 1, m] = (2 * m + 1) * n0[2] * V[m, m]
            W[m + 1, m] = (2 * m + 1) * n0[2] * W[m, m]

        for n in range(m + 2, n_max + 2):
            V[n, m] = (
                (2 * n - 1) * n0[2] * V[n - 1, m] - (n + m - 1) * rho * V[n - 2, m]
            ) / (n - m)
            W[n, m] = (
                (2 * n - 1) * n0[2] * W[n - 1, m] - (n + m - 1) * rho * W[n - 2, m]
            ) / (n - m)

    # calculate accelerations ax, ay, az
    force = np.zeros(3)

    for m in range(0, m_max + 1):
        for n in range(m, n_max + 1):
            if m == 0:
                C = CS[n, 0]
                force -= C * np.array([V[n + 1, 1], W[n + 1, 1], (n + 1) * V[n + 1, 0]])
            else:
                C = CS[n, m]
                S = CS[m - 1, n]
                Fac = 0.5 * (n - m + 1) * (n - m + 2)

                force += np.array(
                    [
                        (-C * V[n + 1, m + 1] - S * W[n + 1, m + 1]) / 2
                        + Fac * (C * V[n + 1, m - 1] + S * W[n + 1, m - 1]),
                        (-C * W[n + 1, m + 1] + S * V[n + 1, m + 1]) / 2
                        + Fac * (-C * W[n + 1, m - 1] + S * V[n + 1, m - 1]),
                        (n - m + 1) * (-C * V[n + 1, m] - S * W[n + 1, m]),
                    ]
                )

    force *= GM / (R_earth**2)

    return force


@njit
def F_geo_MEMED(
    time: float,
    z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Earth's gravitational potential for a position in ITRS coordinates using
    an 8x8 spherical harmonics model. This functon is a wrapper around F_geo_ITRS. It
    converts the position vector from MEMED to ITRS, computes the acceleration, and then
    converts the acceleration vector back to MEMED.

    Parameters
    ----------
    time : float
        Reference epoch for the MEMED coordinate system, expressed as a Modified Julian
        Date (MJD) in the UT1 time system.
    z : NDArray[np.float64]
        Position vector at which the Earth's gravitational potential will be evaluated.
        Should have shape (3, ) and be expressed in units of km in the MEMED coordinate
        system.

    Returns
    -------
    acc : NDArray[np.float64]
        Acceleration vector due to Earth's gravity. The output has shape (3, ) and is in
        units of km / s\\ :sup:`2`.
    """

    z1 = MEMEDToITRS(time, z)
    f = F_geo_ITRS(z1)
    f1 = ITRSToMEMED(time, f)
    return f1
