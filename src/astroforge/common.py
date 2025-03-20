# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Utilities that are common to many astrodynamics applications (e.g. approximate 
Sun and Moon positions)
"""

from numba import njit
import numpy as np
from numpy import pi
from numpy.typing import NDArray

from .coordinates import Rx

__all__ = [
    "R_moon",
    "R_moon_MEMED",
    "R_sun",
    "R_sun_MEMED",
]


@njit
def R_moon(
    time: float,
) -> NDArray[np.float64]:
    """
    Low fidelity position of Moon in GCRS (Montenbruck & Gill, p. 70).  The
    input time is given in MJD.  The output units are in km.
    The following relations allow the computation of lunar longitude and
    latitude with a typical accuarcy of several arcminutes and about 500 km
    in the lunar distance.  The calculation is based on five fundamental
    arguments:  the mean longitude, L0, of the Moon; the Moon's mean anomaly,
    l; the Sun's mean anomaly, lprime; the mean angular distance of the
    Moon' from the ascending node, F; and the difference, D, betwen the mean
    longitudes of the Sun and the Moon.  The longitude of the ascending node,
    Omega, is not explicity employed (Omega = L0 - F).
    """

    T = (time - 51544.5) / 36525.0

    # The following angles are in degrees
    eps = 23.43929111 * pi / 180  # obliquity
    L0 = 218.31617 + 481267.88088 * T - 1.3972 * T
    l = 134.96292 + 477198.86753 * T
    lprime = 357.52543 + 35999.04944 * T
    F = 93.27283 + 483202.01873 * T
    D = 297.85027 + 445267.11135 * T

    lambda0 = (
        L0
        + (22640 / 3600) * np.sin(pi * l / 180)
        + (796 / 3600) * np.sin(2 * pi * l / 180)
        - (4586 / 3600) * np.sin((l - 2 * D) * pi / 180)
        + (2370 / 3600) * np.sin(2 * pi * D / 180)
        - (668 / 3600) * np.sin(pi * lprime / 180)
        - (412 / 3600) * np.sin(2 * pi * F / 180)
        - (212 / 3600) * np.sin(2 * (l - D) * pi / 180)
        - (206 / 3600) * np.sin((l + lprime - 2 * D) * pi / 180)
        + (192 / 3600) * np.sin((l + 2 * D) * pi / 180)
        - (165 / 3600) * np.sin((lprime - 2 * D) * pi / 180)
        + (148 / 3600) * np.sin((l - lprime) * pi / 180)
        - (125 / 3600) * np.sin(pi * D / 180)
        - (110 / 3600) * np.sin((l + lprime) * pi / 180)
        - (55 / 3600) * np.sin(2 * (F - D) * pi / 180)
    )

    beta = (
        (18520 / 3600)
        * np.sin(
            (
                F
                + lambda0
                - L0
                + (412 / 3600) * np.sin(2 * pi * F / 180)
                + (541 / 3600) * np.sin(pi * lprime / 180)
            )
            * pi
            / 180
        )
        - (526 / 3600) * np.sin((F - 2 * D) * pi / 180)
        + (44 / 3600) * np.sin((l + F - 2 * D) * pi / 180)
        - (31 / 3600) * np.sin((-l + F - 2 * D) * pi / 180)
        - (25 / 3600) * np.sin((-2 * l + F) * pi / 180)
        - (23 / 3600) * np.sin((lprime + F - 2 * D) * pi / 180)
        + (21 / 3600) * np.sin((-l + F) * pi / 180)
        + (11 / 3600) * np.sin((-lprime + F - 2 * D) * pi / 180)
    )

    r = (
        385000
        - 20905 * np.cos(l * pi / 180)
        - 3699 * np.cos((2 * D - l) * pi / 180)
        - 2956 * np.cos(2 * pi * D / 180)
        - 570 * np.cos(2 * pi * l / 180)
        + 246 * np.cos(2 * (l - D) * pi / 180)
        - 205 * np.cos((lprime - 2 * D) * pi / 180)
        - 171 * np.cos((l + 2 * D) * pi / 180)
        - 152 * np.cos((l + lprime - 2 * D) * pi / 180)
    )

    n = Rx(-eps) @ np.array(
        [
            r * np.cos(lambda0 * pi / 180) * np.cos(beta * pi / 180),
            r * np.sin(lambda0 * pi / 180) * np.cos(beta * pi / 180),
            r * np.sin(beta * pi / 180),
        ]
    )

    return n


@njit
def R_moon_MEMED(
    time: float,
) -> NDArray[np.float64]:
    """
    Low fidelity position of Moon in MEMED (Montenbruck & Gill, p. 70).  The
    input time is given in MJD.  The output units are in km.
    The following relations allow the computation of lunar longitude and
    latitude with a typical accuarcy of several arcminutes and about 500 km
    in the lunar distance.  The calculation is based on five fundamental
    arguments:  the mean longitude, L0, of the Moon; the Moon's mean anomaly,
    l; the Sun's mean anomaly, lprime; the mean angular distance of the
    Moon' from the ascending node, F; and the difference, D, betwen the mean
    longitudes of the Sun and the Moon.  The longitude of the ascending node,
    Omega, is not explicity employed (Omega = L0 - F).
    """

    T = (time - 51544.5) / 36525.0

    # The following angles are in degrees
    eps = 23.43929111 * pi / 180  # obliquity
    L0 = 218.31617 + 481267.88088 * T - 1.3972 * T
    l = 134.96292 + 477198.86753 * T
    lprime = 357.52543 + 35999.04944 * T
    F = 93.27283 + 483202.01873 * T
    D = 297.85027 + 445267.11135 * T

    lambda0 = (
        L0
        + (22640 / 3600) * np.sin(pi * l / 180)
        + (796 / 3600) * np.sin(2 * pi * l / 180)
        - (4586 / 3600) * np.sin((l - 2 * D) * pi / 180)
        + (2370 / 3600) * np.sin(2 * pi * D / 180)
        - (668 / 3600) * np.sin(pi * lprime / 180)
        - (412 / 3600) * np.sin(2 * pi * F / 180)
        - (212 / 3600) * np.sin(2 * (l - D) * pi / 180)
        - (206 / 3600) * np.sin((l + lprime - 2 * D) * pi / 180)
        + (192 / 3600) * np.sin((l + 2 * D) * pi / 180)
        - (165 / 3600) * np.sin((lprime - 2 * D) * pi / 180)
        + (148 / 3600) * np.sin((l - lprime) * pi / 180)
        - (125 / 3600) * np.sin(pi * D / 180)
        - (110 / 3600) * np.sin((l + lprime) * pi / 180)
        - (55 / 3600) * np.sin(2 * (F - D) * pi / 180)
    )

    beta = (
        (18520 / 3600)
        * np.sin(
            (
                F
                + lambda0
                - L0
                + (412 / 3600) * np.sin(2 * pi * F / 180)
                + (541 / 3600) * np.sin(pi * lprime / 180)
            )
            * pi
            / 180
        )
        - (526 / 3600) * np.sin((F - 2 * D) * pi / 180)
        + (44 / 3600) * np.sin((l + F - 2 * D) * pi / 180)
        - (31 / 3600) * np.sin((-l + F - 2 * D) * pi / 180)
        - (25 / 3600) * np.sin((-2 * l + F) * pi / 180)
        - (23 / 3600) * np.sin((lprime + F - 2 * D) * pi / 180)
        + (21 / 3600) * np.sin((-l + F) * pi / 180)
        + (11 / 3600) * np.sin((-lprime + F - 2 * D) * pi / 180)
    )

    r = (
        385000
        - 20905 * np.cos(l * pi / 180)
        - 3699 * np.cos((2 * D - l) * pi / 180)
        - 2956 * np.cos(2 * pi * D / 180)
        - 570 * np.cos(2 * pi * l / 180)
        + 246 * np.cos(2 * (l - D) * pi / 180)
        - 205 * np.cos((lprime - 2 * D) * pi / 180)
        - 171 * np.cos((l + 2 * D) * pi / 180)
        - 152 * np.cos((l + lprime - 2 * D) * pi / 180)
    )

    n = Rx(-eps) @ np.array(
        [
            r * np.cos((lambda0 + 1.3971 * T) * pi / 180) * np.cos(beta * pi / 180),
            r * np.sin((lambda0 + 1.3971 * T) * pi / 180) * np.cos(beta * pi / 180),
            r * np.sin(beta * pi / 180),
        ]
    )

    return n


@njit
def R_sun(
    time: float,
) -> NDArray[np.float64]:
    """
    Low fidelity position of Sun in GCRS (Montenbruck & Gill, p. 70).  The
    input time is given in MJD.  The output units are in km.  The
    longitude, lambda, and the position vector, r, refer to the mean equinox
    and ecliptic of J2000.
    A modification of M to match JPL430 was necessary to have the correct
    sidereal year.From the mean longitude referred to the mean ecliptic and
    the equinox J2000 given in Simon et al., 1994, Astron. Astrophys., 282,
    663. It is possible tha Montenbruck & Gill mistakenly used the
    anomalistic year instead of the sidereal year for lambda.
    """

    T = (time - 51544.5) / 36525.0

    # the following angles are in degrees
    eps = 23.43929111  # obliquity
    Omega_omega = 282.9400
    M = 357.5256 + 35999.049 * T

    Lambda = (
        Omega_omega
        + M
        + 0.32385652710218 * T
        + (6892 / 3600) * np.sin(pi * M / 180)
        + (72 / 3600) * np.sin(2 * pi * M / 180)
    )
    r = 1e6 * (
        149.619 - 2.499 * np.cos(pi * M / 180) - 0.021 * np.cos(2 * pi * M / 180)
    )

    n = np.array(
        [
            r * np.cos(pi * Lambda / 180),
            r * np.sin(pi * Lambda / 180) * np.cos(pi * eps / 180),
            r * np.sin(pi * Lambda / 180) * np.sin(pi * eps / 180),
        ]
    )

    return n


@njit
def R_sun_MEMED(
    time: float,
) -> NDArray[np.float64]:
    """
    Low fidelity position of Sun in MEMED (Montenbruck & Gill, p. 70).  The
    input time is given in MJD.  The output units are in km.  The
    longitude, lambda, and the position vector, r, refer to the mean equinox
    and ecliptic of J2000.
    A modification of M to match JPL430 was necessary to have the correct
    sidereal year.From the mean longitude referred to the mean ecliptic and
    the equinox J2000 given in Simon et al., 1994, Astron. Astrophys., 282,
    663. It is possible tha Montenbruck & Gill mistakenly used the
    anomalistic year instead of the sidereal year for lambda.
    """

    T = (time - 51544.5) / 36525.0

    # the following angles are in degrees
    eps = 23.43929111  # obliquity
    Omega_omega = 282.9400
    M = 357.5256 + 35999.049 * T

    Lambda = (
        Omega_omega
        + M
        + 1.72095652710218 * T
        + (6892 / 3600) * np.sin(pi * M / 180)
        + (72 / 3600) * np.sin(2 * pi * M / 180)
    )
    r = 1e6 * (
        149.619 - 2.499 * np.cos(pi * M / 180) - 0.021 * np.cos(2 * pi * M / 180)
    )

    n = np.array(
        [
            r * np.cos(pi * Lambda / 180),
            r * np.sin(pi * Lambda / 180) * np.cos(pi * eps / 180),
            r * np.sin(pi * Lambda / 180) * np.sin(pi * eps / 180),
        ]
    )

    return n
