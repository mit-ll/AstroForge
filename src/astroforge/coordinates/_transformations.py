# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import warnings
from typing import Callable, Tuple

import numpy as np
from numba import njit
from numpy import pi
from numpy.typing import NDArray

from ..constants import GM as GM_Earth
from ._base_rotations import Rx, Ry, Rz
from ._utilities import nutate, polarmotion

__all__ = [
    "CIRSToMEMED",
    "CIRSToTETED",
    "ITRSToMEMED",
    "ITRSToTIRS",
    "ITRSToTETED",
    "ITRSToLatLonAlt",
    "ITRSToSEZ",
    "LatLonAltToITRS",
    "MEMEDToCIRS",
    "MEMEDToITRS",
    "PosVelConversion",
    "PosVelToFPState",
    "SEZToAzElRange",
    "TETEDToCIRS",
    "TETEDToITRS",
    "TIRSToITRS",
    "cartesian_to_keplerian",
    "keplerian_to_cartesian",
    "true_anomaly_from_mean_anomaly",
    "true_anomaly_from_eccentric_anomaly",
    "mean_anomaly_from_true_anomaly",
    "eccentric_anomaly_from_true_anomaly",
    "eccentric_anomaly_from_mean_anomaly",
    "ConvergenceException",
]


class ConvergenceException(Exception):
    pass


class PrecisionWarning(UserWarning):
    pass


@njit
def CIRSToMEMED(
    time: float,
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Converts a cartesian vector from the Celestial Intermediate Reference System (CIRS)
    to the Mean Equator Mean Equinox of Date (MEMED) coordinate system. See
    `Kaplan (2005) <Kaplan>`_ for more information.

    Parameters
    ----------
    time : float
        Reference epoch for the MEMED coordinate system, specified as Modified Julian Date (MJD),
        in the UT1 time system.
    X : NDArray[np.float64]
        Shape (3,) cartesian vector to be rotated from CIRS to MEMED

    Returns
    -------
    NDArray[np.float64]
        Rotated vector


    .. [Kaplan]: https://aa.usno.navy.mil/downloads/Circular_179.pdf

    """

    T = (time - 51544.5) / 36525.0
    epsilon = (pi / (180 * 3600)) * (
        -0.014506
        - 4612.156534 * T
        - 1.3915817 * T**2
        + 0.00000044 * T**3
        + 0.000029956 * T**4
        + 0.0000000368 * T**5
    )
    Y = Rz(epsilon) @ X
    return Y


@njit
def CIRSToTETED(
    time: float,
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Converts a cartesian vector from the Celestial Intermediate Reference System (CIRS)
    to the True Equator True Equinox of Date (TETED) coordinate system. See
    `Kaplan (2005) <Kaplan>`_ for more information.

    Parameters
    ----------
    time : float
        Reference epoch for the TETED coordinate system, specified as Modified Julian Date (MJD),
        in the UT1 time system.
    X : NDArray[np.float64]
        Shape (3,) cartesian vector to be rotated from CIRS to TETED

    Returns
    -------
    NDArray[np.float64]
        Rotated vector
    """

    T = (time - 51544.5) / 36525.0
    (dpsi, deps, TrueObliquity, _) = nutate(time)
    eps = TrueObliquity - deps
    F = (
        (
            335779.526232
            + 1739527262.8478 * T
            - 12.7512 * T**2
            - 0.001037 * T**3
            + 0.00000417 * T**4
        )
        * pi
        / 180
        / 3600
    )
    D = (
        (
            1072260.70369
            + 1602961601.2090 * T
            - 6.3706 * T**2
            + 0.006593 * T**3
            - 0.00003169 * T**4
        )
        * pi
        / 180
        / 3600
    )
    Omega = (
        (
            450160.398036
            - 6962890.5431 * T
            + 7.4722 * T**2
            + 0.007702 * T**3
            - 0.00005939 * T**4
        )
        * pi
        / 180
        / 3600
    )

    epsilon = (pi / (180 * 3600)) * (
        -0.014506
        - 4612.156534 * T
        - 1.3915817 * T**2
        + 0.00000044 * T**3
        + 0.000029956 * T**4
        + 0.0000000368 * T**5
        - dpsi * np.cos(eps) * 3600 * 180 / pi
        - 0.00264096 * np.sin(Omega)
        - 0.00006352 * np.sin(2 * Omega)
        - 0.00001175 * np.sin(2 * F - 2 * D + 3 * Omega)
        - 0.00001121 * np.sin(2 * F - 2 * D + Omega)
        + 0.00000455 * np.sin(2 * F - 2 * D + 2 * Omega)
        - 0.00000202 * np.sin(2 * F + 3 * Omega)
        - 0.00000198 * np.sin(2 * F + Omega)
        + 0.00000172 * np.sin(3 * Omega)
        + 0.00000087 * T * np.sin(Omega)
    )

    Y = Rz(epsilon) @ X
    return Y


@njit
def ITRSToMEMED(
    time: float,
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate a cartesian vector from the International Terrestrial Reference System (ITRS)
    to the Mean Equator Mean Equinox of Date (MEMED) coordinate system. See
    `Kaplan (2005) <Kaplan>`_ for more information.

    Parameters
    ----------
    time : float
        Reference epoch for the MEMED coordinate system, specified as Modified Julian Date (MJD),
        in the UT1 time system.
    X : NDArray[np.float64]
        Shape (3,) cartesian vector to be rotated from ITRS to MEMED

    Returns
    -------
    NDArray[np.float64]
        Rotated vector
    """

    omega = 1.00273781191135448  # rev/day
    T = time - 51544.5

    #  ITRSToTIRS (polar motion)
    #  TIRSToCIRS (earth rotation angle)
    #  CIRSToGCRS (bias, precession, nutation)
    angle = 2 * pi * (0.7790572732640 + omega * T)

    X1 = Rz(-angle) @ X
    Y = CIRSToMEMED(time, X1)
    return Y


def ITRSToTIRS(
    mjd: float,
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate a cartesian vector from the International Terrestrial Reference System (ITRS)
    to the Terrestrial Intermediate Reference System(TIRS) coordinate system. TIRS differs from
    ITRS by the polar motion. See `Kaplan (2005) <Kaplan>`_ for more information.

    Parameters
    ----------
    time : float
        Time at which to evaluate the polar motion, specified as Modified Julian Date (MJD),
        in the UT1 time system.
    X : NDArray[np.float64]
        Shape (3,) cartesian vector to be rotated from ITRS to TIRS

    Returns
    -------
    NDArray[np.float64]
        Rotated vector
    """

    xp, yp = polarmotion(mjd)
    if (isinstance(xp, float)) and (isinstance(yp, float)):
        Y = Ry(xp * pi / 180 / 3600) @ Rx(yp * pi / 180 / 3600) @ X
    else:
        raise TypeError(
            "The output of polarmotion for this operation must be type `float` "
            "to support usage in rotation matrix methods."
        )
    return Y


@njit
def ITRSToTETED(
    time: float,
    X: NDArray[np.float64],
    V: NDArray[np.float64] | None = None,
) -> NDArray[np.float64] | Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotate a cartesian vector from the International Terrestrial Reference System (ITRS)
    to the True Equator True Equinox of Date (TETED) coordinate system. See
    `Kaplan (2005) <Kaplan>`_ for more information.

    Parameters
    ----------
    time : float
        Time epoch for the conversion, specified as Modified Julian Date (MJD),
        in the UT1 time system.
    X : NDArray[np.float64]
        Shape (3,) position vector to be rotated
    V : Optional[NDArray[np.float64]]
        Shape (3,) velocity vector to be rotated

    Returns
    -------
    Y : NDArray[np.float64]
        Rotated vector

    V : NDArray[np.float64]
        Rotated velocity vector. Only provided if an input velocity vector is given.
    """

    omega = 1.00273781191135448  # rev/day
    T = time - 51544.5

    # ITRSToTIRS (polar motion)
    X2 = ITRSToTIRS(time, X)
    # TIRSToCIRS (earth rotation angle)
    # CIRSToGCRS (bias, precession, nutation)
    angle = 2 * pi * (0.7790572732640 + omega * T)

    X1 = Rz(-angle) @ X2
    Y = CIRSToTETED(time, X1)

    if V is not None:
        V_trans = CIRSToTETED(
            time,
            Rz(-angle) @ V + omega * 2 * pi / 86400 * np.array([-X1[1], X1[0], 0]),
        )
        return (Y, V_trans)
    else:
        return Y


@njit
def ITRSToLatLonAlt(
    z: NDArray[np.float64],
) -> Tuple[float, float, float]:
    """Converts a cartesian vector from the International Terrestrial Reference System (ITRS)
    to geodetic latitute, longitude, and altitude. This method is a wrapper around `astropy`
    utilities.

    .. note::
        Output latitude and longitude are in degrees. Output altitude is in kilometers.

    Parameters
    ----------
    z : NDArray[np.float64]
        Shape (3,) cartesian vector in ITRS

    Returns
    -------
    lat : float
        Geodetic latitude (degrees)

    lon : float
        Longitude (degrees)

    alt : float
        Height above the WGS84 reference ellipsoid (km)

    """

    # almost a direct copy from https://github.com/liberfa/erfa `gc2gd`
    # WGS84 parameters
    R_earth = 6378.137  # equatorial radius
    f = 1 / 298.257223563  # flattening

    # Functions of ellipsoid parameters (with further validation of f)
    aeps2 = R_earth**2 * 1e-32
    e2 = (2.0 - f) * f
    e4t = e2**2 * 1.5
    ec2 = 1.0 - e2
    assert ec2 > 0, "Invalid WGS84 parameters"
    ec = np.sqrt(ec2)
    b = R_earth * ec

    # cartesian components
    x, y, z = z

    # distance from polar axis squared
    p2 = x**2 + y**2

    # longitude
    lon = np.arctan2(y, x) if p2 > 0.0 else 0.0

    # unsigned z-coordinate
    absz = np.abs(z)

    if p2 > aeps2:
        # distance from polar axis
        p = np.sqrt(p2)

        # normalization
        s0 = absz / R_earth
        pn = p / R_earth
        zc = ec * s0

        # prepare Newton correction factors
        c0 = ec * pn
        c02 = c0**2
        c03 = c02 * c0
        s02 = s0**2
        s03 = s02 * s0
        a02 = c02 + s02
        a0 = np.sqrt(a02)
        a03 = a02 * a0
        d0 = zc * a03 + e2 * s03
        f0 = pn * a03 - e2 * c03

        # prepare Halley correction factors
        b0 = e4t * s02 * c02 * pn * (a0 - ec)
        s1 = d0 * f0 - b0 * s0
        cc = ec * (f0**2 - b0 * c0)

        # evaluate latitude and height
        lat = np.arctan(s1 / cc)
        s12 = s1**2
        cc2 = cc**2
        height = (p * cc + absz * s1 - R_earth * np.sqrt(ec2 * s12 + cc2)) / np.sqrt(
            s12 + cc2
        )
    else:
        lat = np.pi / 2.0
        height = absz - b

    return lat * 180 / np.pi, lon * 180 / np.pi, height


@njit
def ITRSToSEZ(
    X: NDArray[np.float64],
    X0: NDArray[np.float64],
    lat: float,
    lon: float,
) -> NDArray[np.float64]:
    """Converts cartesian vector in the International Terrestrial Reference System (ITRS)
    to a South-East-Zenith (SEZ) relative to a specific geodetic location.

    Parameters
    ----------
    X : NDArray[np.float64]
        Vector to be rotated, in ITRS (km)
    X0 : NDArray[np.float64]
        Reference location, in ITRS (km)
    lat : float
        Geodetic latitude of the reference location (degrees)
    lon : float
        Longitude of the reference location (degrees)

    Returns
    -------
    NDArray[np.float64]
        Rotated vector in SEZ coordinate frame
    """

    #  X0 is the location of the reference site in ITRS
    #  lat is its geodetic latitude in degrees
    #  long is its geodetic longitude in degrees

    X1 = X - X0
    X2 = Rz(lon * pi / 180) @ X1

    Y = Ry(pi / 2 - lat * pi / 180) @ X2
    return Y


@njit
def LatLonAltToITRS(
    lat: float,
    lon: float,
    alt: float,
) -> NDArray[np.float64]:
    """Convert a geodetic latitude, longitude, and altitude to a cartesian position
    vector in the International Terrestrial Reference System (ITRS).

    .. note::

        Latitude and longitude inputs should be specified in degrees.

    Parameters
    ----------
    lat : float
        Geodetic latitude of the site (degrees)
    lon : float
        Longitude of the site (degrees)
    alt : float
        Height above the WGS84 reference ellipsoid (km)

    Returns
    -------
    NDArray[np.float64]
        ITRS position vector
    """

    #  lat is its geodetic latitude in degrees
    #  long is its geodetic longitude in degrees
    R_earth = 6378.137
    f = 1 / 298.257223563

    N = R_earth / np.sqrt(1 - f * (2 - f) * np.sin(lat * pi / 180) ** 2)
    x = np.array(
        [
            (N + alt) * np.cos(lat * pi / 180) * np.cos(lon * pi / 180),
            (N + alt) * np.cos(lat * pi / 180) * np.sin(lon * pi / 180),
            ((1 - f) ** 2 * N + alt) * np.sin(lat * pi / 180),
        ]
    )

    return x


@njit
def MEMEDToCIRS(
    time: float,
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Converts a cartesian vector from the Mean Equator Mean Equinox of Date (MEMED)
    coordinate system to the Celestrial Intermediate Reference System (CIRS). CIRS is
    a geocentric coordinate system whose x-axis is in the direction of the Celestial
    Intermediate Origin (CIO) and whose z-axis is in the direction of the Celestial
    Intermediate Pole (CIP). See `Kaplan (2005) <Kaplan>`_ for more information.

    Parameters
    ----------
    time : float
        Reference epoch for the MEMED coordinate system, specified as a Modified Julian
        Date (MJD) in the UT1 time system.
    X : NDArray[np.float64]
        Shape (3,) vector to be rotated

    Returns
    -------
    NDArray[np.float64]
        Rotated vector in the CIRS system
    """

    T = (time - 51544.5) / 36525.0

    epsilon = (pi / (180 * 3600)) * (
        -0.014506
        - 4612.156534 * T
        - 1.3915817 * T**2
        + 0.00000044 * T**3
        + 0.000029956 * T**4
        + 0.0000000368 * T**5
    )

    Y = Rz(-epsilon) @ X
    return Y


@njit
def MEMEDToITRS(
    time: float,
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert a cartesian vector from the Mean Equator Mean Equinox of Date (MEMED)
    coordinate system to the Internation Terrestrial Reference System (ITRS). See
    `Kaplan (2005) <Kaplan>`_ for more information.

    Parameters
    ----------
    time : float
        Reference epoch for the MEMED coordinate system, specified as a Modified
        Julian Date (MJD) in the UT1 time system.
    X : NDArray[np.float64]
        Shape (3,) vector to be rotated

    Returns
    -------
    NDArray[np.float64]
        Rotated vector
    """

    omega = 1.00273781191135448  # rev/day
    T = time - 51544.5

    #  GCRSToCIRS (bias, precession, nutation)
    X2 = MEMEDToCIRS(time, X)

    #  CIRSToTIRS (earth rotation angle)
    angle = 2 * pi * (0.7790572732640 + omega * T)
    Y = Rz(angle) @ X2

    #  TIRSToITRS (polar motion)
    return Y


def PosVelConversion(
    conversion: Callable,
    mjd: float,
    X0: NDArray[np.float64],
    V0: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Utility for converting a cartesian position and velocity vectors from one
    coordinate system to another. The first argument should be the conversion method.

    Parameters
    ----------
    conversion : Callable
        Method to call on the position and velocity vectors to do the conversion
    mjd : float
        Reference time for the conversion, specified as a Modified Julian Date (MJD)
        in the UT1 time system.
    X0 : NDArray[np.float64]
        Position vector to be rotated
    V0 : NDArray[np.float64]
        Velocity vector to be rotated

    Returns
    -------
    X, V : Tuple[NDArray[np.float64], NDArray[np.float64]]
        Rotated position and velocity vectors

    Examples
    --------

    Rotate an ITRS site location (fixed relative to a rotating Earth) to pos/vel vectors
    in the True Equator True Equinox of Date (TETED) coordinate system:

    >>> # pick a site location
    >>> lat, lon, alt = 42.459629, -71.267319, 0.0
    >>>
    >>> # convert lat/lon/alt to an ITRS position vector
    >>> x_itrs = LatLonAltToITRS(lat, lon, alt)
    >>> v_itrs = np.array([0.0, 0.0, 0.0])  # site isn't moving in earth-fixed coordinates
    >>>
    >>> # time, used as a reference epoch for the TETED coordinate system
    >>> mjd = 51720.0
    >>>
    >>> # do the conversion
    >>> x_teted, v_teted = PosVelConversion(ITRSToTETED, mjd, x_itrs, v_itrs)
    >>> with np.printoptions(suppress=True):
    >>>     print(f"{x_teted = }")
    >>>     print(f"{v_teted = }")
    x_teted = array([-4364.24798307, -1778.39228689,  4283.414358  ])
    v_teted = array([ 0.12968241, -0.31824598, -0.        ])

    """

    eps = 1e-4

    X = conversion(mjd, X0)
    V = (
        conversion(mjd, V0)
        + (conversion(mjd + eps, X0) - conversion(mjd - eps, X0)) / 2.0 / eps / 86400.0
    )

    return X, V


def PosVelToFPState(
    x: NDArray[np.float64],
    v: NDArray[np.float64],
    x_site: NDArray[np.float64],
    v_site: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute observables for a target with a given position and velocity as
    seen by a sensor with a particular position and velocity.

    .. note::

        This method was originally intended to compute angles-only observations
        from the sensor location, thus the acryonym "FP" in the method name, which
        stands for Focal Plane. It has since been adapted to also compute the range
        and range-rate, which are observables typical for radar sensors.

    Parameters
    ----------
    x : NDArray[np.float64]
        Position vector of the target(s). Should have shape (3, ) or (N, 3).
    v : NDArray[np.float64]
        Velocity vector of the target(s). Should have shape (3, ) or (N, 3).
    x_site : NDArray[np.float64]
        Position vector of the sensor. Should have shape (3, ) or (N, 3).
    v_site : NDArray[np.float64]
        Velocity vector of the sensor. Should have shape (3, ) or (N, 3).

    Returns
    -------
    fpstate : NDArray[np.float64]
        2D array of shape (N, 4) with the columns corresponding to right ascension
        (:math:`\\alpha`), declination (:math:`\\delta`), and their rates of change
        (:math:`\\dot{\\alpha}, \\dot{\\delta}`).

    r : NDArray[np.float64]
        1D array of shape (N, ) of range values (:math:`r`)

    r_rate : NDArray[np.float64]
        1D array of shape (N, ) of range-rate values (:math:`\\dot{r}`)

    Examples
    --------

    Compute observations from a ground-based sensor:

    >>> # make up a site location
    >>> lat, lon, alt = 42.459629, -71.267319, 0.0
    >>>
    >>> # time of the observation
    >>> mjd = 51720.0
    >>>
    >>> # rotate site location into inertial coordinates
    >>> x_site_itrs = af.coordinates.LatLonAltToITRS(lat, lon, alt)
    >>> v_site_itrs = np.zeros(3)
    >>> x_site, v_site = PosVelConversion(ITRSToTETED, mjd, x_site_itrs, v_site_itrs)
    >>>
    >>> # make up some satellite ephemeris data
    >>> alt = 400.0
    >>> r = af.R_earth + alt  # note: R_earth is *equatorial* radius
    >>> v = np.sqrt(af.GM / r)
    >>> xdir = x_site / np.sqrt(x_site @ x_site)   # unit vector pointing from Earth center to site
    >>> vdir = np.cross(xdir, np.array([1.0, 0.0, 0.0]))
    >>> vdir /= np.sqrt(vdir @ vdir)
    >>>
    >>> x = r * xdir
    >>> v = v * vdir
    >>>
    >>> # compute observation
    >>> angles, rho, rhodot = af.coordinates.PosVelToFPState(x, v, x_site, v_site)
    >>> with np.printoptions(suppress=True):
    >>>     print(f"{angles = }")
    >>>     print(f"{rho  = }")
    >>>     print(f"{rhodot = }")
    angles = array([[-2.75464514,  0.73771759, -0.02276666,  0.00969875]])
    rho  = array([409.70091677])
    rhodot = array([0.])

    .. note::

        The range is not exactly 400 km because the Earth is not a perfect sphere. The radius of the
        Earth at the latitude used in this examples is less than the equatorial radius, and therefore
        the satellite is further away.

    """

    if x.ndim == 2:
        N = x.shape[0]
        x = x.T
        v = v.T
        x_site = x_site.T
        v_site = v_site.T
    else:
        N = 1
        x = np.atleast_2d(x).T
        v = np.atleast_2d(v).T
        x_site = np.atleast_2d(x_site).T
        v_site = np.atleast_2d(v_site).T

    fp_state = np.zeros((N, 4))

    dr = x - x_site
    dv = v - v_site

    r = np.sqrt(np.sum(dr**2, axis=0))

    e = dr / r  # will broadcast

    r_rate = np.sum(dv * e, axis=0)
    dv -= r_rate * e
    ep = dv / r

    fp_state[:, 0] = np.arctan2(e[1], e[0])

    I = (e[2] < 1) & (e[2] > -1)
    temp = np.arcsin(e[2, I])

    fp_state[e[2] >= 1, 1] = np.pi / 2
    fp_state[e[2] <= -1, 1] = -np.pi / 2
    fp_state[I, 1] = temp

    fp_state[:, 2] = (
        -np.sin(fp_state[:, 0]) * ep[0] + np.cos(fp_state[:, 0]) * ep[1]
    ) / np.cos(fp_state[:, 1])
    fp_state[:, 3] = ep[2] / np.cos(fp_state[:, 1])

    return (fp_state, r, r_rate)


@njit
def SEZToAzElRange(
    X: NDArray[np.float64],
) -> Tuple[float, float, float]:
    """Compute the azimuth, elevation, and range given a displacement
    in the SEZ coordinate frame.

    Parameters
    ----------
    X : NDArray[np.float64]
        Displacement vector in SEZ frame. Should have shape (3, )

    Returns
    -------
    az : float
        Azimuth angle (degrees)

    el : float
        Elevation angle (degrees)

    r : float
        Range (km)

    """
    r = float(np.sqrt(X @ X))
    el = (180 / pi) * np.arcsin(X[2] / r)
    az = 180 - (180 / pi) * np.arctan2(X[1], X[0])

    return az, el, r


@njit
def TETEDToCIRS(
    time: float,
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate a cartesian vector from the True Equator True Equinox of Date (TETED)
    coordinate system to the Celestial Intermediate Reference System (CIRS). CIRS is
    a geocentric coordinate system whose x-axis is in the direction of the Celestial
    Intermediate Origin (CIO) and whose z-axis is in the direction of the Celestial
    Intermediate Pole (CIP). See `Kaplan (2005) <Kaplan>`_ for more information.

    Parameters
    ----------
    time : float
        Reference epoch for the TETED frame, expressed as a Modified Julian Date (MJD)
        in the UT1 time system.
    X : NDArray[np.float64]
        Vector to be rotated. Should have shape (3, )

    Returns
    -------
    Y : NDArray[np.float64]
        Rotated vector

    """

    T = (time - 51544.5) / 36525.0
    (dpsi, deps, TrueObliquity, _) = nutate(time)
    eps = TrueObliquity - deps
    F = (
        (
            335779.526232
            + 1739527262.8478 * T
            - 12.7512 * T**2
            - 0.001037 * T**3
            + 0.00000417 * T**4
        )
        * pi
        / 180
        / 3600
    )
    D = (
        (
            1072260.70369
            + 1602961601.2090 * T
            - 6.3706 * T**2
            + 0.006593 * T**3
            - 0.00003169 * T**4
        )
        * pi
        / 180
        / 3600
    )
    Omega = (
        (
            450160.398036
            - 6962890.5431 * T
            + 7.4722 * T**2
            + 0.007702 * T**3
            - 0.00005939 * T**4
        )
        * pi
        / 180
        / 3600
    )

    epsilon = (pi / (180 * 3600)) * (
        -0.014506
        - 4612.156534 * T
        - 1.3915817 * T**2
        + 0.00000044 * T**3
        + 0.000029956 * T**4
        + 0.0000000368 * T**5
        - dpsi * np.cos(eps) * 3600 * 180 / pi
        - 0.00264096 * np.sin(Omega)
        - 0.00006352 * np.sin(2 * Omega)
        - 0.00001175 * np.sin(2 * F - 2 * D + 3 * Omega)
        - 0.00001121 * np.sin(2 * F - 2 * D + Omega)
        + 0.00000455 * np.sin(2 * F - 2 * D + 2 * Omega)
        - 0.00000202 * np.sin(2 * F + 3 * Omega)
        - 0.00000198 * np.sin(2 * F + Omega)
        + 0.00000172 * np.sin(3 * Omega)
        + 0.00000087 * T * np.sin(Omega)
    )

    Y = Rz(-epsilon) @ X

    return Y


def TETEDToITRS(
    time: float,
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate a cartesian vector from the True Equator True Equinox of Date (TETED)
    coordinate system to the International Terrestrial Reference System (ITRS).

    Parameters
    ----------
    time : float
        Reference epoch for the TETED frame, expressed as a Modified Julian Date (MJD)
        in the UT1 time system.
    X : NDArray[np.float64]
        Vector to be rotated, should have shape (3, )

    Returns
    -------
    Y : NDArray[np.float64]
        Rotated vector

    """

    omega = 1.00273781191135448  # rev/day
    T = time - 51544.5

    #  GCRSToCIRS (bias, precession, nutation)
    X2 = TETEDToCIRS(time, X)

    #  CIRSToTIRS (earth rotation angle)
    angle = 2 * pi * (0.7790572732640 + omega * T)

    X3 = Rz(angle) @ X2

    #  TIRSToITRS (polar motion)
    Y = TIRSToITRS(time, X3)

    return Y


def TIRSToITRS(
    mjd: float,
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate a cartesian vector from the Terrestrial Intermediate Reference System (TIRS)
    to the International Terrestrial Reference System (TIRS). TIRS and ITRS only differ by
    the polar motion.

    Parameters
    ----------
    mjd : float
        Time at which to evaluate the polar motion, expressed as a Modified Julian Date
        (MJD) in the UT1 time system.
    X : NDArray[np.float64]
        Vector to be rotated, should have shape (3, )

    Returns
    -------
    Y : NDArray[np.float64]
        Rotated vector
    """

    (xp, yp) = polarmotion(mjd)
    if isinstance(xp, float) and isinstance(yp, float):
        Y = Rx(-yp * pi / 180 / 3600) @ Ry(-xp * pi / 180 / 3600) @ X
    else:
        raise TypeError(
            "The output of polarmotion for this operation must be type `float` "
            "to support usage in rotation matrix methods."
        )
    return Y


def cartesian_to_keplerian(
    pos: NDArray[np.float64],
    vel: NDArray[np.float64],
    GM: float = GM_Earth,
) -> dict:
    """Convert Cartesian coordinates to Keplerian elements

    Parameters
    ----------
    pos : NDArray[np.float64]
        Position vector wrt attractor center, in km
    vel : NDArray[np.float64]
        Velocity vector, in km/s
    GM : float, optional
        Standard gravitational parameter (the product of the gravitational constant and the mass of a given astronomical
        body such as the Sun or Earth) in km^3/s^2, by default 398600.4418

    Returns
    -------
    dict
        Keplerian elements dictionary

    References
    ----------
    .. [1] Schwarz, Rene (2017). "Cartesian State Vectors -> Keplerian Orbit
    Elements". https://downloads.rene-schwarz.com/download/M002-Cartesian_State_Vectors_to_Keplerian_Orbit_Elements.pdf
    """

    # Replace position zeros w/ 1e-21
    if np.any(pos == 0):
        warnings.warn(
            "Zero(s) detected in position array. Any zero will be replaced with 1e-21 to avoid "
            "any divide-by-zero issues.",
            PrecisionWarning,
        )
        pos[pos == 0] = 1e-21

    # Calculate magnitude of pos & vel vectors
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)

    # Calculate specific angular momentum vector
    h = np.cross(pos, vel)

    # Calculate orbital inclination
    inclination_rad = np.arccos(h[2] / np.linalg.norm(h))

    # Calculate eccentricity vector & eccentricity
    ecc_vec = ((v**2 - GM / r) * pos - np.dot(pos, vel) * vel) / GM
    ecc = np.linalg.norm(ecc_vec)

    # Calculate semi-major axis
    semi_major_axis_km = 1 / ((2 / r) - (v**2 / GM))

    # Calculate longitude of ascending node
    n = np.cross(np.array([0, 0, 1]), h)
    n_mag = np.linalg.norm(n)
    if n_mag == 0:
        raan_rad = 0.0
    else:
        raan_rad = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            raan_rad = 2 * np.pi - raan_rad

    # Calculate argument of perigee
    internal_calc = np.dot(n, ecc_vec) / (n_mag * ecc)
    if (internal_calc > 1.0) and np.isclose(internal_calc, 1):
        internal_calc = 1.0
    argp_rad = np.arccos(internal_calc)
    if ecc_vec[2] < 0:
        argp_rad = 2 * np.pi - argp_rad

    # Calculate True Anomaly
    internal_calc = np.dot(ecc_vec, pos) / (ecc * r)
    if (internal_calc > 1.0) and np.isclose(internal_calc, 1):
        internal_calc = 1.0
    true_anomaly_rad = np.arccos(internal_calc)
    if np.dot(pos, vel) < 0:
        true_anomaly_rad = 2 * np.pi - true_anomaly_rad
    # Calculate Eccentric Anomaly
    ecc_anomaly_rad = 2 * np.arctan2(
        np.sqrt(1 - ecc) * np.tan(true_anomaly_rad / 2), np.sqrt(1 + ecc)
    )
    # Calculate Mean Anomaly
    mean_anomaly_rad = ecc_anomaly_rad - ecc * np.sin(ecc_anomaly_rad)

    return {
        "inclination_rad": inclination_rad,
        "raan_rad": raan_rad,
        "argp_rad": argp_rad,
        "eccentricity": ecc,
        "semi_major_axis_km": semi_major_axis_km,
        "eccentric_anomaly_rad": ecc_anomaly_rad,
        "true_anomaly_rad": true_anomaly_rad,
        "mean_anomaly_rad": mean_anomaly_rad,
    }


def keplerian_to_cartesian(
    inclination_rad: float,
    raan_rad: float,
    argp_rad: float,
    ecc: float,
    semi_major_axis_km: float,
    mean_anomaly_rad: float,
    GM: float = GM_Earth,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert Keplerian elements to Cartesian coordinates

    Parameters
    ----------
    inclination_rad : float
        Orbital inclination in radians
    raan_rad : float
        Right ascension of the ascending node in radians
    argp_rad : float
        Argument of pericenter in radians
    ecc : float
        Eccentricity, a number between 0 and 1
    semi_major_axis_km : float
        Semi-major axis in km
    mean_anomaly_rad : float
        Mean anomaly in radians
    GM : float, optional
       Standard gravitational parameter (the product of the gravitational constant and the mass of a given astronomical
       body such as the Sun or Earth) in km^3/s^2, by default 398600.4418

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        First element of tuple is X : (N,3), position vector. Coordinate system is TETED. Units are km.
        Second element of tuple is V : (N,3), velocity vector. Coordinate system is TETED. Units are km/s.
    """
    E0 = mean_anomaly_rad
    E = mean_anomaly_rad + ecc * np.sin(mean_anomaly_rad)
    count = 0
    while (np.sum((E - E0) ** 2) > 1.0e-12) and (count < 100):
        E0 = E
        E = mean_anomaly_rad + ecc * np.sin(E)
        count += 1

    nu = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E / 2))
    p = semi_major_axis_km * (1 - ecc**2)

    rx = p * np.cos(nu) / (1 + ecc * np.cos(nu))
    ry = p * np.sin(nu) / (1 + ecc * np.cos(nu))
    vx = -np.sqrt(GM / p) * np.sin(nu)
    vy = np.sqrt(GM / p) * (ecc + np.cos(nu))

    ci = np.cos(inclination_rad)
    si = np.sin(inclination_rad)
    cO = np.cos(raan_rad)
    sO = np.sin(raan_rad)
    co = np.cos(argp_rad)
    so = np.sin(argp_rad)

    a11 = cO * co - sO * so * ci
    a12 = -cO * so - sO * co * ci
    a21 = sO * co + cO * so * ci
    a22 = -sO * so + cO * co * ci
    a31 = so * si
    a32 = co * si

    X = np.array([a11 * rx + a12 * ry, a21 * rx + a22 * ry, a31 * rx + a32 * ry])
    V = np.array([a11 * vx + a12 * vy, a21 * vx + a22 * vy, a31 * vx + a32 * vy])

    return X, V


def true_anomaly_from_mean_anomaly(
    e: float,
    M: float,
    tolerance: float = 1.0e-14,
    max_iterations: int = 100,
) -> float:
    """True anomaly from mean anomaly

    Parameters
    ----------
    e : float
        Eccentricity, a number between 0 and 1
    M : float
        Mean anomaly in radians
    tolerance : float, optional
        Numerical solver convergence tolerance, by default 1.0e-14
    max_iterations : int, optional
        Maximum number of iterations for the numerical solver, by default 100

    Returns
    -------
    float
        True anomaly in radians
    """
    E = eccentric_anomaly_from_mean_anomaly(
        e,
        M,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    v = true_anomaly_from_eccentric_anomaly(e, E)
    return v


def true_anomaly_from_eccentric_anomaly(
    e: float,
    E: float,
) -> float:
    """True anomaly from eccentric anomaly

    Parameters
    ----------
    e : float
        Eccentricity, a number between 0 and 1
    E : float
        Eccentric anomaly in radians

    Returns
    -------
    float
        True anomaly in radians

    References
    ----------
    .. [1] Broucke, R.; Cefola, P. (1973). "A Note on the Relations between True and Eccentric Anomalies in the Two-Body
        Problem". Celestial Mechanics. 7 (3): 388-389.
    """
    beta = e / (1 + np.sqrt(1 - e**2))
    v = E + 2 * np.arctan(beta * np.sin(E) / (1 - beta * np.cos(E)))
    return v


def mean_anomaly_from_true_anomaly(
    e: float,
    v: float,
) -> float:
    """Mean anomaly from true anomaly

    Parameters
    ----------
    e : float
        Eccentricity, a number between 0 and 1
    v : float
        True anomaly in radians

    Returns
    -------
    float
        Mean anomaly in radians

    References
    ----------
    .. [1] Smart, W. M. (1977). "Textbook on Spherical Astronomy" (sixth ed.). Cambridge University Press, Cambridge.
        p. 113.
    """
    E = eccentric_anomaly_from_true_anomaly(e, v)
    M = E - e * np.sin(E)
    return M


def eccentric_anomaly_from_true_anomaly(
    e: float,
    v: float,
) -> float:
    """Eccentric anomaly from true anomaly

    Parameters
    ----------
    e : float
        Eccentricity, a number between 0 and 1
    v : float
        True anomaly in radians

    Returns
    -------
    float
        Eccentric anomaly in radians

    References
    ----------
    .. [1] Tsui, James Bao-yen (2000). "Fundamentals of Global Positioning System receivers: A software approach"
        (3rd ed.). John Wiley & Sons. p. 48.
    """
    E = np.mod(
        np.arctan2(np.sqrt(1 - e**2) * np.sin(v), e + np.cos(v)),
        2 * np.pi,
    )
    return E


def eccentric_anomaly_from_mean_anomaly(
    e: float,
    M: float,
    tolerance: float = 1.0e-14,
    max_iterations: int = 100,
) -> float:
    """Eccentric anomaly from mean anomaly

    Parameters
    ----------
    e : float
        Eccentricity, a number between 0 and 1
    M : float
        Mean anomaly in radians
    tolerance : float, optional
        Numerical solver convergence tolerance, by default 1.0e-14
    max_iterations : int, optional
        Maximum number of iterations for the numerical solver, by default 100

    Returns
    -------
    float
        Eccentric anomaly in radians

    References
    ----------
    .. [1] Murison, Marc A. "A Practical Method for Solving the Kepler Equation", 2006.
    """

    def eps3(
        e: float,
        M: float,
        x: float,
    ) -> float:
        """Numerical solver terms"""
        t1 = np.cos(x)
        t2 = e * t1 - 1
        t3 = np.sin(x)
        t4 = e * t3
        t5 = t4 + M - x
        t6 = t5 / (1 / 2 * t5 * t4 / t2 + t2)
        return t5 / ((1 / 2 * t3 - 1 / 6 * t1 * t6) * e * t6 + t2)

    Mnorm = np.fmod(M, 2 * np.pi)
    E = 0.0
    E0 = M + (
        -1 / 2 * e**3 + e + (e**2 + 3 / 2 * np.cos(M) * e**3) * np.cos(M)
    ) * np.sin(M)
    dE = tolerance + 1
    count = 0
    while dE > tolerance:
        E = E0 - eps3(e, Mnorm, E0)
        dE = abs(E - E0)
        E0 = E
        count += 1
        if count == max_iterations:
            raise ConvergenceException(
                "Eccentric anomaly calculation failed to converge."
            )
    return E
