# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Other coordinates methods
"""

import json
import os
from importlib import resources
from typing import Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from . import iers

__all__ = [
    "polarmotion",
    "dut1utc",
    "rmfu",
    "obliquity",
    "rmat",
    "nutate",
    "shadow",
]


class PolarMotionData:
    loaded: bool
    mjd: NDArray[np.float64]
    xp: NDArray[np.float64]
    yp: NDArray[np.float64]
    min_mjd: float
    max_mjd: float

    def __init__(self):
        self.loaded = False

    def load(self):
        if not os.path.exists(iers.POLARMOTION_FILENAME):
            iers.setup_iers()

        f = np.load(iers.POLARMOTION_FILENAME)
        self.mjd = f["mjd"]
        self.xp = f["xp"]
        self.yp = f["yp"]
        self.min_mjd = self.mjd.min()
        self.max_mjd = self.mjd.max()
        self.loaded = True


class UTCData:
    loaded: bool
    mjd: NDArray[np.float64]
    dt: NDArray[np.float64]
    min_mjd: float
    max_mjd: float

    def __init__(self):
        self.loaded = False

    def load(self) -> None:
        if not os.path.exists(iers.UT1_UTC_FILENAME):
            iers.setup_iers()

        f = np.load(iers.UT1_UTC_FILENAME)
        self.mjd = f["mjd"]
        self.dt = f["ut1utc"]
        self.min_mjd = min(self.mjd)
        self.max_mjd = max(self.mjd)
        self.loaded = True


class NutationData:
    loaded: bool
    Si: NDArray[np.float64]
    dSi: NDArray[np.float64]
    Ci: NDArray[np.float64]
    dCi: NDArray[np.float64]
    mults: NDArray[np.float64]

    def __init__(self):
        self.loaded = False

    def load(self) -> None:
        # get the path to the packaged data file
        root = resources.files("astroforge")
        fname = root / "coordinates" / "nutation_data.json"
        with open(str(fname), "r") as f:
            data = json.load(f)
            for key in ("Si", "dSi", "Ci", "dCi", "mults"):
                setattr(self, key, np.array(data[key]))

        self.loaded = True


_pm_data: PolarMotionData = PolarMotionData()
_utc_data: UTCData = UTCData()
_nutation_data: NutationData = NutationData()


def polarmotion(
    mjd: float | NDArray[np.float64],
    bounds_check: bool = True,
) -> Tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Computes the motion of the Earth's rotational axis as a function of time.

    For more information, see [#]_

    Parameters
    ----------
    mjd : float | NDArray[np.float64]
        Time(s) at which to compute polar motion. Time is specified as a Modified
        Julian Date (MJD) in the UT1 time system.

        .. note::
            UT1 differs from UTC by < 1 second, which may or may not matter for
            your application. See [#]_ for more information.
    bounds_check : bool, optional
        Flag for turning on or off bounds checking the interpolation of the IERS
        data, by default True

    Returns
    -------
    dx, dy : Tuple[float | NDArray[np.float64], float | NDArray[np.float64]]
        Polar motion value(s) in the x- and y-directions (arcsec)

    References
    ----------
    .. [#] https://www.iers.org/IERS/EN/Science/EarthRotation/PolarMotion.html
    .. [#] https://www.iers.org/IERS/EN/Science/EarthRotation/UT1LOD.html

    Examples
    --------
    Basic usage:

    >>> x, y = polarmotion(59025.0)
    >>> print(x, y)
    0.155409 0.434462

    Multiple time inputs are supported with `numpy` arrays:

    >>> x, y = polarmotion(np.array([59025.0, 59026.0]))
    >>> print(x, y)
    [0.155409 0.156978] [0.434462 0.433877]

    If you know the query times are within the bounds of the IERS data interpolation,
    you can skip the bounds check and speed up the computation slightly:

    >>> x, y = polarmotion(59025.0, bounds_check=False)
    >>> print(x, y)
    0.155409 0.434462
    """

    if not _pm_data.loaded:
        _pm_data.load()

    return _polarmotion_wrapped(
        mjd,
        _pm_data.mjd,
        _pm_data.xp,
        _pm_data.yp,
        bounds_check=bounds_check,
        min_mjd=_pm_data.min_mjd,
        max_mjd=_pm_data.max_mjd,
    )


@njit
def _polarmotion_wrapped(
    mjd: float | NDArray[np.float64],
    tab_mjd: NDArray[np.float64],
    tab_xp: NDArray[np.float64],
    tab_yp: NDArray[np.float64],
    bounds_check: bool = True,
    min_mjd: float = -np.inf,
    max_mjd: float = np.inf,
) -> Tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    if bounds_check:
        if np.min(np.asarray(mjd)) < min_mjd or np.max(np.asarray(mjd)) > max_mjd:
            raise ValueError(
                (
                    "Requested UT1-UTC value at supplied time is outside "
                    "the bounds on the *current* IERS file."
                )
            )

    out = (
        np.interp(mjd, tab_mjd, tab_xp),
        np.interp(mjd, tab_mjd, tab_yp),
    )
    return out


def dut1utc(
    mjd: float | NDArray[np.float64],
    bounds_check: bool = True,
) -> float | NDArray[np.float64]:
    """Interpolates the IERS data for the UT1-UTC difference to the time(s) given.

    For more information, see [#]_

    Parameters
    ----------
    mjd : float | NDArray[np.float64]
        Time(s) at which to compute the UT1-UTC offset
    bounds_check : bool, optional
        Flag for turning on or off bounds checking the interpolation of the IERS
        data, by default True

    Returns
    -------
    delta : float | NDArray[np.float64]
        UT1-UTC offset at the time(s) given

    References
    ----------
    .. [#] https://www.iers.org/IERS/EN/Science/EarthRotation/UT1LOD.html

    Examples
    --------
    Basic usage:

    >>> dut1utc(59025.0)
    -0.2426

    Multiple times are supported with `numpy` arrays:

    >>> dut1utc(np.array([59025.0, 59026.0]))
    array([-0.2426   , -0.2418664])

    """

    if not _utc_data.loaded:
        _utc_data.load()

    return _dut1utc_wrapped(
        mjd,
        _utc_data.mjd,
        _utc_data.dt,
        bounds_check=bounds_check,
        min_mjd=_utc_data.min_mjd,
        max_mjd=_utc_data.max_mjd,
    )


@njit
def _dut1utc_wrapped(
    mjd: float | NDArray[np.float64],
    tab_mjd: NDArray[np.float64],
    tab_dt: NDArray[np.float64],
    bounds_check: bool = True,
    min_mjd: float = -np.inf,
    max_mjd: float = np.inf,
) -> float | NDArray[np.float64]:
    if bounds_check:
        if np.min(np.asarray(mjd)) < min_mjd or np.max(np.asarray(mjd)) > max_mjd:
            raise ValueError(
                (
                    "Requested UT1-UTC value at supplied time is outside "
                    "the bounds on the *current* IERS file."
                )
            )

    return np.interp(mjd, tab_mjd, tab_dt)


@njit
def rmfu(
    n: int,
    c: float,
    s: float,
) -> NDArray[np.float64]:
    """Create a rotation matrix about axis `n`, given the sine `s` and
    cosine `c` of the rotation angle.

    Parameters
    ----------
    n : int
        Primary axis of rotation, must be within (0, 1, 2)
    c : float
        Cosine of the rotation angle
    s : float
        Sine of the rotation angle

    Returns
    -------
    R : NDArray[np.float64]
        Rotation matrix

    Raises
    ------
    ValueError
        Raised if the rotation axis is not in (0, 1, 2)
    """

    if n not in (0, 1, 2):
        raise ValueError(f"n not in (0, 1, 2); {n} is not valid")

    rmx = np.zeros((3, 3))
    rmx[n, n] = 1.0
    if n == 0:
        l = 1
        m = 2
    elif n == 1:
        l = 2
        m = 0
    else:  # n == 2
        l = 0
        m = 1

    rmx[l, l] = c
    rmx[m, m] = c
    rmx[l, m] = s
    rmx[m, l] = -s

    return rmx


@njit
def obliquity(mjd_tdb: float) -> float:
    """Compute the mean obliquity of the ecliptic based at the time given.

    Parameters
    ----------
    mjd_tdb : float
        Time at which to compute the obliquity. Time is specified as a Modified
        Julian Date (MJD) in the TDB time system.

    Returns
    -------
    obl : float
        Mean obliquity of the ecliptic in radians

    Examples
    --------

    Basic usage:

    >>> obliquity(59025.0)
    0.4090460953738584

    """

    # define some constants
    RAD2ASEC = 180.0 / np.pi * 3600
    ASEC2RAD = 1.0 / RAD2ASEC
    MJD_J2000 = 51544.5
    JCENTURY_LEN = 36525.0

    T = (mjd_tdb - MJD_J2000) / JCENTURY_LEN

    # Calculate the mean obliquity
    oblq = (
        84381.406
        - 46.836769 * T
        - 0.0001831 * T**2
        + 0.00200340 * T**3
        - 0.576e-6 * T**4
        - 4.34e-8 * T**5
    )
    oblq *= ASEC2RAD

    return oblq


@njit
def rmat(n: int, a: float) -> NDArray[np.float64]:
    """Create a rotation matrix about primary axis `n` with rotation
    angle `a`.

    Parameters
    ----------
    n : int
        Primary axis of rotation

    a : float
        Angle of rotation (radians)

    Returns
    -------
    rmx : NDArray[np.float64]
        Rotation matrix

    """

    c = np.cos(a)
    s = np.sin(a)
    rmx = rmfu(n, c, s)

    return rmx


def nutate(mjd: float) -> Tuple[float, float, float, NDArray[np.float64]]:
    """Compute the nutation of the Earth's rotational axis at the time given.

    The first two outputs are the nutation angles in longitude and obliquity:
    :math:`\\Delta\\psi`, :math:`\\Delta\\epsilon`.

    The third output is the `true obliquity` at the time given:
    :math:`\\epsilon' = \\epsilon + \\Delta\\epsilon`

    The final output is the total nutation matrix.

    Args
    ----
        mjd : float
        Time at which nutation is computed. Time is specified as a Modified
        Julian Date (MJD) in the UT1 time system.

    Returns
    -------
        dpsi : float
            Nutation in longitude (radians); :math:`\\Delta\\psi`

        deps : float
            Nutation in obliquity (radians); :math:`\\Delta\\epsilon`

        true_obliquity : float
            True obliquity of the ecliptic (radians); :math:`\\epsilon'`

        nutation_matrix : NDArray[np.float64]
            Rotation matrix

    References
    ----------
    .. [#] https://aa.usno.navy.mil/downloads/Circular_179.pdf

    Examples
    --------
    .. code-block:: python

        >>> dpsi, deps, true_oblq, R = nutate(59025.0)
        >>> print(dpsi, deps, true_oblq)
        -8.137262475393194e-05 -1.4786332984509557e-06 0.4090446167405599

        >>> with np.printoptions(suppress=True):
                print(R)
        [[ 1.          0.00007466  0.00003236]
        [-0.00007466  1.          0.00000148]
        [-0.00003236 -0.00000148  1.        ]]

    """
    if not _nutation_data.loaded:
        _nutation_data.load()

    return _nutate_wrapped(
        mjd,
        _nutation_data.Si,
        _nutation_data.dSi,
        _nutation_data.Ci,
        _nutation_data.dCi,
        _nutation_data.mults,
    )


@njit
def _nutate_wrapped(
    mjd: float,
    Si: NDArray[np.float64],
    dSi: NDArray[np.float64],
    Ci: NDArray[np.float64],
    dCi: NDArray[np.float64],
    mults: NDArray[np.float64],
) -> Tuple[float, float, float, NDArray[np.float64]]:
    radsec = 2 * np.pi / (360 * 3600)

    T = ((mjd + 2400000.5) - 2451545.0) / 36525.0

    v = np.zeros((5,))

    v[0] = 485866.733 + 1717915922.633 * T + 31.310 * T * T + 0.064 * T * T * T
    v[1] = 1287099.804 + 129596581.224 * T - 0.577 * T * T - 0.012 * T * T * T
    v[2] = 335778.877 + 1739527263.137 * T - 13.257 * T * T + 0.011 * T * T * T
    v[3] = 1072261.307 + 1602961601.328 * T - 6.891 * T * T + 0.019 * T * T * T
    v[4] = 450160.280 - 6962890.539 * T + 7.455 * T * T + 0.008 * T * T * T

    v *= radsec
    dpsi = 0.0
    deps = 0.0

    for i in range(106):
        arg = 0
        for j in range(5):
            arg += mults[j, i] * v[j]

        amppsi = Si[i] / 1.0e4 + dSi[i] * T / 1.0e5
        ampeps = Ci[i] / 1.0e4 + dCi[i] * T / 1.0e5
        dpsi = dpsi + amppsi * np.sin(arg)
        deps = deps + ampeps * np.cos(arg)

    dpsi = dpsi * radsec
    deps = deps * radsec
    MeanObliquity = obliquity(mjd)
    TrueObliquity = MeanObliquity + deps
    TA = rmat(0, MeanObliquity)
    TB = rmat(2, -dpsi)
    TC = TB @ TA
    TA = rmat(0, -TrueObliquity)

    nutation_matrix = TA @ TC

    return dpsi, deps, TrueObliquity, nutation_matrix


@njit
def shadow(n: NDArray[np.float64], z: NDArray[np.float64]) -> int:
    """Compute whether the satellite is in shadow

    Parameters
    ----------
    n : NDArray[np.float64]
        Unit vector pointing from the Earth to the Sun
    z : NDArray[np.float64]
        ECI position of the satellite (km)

    Returns
    -------
    test : int
        Flag for if the satellite is in Earth's shadow
        0 = shadow, 1 = not in shadow
    """

    R_earth = 6378.137

    test = int((((z**2).sum() - (n * z).sum() ** 2 > R_earth**2) | ((n * z).sum() > 0)))

    return test
