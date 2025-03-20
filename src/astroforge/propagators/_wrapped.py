# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple

from ._propagator import propagator
from ..force_models import F_mp, F_mp_srp
from ..coordinates import dut1utc

__all__ = [
    "mp",
    "mp_srp",
]

_DEFAULT_ATOL = 1e-9
_DEFAULT_RTOL = 1e-9


def mp(
    x0: NDArray[np.float64],
    v0: NDArray[np.float64],
    T: NDArray[np.float64],
    **kwargs,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagates an orbital state with a medium-fidelity force model. See
    :func:`~astroforge.force_models.F_mp` for details about the force model used by this
    propagator.

    Parameters
    ----------
    x0 : NDArray[np.float64]
        Initial position vector; should have shape (3, ) and be in units of km.
    v0 : NDArray[np.float64]
        Initial velocity vector; should haev shape (3, ) and be in units of km/s.
    T : NDArray[np.float64]
        Times at which the orbit propagation should be output. The first time in the
        array must correspond to the epoch of the input position and velocity vectors.
        Time should be expressed as a set of Modified Julian Dates (MJDs) in the UTC
        time system.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        Position and velocity vectors at the times given.
    """
    y0 = np.hstack((x0, v0))

    T = _utc_to_ut1(T)
    kwargs = _add_tolerances_to_kwargs(kwargs)

    # numerical integration of the indicated force model
    out = propagator(F_mp, y0, T, **kwargs)

    # parse and return output
    X = out[:, :3]
    V = out[:, 3:6]
    return (X, V)


def mp_srp(
    x0: NDArray[np.float64],
    v0: NDArray[np.float64],
    alpha: NDArray[np.float64],
    T: NDArray[np.float64],
    **kwargs,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagates an orbital state with a medium-fidelity force model which includes
    solar radiation pressure. See :func:`~astroforge.force_models.F_mp_srp` for details
    about the force model used by this propagator.

    Parameters
    ----------
    x0 : NDArray[np.float64]
        Initial position vector; should have shape (3, ) and be in units of km.
    v0 : NDArray[np.float64]
        Initial velocity vector; should have shape (3, ) and be in units of km/s.
    T : NDArray[np.float64]
        Times at which the orbit propagation should be output. The first time in the
        array must correspond to the epoch of the input position and velocity vectors.
        Time should be expressed as a set of Modified Julian Dates (MJDs) in the UTC
        time system.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        Position and velocity vectors at the times given.
    """
    y0 = np.hstack((x0, v0, alpha))

    T = _utc_to_ut1(T)
    kwargs = _add_tolerances_to_kwargs(kwargs)

    # numerical integration of the indicated force model
    out = propagator(F_mp_srp, y0, T, **kwargs)

    # parse and return output
    X = out[:, :3]
    V = out[:, 3:6]
    return (X, V)


def _utc_to_ut1(T):
    # convert time to SI units (seconds), and convert from UTC to UT1
    T = T.copy()
    T += dut1utc(T) / 86400
    T *= 86400
    return T


def _add_tolerances_to_kwargs(kwargs):
    if "atol" not in kwargs:
        kwargs["atol"] = _DEFAULT_ATOL
    if "rtol" not in kwargs:
        kwargs["rtol"] = _DEFAULT_RTOL
    return kwargs
