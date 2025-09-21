# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
_propagator.py
"""

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

__all__ = [
    "propagator",
]


def propagator(
    fm: Callable,
    x0: NDArray[np.float64],
    t: NDArray[np.float64],
    **kwargs,
) -> NDArray[np.float64]:
    """Propagates an orbital state with a given force model

    Parameters
    ----------
    fm : Callable
        Force model to use for propagation
    x0 : NDArray[np.float64]
        Orbital state vector; should have shape (N, ) where N is the number of
        parameters in the state.
    t : NDArray[np.float64]
        Times to at which the orbit propagation should be evaluated. The first time must
        correspond to the epoch of the initial position and velocity vector given as
        input.

    Returns
    -------
    NDArray[np.float64]
        State vector at the times requested

    Raises
    ------
    RuntimeError
        Raised if propagation fails for any reason
    """

    atol = kwargs.pop("atol", 1e-9)
    rtol = kwargs.pop("rtol", 1e-9)

    out = solve_ivp(fm, (t[0], t[-1]), x0, t_eval=t, atol=atol, rtol=rtol, **kwargs)

    if out.success:
        return out.y.T
    else:
        raise RuntimeError(
            f"Numeric integration failed with message:\n\t{out.message}\n"
        )
