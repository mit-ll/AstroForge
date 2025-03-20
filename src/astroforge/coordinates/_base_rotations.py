# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Base 3D rotation matrix definitions
"""

import numpy as np
from numpy.typing import NDArray
from numba import njit

__all__ = [
    "Rx",
    "Ry",
    "Rz",
]


@njit
def Rx(
    angle: float,
) -> NDArray[np.float64]:
    """Create a rotation matrix about the first axis by the given rotation angle.
    The rotation matrix is given by:

    .. math::

        \\begin{bmatrix}
            1.0 & 0.0               & 0.0               \\\\
            0.0 & \\cos (\\theta)   & \\sin (\\theta)   \\\\
            0.0 & -\\sin (\\theta)  & \\cos (\\theta)
        \\end{bmatrix}

    Parameters
    ----------
    angle : float
        Rotation angle (radians)

    Returns
    -------
    R : NDArray[np.float64]
        Rotation matrix
    """
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), np.sin(angle)],
            [0.0, -np.sin(angle), np.cos(angle)],
        ]
    )
    return R


@njit
def Ry(
    angle: float,
) -> NDArray[np.float64]:
    """Create a rotation matrix about the second axis by the given rotation angle.
    The rotation matrix is given by:

    .. math::

        \\begin{bmatrix}
            \\cos (\\theta) & 0.0   & -\\sin (\\theta)   \\\\
            0.0             & 1.0   & 0.0               \\\\
            \\sin (\\theta) & 0.0   & \\cos (\\theta)
        \\end{bmatrix}

    Parameters
    ----------
    angle : float
        Rotation angle (radians)

    Returns
    -------
    R : NDArray[np.float64]
        Rotation matrix

    """

    R = np.array(
        [
            [np.cos(angle), 0.0, -np.sin(angle)],
            [0.0, 1.0, 0.0],
            [np.sin(angle), 0, np.cos(angle)],
        ]
    )
    return R


@njit
def Rz(
    angle: float,
) -> NDArray[np.float64]:
    """Create a rotation matrix about the third axis by the given rotation angle.
    The rotation matrix is given by:

    .. math::

        \\begin{bmatrix}
            \\cos (\\theta)     & \\sin (\\theta)   & 0.0 \\\\
            -\\sin (\\theta)    & \\cos (\\theta)   & 0.0 \\\\
            0.0                 & 0.0               & 1.0
        \\end{bmatrix}

    Parameters
    ----------
    angle : float
        Rotation angle (radians)

    Returns
    -------
    R : NDArray[np.float64]
        Rotation matrix

    """
    R = np.array(
        [
            [np.cos(angle), np.sin(angle), 0.0],
            [-np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return R
