# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import pathlib

_datadir = pathlib.Path.home() / ".astroforge" / "data"

# some constants
CURRENT_IERS_FILENAME = _datadir / "current_IERS_file.txt"
POLARMOTION_FILENAME = _datadir / "polarmotion_data.npz"
UT1_UTC_FILENAME = _datadir / "ut1utc_data.npz"
LEAPSECOND_FILENAME = _datadir / "leapsecond_data.npz"
IERS_URL = "https://datacenter.iers.org/data/9/finals2000A.all"

from ._iers_utils import download_iers, parse_iers, setup_iers

__all__ = [
    "download_iers",
    "parse_iers",
    "setup_iers",
    "CURRENT_IERS_FILENAME",
    "POLARMOTION_FILENAME",
    "UT1_UTC_FILENAME",
    "LEAPSECOND_FILENAME",
    "IERS_URL",
]
