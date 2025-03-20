# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import os
import sys
from datetime import datetime

import numpy as np
import requests

from . import (
    CURRENT_IERS_FILENAME,
    IERS_URL,
    LEAPSECOND_FILENAME,
    POLARMOTION_FILENAME,
    UT1_UTC_FILENAME,
    _datadir,
)


def download_iers():
    """Download and save the IERS Bulletin A file.

    The Bulletin A file can be found `here <https://datacenter.iers.org/data/9/finals2000A.all>`_.
    """

    print(f"downloading IERS file from \n\t{IERS_URL}")

    # format filename
    dt = datetime.now()
    newfname = _datadir / f"finalsAll_{dt.year:4d}-{dt.month:02d}-{dt.day:02d}.txt"
    print(f"Saving to {newfname}")

    if not os.path.exists(_datadir):
        os.makedirs(_datadir)

    # download the IERS file
    try:
        with open(newfname, "wb") as f:
            response = requests.get(IERS_URL, stream=True)
            total_length = response.headers.get("content-length")

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(100 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ("=" * done, " " * (100 - done)))
                    sys.stdout.flush()

        print()

    except:
        raise Exception("Failed to download IERS file")

    # link the downloaded file to be the "current" file
    try:
        fname = CURRENT_IERS_FILENAME
        print(f"Making symbolic link to {fname}")

        if os.path.exists(fname) and os.path.islink(fname):
            # NOTE: I understand there is technically a race condition here
            # (it is possible that another process creates a link to the file
            # between the time I unlink it and relink it). That's not likely,
            # so I'm not worrying about it
            os.unlink(fname)

        os.symlink(newfname, fname)
    except:
        raise Exception("Failed to create symlink")


def parse_iers():
    """Parse the IERS file into more readily usable intermediate files.

    This function saves three files with
        1) polar motion data
        2) UT1-UTC data
        3) leap second data

    """

    # parse the MJD, pm-x, pm-y, ut1-utc data from the file
    with open(CURRENT_IERS_FILENAME, "r") as f:
        lines = f.readlines()

    N = len(lines)
    mjd = np.nan * np.ones((N,))
    xp = np.nan * np.ones((N,))
    yp = np.nan * np.ones((N,))
    ut1utc = np.nan * np.ones((N,))

    for n, line in enumerate(lines):
        try:
            mjd[n] = float(line[6:16])
            xp[n] = float(line[18:28])
            yp[n] = float(line[37:47])
            ut1utc[n] = float(line[58:68])
        except:
            pass

    # determine leap seconds
    with np.errstate(invalid="ignore"):
        leapseconds = 12 + ut1utc - np.unwrap(ut1utc * 2 * np.pi) / 2 / np.pi

    # save to file
    np.savez(
        POLARMOTION_FILENAME,
        mjd=mjd[~np.isnan(xp)],
        xp=xp[~np.isnan(xp)],
        yp=yp[~np.isnan(yp)],
    )
    np.savez(
        UT1_UTC_FILENAME, mjd=mjd[~np.isnan(ut1utc)], ut1utc=ut1utc[~np.isnan(ut1utc)]
    )
    np.savez(
        LEAPSECOND_FILENAME,
        mjd=mjd[~np.isnan(leapseconds)],
        leapseconds=leapseconds[~np.isnan(leapseconds)].astype(int),
    )


def setup_iers(download=True):
    """Utility function for setting up the IERS utilities.

    Parameters
    ----------
    download : bool, optional
        Flag indicating whether or not to download the IERS bulletin A file, by default True

    Raises
    ------
    FileNotFoundError
        Raised if there is not a local copy of the IERS file.
    """

    if download:
        download_iers()

    if not os.path.exists(CURRENT_IERS_FILENAME):
        raise FileNotFoundError(
            (
                f"Trying to access the *current* IERS file at\n"
                f" {CURRENT_IERS_FILENAME}\n"
                f"but that file does not exist"
            )
        )

    parse_iers()
