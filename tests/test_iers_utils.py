# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import os
from unittest import mock
import pytest
from requests import Response

from astroforge.coordinates.iers import (
    download_iers,
    parse_iers,
    setup_iers,
    CURRENT_IERS_FILENAME,
    LEAPSECOND_FILENAME,
    POLARMOTION_FILENAME,
    UT1_UTC_FILENAME,
    _datadir,
)
from astroforge.coordinates._utilities import PolarMotionData, UTCData


@pytest.mark.download
def test_chain():
    setup_iers(download=True)
    assert os.path.exists(CURRENT_IERS_FILENAME)
    assert os.path.exists(LEAPSECOND_FILENAME)
    assert os.path.exists(POLARMOTION_FILENAME)
    assert os.path.exists(UT1_UTC_FILENAME)

    # make sure we can load the downloaded data
    data = PolarMotionData()
    data.load()
    assert data.loaded

    data = UTCData()
    data.load()
    assert data.loaded

    # run again to test the branch that doesn't download the data
    setup_iers(download=False)

    assert os.path.exists(CURRENT_IERS_FILENAME)
    assert os.path.exists(LEAPSECOND_FILENAME)
    assert os.path.exists(POLARMOTION_FILENAME)
    assert os.path.exists(UT1_UTC_FILENAME)


@pytest.mark.download
def test_chain_without_setup():
    # We don't really want to delete and re-download files
    # unnecessarily, so we'll mock those actions instead.
    import astroforge

    src_module = "astroforge.coordinates._utilities"

    with mock.patch(f"{src_module}.os.path.exists", mock.MagicMock(return_value=False)):
        with mock.patch(
            f"{src_module}.iers.setup_iers",
            mock.MagicMock(return_value=None),
        ):
            data = PolarMotionData()
            data.load()
            assert data.loaded

            utc = UTCData()
            utc.load()
            assert utc.loaded

            assert astroforge.coordinates._utilities.iers.setup_iers.call_count == 2

    assert os.path.exists(CURRENT_IERS_FILENAME)
    assert os.path.exists(LEAPSECOND_FILENAME)
    assert os.path.exists(POLARMOTION_FILENAME)
    assert os.path.exists(UT1_UTC_FILENAME)


@pytest.mark.download
def test_download_iers_issues():
    src_module = "astroforge.coordinates.iers._iers_utils"
    original_os_exists = os.path.exists

    def mock_os_exists(arg):
        if arg == _datadir:
            return False
        else:
            return original_os_exists(arg)

    def mock_request_fail(arg, stream=True):
        raise Exception

    def mock_request_response(arg, stream=True):
        return Response()

    def mock_symlink_fail(arg):
        raise Exception

    def mock_symlink_false(arg):
        return False

    with mock.patch(f"{src_module}.os.path.exists", mock_os_exists):
        with mock.patch(f"{src_module}.os.makedirs", mock.MagicMock(return_value=None)):
            with mock.patch(f"{src_module}.requests.get", mock_request_fail):
                with pytest.raises(Exception) as e:
                    download_iers()

                assert "Failed to download IERS file" in str(e)
                assert os.makedirs.called

            with mock.patch("builtins.open", mock.mock_open()):
                with mock.patch("requests.get", mock_request_response):
                    with mock.patch("os.path.islink", mock_symlink_fail):
                        with pytest.raises(Exception) as e:
                            download_iers()

                        assert "Failed to create symlink" in str(e)

                    with mock.patch("os.path.islink", mock_symlink_false):
                        with mock.patch(
                            "os.symlink", mock.MagicMock(return_value=None)
                        ):
                            download_iers()

                            assert os.symlink.called

    assert os.path.exists(CURRENT_IERS_FILENAME)
    assert os.path.exists(LEAPSECOND_FILENAME)
    assert os.path.exists(POLARMOTION_FILENAME)
    assert os.path.exists(UT1_UTC_FILENAME)


def test_iers_file_not_exist():
    src_module = "astroforge.coordinates.iers._iers_utils"
    original_os_exists = os.path.exists

    def mock_os_exists(arg):
        if arg == CURRENT_IERS_FILENAME:
            return False
        else:
            return original_os_exists(arg)

    with mock.patch(f"{src_module}.os.path.exists", mock_os_exists):
        with pytest.raises(FileNotFoundError):
            setup_iers(download=False)

    assert os.path.exists(CURRENT_IERS_FILENAME)
    assert os.path.exists(LEAPSECOND_FILENAME)
    assert os.path.exists(POLARMOTION_FILENAME)
    assert os.path.exists(UT1_UTC_FILENAME)
