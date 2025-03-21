===================================
Setting up the IERS Bulletin A File
===================================

The International Earth Rotation and Reference Systems Service (IERS) maintains a
dataset of parameters that describe precisely how the Earth is oriented at any
particular moment. AstroForge uses the IERS Bulletin A file for many coordinate
conversion. You can find more information about this data on the 
`IERS webpage`_, or `download the data directly`_. 

In most circumstances, AstroForge users do not need to explicitly setup the IERS file.
The first time AstroForge does a coordinate conversion that requires IERS data it will
try to download the file from the internet. This guide is to help when that process
does not work for whatever reason. 

.. _automated_instructions:

Running AstroForge's Automated Process
--------------------------------------
The first troubleshooting step should be to re-run the automated process for downloading
and installing the IERS file. To do so, simply run the following in python:

.. code-block:: python

    from astroforge.coordinates.iers import setup_iers
    setup_iers(download=True)

If that is successful, you should see output that looks like the following (though the 
file paths will differ based on your system, username, and the current date):

    >>> from astroforge.coordinates.iers import setup_iers
    >>> setup_iers(download=True)
    downloading IERS file from
            https://datacenter.iers.org/data/9/finals2000A.all
    Saving to /home/<USER>/.astroforge/data/finalsAll_2023-10-27.txt
    [===========================================================================]
    Making symbolic link to /home/<USER>/.astroforge/data/current_IERS_file.txt

.. note::

    | The default file location for unix/linux users is: ``/home/<USER>/.astroforge/data``.
    | The default file location for windows users is: ``C:\Users\<USER>\.astroforge\data``.

If the progress bar never appears and the automated download process times out, check
your internet connection and proxy settings--- something is blocking python from
reaching out to the internet. If that still doesn't work, move on to the 
:ref:`next section <manual_instructions>`.

If the download and parsing succeeded, there should be some data files in the download
location. They consist of the raw IERS Bulletin A file (named ``finalsAll_<date>.txt``),
three .npz files, and a symbolic link pointing ``current_IERS_file.txt`` to the latest
raw data file. The three .npz files are intermediate data products that were generated
by the automatic setup process. 

.. note::

    AstroForge never deletes *old* IERS files. It instead saves new files with the data
    and then updates the symbolic link.

.. code-block:: console

    $ ls -1 ~/.astroforge/data/
    current_IERS_file.txt
    finalsAll_2023-10-27.txt
    leapsecond_data.npz
    polarmotion_data.npz
    ut1utc_data.npz

.. _manual_instructions:

Manually Download and Install
-----------------------------

If the automated process isn't working, you can try downloading and installing the IERS
file yourself. The manual process isn't that different than the automated one. 

First, download the `IERS file`_. Then, create the data directory:
``$HOME/.astroforge/data`` on unix, or ``C:\Users\<USER>\.astroforge\data`` on windows.
Place the newly downloaded IERS file in the data directory and create a symbolic link to
it with the name ``current_IERS_file.txt``. On unix, these steps can be achieved with
the following at the command line (remember to replace content in the angle brackets
with the *actual* IERS filename).

.. code-block:: console

    $ mkdir -p $HOME/.astroforge/data
    $ cp <path/to/IERS/file> $HOME/.astroforge/data/
    $ ln -s $HOME/.astroforge/data/<IERS_filename> $HOME/.astroforge/data/current_IERS_file.txt

Finally, launch python and parse the IERS data file manually:

.. code-block:: python
    
    from astroforge.coordinates.iers import setup_iers
    setup_iers(download=False)

After parsing the raw datafile, the data directory should have the same .npz files shown
above. You can also test that it worked by using an AstroForge function that requires
these files, such as :func:`~astroforge.coordinates.polarmotion`:

    >>> from astroforge.coordinates import polarmotion
    >>> x, y = polarmotion(51720.0)
    >>> print(x, y)
    0.108198 0.28807

If none of these steps succeed, feel free to open a discussion about it on `AstroForge's discussion page`_. 

.. _download the data directly: https://datacenter.iers.org/data/9/finals2000A.all
.. _IERS file: https://datacenter.iers.org/data/9/finals2000A.all
.. _IERS webpage: https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html
.. _AstroForge's discussion page: https://github.com/mit-ll/AstroForge/discussions