============================
Reference API
============================

All the gory details that you would need to use AstroForge to its fullest.

Constants
=========
AstroForge exports a few useful constants:

.. currentmodule:: astroforge.constants

.. autosummary::
    :toctree: generated/

    R_earth
    GM
    J2
    Omega
    Rgeo
    Vgeo
    Ageo
    ss_GM
    c

Moon and Sun Positions
======================
Given an input time in Modified Julian Date (MJD) format, AstroForge can 
calculate the low fidelity positions of the sun and moon in GCRS or MEMED 
coordinates:

.. currentmodule:: astroforge.common

.. autosummary::
    :toctree: generated/

    R_moon
    R_moon_MEMED
    R_sun
    R_sun_MEMED


Force Models
============

AstroForge utilizes Ordinary Differential Equation (ODE) solvers for propagating orbital
states through time. In order to do so, one must specify the equations of motion by 
providing a function that evaluates the derivative of the state. Within the AstroForge,
there exist both low-level utilities for creating such a function, as well as a few
high-level pre-built force models that can be used off-the-shelf.

The following are the low-level utilities:

.. currentmodule:: astroforge.force_models

.. autosummary::
    :toctree: generated/

    F_geo_ITRS
    F_geo_MEMED
    F_J2
    F_third

And these are the high-level, built-in force models:

.. currentmodule:: astroforge.force_models

.. autosummary::
    :toctree: generated/

    F_mp
    F_mp_srp
    kepler

Propagators
===========

The :func:`~astroforge.force_models.F_mp` and :func:`~astroforge.force_models.F_mp_srp`
force models can be propagated directly, or a custom force model can be used with 
:func:`~astroforge.propagators.propagator`:

.. currentmodule:: astroforge.propagators

.. autosummary::
    :toctree: generated/
    
    mp
    mp_srp
    propagator

Coordinate Conversions
======================
Much of this package is devoted to carefully handling the coordinate conversions
necesary for precise orbit determination. Each of the coordinate conversion utilities
are summarized below: 

.. currentmodule:: astroforge.coordinates

.. autosummary::
    :toctree: generated/

    CIRSToMEMED
    CIRSToTETED
    dut1utc
    ITRSToMEMED
    ITRSToTIRS
    ITRSToTETED
    ITRSToLatLonAlt
    ITRSToSEZ
    LatLonAltToITRS
    obliquity
    MEMEDToCIRS
    MEMEDToITRS
    nutate
    polarmotion
    PosVelConversion
    PosVelToFPState
    rmat
    rmfu
    Rx
    Ry
    Rz
    SEZToAzElRange
    shadow
    TETEDToCIRS
    TETEDToITRS
    TIRSToITRS

IERS Utilities
==============

Many of these coordinate conversions rely on precise data about the orientation of the Earth,
which is contained within the International Earth Rotation and Reference Systems Service (IERS)
Bulletin A file, found `here <https://datacenter.iers.org/data/9/finals2000A.all>`_. AstroForge
has some utilities for automatically downloading and parsing this data. 

.. currentmodule:: astroforge.coordinates.iers

.. autosummary::
    :toctree: generated/
    
    download_iers
    parse_iers
    setup_iers

