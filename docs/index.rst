.. admonition:: Join the Discussion

    Feel free to share ideas and ask questions over at `AstroForge's discussion page`_.

    .. _AstroForge's discussion page: https://github.com/mit-ll/AstroForge/discussions

========================================
Welcome to AstroForge's documentation!
========================================

AstroForge is a Python library of astrodynamics tools. It is meant to be used for satellite state
propagation and the computation of observations from ground-based or space-based sensors. Precise
orbit fitting requires pristine knowledge of the observer's site location in inertial coordinates,
which in turn requires methodical handling of coordinate systems and the conversions between them.
As such, much of this library is devoted to coordinate conversions.

Example Usage
=============
Let's start with a simple example of propagating a satellite in Low Earth Orbit (LEO) and computing
the angles-only observations of the satellite from a ground-based sensor.

Defining the Orbit
------------------

AstroForge has some constants defined for us, such as the equatorial radius (:data:`~astroforge.constants.R_earth`) and
gravitational parameter (:data:`~astroforge.constants.GM`) of the Earth. We'll use those to define the position and velocity
vectors for a satellite in a circular equatorial orbit with an altitude of 400 km. 

.. tab-set::

    .. tab-item:: Code

        .. code-block:: python
            :caption: Setup an orbital state to propagate

            import numpy as np
            import astroforge as af

            alt = 400.0
            r = alt + af.R_earth
            v = np.sqrt(af.GM / r)

            x = np.array([r, 0.0, 0.0])
            v = np.array([0.0, v, 0.0])

            with np.printoptions(suppress=True):
                print(f"{x = }")
                print(f"{v = }")

    .. tab-item:: Output

        .. code-block:: console

            x = array([6778.137,    0.   ,    0.   ])
            v = array([0.        , 7.66855818, 0.        ])

Propagating the Orbit
---------------------

We can propagate that orbital state using a medium-fidelity propagator within
AstroForge. Alternatively, you can write your own force model and provide that as the
first argument to ``astroforge.propagators.propagator``.

.. tab-set::

    .. tab-item:: Code

        .. code-block:: python
            :caption: Propagate using a medium-fidelity force model

            force_model = af.force_models.F_mp
            propagator = af.propagators.propagator

            x0 = np.hstack([x, v])
            t = np.array([51720.0, 51721.0]) * 86400.0

            states = propagator(force_model, x0, t)
            assert (states[0] == x0).all()

            xf, vf = states[1][:3], states[1][3:]
            with np.printoptions(suppress=True):
                print(f"{xf = }")
                print(f"{vf = }")

    .. tab-item:: Output

        .. code-block:: console

            xf = array([-5406.18389055, -4058.16278305,     0.00903747])
            vf = array([ 4.61189691, -6.1527153 , -0.00000629])

The output of the propagator is the satellite state at the times given, so the first element
of the states vector should be exactly equal to the input.

.. admonition:: Note

    The :func:`astroforge.force_models.F_mp` function needs the absolute time (MJD) in
    order to compute the position of the Sun and Moon, but the numerical integration
    uses SI units. So, the time input here is the MJD converted to seconds.

Compute Observations
--------------------

Finally, let's compute observations from a ground-based sensor. In this case we'll model
the sensor as an optical telescope, which reports inertial angles-only measurements of
the satellite--- right ascension (:math:`\alpha`), declination (:math:`\delta`), and their
rates of change (:math:`\dot{\alpha}, \dot{\delta}`). `This article 
<https://en.wikipedia.org/wiki/Equatorial_coordinate_system>`_ has more information, if 
you are interested.

.. tab-set::

    .. tab-item:: Code

        .. code-block:: python
            :caption: Compute angles-only observations from ground-based sensor

            # make up a site location
            lat, lon, alt = 42.459629, -71.267319, 0.0

            # rotate site location into inertial coordinates
            x_site_itrs = af.coordinates.LatLonAltToITRS(lat, lon, alt)
            v_site_itrs = np.zeros(3)
            x_site, v_site = af.coordinates.PosVelConversion(
                af.coordinates.ITRSToTETED,
                t[1] / 86400.0,
                x_site_itrs,
                v_site_itrs,
            )

            angles, _, _ = af.coordinates.PosVelToFPState(xf, vf, x_site, v_site)
            ra = angles[0, 0] * 180 / np.pi
            dec = angles[0, 1] * 180 / np.pi
            rates = angles[0, 2:4] * 180 * 3600 / np.pi

            print(f"Angles (ra, dec) = ({ra:.6f}, {dec:.6f}) deg")
            print(f"Rates (ra-rate, dec-rate) = ({rates[0]:.3f}, {rates[1]:.3f}) arcsec/sec")

    .. tab-item:: Output

        .. code-block:: console

            Angles (ra, dec) = (-115.952570, -60.208808) deg
            Rates (ra-rate, dec-rate) = (553.427, 119.284) arcsec/sec   



Documentation Contents
======================

.. toctree::
    :maxdepth: 2

    How-To Guides <how_to_guides>
    Reference <reference>