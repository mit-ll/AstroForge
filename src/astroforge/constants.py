# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""

This submodule contains constants that are used elsewhere in the AstroForge module
and are exposed directly to the user.
"""


from math import pi, sqrt

__all__ = [
    "twopi",
    "R_earth",
    "GM",
    "J2",
    "Omega",
    "Rgeo",
    "Vgeo",
    "Ageo",
    "ss_GM",
    "c",
]

# generic constants
twopi = 2 * pi

# --- Earth parameters ---
#: Equatorial radius of the Earth (km)
R_earth = 6378.137
#: Gravitational parameter of the Earth (km\ :sup:`3` / s\ :sup:`2`)
GM = 398600.4418
#: Oblateness of the Earth
J2 = 4.84165368e-4 * sqrt(5)
#: Rotation rate of the Earth (rad/s)
Omega = twopi / 86164.0989

# --- GEO parameters ---
#: Orbital radius of the GEO belt (km)
Rgeo = (GM / Omega**2) ** (1 / 3)
#: Orbital velocity for a satellite in GEO (km/s)
Vgeo = sqrt(GM / Rgeo)
#: Primary acceleration for a satellite in GEO (km / s\ :sup:`2``)
Ageo = GM / Rgeo**2

# --- other bodies ---
#: Gravitational parameter of the Moon (km\ :sup:`3` / s\ :sup:`2`)
GM_moon = 4.9027779e3
#: Gravitational parameter of the Sun (km\ :sup:`3` / s\ :sup:`2`)
GM_sun = 1.32712440018e11
#: Gravitational parameters for the rest of the solar system (km\ :sup:`3` / s\ :sup:`2`)
ss_GM = {
    "mercury": 22032,
    "venus": 324859,
    "emb": 398600.435436 * (1 + 1 / 81.30056907419062),
    "mars": 42828.375214,
    "jupiter": 126712764.800000,
    "saturn": 37940585.200000,
    "uranus": 5793939,
    "neptune": 6836529,
    "moon": GM_moon,
    "sun": GM_sun,
}

#: Speed of light (km/s)
c = 299792.458
