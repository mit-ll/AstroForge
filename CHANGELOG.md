# Changelog

## Version 0.2 (development)
- Modified `pyproject.toml` to use `uv` for package management
- Removed `setup.cfg` and `setup.py`
- Replaced call to `astropy` for converting geocentric cartesian coordinates
  to geodetic latitude/longitude/altitude with the algorithm implemented in
  https://github.com/liberfa/erfa. Removed `astropy` as a dependency.
- Set default `atol` and `rtol` in `solve_ivp` to 1e-9 for both.
- Remove unnecessary dimension on output of `keplerian_to_cartesian`
- Fix failing `test_ea_from_ma_divergence` test
