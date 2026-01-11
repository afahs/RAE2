# RAE2Sim

Simulate the RAE2 upper/lower V receiver power given spacecraft positions, beam patterns, and sky maps.

## Inputs

Positions CSV (`--positions`) must include:
- `time`: timestamp or index
- `x_km`, `y_km`, `z_km`: spacecraft position in km, lunar-centered B1950 frame
- `ex`, `ey`, `ez`: Earth unit vector in the same frame

Beam CSVs (`--e-plane`, `--h-plane`) must include:
- `theta_deg`: polar angle from antenna boresight
- `gain`: linear gain (or dB if `--beam-db` is set)

Skymaps (`--skymap`) are Healpy FITS maps in Kelvin. Provide one map per frequency
or a single map that will be reused for all frequencies.

## Output

The output CSV (`--output`) includes:
- `time`, `freq_hz`
- `power_upper_k`, `power_lower_k` (antenna temperature in K, plus optional system temperature)
- input position vectors, `system_temperature_k`, and beam/occultation metadata

## Python API

`simulate.py` exposes `simulate(...)` that accepts a pandas DataFrame for positions if you prefer
to call it from an interactive session.

## Usage

```bash
python simulate.py \
  --positions positions.csv \
  --skymap map_10mhz.fits --skymap map_15mhz.fits \
  --freqs 1.0e7,1.5e7 \
  --e-plane e_plane.csv \
  --h-plane h_plane.csv \
  --system-temperature 200.0 \
  --output rae2sim_output.csv
```

If no beam files are provided, a default dipole pattern is used.
