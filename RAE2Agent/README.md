# RAE2Agent Occultation Pipeline

This folder contains a standalone occultation analysis pipeline for the RAE2
interpolated master file. It searches for on/off lunar occultation events for
planets and fixed point sources, then measures the dip significance per band
and with bands combined.

An alternate pipeline (`occultation_search.py`) runs a global blocked-vs-visible
analysis across the full dataset, includes random sky points for comparison, and
can optionally include planets using a coarse ephemeris grid.

## Quick Start

1) Activate the environment (per project AGENTS.md):
```
conda activate luseepy_env
```

2) Run the analysis:
```
python RAE2Agent/occultation_analysis.py \
  --data /global/cfs/projectdirs/m4895/RAE2Data/interpolatedRAE2MasterFile.csv
```

3) Run the alternate global search:
```
python RAE2Agent/occultation_search.py \
  --data /global/cfs/projectdirs/m4895/RAE2Data/interpolatedRAE2MasterFile.csv
```

## Notes

- Time is read from the `time` column and used to set the index.
- Occultation geometry is computed in FK4 B1950 to match the spacecraft frame.
- `rv2_coarse` (lower V) is the default data column; add `--data-cols rv2_coarse rv1_coarse`
  to analyze both channels.
- Earth-Moon distance is fixed at 384400 km and Sun/Earth limb filters are kept.
- Plots and tables are written under `RAE2Agent/output/<target>/`.
- `right_ascension` / `declination` columns are currently treated as metadata.

## Outputs

Each target produces:
- `tables/events_all.csv` and `tables/events_filtered.csv`
- `tables/significance_per_band_<data_col>.csv`
- `tables/aggregate_per_band_<data_col>.csv`
- `tables/event_combined_z_<data_col>.csv`
- Optional combined-band tables and plots
- `plots/zscores_per_band_<data_col>.png`
- `plots/zscores_combined_<data_col>.png` (if enabled)

Event-level random baseline outputs:
- `event_combined_z_all_<data_col>.csv`
- `event_random_percentile_per_band_<data_col>.csv`
- `plots/<data_col>/event_random_percentile_<target>.png`

The global search produces:
- `summary_per_band_<data_col>.csv`
- `summary_combined_<data_col>.csv`
- `random_per_band_<data_col>.csv`
- `real_per_band_<data_col>.csv`
- `random_comparison_per_band_<data_col>.csv`
- `random_comparison_combined_<data_col>.csv`
- `plots/<data_col>/random_compare_band_*_planets.png`
- `plots/<data_col>/random_compare_band_*_nonplanets.png`
- `plots/<data_col>/random_compare_combined_planets.png`
- `plots/<data_col>/random_compare_combined_nonplanets.png`
- `plots/<data_col>/random_percentile_<target>.png`
- `perm_null_per_band_<data_col>.csv`
- `permutation_comparison_per_band_<data_col>.csv`

Notes for the global search:
- Sun/Earth limb filtering is applied by default (use `--no-limb-filter` to disable).
- Planet ephemerides and Sun geometry are computed on a coarse time grid set by `--planet-step-minutes`.
- Optional time downsampling is available via `--downsample-minutes`.
- Permutation null tests are available via `--permute-reps` and `--permute-window-minutes`.

Notes for the event-based pipeline:
- Random sky baselines are controlled by `--random-sky` and `--random-seed`.
- Planets can be skipped with `--no-planets` and use a coarse ephemeris grid via `--planet-step-minutes`.
- Event z-scores can be stabilized with `--min-samples`.
- Runtime controls: `--n-processes` (parallel targets), `--event-stride`/`--max-events`,
  and `--random-event-stride`/`--random-max-events` to limit random baseline cost.

An aggregate summary across all targets is saved to:
- `RAE2Agent/output/summary_aggregates.csv`

## Extending Targets

Edit `build_default_targets()` in `RAE2Agent/occultation_analysis.py`:
- Add fixed sources with ICRS (J2000) coordinates.
- Add planets by name (must be recognized by `astropy.coordinates.get_body`).
