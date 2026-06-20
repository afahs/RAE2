# Script Inventory

The core, reusable interface is `python -m rylevonberg.pipeline`.

The `scripts/` directory contains higher-level analyses used during the RAE-2
work. Many scripts write to `outputs/` and some require local assets that are
not part of this repository.

## Generally Reusable

These scripts mostly depend on the master CSV, package code, and config files:

- `scripts/run_sun_whole_dataset_validation.py`
- `scripts/run_solar_detection_confirmation_suite.py`
- `scripts/run_candidate_confirmation_checks.py`
- `scripts/run_planetary_confirmation_survey.py`
- `scripts/run_focused_validation.py`

Pass explicit `--data`, `--output-root`, or `--out-dir` values when running
them outside the original workstation.

## Extra Local Assets

Some scripts need one or more of these optional assets:

- digitized antenna beam cuts, set with `RAE2_ANTENNA_DIGITIZATION_DIR`
- PySM sky maps, set with `RAE2_SKY_MAP_DIR`
- Mie forward-model tables, set with `RAE2_MIE_DIR`
- individual Mie CSV overrides, set with `RAE2_MIE_POINT_CSV` or
  `RAE2_MIE_TIME_CSV`
- burst-receiver CSV exports, set with `RAE2_BURST_RECEIVER_CSV_DIR`
  or `RAE2_BURST_RECEIVER_CSV`

Scripts with hard dependencies on those assets include diffuse-beam forward
models, Novaco-Brown map builders, Mie sampling simulations, and burst-receiver
source analyses. If a script still contains an absolute path default, treat it
as documentation of the original local layout and override the path before
running.
