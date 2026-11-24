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

## Primary Visual Detection Review

- `scripts/build_occultograms.py` is the current morphology-first review. It
  writes event-by-time heatmaps, disappearance/reappearance-folded profiles,
  event/lunar-block quantiles, signed event contrasts, and fixed ±10/±20
  minute controls. Its default target order is Earth, Sun, Fornax A, and Virgo
  A; use `--sources` to add outer planets or plane-confused bright sources.
- `scripts/compact_occultogram_outputs.py` migrates legacy occultogram NPY/CSV
  tables to dictionary-encoded, Zstandard-compressed Parquet. With
  `--delete-originals`, it removes only the three superseded tables and only
  after row, value, event-key, and round-trip validation succeeds.
- `scripts/compact_occultogram_inputs.py` extracts the six time-series/event
  tables needed for a rerun from the old experiment tree, stores them under
  `runtime_inputs/occultogram_v1/`, and validates every value after the
  Parquet round trip.
- `scripts/run_earth_fixed_zero_methods.py` applies four positive-control
  demonstrations with the edge fixed at the ephemeris: robust Huber edge
  amplitudes, lunar-block label permutations, multiscale persistence, and a
  cross-frequency Galactic-foreground residual. It writes compact Parquet and
  PNG products; it does not grade other sources.
- `scripts/run_earth_lowrank_covariance.py` performs the next Earth-only
  benchmark. It uses nested lunar-block cross-fitting for low-rank foreground
  removal, OAS-shrunk time/frequency covariance with robust Student-t fits,
  exact enumeration of the +/-20-minute block-control null, and attenuated-
  Earth recovery. Antennas are fitted separately and agreement is not a gate.
- `scripts/run_extrasolar_lowrank_occultograms.py` applies the validated
  low-rank foreground-removal stage to event-level Fornax A and Virgo A
  occultograms. It aligns channel-specific records into physical events,
  keeps the edge fixed at zero, chooses factor rank with nested lunar-block
  control cross-validation, and writes only compact Parquet/PNG/JSON products.
  Antennas are fitted separately. This is a morphology experiment, not an
  automatic detection grader.
- `scripts/run_fornax_covariance_fixedzero.py` performs the covariance-aware
  Fornax A sensitivity test. It uses control-only spectral directions,
  block-held-out time/frequency covariance, robust Student-t GLS, fixed-zero
  step/ramp templates, an exact four-shift block-control null, family-wise
  correction across methods, and leave-one-frequency-out diagnostics. It
  writes compact Parquet/PNG/JSON products. Its current result is a candidate
  fixed-time excess, not a detection claim.
- `scripts/run_fornax_blanksky_comparison.py` applies the target-frozen Fornax
  configurations to 24 nearby sky positions on 2- and 5-degree rings. It
  predicts each control's lunar occultations on Perlmutter, retains only
  compact lunar-block profiles, compares raw and cross-fitted low-rank
  branches, and writes selected target/strongest/typical event occultograms.
  No blank position may reselect the antenna, spectral template, or width.
  Outputs are compressed Parquet, PNG, JSON, and Markdown only.
- `scripts/build_all_frequency_occultation_profile_grids.py` is retained only
  for reproducing legacy median-profile figures. Do not use its robust
  sample-count error bars as the basis of a detection claim.

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
