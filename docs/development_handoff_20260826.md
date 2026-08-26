# RAE-2 Development Handoff — 2026-08-26

## Repository topology

- `/global/u2/a/afahs/RAE2` is the original `main` worktree. It is intentionally
  analysis-heavy (about 7.9 GB), has many historical tracked deletions, and is
  not safe to commit or merge wholesale.
- `/global/u2/a/afahs/RAE2_github_upload` is a linked, source-only Git worktree
  (about 3.5 MB). It was clean at the start of this audit and is the right base
  for continued GitHub development.
- Maintenance work from this audit is isolated on
  `codex/rae2-cleanup-20260826`; nothing was pushed to GitHub.

## Verified state

- Supported Perlmutter environment: `module load python/3.12`.
- Source-only worktree after fixes: `99 passed`.
- Live analysis tree after the same fixes: `119 passed`.
- Both trees compile successfully under Python 3.12.
- The default `/usr/bin/python3` is Python 3.6.15 and must not be used for this
  project.
- The package is not installed in the stock Python module. Either install it in
  an isolated environment as documented in `README.md`, or use `PYTHONPATH=.`
  when running a repository script directly.

## Correctness fixes applied

1. `find_limb_transitions` now requires two adjacent finite limb angles and
   valid timestamps. Previously, a missing angle was converted to `False` by
   comparison and could create a false disappearance/reappearance pair.
2. `add_offsource_pvalues` now compares frequency bands numerically when
   possible. Previously, equivalent values such as integer `7` and float `7.0`
   failed a string comparison and silently removed all controls for that row.

Both fixes have regression tests.

## Open audit findings

These are not changed because they affect scientific selection or historical
result reproducibility and need an explicit analysis decision.

1. `validate_timeseries` labels every value at or above each group's 99.99th
   percentile as `saturation_or_nonpositive`. This guarantees that legitimate
   maxima can be rejected even when no instrument saturation occurred. Replace
   the percentile rule with a documented receiver limit, or rename/split the
   flag and validate its impact.
2. `offset_unit_vectors_tangent` uses a first-order tangent-plane displacement
   followed by normalization. At the 10–20 degree moving-control offsets used
   by the Sun analysis, the realized great-circle separation is slightly
   smaller than requested. An exact spherical offset would change control
   tracks and requires regeneration of affected products.
3. `aggregate_source_level` sets `randomized_empirical_p` to the minimum
   event-level empirical p-value in a channel. This is not a stack-level
   empirical p-value and is sensitive to multiple comparisons. Detection
   grading should use a control distribution of the complete stack statistic.
4. Older plot inventories and handoff notes point to PNGs removed by a previous
   generated-output cleanup. The confirmation CSVs remain; source inspection
   plots were rebuilt during this audit. Treat the rebuilt README as current.
5. The live analysis tree contains 12 scripts and 7 tests that are not in the
   source-only worktree. They all pass in the live tree, but should be reviewed
   and ported in a dedicated commit rather than copied wholesale.

## Unpublished live-only files

Scripts:

- `investigate_earth_noise_geography.py`
- `investigate_lower_v_lunar_reflection_interference.py`
- `ocr_warwick_dulk_riddle_and_match_rae2.py`
- `plot_all_frontside_signal_vs_water.py`
- `plot_jupiter_burst_count_vs_cml.py`
- `plot_lunar_reflection_data_views.py`
- `plot_lunar_reflection_interference_results.py`
- `plot_pass_signal_vs_visible_water.py`
- `plot_warwick_dulk_riddle_catalog_phase_cml.py`
- `plot_wdr_rae2_5min_all_frequency_samples.py`
- `run_naive_subspacecraft_land_ocean.py`
- `run_rae2_lunar_reflection_feasibility.py`

Tests observed only in the live tree:

- `test_all_frontside_signal_vs_water.py`
- `test_earth_noise_geography.py`
- `test_lunar_reflection_interference.py`
- `test_naive_subspacecraft_land_ocean.py`
- `test_visible_water_signal.py`
- `test_warwick_dulk_riddle_catalog_phase_cml.py`
- `test_warwick_dulk_riddle_ocr_match.py`

## Curated figures

The reviewed plot set and provenance notes are in:

`/global/u2/a/afahs/RAE2/RyleVonberg/outputs/curated_best_plots_20260826`

Jupiter is labeled as potential/unconfirmed, the IMP-6 absolute map scale is
labeled provisional, and the Sun figures carry the newer control caveat in the
folder README.

## Recommended next development sequence

1. Review and merge `codex/rae2-cleanup-20260826` into the normal development
   branch after inspecting the two correctness fixes.
2. Port the 19 live-only scripts/tests in small topical commits, starting with
   the Earth geography and Jupiter catalog work; run the full suite after each.
3. Resolve the percentile-based saturation flag before regenerating detection
   claims.
4. Define a stack-level empirical null and rerun Sun/Jupiter grading.
5. Regenerate moving-source controls if the tangent offset is replaced with an
   exact spherical construction.
6. Keep generated outputs out of Git and record every promoted figure's source
   table, run configuration, code commit, and caveat in its README.
