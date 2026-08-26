# RAE-2 Development Handoff — 2026-08-26

## Repository state

- Live analysis tree: `/global/u2/a/afahs/RAE2/RyleVonberg`
- Source-only Git worktree: `/global/u2/a/afahs/RAE2_github_upload`
- Approved correctness fixes were fast-forwarded into `publish-rylevonberg` at
  `ffd34a0`.
- Continued code work is isolated on `codex/rae2-development-20260826`
  through `b3594f6`; the branch also includes the final handoff update.
- Nothing was pushed to GitHub.
- Use `module load python/3.12`; Perlmutter's stock `/usr/bin/python3` is
  Python 3.6 and is unsupported for this project.

## Validation

- Source-only worktree: `129 passed`.
- Live analysis tree: `129 passed`.
- Both trees are free of `__pycache__` and `.pytest_cache` after validation.
- `validate_timeseries` is intentionally unchanged; its current behavior was
  explicitly accepted for this development sequence.

## Completed correctness and audit work

1. `find_limb_transitions` ignores missing adjacent limb samples.
2. Off-source frequency matching treats equivalent numeric band dtypes
   consistently.
3. Moving-source controls now use an exact spherical exponential-map offset.
   A 12° east / 16° north request is tested to produce an exact 20° great-circle
   separation.
4. Source-level randomized p-values now compare the real stack with complete
   randomized stack realizations. Event-level minimum p-values are no longer
   substituted for a stack null.
5. Every randomized summary records `control_id` and `n_input_events`;
   incomplete/capped realizations are rejected during aggregation.
6. Optional missing scoring columns no longer crash source aggregation.
7. Detection grading no longer emits all-NaN reduction warnings when both
   methods are unavailable.
8. Focused validation reuses verified Cyg event/off-source and complete-null
   caches, with force flags for intentional regeneration. The 109 MB Cyg limb
   state intermediary is no longer required for cache reuse and is written
   only with `--write-cyg-limb-states`.
9. All 12 live-only scripts and 7 live-only tests were audited and ported in
   topical commits. Three Perlmutter-only fallback paths were replaced with the
   portable `RAE2_MASTER_CSV` / `data/interpolatedRAE2MasterFile.csv`
   convention.

## Complete-control regrade

Twenty full randomized realizations were generated per analysis. The minimum
resolvable empirical p-value is therefore `1/21 = 0.047619`.

- Earth: `positive_control_confirmed`; strongest stack SNR `19.27`,
  randomized p `0.047619`.
- Sun: focused classification `systematic_like` and unified status
  `likely_systematic`; strongest stack SNR `-8.47`, randomized p
  `0.095238`, moving off-ephemeris p `0.235294`.
- Jupiter: `unresolved`; strongest stack SNR `3.06`, randomized p
  `0.047619`. This remains a potential/episodic diagnostic, not a detection.
- Cyg A: `unresolved`; the strongest 900 s stack is large but does not pass
  all timing/randomized/off-source gates.

Authoritative tables:

- `outputs/summary/source_level_results.csv`
- `outputs/summary/source_final_summary.csv`
- `outputs/sun_focused_validation/sun_source_level_results.csv`
- `outputs/sun_focused_validation/controls/sun_randomized_stack_summary.csv`
- `outputs/focused_validation/jupiter_randomized_stack_summary_600s.csv`
- `outputs/focused_validation/jupiter_randomized_stack_summary_900s.csv`

## Curated figures

The selected eight-figure set is in:

`/global/u2/a/afahs/RAE2/RyleVonberg/outputs/curated_best_plots_20260826`

It contains Earth occultation profiles and diffraction comparisons, relative
and provisional absolute sky maps, two potential/unconfirmed Jupiter
diagnostics, and two explicitly Sun-labelled control figures. Read that
folder's `README.md` for provenance and scientific caveats.

## Cleanup completed

Deleted only products demonstrated to be temporary or superseded:

- `outputs/earth_noise_geography_smoke`
- `outputs/lower_v_lunar_reflection_interference_smoke`
- `outputs/lower_v_lunar_reflection_interference_smoke2`
- `outputs/upper_v_novaco_brown_scan_maps_raw_master_smoke_v1`
- `outputs/upper_v_novaco_brown_scan_maps_rv1_coarse_earthbeam90_smoke_v1`
- `outputs/upper_v_novaco_brown_scan_maps_rv1_fine_sunveto10db_destripe_smoke_v1`
- `outputs/upper_v_novaco_brown_scan_maps_smoke_v1`
- `outputs/focused_validation/controls/cyg_a_offsource/limb_visibility_states.csv`

The directories were smoke runs capped at 50,000 rows, minimal bands, or a
handful of events and were superseded by production runs. The Cyg limb-state
CSV was an unused 109 MB cache intermediary; the reusable predicted-event
table remains. Deleted products can be regenerated from their scripts. Raw
data, production outputs, and historical reports were preserved.

The refreshed cleanup report is:

`reports/cleanup_generated_outputs/cleanup_generated_outputs_report.md`

Current inventory: 480 image-like files (129,539,960 bytes), 96 NPY tables
(5,605,071,538 bytes), five CSVs above 10 MB (196,444,752 bytes), and no empty
output directories.

## Development commits

- `ec832f0` — exact moving controls and stack-level nulls
- `68c4cc5` — Earth geography diagnostics
- `7e1a356` — lunar reflection diagnostics
- `51f1e70` — Jupiter catalog diagnostics
- `843bbbb` — resumable focused-validation caches
- `b3594f6` — complete randomized stack controls and NaN-safe grading

## Recommended next sequence

1. Review `codex/rae2-development-20260826` and merge it into
   `publish-rylevonberg`; push only after review.
2. If finer p-value resolution is needed, raise complete randomized controls
   above 20 and run via a checkpointed/batch implementation.
3. Investigate why the large Cyg A stack fails the full control set before
   treating it as source-like.
4. Keep Jupiter labelled potential/unconfirmed and pursue the episodic
   Io-phase/burst tests rather than a continuum detection claim.
5. Keep the IMP-6 absolute sky-map calibration labelled provisional until an
   independent calibration audit is completed.
