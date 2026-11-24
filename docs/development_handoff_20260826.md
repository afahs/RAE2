# RAE-2 Development Handoff — 2026-08-26

## Earth low-rank/covariance addendum

- `scripts/run_earth_lowrank_covariance.py` was added and run on Perlmutter
  against 2,558,541 Earth profile rows in nine lunar blocks. Its compact output
  is `/pscratch/sd/a/afahs/RAE2/earth_lowrank_covariance_v1` (about 4.9 MB).
- All foreground, covariance, and Earth-template quantities are lunar-block
  cross-fitted. The two antennas are estimated separately and cross-antenna
  agreement is explicitly not required.
- Low-rank removal preserves the visible 0.45, 3.93, 6.55, and 9.18 MHz Earth
  edges while reducing much of the 0.70--2.20 MHz common foreground. The 4.70
  MHz asymmetry remains: weak RV1, positive RV2.
- The exact null enumerates all 512 choices of the +/-20-minute control across
  nine blocks (`p_min = 1/513`). Every Earth pipeline exceeds every control
  combination, but attenuated recovery has no universal winner: covariance is
  strongest for weak RV1 injections and raw fixed-zero is strongest for RV2.
- No other source was processed in this run.

## Storage and inference follow-up

- The live morphology-first outputs are now on scratch at
  `/pscratch/sd/a/afahs/RAE2/occultograms_morphology_v1` and
  `/pscratch/sd/a/afahs/RAE2/occultograms_outer_planets_v1`, about 99 MB each.
  Their 5.64 GiB of object NPY/CSV tables were replaced with validated,
  Zstandard-compressed Parquet.
- The six tables needed to rebuild the occultograms are preserved under
  `runtime_inputs/occultogram_v1` in the live tree. The builder now uses these
  paths by default and maps Saturn/Uranus/Neptune to the outer-planet catalog.
- The former 6.2 GB `outputs/` tree was first moved to scratch, then permanently
  deleted after explicit approval and a successful compile of the retained
  Python tree. It is not recoverable. Historical output paths cited later in
  this handoff are provenance only and no longer exist.
- `docs/statistical_detection_plan.md` supersedes the old sample-level SNR
  framing with robust change-point localization, lunar-block permutation,
  multiscale edges, foreground modeling, and injection/recovery.

## Morphology-first occultogram addendum

- `scripts/build_occultograms.py` is now the primary source-review product. It
  shows one independent event per heatmap row, folds reappearance into the
  disappearance sign, places fixed ±10/±20 minute controls beside the target,
  and reports event/lunar-block quantiles rather than a sample-count SNR.
- The event profile table retains raw power and outer-sideband-normalized power;
  no trendline or diffuse model is subtracted. Finite samples are kept by
  default even when legacy amplitude/jump flags reject them, with
  `--validity-mode pipeline` available as a sensitivity run.
- Earth bands 0.70, 0.90, 1.31, and 2.20 MHz are marked
  Galaxy-dominated. Their smooth structure is not evidence for five additional
  Earth detections.
- Extrasolar defaults are Fornax A and Virgo A because they have much cleaner
  Galactic foreground geometry than Cas A, Cyg A, and Tau A. Fixed-source
  catalog rows containing J2000 coordinates are now explicitly `icrs`; the
  B1950 Galactic-center control remains `fk4`.
- The generic SNR summary, old stacked profile, Sun timing/window/control
  figures, and superseded ±900 s Earth grid were removed from the active review
  folder. Earth diffraction/map products and Jupiter burst diagnostics remain.

## Repository state

- Live analysis tree: `/global/u2/a/afahs/RAE2/RyleVonberg`
- Source-only Git worktree: `/global/u2/a/afahs/RAE2_github_upload`
- Approved correctness fixes were fast-forwarded into `publish-rylevonberg` at
  `ffd34a0`.
- Continued code work is isolated on `codex/rae2-development-20260826`
  through `b312e9b`; the branch also includes the final handoff update.
- The source-only worktree is clean. The outer `/global/u2/a/afahs/RAE2`
  checkout remains intentionally unnormalized: 3,671 tracked deletions, four
  modifications, and 27 untracked entries spanning unrelated legacy
  `RAE2Agent` and digitization work pre-dated this audit. They were preserved
  rather than reset, staged, or deleted without project-owner review.
- Nothing was pushed to GitHub.
- Use `module load python/3.12`; Perlmutter's stock `/usr/bin/python3` is
  Python 3.6 and is unsupported for this project.

## Validation

- Source-only worktree: `130 passed`.
- Live analysis tree: `130 passed`.
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
- `outputs/focused_validation/controls/cyg_a_offsource/limb_visibility_states.npy`
- `outputs/sun_focused_validation/controls/sun_offephemeris_predicted_events.npy`
- `outputs/finite_duration_stackfit_fornax_allfreq_v1/finite_duration_stack_profiles.npy`

The directories were smoke runs capped at 50,000 rows, minimal bands, or a
handful of events and were superseded by production runs. The Cyg limb-state
tables were unused cache intermediaries; the reusable predicted-event table
remains. The Sun NPY was stale by 100 rows after exact-geometry regeneration,
so the current CSV was preserved. The 416.7 MB Fornax array was an old,
regenerable sample-level intermediate outside the active Earth/Sun/Jupiter
sequence; its run configuration, binned stacks, summary, and report remain.
Deleted products can be regenerated from their scripts. Raw data, final
summaries, curated plots, and historical reports were preserved.

The refreshed cleanup report is:

`reports/cleanup_generated_outputs/cleanup_generated_outputs_report.md`

Current inventory: 480 image-like files (129,539,960 bytes), 93 NPY tables
(5,114,187,425 bytes), four CSVs above 10 MB (87,451,935 bytes), and no empty
output directories.

## Development commits

- `ec832f0` — exact moving controls and stack-level nulls
- `68c4cc5` — Earth geography diagnostics
- `7e1a356` — lunar reflection diagnostics
- `51f1e70` — Jupiter catalog diagnostics
- `843bbbb` — resumable focused-validation caches
- `b3594f6` — complete randomized stack controls and NaN-safe grading
- `b312e9b` — remove the unused Cyg limb-state cache dependency

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

## Extrasolar foreground-subtraction addendum

`scripts/run_extrasolar_lowrank_occultograms.py` was added and run for Fornax A
and Virgo A. It produces event-level raw/low-rank occultograms with a fixed
zero edge, nested lunar-block control cross-validation, independent antenna
fits, and compact Parquet outputs. A missing-tracer bug found during the first
run was corrected: unavailable standardized tracers now contribute their
training median instead of deleting the target row. The corrected products
retain the full event populations.

Fornax A uses 1,799 events in nine lunar blocks. Common foreground slopes are
reduced, but no coherent zero-centered edge appears. Virgo A uses only 34
events in six blocks and remains sparse and outlier-sensitive. Neither result
supports a detection claim. The review and products are in
`/pscratch/sd/a/afahs/RAE2/extrasolar_lowrank_occultograms_v1`; the next
recommended experiment is the covariance-aware fixed-zero fit on Fornax A.

## Fornax covariance addendum

`scripts/run_fornax_covariance_fixedzero.py` completed the recommended Fornax
test. It uses five non-tracer bands, control-only spectral templates, OAS
time/frequency covariance, robust Student-t GLS, four fixed-zero edge widths,
and all 262,144 combinations of four off-time controls across nine lunar
blocks. The complete all-antenna/method family-wise p is `0.000980`, driven by
an RV1 raw 240 second ramp. RV1 becomes null after low-rank subtraction
(`p = 0.435`); RV2 raw and low-rank are weaker (`p = 0.0147` and `0.0447`).

Treat this as a promising fixed-time excess, not a Fornax A detection. The
event-level morphology is not Earth-like, the winning feature is broad, and
temporal controls cannot exclude diffuse Galactic or lunar/beam structure at
the catalog position. The next required experiment is the frozen pipeline on
matched nearby blank-sky positions with similar lunar-limb and Galactic
geometry. Products and the full review are in
`/pscratch/sd/a/afahs/RAE2/fornax_covariance_fixedzero_v2`.

## Fornax matched blank-sky addendum

`scripts/run_fornax_blanksky_comparison.py` completed the required spatial
control. It predicted and processed 24 local sky positions on 2- and 5-degree
rings around Fornax A using the same five shifts, nine bands, two independent
antennas, lunar-block cross-fitting, and target-frozen covariance candidates.
Only compact block profiles and selected review occultograms were retained.

The frozen RV1 raw 240 second ramp places Fornax first of 25 positions: target
score `1.6575`, strongest blank `1.1273`, and spatial empirical `p = 0.04`, the
minimum possible with 24 blanks. Frozen RV1 sharp-step and RV2 raw candidates
also rank first. Low-rank localization is weaker: RV1 ranks fifth (`p = 0.20`)
and RV2 ranks second (`p = 0.08`). The event heatmaps do not show an Earth-like
vertical edge, and raw target medians mostly overlap the blank-sky envelope.
Keep the status as a localized raw candidate, not a Fornax A detection.
Products are in `/pscratch/sd/a/afahs/RAE2/fornax_blanksky_comparison_v1` and
the complete directory is about 21 MB with no CSV or NPY outputs.
