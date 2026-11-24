# Source sensitivity fixes and validation — 20 October 2026

All five audited defects are fixed and covered by regression tests. The full
package suite passes: **185 tests**, with three existing pandas/NumPy timedelta
deprecation warnings. The real-background experiment demonstrates improved
recovery for strong persistent occultation steps, but **does not demonstrate a
general improvement for weak sources or bursts**.

## Changes

1. Ingest no longer treats a sample percentile as an instrument saturation
   limit. Tied bright plateaus and real sharp edges survive validation.
   Percentile and jump flags remain advisory; an explicit positive
   `saturation_power` limit can reject instrument saturation. Nonfinite,
   nonpositive, duplicate, and gap-invalid samples remain excluded. Strict
   masking no longer silently clips the upper power tail by default.
2. Default stacking jointly fits a linear background and the predicted step,
   then subtracts only the background. A noiseless injected amplitude of 10
   recovers 10, compared with about 2.72 using the old sideband linear fit.
3. Flat or missing sidebands no longer discard raw occultogram profiles.
   Unavailable normalization is recorded explicitly, with NaN normalized
   values and separate raw/normalized event counts.
4. Ordinary and accelerated/control stacks share extraction, masking,
   detrending, and scoring. Processing metadata rejects stale or incompatible
   control caches. Focused validation rebuilds target stacks and refreshes
   retained sample quality flags.
5. Solar windowing and related scan-phase arithmetic explicitly convert
   timestamps to nanoseconds. Tests cover seconds, milliseconds,
   microseconds, and nanoseconds.

The first real-background run exposed an additional scoring interaction:
estimating noise from a profile containing the recovered signal suppresses
the score as amplitude increases. Both stack paths now estimate noise from
the fitted residual. The regression suite checks monotonic response to
injected amplitude. The processing version is `source_preserving_v3`.
The score still requires empirical matched controls; it is not a Gaussian
significance for correlated spacecraft samples.

## Real-background experiment

Input master: `/global/cfs/cdirs/m4895/Data/RAE2/interpolatedRAE2MasterFile.csv`
(3,155,140,923 bytes; 13,701,575 rows scanned).
Event tables: `/global/u2/a/afahs/RAE2/RyleVonberg/runtime_inputs/occultogram_v1`.
Execution used an interactive CPU allocation, four workers, and isolated code
under `/pscratch/sd/a/afahs/RAE2/audit_fixes_20261020_run1`.

The experiment selected one physical event per source per UTC day without
examining power: 64 days each for Earth, Fornax A, and Sun, and 10 for Jupiter.
Windows were shifted by +/-1,200 seconds and extended +/-600 seconds around
the shifted center. Half of each source's days calibrated thresholds; the
other half evaluated recovery. A seed of 20261020 fixed the selection.
All nine frequency bands and both coarse receivers were included where
samples supported an estimate. The master scan retained 87,776 raw rows.

Persistent steps and 180-second bursts were added to raw power before sample
validation. Amplitudes were 0.25, 0.5, 1, 2, 4, and 8 times the local robust
background residual scale. Each pipeline's detection threshold was separately
calibrated at the empirical 95th percentile of its uninjected calibration
days, requiring at least 10 usable calibration windows per channel.
Missing evaluation scores counted as non-recoveries.

There were 44,016 before/after measurements. The paired comparison retained
1,515 evaluation channel-windows per amplitude/shape, spanning 76 distinct
UTC days. These are correlated receiver/window trials, not 1,515 independent
sources. Jupiter lacked enough calibration days and was excluded from the
reported detection rates. In total, 702 paired shape/level rows lacked
supported thresholds, and 456 candidate channel-windows lacked adequate
sample support or a nonzero background scale.

| Amplitude / background scale | Persistent step before | Persistent step after | Burst before | Burst after |
|---:|---:|---:|---:|---:|
| 0.25 | 9.37% | 8.18% | 10.10% | 7.79% |
| 0.5 | 12.41% | 9.97% | 12.81% | 9.57% |
| 1 | 19.87% | 16.11% | 18.28% | 14.85% |
| 2 | 38.35% | 34.39% | 31.95% | 27.92% |
| 4 | 65.81% | 74.85% | 58.75% | 47.59% |
| 8 | 78.94% | 95.58% | 81.72% | 63.37% |

Held-out uninjected false-positive rates were **5.68% before and 6.34% after**.
Thus the two finite-sample calibrations do not achieve exactly identical
false-positive rates. Median persistent-step amplitude transfer improves
from about 27% to 100%; that is an amplitude preservation result, not a
3.7-fold detection sensitivity gain.

Paired UTC-day bootstrap intervals (2,000 resamples, thresholds held fixed):

| Comparison | Recovery difference | 95% interval |
|---|---:|---:|
| Persistent step at 1x | -3.76 percentage points | -6.03 to -1.15 |
| Persistent step at 4x | +9.04 percentage points | +6.08 to +12.23 |
| Persistent step at 8x | +16.63 percentage points | +13.64 to +19.86 |
| Burst at 8x | -18.35 percentage points | -21.47 to -14.93 |
| Uninjected false positives | +0.66 percentage points | -0.75 to +2.30 |

These intervals omit calibration-threshold uncertainty. The first pass
before the residual-noise scoring correction is preserved under
`audit_20261020/first_pass_v2`; it must not be confused with the final results.
The final pass reused the unchanged baseline measurements after verifying
the selected windows and master path/size.

## Interpretation and limits

The fixes remove demonstrable mechanisms that erase or miswindow real
signals. Strong step recovery improves, especially in Sun and Fornax A
backgrounds. Nevertheless, weak step recovery declines by approximately
1–4 percentage points in this experiment. Burst recovery declines across
the tested amplitudes. Preserving amplitude also preserves background
structure, and a continuum step template is not optimized for a short
burst. This experiment does not isolate the contribution of each change.
The existing explicit legacy baseline options remain available for
controlled comparisons; matched controls must use the same option.

This is end-to-end through raw-power injection, validation, masking,
detrending, and the stack score, evaluated one event window at a time.
It does not rerun source geometry, the full-mission multi-event source
grading pipeline, or a dedicated burst detector. Quality statistics are
window-local. Off-time backgrounds may contain real emission or instrument
structure. Thresholds are exploratory per-channel thresholds, not final
multiple-search-adjusted discovery criteria. No new source detection or
full-mission detection limit is established here.

Before adopting the new defaults for weak-source or burst discovery, the
next scientific validation should compare baseline/template choices using
fresh calibration and evaluation days, including a dedicated burst
template. Existing source claims should be recomputed with matched controls;
historical reports and the live remote project were not overwritten.

## Reproduction and artifacts

Run `scripts/run_end_to_end_sensitivity.py --help` for the benchmark CLI.
Use the master and event paths above, an untouched pre-fix package, and
`--windows-per-source 64 --workers 4 --seed 20261020`. Omit
`--reuse-before-results` for a completely fresh before/after run.

The workspace contains:

- `audit_20261020/pre_fix`: preserved original package and solar/drift scripts.
- `audit_20261020/release_tests.log`: full 185-test result.
- `audit_20261020/core_fixes.diff` and `fixed_source_hashes.json`: source provenance.
- `audit_20261020/real_background_results/run_manifest.json`: inputs and run scope.
- `audit_20261020/real_background_results/prespecified_windows.csv`: selected windows.
- `audit_20261020/real_background_results/paired_measurements.parquet`: measurements.
- `audit_20261020/real_background_results/channel_recovery.csv`: channel thresholds/rates.
- `audit_20261020/real_background_results/paired_recovery_summary.csv`: aggregate rates and intervals.
- `audit_20261020/real_background_results/source_recovery_summary.csv`: source-specific rates.
- `audit_20261020/real_background_results/recovery_comparison.png`: comparison figure.
- `audit_20261020/summarize_recovery.py`: summary/figure reproduction.

Run the package tests with `python -m pytest tests` from
`remote_upload_snapshot`, with its declared dependencies installed.
