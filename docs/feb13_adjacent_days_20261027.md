# February 13 predicted from February 12 and 14 only

**With all February 13 spacecraft coordinates withheld, five methods predict the solar-center occultation at approximately 08:17–09:04 UTC, lasting 47.1 minutes.** This agrees with the archived February 13 trajectory and disagrees with the approximately 08:21–09:02 markers in Weber et al.'s Figure 2. A plot-label or marker-placement error remains possible. Corruption confined to the February 13 coordinates is a poor explanation of the discrepancy.

The test also uncovered and fixed a real midnight rollover bug in our native tape decoder. The final results below use a separately rebuilt coordinate cache. Correcting the decoder changes the disputed-event predictions by no more than **0.136 seconds**. The suspicious midnight jumps were a decoding artifact and must not be cited as evidence that the original XYZ values were corrupt.

## Entire-day exclusion and prediction

The preceding [orbit check](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/docs/feb13_independent_records_20261027.md) withheld the eclipse interval but still used other February 13 arcs. This stronger test excludes the entire UTC day **before quality filtering, seed selection, orbit fitting, or frequency estimation**. The only spacecraft coordinates available to the target prediction are from February 12 and February 14. All fitted states and predicted trajectories are saved before comparing with the withheld day.

The working interpretation remains Moon-centered kilometers in **FK4/B1950**, transformed with the observation epoch. Solar contacts use the saved independent JPL Sun directions and the established 1,737.4 km lunar radius. The crossing is for the solar center. It is not a frequency-dependent radio disappearance or the first/last contact of the finite optical disk.

| Training and prediction method | Ingress UTC, February 13 | Egress UTC | Duration |
|---|---|---|---:|
| February 12 only: forward two-body propagation | 08:16:56.71 | 09:04:03.88 | 47.120 min |
| February 14 only: backward two-body propagation | 08:17:03.29 | 09:04:06.58 | 47.055 min |
| Both days: joint two-body fit | 08:16:59.84 | 09:04:05.37 | 47.092 min |
| Both days: interpolate between independently fitted orbits | 08:16:59.45 | 09:04:05.07 | 47.094 min |
| Both days: empirical harmonic interpolation | 08:16:59.88 | 09:04:05.75 | 47.098 min |
| Withheld February 13 archive, comparison only | 08:17:00.01 | 09:04:06.10 | 47.101 min |
| Paper markers, approximate readings | 08:21 | 09:02 | ~41 min |

The five predictions span **6.58 seconds at ingress and 2.70 seconds at egress**. These are model-consistency measures, not confidence intervals or estimates of absolute tracking accuracy. All methods share the neighboring archive, adopted frame, and contact geometry. A coherent error affecting several days or a common frame error can survive this test.

![Entire-day exclusion and predicted contacts](C:/Users/Adam/Documents/ChatGPT/RAE2/feb13_adjacent_days_20261027/adjacent_day_prediction.png)

The top panel shows available neighboring-day data, not every row retained in a fit. Shading around the paper markers denotes a ±60-second reading allowance, not a measured or statistical error bar.

The joint fit uses 358 thinned samples after interval-quality filtering. The two-body fits use a spherical Moon and robust losses; they omit lunar harmonics, third-body tides, and radiation pressure. The harmonic alternative fits two harmonics with coefficients changing linearly in time, estimating its frequency from the neighboring days. This avoids relying exclusively on the two-body dynamics. It also avoids straight-line interpolation of XYZ across several revolutions, which would not represent an orbit.

## The midnight bug and its correction

A logical tape record holds eight samples at one frequency. Some records begin before midnight and end after it, with milliseconds-of-day returning to zero while the record retains its starting date. Our decoder previously repeated that date for every sample. Sorting then moved post-midnight samples approximately one day backward.

For example, `RV-DAT/FILE_DS4293_055.DAT`, logical record 684 (zero-based), has date word `740214` and sample times ending `86351922, 90158, 228394` milliseconds. Those last two samples belong to February **15**, not February 14. The adjacent records and the next date header independently support the rollover. Nine records at this boundary account for the affected segment. Their decoded positions match the neighboring orbit at time +1 day, with a median residual of about 0.61 km instead of 5,602 km.

The same defect explains the initially unmatched February 17 00:01:59 control contact: those samples belong to February 18. Its disappearance from the corrected February 17 contact list is a date correction, not a quality cut designed to remove a failed prediction.

The corrected [decoder](C:/Users/Adam/Documents/ChatGPT/RAE2/geometry_beam_investigation_20261026/archive.py) unwraps a midnight transition within each logical record, ignores invalid/empty sample slots, and resets the offset at the next record. Calendar transitions use elapsed days, so month, year, and leap-day boundaries are handled. Small backward timestamp fluctuations are not interpreted as day rollovers.

Across the input archive, **10,408 valid frequency blocks** require this correction. The regenerated coordinate cache contains 1,999,428 unique epochs, the same row count as the original. The raw tape is unchanged. The prior cache and earlier result tables are preserved for comparison; the current analysis uses `feb13_adjacent_days_20261027/rollover_corrected_geometry.npy`.

**Earlier power and coordinate caches do not become corrected merely because the decoder changed.** The original cache now carries a rollover notice. New science runs must rebuild their inputs with the corrected reader, and earlier results using the affected midnight intervals need reassessment. This report reruns the adjacent-day coordinate analysis, not all previous source searches.

## Robustness and control days

Before correcting the decoder, robust fitting already limited the influence of the misplaced samples. Rebuilding the input changes any target ingress/egress prediction by at most **0.136 seconds**. With the corrected data, additionally removing the first two hours of both adjacent days changes any target contact by at most **0.678 seconds**. The secondary screen removing adjacent-day positions over 100 km from the robust joint fit makes no meaningful change. No plausible-radius jump faster than 3 km/s across gaps up to 180 seconds remains in the February 12–14 diagnostic window.

Six other days were withheld in full: February 9, 10, 11, 15, 16, and 17. Each was fitted from its own preceding/following days; none used February 13 for training. These are archive-consistency controls, not independent tracking observations. All **33 available native control contacts** have a predicted contact of the same type within the preselected ten-minute matching window.

| Method | Median absolute contact difference | 95th percentile | Maximum |
|---|---:|---:|---:|
| Previous day only | 2.03 s | 35.06 s | 36.53 s |
| Next day only | 2.43 s | 26.16 s | 28.53 s |
| Joint two-body fit | 2.57 s | 17.72 s | 22.81 s |
| Blended orbits | 1.27 s | 16.01 s | 19.34 s |
| Harmonic interpolation | 1.25 s | 18.84 s | 24.40 s |

Only contacts bracketed by valid observed intervals enter the native comparison; gaps do not constitute failed predictions or independent validation. The less favorable control cases caution against interpreting the few-second model spread for February 13 as absolute accuracy. They remain much smaller than the roughly six-minute duration discrepancy.

## Could the plot be labeled incorrectly?

Yes. The original [Weber, Fainberg and Stone (1976) paper](C:/Users/Adam/Documents/ChatGPT/RAE2/solar_lf_correction_20261027/weber1976.pdf), Figure 2 on PDF page 2, visibly says **13 FEBRUARY 1974**. Its caption identifies the vertical lines as sunset and sunrise. This confirms what is printed, not that the annotation is correct. The approximate readings used here are 08:21 and 09:02. A date OCR error in a text extraction is not evidence that the published date differs.

Three possibilities were tested:

1. **A uniform clock-label offset does not resolve it.** The best common shift is about +57.2 seconds, after which the two contacts still disagree in opposite directions by about 183 seconds each. A shift cannot change the interval's duration.
2. **A different date alone has no close match in the available contact catalog.** Among 125 complete candidate pairs near the printed clock time, including 15 within two weeks of February 13, none places both edges within ±60 seconds of the markers. The nearest alternative nearby date is February 15 at about 08:22:31–09:09:21. June 29 has a similar 40.89-minute width but would also need roughly a three-minute clock shift. This is a many-date coincidence, not an identification of the figure's actual date. Missing coverage prevents an exhaustive rejection of a wrong-date label.
3. **A time-scale error could mathematically reconcile the two markers, but requires substantial distortion.** Each plotted 30-minute interval would need to represent about 34.46 actual minutes, a factor of 1.149. This is the scale needed to force agreement, not evidence that such an error occurred. Incorrect individual marker placement also remains possible.

The date search uses the existing native B1950 solar-contact catalog. The decoder correction was checked against all coordinate rows from 06:00–12:00 UTC, covering every candidate edge: those rows are exactly unchanged. Therefore the midnight bug does not invalidate this restricted date-label comparison.

The result supports continuing with FK4/B1950 while excluding the disputed figure markers from calibration. It does **not** prove the published markers wrong. The decisive missing evidence remains an independently timed shadow measurement or an absolute tracking solution. Routine hourly mission plots with Sun/Earth occultation bars could first determine whether the discrepancy was introduced when the journal figure was made; their retrieval details are in the preceding independent-record report.

## Reproduction and evidence

The [analysis directory](C:/Users/Adam/Documents/ChatGPT/RAE2/feb13_adjacent_days_20261027) contains saved fit states, training rows, predictions, event and control tables, date-label checks, raw midnight record context, and source hashes. `before_rollover_fix` preserves the initial results. Numerical results in this report refer to the corrected cache.

**30 tests passed in 4.38 seconds:** the 18 existing investigation tests, five whole-day propagation/exclusion tests, and seven new rollover regression cases. The propagation tests compare against analytic circular motion and a separate numerical ODE integrator; the exclusion test makes every target-day coordinate NaN; the robustness test injects an opposite-phase segment. Rollover tests cover actual February tape times, empty slots, logical-record resets, calendar boundaries, and small backward clock changes. The retained output is `pytest_results.txt`.

```powershell
$env:PYTHONPATH='C:/Users/Adam/Documents/ChatGPT/RAE2/.audit_deps'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$auditPython='C:/Users/Adam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $auditPython feb13_adjacent_days_20261027/rebuild_geometry.py
& $auditPython feb13_adjacent_days_20261027/predict_day.py
& $auditPython feb13_adjacent_days_20261027/predict_day.py --controls
& $auditPython feb13_adjacent_days_20261027/check_labels.py
& $auditPython feb13_adjacent_days_20261027/summarize.py
& $auditPython -m pytest geometry_beam_investigation_20261026/test_archive_rollover.py geometry_beam_investigation_20261026/test_investigation.py feb13_adjacent_days_20261027/test_predict_day.py -q
```
