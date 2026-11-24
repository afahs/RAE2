# Fresh point-source investigation — 20 October 2026

No defensible point-source detection emerged. Fornax A has a temporal excess,
but six nearby comparison directions score higher. Virgo A has a similar
localization problem, and a separate promising Virgo result failed a frozen
check on unused days. The most useful improvement was recovering about 4.02
million fine-channel measurements excluded by an earlier table-building rule;
these measurably improve injected-source recovery.

This investigation recomputed events and measurements from mission tables.
Previous source detections, fitted profiles, and event lists were not inputs.
No solar-burst search was included. Scripts and results are in
`../../point_source_investigation_20261020/`, the experiment directory below.

## Data and independent methods

The shared master is
`/global/cfs/cdirs/m4895/Data/RAE2/interpolatedRAE2MasterFile.csv`:
3,155,140,923 bytes and 13,701,575 rows, spanning July 1973–January 1976.
The separate local decoded table is
`C:/Users/Adam/Documents/PhD/Diffraction/RAE2MasterFile.csv`:
3,621,177,542 bytes and 17,503,360 rows. All rows were scanned, but the source
experiment uses at most 192 blindly selected UTC days per source, one event
per day. It is not an analysis of every available occultation.

The search covers Fornax A, Virgo A, Cygnus A, Cassiopeia A, and Taurus A,
nine frequency bands, and both antennas' fine and coarse measurements.
Catalog ICRS directions were transformed to FK4/B1950, retaining the existing
spacecraft-frame interpretation as an unverified assumption.

Four estimators were compared on native samples:

1. Median power contrast between the two sides of the event, using 60–420 s.
2. A simultaneous occultation step and linear background fit.
3. A simultaneous occultation step and quadratic background fit.
4. A 240 s transition ramp fitted with a linear background.

Fits use ±600 s windows and equal total weight for each frequency dwell.
Zero fine readings are missing telemetry, and duplicate channel/timestamp
records are excluded. There is no bright-tail or jump clipping and no fitted
timing shift. Temporal controls at ±1,800 s set the background scale. A fifth,
exploratory approach subtracts a robust local spatial plane learned only from
comparison directions; the tested target is excluded from that fit.

Each source has 24 comparison directions on 3° and 7° rings. Their events are
matched by day, disappearance/reappearance, and nearest crossing. Source scores
aggregate lunar-block means. A common set of 4,095 block-sign permutations
accounts for searching methods and channels, with a factor of five for the
five sources. The expanded search includes both original and restored fine
data versions: 216 method/band/stream combinations per source. Allowing the
two background-analysis families adds another conservative factor of two.
Alternating-block selection and testing, separate event directions, and
Earth-hidden subsets provide further diagnostics.

These scores are not Gaussian sigma. The permutation null assumes block-level
symmetry and does not exclude geometry-dependent foregrounds. Nearby fields
are correlated and have different sample support; their rank is a localization
diagnostic, not a calibrated discovery probability.

## Point-source results

Rank 1 would mean the target exceeds all 24 comparison fields. These are the
final expanded-search results, including restored fine measurements.

| Source | Strongest combination | Usable blocks | Temporal search p, including both families | Spatial rank | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Fornax A | Upper fine, restored, 0.90 MHz; median contrast | 23 | 0.0195 | 7/25 | Temporal excess, but not localized to Fornax |
| Virgo A | Lower fine, original, 1.31 MHz; median contrast | 12 | 0.0732 | 14/25 | Nearby fields commonly score higher |
| Cygnus A | Lower fine, restored, 4.70 MHz; quadratic fit | 15 | 1.0 | 8/25 | No persuasive excess |
| Taurus A | Lower fine, restored, 0.45 MHz; median contrast | 13 | 1.0 | 21/25 | No persuasive excess |
| Cassiopeia A | Insufficient coverage | 1 | — | — | Only four valid days; not a useful detection test |

The geometric selection found 650 available days for Fornax, 269 for Virgo,
370 for Cygnus, 414 for Taurus, and four for Cassiopeia. It selected 192 days
for each of the first four. Available power and control coverage further limit
the usable blocks in each channel. Fornax's Earth-hidden subset has only one
block, so it cannot establish a convincing Fornax result independent of
terrestrial emission. The profiles also contain broad background gradients.

The local spatial-plane approach initially singled out Virgo at 6.55 MHz,
upper coarse, with the quadratic fit. It ranked first among nearby fields,
but its search p was about 0.066 after allowing both analysis families.
The source, frequency, stream, and estimator were then frozen. Of 77 previously
unused Virgo days, 39 supported measurements across 12 blocks. The raw check
gave one-sided p = 0.693 and the spatial-residual check p = 0.354, both ranking
9/25. The lead did not reproduce. These unused days share lunar blocks with
the exploratory sample, so they are not an independent observing-season test.

As a positive control, the same pipeline recovered the Earth's occultation
signal at 9.18 MHz in lower coarse data, with within-search p = 0.000488 and
positive disappearance and reappearance contrasts. Both alternating-block
checks were positive (p = 0.000732 and 0.00220). This checks timing/sign and
basic recovery, but does not calibrate faint astronomical-source sensitivity.

Relevant files: `expanded_results/final_candidates.csv`,
`expanded_results/final_spatial_comparison.png`,
`results/selected_source_profiles.png`, `virgo_reserved/confirmation.json`,
and `earth_control/`.

## Two concrete data problems

### Invalid orbital positions could manufacture occultations

The approximately 15 s geometry cache contains 2,143 rows at or inside the
lunar radius, including 1,117 zero-position rows. The first independent
predictor checked only the position before a crossing. The Earth control
exposed crossings into invalid geometry. That initial run is superseded.

The final event selection checks both bracket radii, finite coordinates,
gaps no longer than 60 s, inferred speed at most 5 km/s, and crossing-time
agreement within 5 s between approximately 15 s and 30 s geometry grids.
These checks precede the blind day selection and power analysis. Final
selected events have no unphysical brackets under these checks.

The production package had the analogous failure: clipping the apparent Moon
radius for an impossible spacecraft position could yield a finite limb angle.
`rylevonberg/geometry.py::moon_limb_angle_deg` now returns NaN for nonfinite
radii or positions at/inside the Moon. Two regression tests cover the repair.
The additional speed/gap/cadence gates are in the independent investigation,
not yet a full production-ingestion rewrite. Historical event tables have
not been regenerated.

### Coarse-channel gating discarded valid fine telemetry

`C:/Users/Adam/Documents/PhD/Diffraction/CleanFileGenerator.ipynb` filters
whole rows using upper-coarse values below `1e10` and unequal to zero.
A missing coarse value fails this rule even when fine measurements are valid.
The notebook also interpolates geometry by row and fills remaining geometry
NaNs with zero, explaining why the derived geometry needs explicit checks.

Extracting native finite-positive fine measurements directly from the decoded
table, without a coarse-channel gate, produced:

| Stream | Original master | Restored extraction | Additional measurements |
| --- | ---: | ---: | ---: |
| Upper fine | 1,971,344 | 3,982,106 | 2,010,762 |
| Lower fine | 1,969,866 | 3,979,120 | 2,009,254 |

Each stream now contains about 2.02 times as many measurements. This is a
sample-count increase, not a twofold sensitivity gain. A sampled provenance
check matched 2,739 unique retained timestamp/channel records to the decoded
table; every comparable finite power value agreed exactly. Geometry differed,
as expected after interpolation. The notebook rule and inventory do not prove
the exact lineage of every row between these separately saved products.

The restored fine cache is saved independently; neither source CSV was
overwritten. Original coarse channels remain subject to historical filtering.
This is not a complete reconstruction from original telemetry.

## Measured recovery improvement

Persistent steps and 240 s ramps were injected into native off-time data.
Even lunar blocks calibrate per-channel 95th-percentile thresholds; odd blocks
evaluate recovery. The final comparison pairs the same windows and the same
injected power between original and restored fine data, with common support
across all four estimators: 8,680 evaluation trials per method/shape/level.
Each version uses its own scoring normalization. This avoids calling a change
in injected amplitude a sensitivity improvement.

For step injections at twice the **original fine-data control scale**:

| Estimator | Original recovery | Restored recovery | Original false positives | Restored false positives |
| --- | ---: | ---: | ---: | ---: |
| Step + linear background | 15.15% | 21.32% | 5.44% | 5.58% |
| Step + quadratic background | 15.16% | 20.70% | 5.68% | 5.66% |
| Median contrast | 8.40% | 13.23% | 2.89% | 3.04% |
| Ramp + linear background | 5.24% | 11.04% | 1.82% | 3.08% |

The linear-step gain is 6.18 percentage points. A 2,000-resample paired
lunar-block bootstrap gives a conditional 95% interval of 5.20–7.19 points,
resampling 13 evaluation blocks with thresholds held fixed. It does not
include uncertainty in threshold calibration. Its false-positive change is
0.14 points, with interval −0.26 to +0.55 points. At one original control
scale, linear-step recovery improves from 9.06% to 11.04%.

Different estimators have different achieved false-positive rates, so the
table is not an exact equal-false-alarm ranking of all methods. In particular,
the ramp estimator's false-positive rate also rises. These are window-level
injection results, not Jy limits or calibrated detection power for the entire
source-level search. Spatial-plane subtraction has not been calibrated with
injections propagated into overlapping comparison windows; it could absorb
real extended/shared emission and remains exploratory.

Relevant files: `expanded_results/common_amplitude_fine_recovery.csv`,
`expanded_results/fine_recovery_block_intervals.csv`, and
`expanded_results/fine_data_recovery_comparison.png`.

## Validation and reproduction

The production suite passes **187 tests** (three existing pandas/NumPy
timedelta deprecation warnings). The independent analysis passes **16 tests**,
covering geometry, sparse sampling, signal-preserving fits, zero fine telemetry,
search multiplicity, and robust spatial fitting. Logs are `package_tests.log`
and `analysis_tests.log` in the experiment directory.

The final remote working directory is
`/pscratch/sd/a/afahs/RAE2/point_source_fresh_20261020/v2`.
Its parent `cache/` contains original native band arrays and sampled geometry.
The original, invalid first-pass source results in the parent directory must
not be used. `search_initial_geometry.py` preserves that implementation.

Reproduction order, with original and restored caches in the script paths:

1. `run_v2.sh`: rebuild validated events, run the four original streams,
   injections, geometry checks, profiles, and Earth control.
2. `recover_fine.py`: extract fine telemetry from the local decoded master;
   copy `recovered_fine_cache/` into the remote working directory.
3. `run_recovered.sh`: apply the same events to restored fine streams and
   combine both data versions with search-wide calibration.
4. `python spatial_residual.py --input results --output results/spatial_foreground`:
   run the exploratory local spatial-plane comparison.
5. `final_checks.sh`: inject identical power using the original cache as the
   amplitude reference, and run the frozen Virgo reserved-day check.
6. `plot_results.py` and `summarize_final.py`: create comparison tables,
   bootstrap intervals, and plots from the downloaded artifacts.

`EXPERIMENT.md` records initial choices and subsequent input-driven and
exploratory additions. `provenance.json` records script/result hashes and
`geometry_fix.diff` records the production geometry repair. The optional
injection-reference argument was added after the original source searches;
`search_before_common_amplitude.py` preserves their search implementation.

The next work with the clearest scientific value is validating the coordinate
frame and reconstructing geometry from trustworthy ephemerides, then testing
a joint antenna-response/diffuse-background model with injections through the
complete spatial analysis. The present results do not establish absence of
Fornax emission; they establish that the tested excesses are not sufficient
to identify a point source.
