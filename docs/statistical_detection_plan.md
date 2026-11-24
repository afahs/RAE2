# RAE-2 Occultation Detection: Statistical Follow-up

## Why the old SNR-style metrics are not claim statistics

The time bins within an occultation window are correlated, the fixed-time
controls overlap the target window, the Galactic foreground is nonstationary,
and occasional bursts/outliers are heavy-tailed. Treating bins as independent
therefore understates uncertainty. A scalar before/after contrast also cannot
distinguish a localized occultation edge from a broad foreground slope.

The independent units for inference should be events and, conservatively,
29.53059-day lunar blocks. Statistical evidence should confirm the visible
morphology rather than replace it.

## Recommended first analysis

### 1. Robust fixed-zero change-point fit

For every event and channel, fold disappearance/reappearance to the same
physical sign and fit two models to the raw profile:

- Null: a smooth local foreground/instrument baseline.
- Source: the same baseline plus an occultation edge with amplitude `A`, fixed
  at the predicted time `t = 0`.

Use Student-t residuals or a Huber loss, and convolve the edge with the time
bin, lunar-limb speed, and (for the Sun) finite source diameter. A real source
should produce correctly signed amplitudes at zero across event types, lunar
blocks, antennas, and nearby window sizes. Galactic slopes should be absorbed
by the local baseline or fail the time-shift and multiscale controls.

Aggregate one amplitude estimate per event, then one robust estimate per lunar
block. Report the effect size, a block-bootstrap interval, and the fraction of
correctly signed events. This is the closest statistical analogue to the
visual occultogram.

### 2. Matched empirical-null permutation test

Apply exactly the same fixed-zero edge statistic at the predicted time,
the fixed +/-10 and +/-20 minute shifts, wrong-antenna pairings, neighboring
frequencies, and matched blank-sky positions. Exchange the predicted/control
labels within event or lunar block to obtain an empirical null.

Use a maximum statistic across the predeclared frequencies, antennas, lags,
and edge scales. This controls the look-elsewhere effect. A cluster-based
permutation version can reward a compact, contiguous time-frequency edge
without treating every pixel as an independent test.

Earth must be recovered when its labels are hidden; control positions must not
be recovered. Only then should the procedure be applied to other targets.

## Foreground-aware extensions

### 3. Low-rank cross-frequency Galactic model

The Galactic contribution is correlated across frequency and changes smoothly
with viewing geometry. Fit a robust low-rank/dynamic-factor model jointly to
all nine bands, using only outer sidebands and fixed-time or blank-sky controls.
The 0.70, 0.90, 1.31, and 2.20 MHz Earth channels can serve as foreground
tracers, not additional Earth detections. Search the residuals for a localized
edge. Never train the foreground model on the central target interval, because
that can subtract a real source.

### 4. Joint hierarchical source model

Fit all events, frequencies, and both antennas together with:

- a shared edge-time distribution centered on the ephemeris;
- a smooth source spectrum across frequency;
- antenna/beam response factors;
- event and lunar-block random effects; and
- heavy-tailed residuals.

Partial pooling can recover a weak repeated source without pretending that
millions of time samples are independent. Validate with leave-one-lunar-block-
out prediction and posterior-predictive checks, not a single Bayes factor or
nominal SNR.

### 5. Multiscale edge persistence

Apply a signed Haar/wavelet or derivative-of-Gaussian edge filter at
predeclared 30, 60, 120, and 240 second scales. A true occultation should peak
at zero with stable localization across nearby scales. A diffuse Galactic
gradient should broaden, drift, or grow monotonically with scale. Calibrate
the maximum response with the matched empirical null above.

### 6. Lunar-limb tomography

Across many lunar chords, solve jointly for a smooth diffuse sky plus sparse
point sources: low-order spherical harmonics or total variation for the
Galaxy, and an L1/sparse component for compact sources. This uses the Moon as
a scanning knife edge rather than stacking each catalog position separately.
Injection/recovery is essential to choose regularization and measure bias.

## Target-specific strategy

- **Sun:** use an extended-disk edge template and a robust quiet/burst mixture.
  Analyze burst intervals separately rather than allowing a few rows to drive
  a continuum claim.
- **Fornax A and Virgo A:** keep these ahead of plane-confused sources because
  they are well away from the Galactic center/plane. Select matched blank-sky
  controls at similar ecliptic and Galactic latitude and similar lunar/beam
  geometry.
- **Jupiter:** use a burst-state or hidden-state model informed by Io phase and
  the burst receiver. A continuum stack is not the most sensitive test.
- **Saturn, Uranus, Neptune:** first perform injected-edge recovery using
  physically plausible flux densities. If recovery probability is low, report
  calibrated upper limits rather than a detection test with little power.
- **Extrasolar wording:** Fornax A and Virgo A are radio galaxies, not stars;
  Cas A and Tau A are remnants. A successful claim should say "extrasolar
  radio source" unless an actual stellar target and flux model are used.

## Detection standard

A source claim should require all of the following:

1. A visible, correctly signed edge localized at zero in the event-by-time
   occultogram.
2. Consistency between disappearance and reappearance and across at least
   three lunar blocks; no single month or event drives the result.
3. A fixed-zero edge that is stronger than identically fitted off-time
   alignments and remains present across predeclared window sizes.
4. Failure of fixed-time, wrong-antenna, neighboring-channel, and matched-sky
   controls to reproduce the effect.
5. Multiplicity-controlled empirical significance plus a block-level effect
   interval; no sample-level Gaussian SNR claim.
6. Blind recovery of Earth and successful injection/recovery at the claimed
   source strength.

## Implementation order

1. Robust fixed-zero change-point fitting plus block permutation/max-statistic
   calibration.
2. Multiscale edge plots and injection/recovery, with Earth as the blinded
   positive control.
3. Low-rank cross-frequency foreground residuals.
4. Joint hierarchical modeling only after the simpler morphology-preserving
   tests behave correctly on positive and negative controls.

## Earth benchmark completed 2026-08-26

`scripts/run_earth_lowrank_covariance.py` now implements items 3 and the
covariance-aware part of item 4 as an Earth-only positive-control benchmark.
Every learned quantity is leave-one-lunar-block-out, and foreground rank is
chosen by nested blank-control prediction rather than Earth strength. The two
antennas are not constrained to agree because their beam/back-lobe responses
are different.

The low-rank residual preserves the established 0.45, 3.93, 6.55, and 9.18
MHz Earth edges while reducing much of the 0.70--2.20 MHz common structure.
The exact empirical null enumerates all `2^9 = 512` choices of the -20 or +20
minute control, giving finite resolution `1/513`. Covariance weighting improves
weak injected-Earth recovery for RV1, while raw fixed-zero fitting is strongest
for RV2; low-rank plus covariance is therefore a sensitivity pipeline, not a
universal replacement. The raw occultogram remains the primary visual product.

## Extrasolar low-rank trial completed 2026-08-26

`scripts/run_extrasolar_lowrank_occultograms.py` applied the foreground stage
to Fornax A and Virgo A with event-level rows, fixed-zero alignment, independent
antenna fits, and lunar-block-held-out controls. Missing tracers are assigned
their training median after standardization so incomplete frequency coverage
does not discard otherwise usable target rows.

For Fornax A, 1,799 events across nine lunar blocks provide a strong visual
test. Subtraction reduces broad correlated gradients but does not reveal a
coherent vertical edge at zero in either antenna. Residual edge summaries are
small and only about half of events have the expected sign. Virgo A has only
34 events across six blocks; it remains sparse, outlier-sensitive, and also
shows no zero-centered edge. Neither source is detected by this experiment.

The next test should apply the covariance-aware fixed-zero fit and matched
off-time block controls to Fornax A first. A scalar fit alone is not sufficient
for a claim: any candidate must also produce localized occultation morphology.

## Fornax A covariance trial completed 2026-08-26

`scripts/run_fornax_covariance_fixedzero.py` applies the joint covariance test
to 1,799 Fornax A events in nine lunar blocks. The spectral directions are
defined from control noise scales, not the target interval. The test evaluates
two directions and four fixed-centered widths in raw and low-rank data with
antennas independent. Its empirical null exactly enumerates all
`4^9 = 262,144` choices of the `-20`, `-10`, `+10`, and `+20` minute controls.

The strongest candidate is RV1 raw with a 240 second ramp. Its family-wise p
within that pipeline is `1/262145`, and the family-wise p after considering
both antennas and both preprocessing branches is `0.000980`. It is positive
in seven of nine blocks and remains unusual when most individual frequencies
are omitted. RV2 is weaker: raw family-wise `p = 0.0147` and low-rank
`p = 0.0447`.

This is not yet a Fornax A detection. The RV1 result disappears after low-rank
foreground removal (`p = 0.435`), the winning morphology is broad rather than
an obvious event-level vertical edge, and off-time controls cannot exclude a
nearby diffuse-sky or lunar/beam feature. The required next control is the
same frozen analysis on matched blank-sky positions around Fornax A with
similar Galactic and lunar-limb geometry, followed by an extended-source
forward model and an independent lunar-block holdout.

## Fornax A matched blank-sky trial completed 2026-08-28

`scripts/run_fornax_blanksky_comparison.py` evaluated the target plus 24 local
sky controls: twelve positions on a 2-degree ring and twelve on a 5-degree
ring. The primary statistic was frozen before examining these positions: RV1
raw data, the control-scaled spectral template, and a 240 second ramp centered
exactly at zero. Each position used its own predicted occultation times and the
same five shifts, nine lunar blocks, channel normalization, held-out covariance
training, and exact temporal-control null. Antennas remained independent.

Fornax ranks first of 25 positions for the primary RV1 raw ramp (`score =
1.6575`, largest blank `1.1273`) and for the frozen RV1 sharp step and RV2 raw
ramp. With 24 controls, the one-sided spatial empirical p-value is limited to
`1/25 = 0.04`. The subtraction branches do not show the same localization:
RV1 low-rank ranks fifth (`p = 0.20`), and RV2 low-rank ranks second (`p =
0.08`).

This result improves the candidate by showing that the raw fixed-zero statistic
is not generic across the immediate sky. It is still not a Fornax A detection.
The target's block-median curves largely lie inside the blank-sky profile
envelope, the event occultograms are dominated by horizontal block structure
rather than an Earth-like vertical edge at zero, and low-rank subtraction
removes most of the spatial distinction. The next confirmatory step should use
a larger predeclared spatial-control set for finer than `0.04` resolution plus
an independent lunar-block holdout; an extended-source/diffuse-sky forward
model should be used to interpret the broad raw feature.
