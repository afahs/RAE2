# Earth timing checks, source recovery under model errors, and the solar plot

**Updated 27 October 2026 after full-text verification:** The intended plot
is now positively identified as Weber, Fainberg and Stone (1976), Figure 2,
showing 13 February 1974. Section 3 replaces the earlier incorrect figure
identification and the subsequent abstract-only interpretation with the
authors' actual discussion and an exploratory reproduction from burst-receiver
summaries. Sections 1–2 and their numerical results are unchanged. Previous
report versions and manifests are preserved in `solar_lf_correction_20261027/`.

26 October 2026. Native RV data confirm terrestrial transitions around the
predicted lunar occultations, particularly at 6.55 and 9.18 MHz. They do not
yet select a unique spacecraft coordinate frame or justify a universal clock
correction. Mismatched injections show that timing errors can mislocalize a
source even when it is extremely bright. Antenna-response errors can reduce
the injected-source response by a factor of about two in the combined stress
test. No new Fornax A, Virgo A or Taurus A detection emerged.

All new code, numerical results, plots and verification logs are in
[`earth_occultation_recovery_20261026`](../../earth_occultation_recovery_20261026/).
Earlier numerical results and production defaults were not overwritten.

## 1. Higher-frequency Earth occultation checks

### Geometry and sampling

The calculation uses the independently decoded native RV cache, not old
stacked profiles or the master CSV. Bands 6, 8 and 9 correspond to 3.93, 6.55
and 9.18 MHz. Coarse and fine samples from both antennas were inspected; the
lower antenna was used only after 7 November 1974, the deployment cutoff
adopted in the earlier experiment.

An independent Earth-Moon ephemeris from Astropy/ERFA supplies the
Moon-to-Earth position. The spacecraft position is subtracted **before**
constructing the Earth direction. This avoids treating a Moon-centered unit
vector as the observer's line of sight. Spacecraft XYZ is tested as B1950 and
as true equator/equinox of date. A uniform Earth disk is occulted by a locally
straight lunar limb. The actual terrestrial brightness distribution is not
uniform, so this is a geometric timing diagnostic, not a complete radio model.

Geometry alone selected one ingress and one egress per day where available:
512 candidate events. A local interpolation quality requirement retained
360 on-time events on 230 days. The final table contains 19,260 individual
frame/receiver/frequency/window fits, including controls 30 minutes before
and after each event. These are repeated measurements, not independent trials.

Within each frequency dwell, positive native powers were summarized by their
median log power. The typical interval between frequency visits is **138.24
seconds**, whereas first-to-last Earth-disk contact is typically **90 seconds**.
Fine samples within a dwell are separated by 7.68 seconds, but they do not
fill the interval before the receiver returns to that frequency. This limits
the timing information in individual events.

An initial implementation required valid geometry at every trial delay and
unnecessarily discarded nearby samples. Its outputs are retained in
`strict_geometry_mask/`. The final calculation uses a degree-six local
position interpolation over 40 minutes, with no power values involved. It
requires at least 40 positions, brackets the evaluated times, rejects gaps
longer than 180 seconds, and rejects a fit if held-out native positions
disagree by more than 0.5 km. The accepted fits had a maximum held-out error
of **0.237 km**, median **0.177 km**, and largest bridged gap **168.96 seconds**.
This validates interpolation consistency, **not absolute orbit accuracy**.

### Results and limits

At 6.55 and 9.18 MHz, on-time finite-disk edge fits outperform paired off-time
controls for both antennas and both sampling products. The monthly-block
bootstrap intervals for these descriptive mean contrasts remain positive.
At 3.93 MHz, the lower-V contrast is clearer; upper-V contrast intervals
include zero. Coarse/fine measurements and nearby days are correlated.

![Selected native Earth examples](../../earth_occultation_recovery_20261026/earth_occultation_examples.png)

The examples are selected by strong upper-V edge-fit improvement and are not
typical events. Some are short bright peaks following emergence rather than
stable plateaus, illustrating why a simple edge fit cannot alone calibrate
the clock. The shaded interval represents approximate Earth-disk contact;
the vertical line is Earth-center contact under B1950.

The two frame hypotheses predict Earth-center contact times with a median
difference of **5.9 seconds**, with 10th-90th percentiles of **-10.2 to +27.9
seconds**. Grazing events have larger differences. At zero clock offset,
B1950 has a small average squared-residual advantage. However, fitting one
global delay on even lunar blocks gives **-16 seconds for B1950** and **-22
seconds for coordinates of date**. On the 11 odd evaluation blocks, their
mean normalized fit improvements are **6.865 and 6.964**, respectively. The
block-bootstrap interval for their difference includes zero.

**Interpretation:** Earth occultations support the gross geometry, but this
test does not resolve frame versus clock versus source-position effects.
There is no justified universal time correction to apply to celestial
sources. The trial-lag grid is two seconds; that is a numerical grid spacing,
not two-second measurement accuracy. The stored per-event profile ranges
are heuristic fit-width diagnostics, not formal confidence intervals.

Relevant outputs: `earth_events.csv`, `earth_edge_fits.csv`,
`earth_local_orbit_validation.csv`, `earth_on_off_contrasts.csv`,
`earth_global_clock_validation.csv`, and `earth_validation_summary.json`.

## 2. Timing and antenna-response errors in point-source recovery

### Fixed experiment

Targets are Fornax A, Virgo A, and Taurus A, using the existing ICRS catalog
directions. Each target is compared with 24 nearby directions on 3- and
7-degree rings. Source injections use native fine samples at 1.31, 3.93 and
6.55 MHz and enter all three physical windows, including off-time controls.
The nominal detector jointly fits the source, a local intercept and slope,
and two smooth beam-weighted Galactic foreground components. Windows extend
15 minutes either side of their centers, longer than in the earlier study.

Even lunar blocks set the channel noise scales and injection units. Those
weights are frozen; an injected signal cannot change its own weights.
Odd blocks supply the reported recovery results:

| Source | Evaluation days | Evaluation lunar blocks | Evaluation windows |
|---|---:|---:|---:|
| Fornax A | 37 | 4 | 642 |
| Virgo A | 36 | 5 | 318 |
| Taurus A | 38 | 8 | 489 |

These archival data were explored previously. This is a separate calibration/
evaluation split within this experiment, not untouched external validation.
The noise weights and source spectrum are defined in channel-dependent
noise/gain units, not Jy. Model scores are not Gaussian significance values;
real background systematics produce very large control scores.

Twenty truth scenarios test constant delays from -60 to +60 seconds,
event-to-event Gaussian timing scatter, beam widths scaled by 0.75 or 1.25,
5- and 10-degree pointing offsets, a 20-degree lower-V offset, halved/doubled
response floors, coordinates-of-date spacecraft positions, and 30/60-second
Gaussian edge broadening. The broadening cases are generic source/edge
stress tests, not fitted diffraction models. Random perturbations use one
fixed reproducible realization, not a completeness Monte Carlo.

### How much response is lost?

The first quantity is the change in the fixed detector's score per unit
injected source amplitude, relative to the nominal injection. It includes
both changed antenna gain and mismatch in the signal shape. A separate
matched-template efficiency measures shape loss relative to an oracle that
knows the injected truth. The oracle is a diagnostic, not a detection method.

| Injected error | Fornax A response retained | Virgo A | Taurus A |
|---|---:|---:|---:|
| Timing scatter, sigma = 10 seconds | 95.1% | 95.7% | 94.8% |
| Timing scatter, sigma = 30 seconds | 85.9% | 86.8% | 86.2% |
| Beam 25% narrower | 67.0% | 78.5% | 80.0% |
| Pointing error of 10 degrees | 95.6% | 95.9% | 99.1% |
| Response floor halved | 87.7% | 75.3% | 71.4% |
| Combined errors | 45.8% | 43.2% | 43.1% |
| Spacecraft XYZ actually of date | 97.8% | 96.4% | 93.3% |

The combined case uses 30-second timing scatter, a 25% narrower beam,
10-degree pointing offsets and half the nominal response floor. Reaching
the **same target score** then requires about **2.18, 2.31 and 2.32 times** the
injected amplitude. These are not discovery thresholds or astrophysical
sensitivity limits. Its matched-template efficiencies are 78.3%, 82.3% and
81.4%; thus much of the total response loss is changed gain, not shape alone.

Constant +/-10-second timing errors reduce injected score response by about
3-8%. At +/-60 seconds, shape efficiency falls to approximately **57-70%**.
Virgo's +30-second scenario slightly increases raw response while reducing
shape efficiency to 68.5%: shifting a sampled edge can change the available
signal energy. A larger raw response does not automatically mean a better
match or a more accurate source position.

![Mismatch response comparison](../../earth_occultation_recovery_20261026/point_source_mismatch_recovery.png)

### Timing errors can cause wrong localization

Constant +/-60-second delays make at least one nearby control direction
outperform the true source direction in the injected-signal-only comparison
for **all three targets**. Fornax also does this at -30 seconds. Increasing
brightness does not correct a wrong timing model.

Full native-power refits at the maximum tested delayed-source amplitude
confirmed target ranks of **8/7 for Fornax**, **4/2 for Virgo**, and **4/4 for
Taurus** for -60/+60 seconds, respectively, out of 25 directions. These
injections were very bright; failure here means incorrect localization,
not absence of injected signal.

The descriptive recovery rule requires the target to rank first and exceed
the baseline spatial-control 95th percentile. With only 24 controls, this is
not a trials-corrected discovery test. First tested grid levels meeting that
rule were:

| Scenario | Fornax A | Virgo A | Taurus A |
|---|---:|---:|---:|
| Nominal | 512 | 64 | 96 |
| 30-second timing scatter | 1024 | 96 | 128 |
| Beam 25% narrower | 768 | 96 | 128 |
| Combined errors | 2048 | 192 | 256 |
| Constant +/-60-second error | None through 2048 | None through 2048 | None through 2048 |

The absolute numbers use this experiment's synthetic channel units, weights,
windows, evaluation subset and recovery rule. They must not be compared
numerically with the earlier 12/16-unit results. The nominal Fornax result
only localizes at a high injected level under this stricter fixed detector;
background mismatch remains substantial. Without injection the target ranks
are **21, 22 and 12 out of 25**, respectively. There is no new point-source
detection or calibrated upper limit.

Relevant outputs: `recovery_mismatch_results.csv`, `recovery_by_channel.csv`,
`recovery_curves.csv`, `recovery_stability.csv`, and
`full_native_scene_replays.csv`.

## 3. Original Figure 2, the authors' explanation, and a data test

### Exact identification

The user's image is **Figure 2 on printed page 298 / PDF page 2** of
Richard R. Weber, Joseph Fainberg and Robert G. Stone (1976), *Low frequency
radio observations of the solar wind near the Moon*, Geophysical Research
Letters 3(6), 297–300, [DOI 10.1029/GL003i006p00297](https://doi.org/10.1029/GL003i006p00297).
The date printed above the original figure, omitted in the user's crop, is
**13 February 1974**. The panels, points, markers and caption match visually.

The complete paper was found in the user's existing local
`C:/Users/Adam/Documents/PhD/Diffraction/Papers/` directory and copied unchanged
to [`weber1976.pdf`](../../solar_lf_correction_20261027/weber1976.pdf).
It was not in the earlier 15-PDF Dropbox inventory. Public NASA metadata had
provided only the abstract; local full-text inspection resolved the attribution.
All four pages were extracted, and the relevant pages 298–300 were inspected
as rendered images. The previously displayed Alexander 1975 Figure 6 is a
different event and frequency range.

![Original Figure 2, including its date](../../solar_lf_correction_20261027/weber1976_figure2.png)

### What the paper actually says

On page 298 the authors explicitly state that the longer high-frequency
quiet interval is **not clearly understood**. Their suggested explanation
is that the weaker high-frequency noise becomes undetectable earlier. They
also report that the sporadic peaks fall with increasing frequency more
steeply than the persistent f^-3 continuum. They do not derive a model that
predicts the eight channel contact times.

The authors distinguish this frequency ordering from another observation:
the 25 kHz noise interval can be displaced from the optical shadow by about
±10 minutes. They argue that this disfavors photoelectrons produced at the
spacecraft as the noise source. Their broader local solar-wind interpretation
uses a leading/trailing difference of roughly 8 degrees in averaged profiles,
consistent with a roughly 4-degree solar-wind aberration, and the near absence
of the component around full Moon. These are evidence for a proposed origin,
not a demonstrated emission mechanism. On page 300 they say the relevant
electron conditions remain undetermined and report negative initial
correlations with magnetic-field direction and magnitude. They also question
a simple local plasma/upper-hybrid frequency explanation for noise reaching
110 kHz. The paper therefore does not support turning each channel dropout
directly into an electron-density measurement.

### Test performed on the exact event

The existing local `BurstRecieverData/F1.tplot` contains all eight relevant
channels on 13 February 1974, for both receivers. It is a **10-minute summary
product**, not the approximately 8-second samples shown in the paper. The
independent RV archive starts at 450 kHz and cannot replace the LF data.

The summary format was verified visually in `RAE-B_Doc.PDF`, PDF page 23.
Time tags denote interval starts. MIN and MAX are the lowest and highest
1-dB histogram bins containing at least 2% of the samples; MAX is neither an
absolute peak nor a cumulative 98th percentile. MODE is the most occupied
bin. A zero can indicate an unavailable statistic. The saved tplot product
does not contain the per-channel sample counts that exist in the original
tape format. T_SUM and T_SUMSQ are encoded logarithmic sums, not directly
usable arithmetic means or variances.

`test_event.py` reads the original IDL saves directly, validates common time
and frequency axes, excludes all duplicated epochs, and masks zero values
and the documented failed BR-1 channels at 55 and 210 kHz. Receivers are
kept separate; the lower antenna was not yet fully deployed on this date.
The analysis uses the median MODE in the four historical quiet bins
08:20–09:00 UTC as each channel's background, then examines the excess of
the 2% MAX statistic over that background in dB. This is an exploratory test
on a historically selected event, not an independently selected confirmation.

![Comparison from existing summaries](../../solar_lf_correction_20261027/figure2_comparison.png)

For the contiguous quiet run containing 08:35 UTC, both receivers give:

| Quiet-bin criterion: MAX above background | 25 kHz quiet bins | 110 kHz quiet bins | Difference |
|---|---:|---:|---:|
| At most 3 dB | 40 min | 60 min | 20 min |
| At most 6 dB | 40 min | 70 min | 30 min |
| At most 10 dB | 40 min | 90 min | 50 min |

These are durations of runs of summary bins, **not precise physical
occultation durations**. They show that the ordering survives in the
archived summaries, and that the apparent separation is sensitive to the
detectability criterion, as the authors' suggestion would allow. They do
not establish that sensitivity alone explains the curves. The increasing
quiet duration with increasing cutoff is partly built into the definition.
Bright tails may be saturated, and the 2% rule discards rare peaks; a fit
that ignored these effects could manufacture a spectral explanation.

The full eight-channel comparison is in `figure2_summary_reproduction.png`.
Numeric results are in `figure2_event_summary.csv`,
`figure2_threshold_sweep.csv`, and `figure2_width_comparison.csv`.
Five regression checks pass, covering interval support, gaps, missing power,
window truncation, duplicate epochs and the failed receiver channels.

### Geometry issues exposed by this check

The saved `RAE_pos_GEI` values have a median vector magnitude near 395 km,
inconsistent with the documented lunar orbit. Reinterpreting those float
bit patterns as IBM hexadecimal floats yields a plausible median radius
near 2803 km. This strongly suggests a conversion error, but that operation
alone does not validate the positions or their epochs. None of those saved
coordinates was used in this test's contact predictions.

The independently decoded RV geometry predicts solar-center limb contacts
at approximately **08:17:00 and 09:04:06 UTC** under the B1950 hypothesis.
The TOD hypothesis changes these by only 5–7 seconds. The paper's markers
read approximately **08:21 and 09:02**. This several-minute discrepancy is
unresolved. The two contacts do not differ by a common offset, so imposing
a universal clock shift is not justified. Finite solar-disk contacts,
original orbit/ephemeris provenance and time conventions require a dedicated
check before using this figure as an absolute timing calibrator. The plots
show both the paper interval and the current RV prediction explicitly.
The quiet-width table does not depend on adopting either contact prediction.

### What can be tested next, and what data it requires

1. **Test the weaker-signal explanation directly.** Compare a shared temporal
   excess-power envelope with channel-dependent amplitudes against a model
   that also allows channel-dependent edge times. Use actual sampling times,
   bandpasses, background distributions, saturation censoring and calibration
   gaps. Check whether fractional-amplitude edges align after correcting
   detectability, and whether that predicts held-out events. The raw
   approximately 7.68-second measurements, or recoverable high-resolution
   plots, are needed for a decisive test. Ten-minute extrema cannot recover
   within-bin ordering or burst occurrence rates.
2. **Recheck the lunar-wake evidence separately.** Repeat the leading/trailing
   new-Moon profiles and the lunar-phase comparison across all eligible
   events, controlling coverage, terrestrial kilometric radiation and
   receiver changes. The 10-minute archive can support a coarse population
   test, but predictions must be integrated over bins: ten minutes spans
   about 16 degrees of orbital phase for a 222-minute orbit. First resolve
   coordinate decoding and time alignment; do not use the unvalidated saved
   GSE coordinates as a shortcut.
3. **Separate a local environment effect from instrumental effects.** Keep
   receivers separate, exclude the failed channels, and model known gain
   calibration interruptions and intermodulation from intense 200–300 kHz
   terrestrial emission. Solar-wind density, speed and temperature can supply
   independent covariates with appropriate spatial/time alignment, but the
   radio summaries alone cannot identify a unique plasma-wave mechanism.

The [historical NASA catalogue](https://ntrs.nasa.gov/api/citations/19900020097/downloads/19900020097.pdf)
lists higher-resolution hourly and daily
burst-receiver plots as NSSDC datasets **73-039A-02C and 73-039A-02D**;
the 10-minute magnetic-tape summaries are **73-039A-02B**. The catalogue
does not establish that the native samples or the relevant microfilm scans
are currently downloadable. No external request for data was sent.

This work adds a documentary identification and a limited LF data test.
It does not claim a new point-source detection, alter production geometry,
or change the Earth and Fornax/Virgo/Taurus numerical results above.

## Validation and implications

All **29 tests passed**: 11 new geometry/mismatch regressions and the 18
earlier investigation tests. There were **2,408 sampled native injection
refits**, with maximum relative projected-response discrepancy 1.17e-13.
An additional **5,796 native window refits** reconstructed 12 complete,
high-amplitude evaluation scenes; all spatial ranks matched and maximum
relative score discrepancy was 1.52e-15. These establish numerical replay
accuracy, not the correctness of the physical beam or statistical null.

The next useful detection change is a bounded timing/edge-width template
family, with the additional search included in control calibration and
localization assessed after allowing timing uncertainty. Response-floor and
beam-shape uncertainty also need explicit treatment; template injections
that reuse the same assumed beam cannot settle them. The Earth check does
not justify changing a production frame or clock convention yet.

`EXPERIMENT.md` records the initial design and changes. `run_reproduce.ps1`
recreates the calculations using the existing native cache. The provenance
manifest hashes the actual inputs, code, figures and report.
