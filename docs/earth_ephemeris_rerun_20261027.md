# Independent Earth ephemerides: repair and native-data rerun

27 October 2026. **The remaining Earth-coordinate mismatch is repaired in the local pipeline. The rerun improves the solar-direction recovery diagnostic when Earth emission is fitted, but does not reveal a convincing Sun, Fornax A, Virgo A, Taurus A or Jupiter detection.**

The first recovered nominal solar injection changes from 24 to 12 synthetic units. The other four sources retain the same first recovered grid points. This is a selected-data injection result, not a factor-of-two improvement in calibrated survey sensitivity.

## What was repaired

Spacecraft XYZ and antenna pointing retain the working FK4/B1950 interpretation. The archived Moon-to-Earth unit direction is no longer used to generate Earth predictions, beam directions, exclusions or the migrated reflection geometry. It normally describes axes of date, whereas spacecraft XYZ uses B1950 axes; some archived Earth entries are independently erroneous. Merely rotating that field would retain those errors.

The corrected path computes independent barycentric Earth and Moon positions at the actual UTC epochs, subtracts them to obtain a Moon-centered position, transforms it into the spacecraft's axes, and then subtracts the spacecraft position. Normalization happens last. This preserves the changing Earth–Moon range and spacecraft parallax. Sun, Jupiter and their moving off-source controls now use the same observer-centered construction.

Archived Earth columns remain available for diagnosis. Missing or nonfinite Earth fields no longer invalidate otherwise usable power readings. New geometry, quality-policy and stack versions prevent the main CLI from silently combining old event predictions or Earth-related rejection flags with the repaired implementation. Old cleaned products must be ingested again; old predicted events and stacks must be rebuilt. The focused-control runner also checks the event version before cache reuse.

The relevant implementation is in `rylevonberg/frames.py`, `events.py`, `ingest.py`, `pipeline.py`, `geometry.py` and `stacking.py`. Earth-beam/profile, Earth-surface and lunar-reflection observation callers have also been migrated. Mathematical helpers that accept explicitly supplied common-frame vectors remain available; they are not an endorsement of directly passing the archived Earth columns.

## Independent accuracy check

The corrected API was compared with the saved JPL Horizons geometric Earth-minus-Moon vectors at 64 epochs across the archive. The median angular difference is **1.570 arcseconds**, the 95th percentile **5.227 arcseconds**, and the maximum **7.887 arcseconds**. The maximum range difference is **7.375 km**. Horizons is independent of the archived Earth direction; the production calculation uses Astropy's built-in ephemeris rather than fitting to those archive columns or to the February 13 paper markers.

The 600-second interpolation used by the large rerun was checked at 1,000 epochs per body against direct evaluation. Maximum direction errors are 0.000587 arcsecond for Earth, 0.000178 for the Sun and 0.000112 for Jupiter. The existing B1950 axis rotation used in the independent experiment differs from the full production FK4 transformation by at most 0.363 arcsecond on the Earth anchors. These checks validate the ephemeris implementation and its numerical approximation; they do not independently certify every archived spacecraft position.

See [validation values](C:/Users/Adam/Documents/ChatGPT/RAE2/earth_ephemeris_rerun_20261027/independent_ephemeris_validation.json) and [epoch comparisons](C:/Users/Adam/Documents/ChatGPT/RAE2/earth_ephemeris_rerun_20261027/independent_jpl_check.csv).

## Paired search design

The input is the previously rebuilt native RV cache with corrected midnight rollover, not the older cache or an interpolated remote master CSV. The source list contains Fornax A, Virgo A, Taurus A, the Sun and Jupiter. The 882 geometry-selected candidate events were frozen before this comparison, with at most one event per UTC day. Earth exclusions were recomputed for both the legacy calculation and independent ephemerides.

The physical search covers **1.31, 3.93 and 6.55 MHz**, upper- and lower-V fine receivers, with lower-V use restricted to the established post-7-November-1974 interval. Each event has a ±20-minute window and two time controls centered 40 minutes earlier/later. The target is compared with 24 directions on 3° and 7° rings. All physical signals, including synthetic injections, enter all three windows. Geometry interpolation permits at most 180-second intervals and retains radius and velocity checks.

Four fits isolate the change: legacy versus independent Earth exclusions, each with or without an explicit Earth-emission nuisance template. Every fit includes a jointly fitted intercept, slope and two diffuse Galactic templates. The Earth nuisance is an approximate visible uniform disk multiplied by the receiver beam response; it is not a complete model of terrestrial radio emission.

Even lunar blocks supply a common uninjected channel calibration and fixed weights. Odd blocks evaluate the search. Calibration units and source windows are shared between the four models. Recovery means the injected target ranks first among 25 directions and exceeds the uninjected spatial-control 95th percentile. The units are training noise divided by approximate beam gain, **not Jy**. Thresholds are first passing points on a discrete grid, not calibrated completeness limits. The data were explored previously, so these are not external validation samples.

The physical run produces **32,398 window/model fits**, of which **20,796** satisfy complete-window and calibration requirements. Corrected evaluation coverage is 43 days for Fornax, 32 for Virgo, 53 for Taurus, 52 for the Sun and 42 for Jupiter. The corrected veto adds one supported evaluation day for Taurus. The extra selected solar event does not survive all receiver/calibration requirements.

## Measured change

This table compares the two fits that include Earth as a nuisance template. Rank 1 would favor the source direction; rank 13/25 does not.

| Source | Uninjected rank, legacy → corrected | First nominal recovered injection, legacy → corrected | Nominal score response change |
|---|---:|---:|---:|
| Sun direction | 14 → 13 | **24 → 12** | −0.043% |
| Fornax A | 15 → 15 | 64 → 64 | −0.157% |
| Virgo A | 13 → 13 | 96 → 96 | +0.071% |
| Taurus A | 21 → 21 | 768 → 768 | +0.589% |
| Jupiter | 15 → 15 | 256 → 256 | −0.023% |

The solar change comes from the fitted foreground/background residual, not an increase in the template's response to source power: that response is essentially unchanged. With Earth used only as a veto, its first recovered grid point stays at 12 units. Thus the repair fixes a bias in one Earth-background model; it does not demonstrate a universal doubling of solar sensitivity.

On identical observation support, Taurus's nominal response changes by only **+0.013%**; most of its +0.589% full-run change comes from the additional day. Other sources already have identical supported windows. Every source fails the localization rule without injection. The block-sign null, allowing all four model choices and five sources, also gives no significant result. Scores in the saved tables are weighted template statistics, not Gaussian standard deviations.

![Recovery and mismatch results](C:/Users/Adam/Documents/ChatGPT/RAE2/earth_ephemeris_rerun_20261027/earth_correction_recovery.png)

## Nine-band check and residual sensitivity risks

A complementary search covers all nine RV bands, **0.45–9.18 MHz**, with upper/lower coarse and fine measurements where coverage permits. It fits signed steps and compares them with both time controls without assuming an audited beam at every frequency. There are 49,792 fitted windows and 126 supported source/band/receiver combinations per Earth version. Some individual uncorrected bootstrap intervals are positive, but the smallest global block-sign p-value across all 252 channel/version choices is **0.3555**. No channel passes this multiple-comparison diagnostic. The sign-symmetry assumption and missing spatial localization still limit this screen.

The veto change leaves supported channels unchanged except for Taurus, whose additional day changes its received-power injection transfer by at most 0.759%. This broad screen measures received-power step recovery, not intrinsic source flux sensitivity or detailed Earth-foreground subtraction.

Timing and beam errors remain substantially more damaging than the repaired Earth mismatch. With corrected Earth-background fitting, a constant ±60-second source timing error retains approximately 61–68% of the nominal target response. A Gaussian edge width of σ = 120 seconds retains 48–50%; the combined stress retains 37–45%. The combined case uses 30-second event timing scatter, a 25% narrower beam, a 10° pointing error and half the assumed beam floor. These are stress assumptions, not measured instrument errors.

At the largest tested injection of 2048 units, a ±60-second timing error gives Fornax target rank **7/25 in both directions**; the signal is strong but assigned elsewhere. Similar localization failures occur for most other delay/source combinations. A few stress curves are nonmonotonic, so a first passing stress grid point must not be treated as a reliable detection limit. A bounded timing and edge-width search, with the same extra trials applied to controls, remains the next useful detection-method improvement.

The solar injection is a steady point-like source at the Sun's direction. It is not a burst search or a model of the low-frequency solar corona or local plasma emission. This work does not repair the problematic literal LF `.tplot` XYZ fields or resolve the February 13 plot annotation by imposing a clock shift.

## Verification, provenance and scope

**202 regression tests pass**: the 198-test package suite plus four experiment regressions. Coverage includes missing/corrupted archive Earth fields, real range and parallax, frame round trips, timestamp storage units/time zones/duplicates, Earth exclusions, stale event/quality/stack caches, ordering overlapping receiver dwells, frozen injection masks, foreground-induced bias and source power in time controls. Three existing pandas timedelta deprecation warnings remain.

The paired physical extraction includes 3,060 native injection refits (maximum relative discrepancy 2.87×10⁻¹³). The nine-band screen adds 444 refits (maximum 9.42×10⁻¹⁴). Full native scene replay adds **87,960 window refits across 184 scenes**, covering all five sources, four models and five injected truths. Every reconstructed spatial rank and recovery decision agrees with the analytic curves. The maximum aggregate score discrepancy is 3.28×10⁻¹¹. Details are in [the replay verification](C:/Users/Adam/Documents/ChatGPT/RAE2/earth_ephemeris_rerun_20261027/native_scene_replays.json).

Replay checks exposed two experiment-harness problems that were corrected before the final results: 19–36 local sample-order reversals per band required sorting before binary window lookup, and zero/nonpositive receiver readings had to remain invalid after injection. The production ingest already sorts times. All physical features, window fits and nine-band results were rebuilt after these fixes. No raw receiver values were rewritten.

Code and results are in [the rerun directory](C:/Users/Adam/Documents/ChatGPT/RAE2/earth_ephemeris_rerun_20261027). Its README and reproduction script specify the dependency order; its manifest hashes the actual inputs, code and outputs. The original Earth columns and historical products remain preserved. This is a repair of the **local pipeline and local native-data analyses**; the remote master CSV, deployed HPC code and every earlier survey product were not regenerated or redeployed.
