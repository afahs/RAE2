# Detection consequences of the coordinate and timing issues

**Later update, 27 October 2026:** the Earth-field path identified below has now been replaced with independent ephemerides. The [repair and paired rerun](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/docs/earth_ephemeris_rerun_20261027.md) supersedes the recommendation to make that repair. This report retains the earlier experiment's results and scope; its recovery units are not directly comparable with the later experiment.

27 October 2026. **Some of these issues can hide a weak source or assign it to the wrong direction. The tested midnight correction does not explain the overall absence of a point-source detection, and the February 13 paper discrepancy is not evidence of a mission-wide timing error.** FK4/B1950 remains the working spacecraft frame.

## Measured defects and their reach

The entire native RV archive was decoded again with the corrected date rollover, including all nine power bands. The new geometry is exactly equal to the separately validated corrected geometry from the February investigation. Original caches and their results remain preserved.

| Issue | New check | Detection consequence |
|---|---|---|
| Logical-record midnight rollover | 10,408 of 2,000,832 frequency blocks were assigned one day early: **0.520%**. They contain 41,464 positive fine readings before duplicate removal. | Local windows can lose valid samples or receive the wrong orbit's samples. A small affected fraction does not guarantee a small effect in a selected or heavily weighted subset. |
| Earth field mixed with spacecraft frame | The current `source_vectors_for_rows(kind="earth")` passes the archived Earth direction directly into subtraction from B1950 XYZ. That direction is in coordinates of date. | The median Earth-contact difference from independent ephemerides is **8.0 s**, the 95th percentile **39.5 s**. Earth modeling and Earth-contamination vetoes can be wrong. This does not move fixed-source or solar ephemerides directly. |
| Occasional erroneous archived Earth directions | After correcting the frame, 1,281 of 1,998,026 valid unit vectors still differ by >0.1°, including 1,279 by >1°. They occur on 45 UTC dates, mostly near midnight; the maximum is **15.13°**. The 99th percentile of the full valid sample is 11.35 arcseconds. | Frame rotation alone does not repair every Earth entry. These sparse intervals can leave Earth contamination unrecognized or create false Earth edges. They do not establish a spacecraft-XYZ error. |
| Literal LF `.tplot` XYZ | All **53,671** finite nonzero vectors in `F1.tplot`, interpreted as km, lie inside the Moon; median radius **395.1 km**. | These values cannot be used directly for occultations. The current inside-Moon guard would reject all of them. Reinterpreting their bits as IBM floats gives a plausible 2802.9 km median radius, but is not a validated coordinate repair. The native RV study uses separately decoded XYZ and is unaffected by this particular saved field. |
| February 13 paper markers | The document markers are absent from the production prediction code; Sun predictions use an ephemeris. | A plotting/annotation problem does not itself corrupt the receiver samples or existing predictions. Forcing those markers into a clock correction could introduce a new error. |

With identical short-gap and velocity screening, the old cache lacks matches for **10/2,166 solar contacts**, **21/4,596 Fornax contacts**, 8/2,038 Virgo contacts, 15/2,913 Taurus contacts, 9/1,806 Jupiter contacts and 9/2,063 Earth contacts. All these missing matches are within half an hour of midnight. Matched contacts have exactly unchanged times. These are geometric contacts on the archive's supported intervals, not numbers of independent detections. One anomalously long launch-period logical record has moved samples as late as 05:01 UT; therefore the rollover inventory must not be treated as a universal 18-minute UTC cut.

The Earth anomalies often resemble an Earth vector evaluated one day away from its row timestamp. On a sparse check of the 44 dates with >1° errors, +1 day is the best of the three tested offsets on 41 dates, −1 day on two, and zero on one; some launch-period anomalies remain poorly matched. This is a diagnostic signature, not an established explanation of the original processing error.

At an **illustrative ±2° Earth-limb veto**, the present mixed-field calculation wrongly excludes 3/2,166 solar contacts and misses contamination on six. Corresponding wrong exclusions are three Fornax, four Virgo, four Taurus and three Jupiter contacts. These counts use a fixed example width, not verified settings of every previous survey. Most events remain unaffected. Replacing the Earth direction with an independent ephemeris is preferable to rotating the occasional bad archive entries and trusting them.

## Rerun of the existing point-source recovery experiment

The original event list, 24 spatial controls, three frequencies, antenna assumptions, 20 injection scenarios and even/odd lunar-block split were retained. Only the native geometry and power caches were replaced. All source and control windows were refit. The corrected run has **5,386 fitted windows**, with **2,871** retained after complete-window and calibration requirements. Evaluation uses 37 days for Fornax, 35 for Virgo and 38 for Taurus.

No source newly meets the descriptive localization criterion. Uninjected ranks remain **21/25 for Fornax, 22/25 for Virgo and 12/25 for Taurus**. The nominal first tested injection levels meeting the original rule remain 512, 64 and 96, respectively, in that experiment's synthetic noise/gain units. They are not Jy or calibrated discovery thresholds.

On common windows with the original channel weights frozen, nominal injection response changes by **−0.12% for Fornax, −15.71% for Virgo and 0% for Taurus**. Thus the date defect can materially bias a subset even though it does not uncover a source here. Virgo also loses one evaluation day in the full rerun. Correcting a defect need not increase a biased sensitivity estimate.

The corrected mismatch results are the fraction of injected score response retained relative to a correctly modeled injection:

| Injected mismatch | Fornax A | Virgo A | Taurus A |
|---|---:|---:|---:|
| XYZ actually of date | 97.8% | 94.2% | 93.3% |
| Timing −60 s | 63.0% | 55.1% | 58.3% |
| Timing +60 s | 60.3% | 60.4% | 59.3% |
| Timing scatter σ = 30 s | 85.9% | 81.0% | 86.2% |
| Beam 25% narrower | 67.0% | 80.1% | 79.5% |
| Combined stress | 45.7% | 36.1% | 43.1% |

The combined case uses 30-second timing scatter, a 25% narrower beam, a 10° pointing offset and half the nominal response floor. It requires **2.19×, 2.77× and 2.32×** more source amplitude for the same target score. These are stress assumptions, not measured instrument errors. FK4/B1950 was already the production default; selecting it now is not a new sensitivity improvement.

For constant ±60-second errors, even injections at the maximum tested level of 2048 favor a different sky direction. Direct native-power refits give target ranks **8/7 for Fornax, 4/2 for Virgo and 4/4 for Taurus**, for negative/positive delays. The signal is present, but the detector can fail to identify its source. This establishes a concrete mechanism for missing a weak source or rejecting an otherwise bright candidate on localization grounds.

## Solar direction and broader timing tests

A separate test injects a steady point-like signal at the Sun, Earth, Jupiter, Fornax, Virgo and Taurus directions into corrected upper-V fine power at 1.31, 3.93 and 6.55 MHz. It uses geometry-selected contacts, a local intercept and slope, approximate RAE-2 beam envelopes and ±20-minute windows. Signals enter the two off-time windows as well as the event window. Even lunar blocks set channel weights; odd blocks measure response. The table below is for the **Sun's direction**, with 42 evaluation days in seven lunar blocks and 342 windows.

| Solar test | Response retained |
|---|---:|
| XYZ interpreted as coordinates of date | 98.2% |
| Detector contact shifted by ±60 s | 93.6–94.6% |
| Detector contact shifted by ±120 s | 88.1–88.9% |
| Detector contact shifted by ±240 s | 78.6–78.8% |
| Paper-like narrowing: ingress +240 s, egress −126 s | 83.5% |
| True edge broadened with Gaussian σ = 300 s | 81.6% |

Thus even a paper-sized discrepancy does **not** erase a bright steady solar-direction source in this broad-window detector, although it could push a marginal source below a threshold. Narrow windows and the more flexible diffuse-background fit can be substantially more sensitive to an edge error, as the point-source experiment demonstrates. The two detectors' percentage losses should not be treated as equivalent measurements.

This is a test of geometric source visibility, not a simulation of the solar radio corona or local solar-wind noise. The 25–110 kHz component in the disputed paper need not follow the optical solar limb. Its frequency-dependent disappearance cannot automatically be treated as a receiver clock error or as the template for a compact celestial source. The available LF summary bins span ten minutes, which also prevents a precise reconstruction of second-scale dropout times from that product. A steady source at the Sun's direction is the relevant injection here; no solar-burst search was performed.

![Injected signal retained](C:/Users/Adam/Documents/ChatGPT/RAE2/detection_impact_20261027/detection_impact.png)

## What this changes

1. Use the rebuilt [native cache](C:/Users/Adam/Documents/ChatGPT/RAE2/detection_impact_20261027/cache/manifest.json) for subsequent work. Previous native-cache analyses need reassessment where rollover windows matter. The remote master CSV and all earlier survey products have not been regenerated in this investigation.
2. Retain B1950 spacecraft XYZ and B1950 antenna pointing, but generate Earth positions independently at the actual timestamps and subtract the spacecraft position in the same frame. Rebuild Earth templates and any Earth-dependent vetoes or foreground subtraction before relying on them for weak sources.
3. Search over justified contact uncertainty and source-edge broadening, with identical searches applied to spatial/time controls. Allowing extra delays without recalibrating the controls would increase false positives.
4. Treat beam provenance and diffuse-background subtraction as larger remaining sensitivity risks than the small difference between plausible B1950 conventions. Do not infer an astrophysical upper limit from the existing nondetections.

## Verification and scope

**25 regression tests pass**, including seven checks in this investigation for rotation consistency, invalid radii, orbit discontinuities, mismatched-step signal transfer, missing-event matching and common sample support for the oracle comparison. The production Earth-field path was inspected but not replaced in this investigation; it remains an identified correction to make before rebuilding dependent analyses.

The corrected experiment was checked with **2,408 injection refits** and **5,760 additional high-amplitude native-window refits**. The broad-window test adds **1,605 native injection refits**; its largest difference from the analytic fixed-weight prediction is below 8×10⁻¹³. Each oracle and detector uses the same retained samples. All nominal template efficiencies equal one. A first short-window coverage pilot is retained separately and is not used for the conclusions; the final test permits ≤180-second geometry intervals with radius/velocity checks and uses longer windows. It never bridges longer gaps. Its broader interpolation is a numerical approximation, not new independent tracking information.

The 600-second body ephemeris interpolation was checked against direct evaluation at 256 archive epochs per body; maximum direction errors are <0.0006 arcsecond. It checks source ephemerides, not the archived spacecraft orbit. Contact matching uses same edge type within ten minutes; missing matches stay missing. Receiver samples and visits are correlated, the beam is approximate, and these experiments are sensitivity diagnostics rather than calibrated completeness measurements.

Scripts, numeric outputs and manifests are in [the analysis directory](C:/Users/Adam/Documents/ChatGPT/RAE2/detection_impact_20261027). The existing [historical contact checks](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/docs/historical_occultation_validation_20261027.md), [FK4 assessment](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/docs/fk4_reassessment_20261027.md) and [LF paper analysis](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/docs/earth_timing_and_mismatch_recovery_20261026.md) supply the documentary context. No absolute tracking orbit has been recovered, and no new astronomical source detection is claimed.
