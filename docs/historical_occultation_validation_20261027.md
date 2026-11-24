# Historical Sun, Earth, and Jupiter checks against FK4/B1950

27 October 2026. This check uses the coordinate archive rebuilt after the midnight rollover correction. It remeasures historical figure markers and adds a newly retrieved catalog of terrestrial occultations. No time shift, coordinate rotation, or event-specific correction is fitted to make the documents agree.

**The dated Earth and Jupiter figures agree with FK4/B1950 within their reading precision. A further 30 terrestrial events provide broader timing consistency. The January solar figure is only a coarse check: three of its eleven available contacts differ by more than the predefined three-minute reading allowance.** These results strengthen the case for treating the February 13 solar markers as unreliable, but do not prove that the original plot, rather than a systematic trajectory error, is responsible.

## Direct geometric-marker comparisons

Every time in this table refers to the event on the date shown. The comparison reads geometric markers from the figure; it does not force frequency-dependent radio dropouts to represent the center of a celestial body.

| Object and event date | Figure markers, approximately UTC | FK4/B1950 prediction, UTC | Prediction minus marker |
|---|---|---|---|
| Earth, 12 December 1973 | 14:58:33–15:32:59 | 14:58:46–15:32:34 | +12 s / -25 s |
| Jupiter, 3 October 1973, upper panel | 04:43:08–05:05:32 | 04:43:21–05:05:31 | +13 s / -2 s |
| Jupiter, same event, lower panels | 04:43:05–05:06:10 | same prediction | +16 s / -39 s |
| Sun, 13 February 1974, disputed | 08:21–09:02 | 08:17:00–09:04:06 | -240 s / +126 s |

The reading allowance is ±60 seconds per marker for these figures. Seconds in the marker column describe the pixel-to-time conversion, not a claimed second-level documentary accuracy. The Jupiter panels are the **same event**, not two independent validations. Their emersion markers differ by about 38 seconds when each panel's own clock ticks are used, demonstrating the limitation of treating printed lines as precise timing records.

The Earth source is Alexander et al. (1975), Figure 5, PDF page 6 of the Dropbox copy. The Jupiter source is Kaiser (1977), Figure 3, PDF page 4 of the local reference copy. The new pixel measurements replace the earlier rough readings of 14:58/15:33 and 04:43:15/05:06:15. The numerical orbit has not been adjusted toward these revised readings. Each source image, tick coordinate, marker coordinate, and reading allowance is retained in `figure_anchors.json`; `axis_measurement_review.png` makes the axis readings inspectable.

Finite source size matters before interpreting small differences. For the Earth event, the visible disk takes approximately two minutes to cross the limb: any part is hidden from 14:57:48 to 15:33:31, while the full disk is hidden from 14:59:45 to 15:31:35. Thus tens of seconds between a radio edge, a printed marker, and the Earth-center crossing need not represent a spacecraft timing error. Jupiter's optical radius changes the corresponding contacts by only about half a second. All direct comparison rows above use source centers consistently.

## The additional solar example: 16 January 1974

Weber, Fainberg and Stone (1976), Figure 1 on PDF page 1, marks six full solar occultation intervals and the end of a seventh clipped by the start of the day. The dynamic spectrum is constructed from ten-minute averages. Its 24-hour time axis is compressed to approximately 915 pixels in the inspected rendering; one pixel represents about 1.6 minutes. We fixed a ±180-second reading allowance before the new numerical comparison and retained it after seeing the results.

| Plotted solar interval, approximately UTC | Native FK4/B1950 interval | Ingress difference | Egress difference |
|---|---|---:|---:|
| Clipped start, ends 00:19:39 | Ends 00:20:05 | — | +26 s |
| 03:29:05–04:06:02 | 03:26:14–04:02:08 | -171 s | **-234 s** |
| 07:06:14–07:43:38 | 07:08:07–07:44:11 | +113 s | +33 s |
| 10:49:52–11:26:30 | 10:50:00–11:26:14 | +8 s | -16 s |
| 14:31:50–15:08:37 | No bracketed native contacts | — | — |
| 18:08:48–18:50:24 | 18:13:47–18:50:20 | **+299 s** | -4 s |
| 21:56:00–22:36:00 | 21:55:40–22:32:23 | -20 s | **-217 s** |

Eight of eleven available edges satisfy the three-minute allowance. The other three differ by 3.90, 4.98, and 3.61 minutes. The interval near 14:30 falls in the documented noon-to-16:00 gap and is not counted as a success or a failure. No interpolation across that long gap is used.

This reproduces the sequence and approximate phase of the January occultations, but **is not a clean minute-precision solar validation**. The very compressed figure, ten-minute intensity averaging, and potentially approximate bar placement limit its diagnostic value. Averaging alone does not prove that the geometric bars were rounded, and we do not enlarge the tolerance retrospectively to declare all edges successful. The January result also means the February figure is not the only solar illustration with several-minute annotation differences.

## Additional terrestrial events from another primary paper

A search following the historical references located NASA report X-695-76-139, accession **19760020664**, by Alexander and Kaiser, *Terrestrial Kilometric Radiation: 2—Emission from the Magnetospheric Cusp and Dayside Magnetosheath*. Its Table 1, PDF page 18, lists 37 dated radio-occultation midtimes. All rows were transcribed from the rendered table, including events with missing archive coverage. The source was retrieved from the [NASA archive](https://ntrs.nasa.gov/api/citations/19760020664/downloads/19760020664.pdf).

These are event midtimes for spatially extended magnetospheric radio sources. They are not independent optical measurements of Earth-center ingress or egress. Comparing them with the midpoint of our geometrical Earth occultation is a **broader consistency check**, not a clock calibration or a strict accuracy test of the frame.

The results are:

- 30 events have both native geometric contacts available.
- Their median absolute midpoint difference is **47 seconds**; the maximum is **2 minutes 53 seconds**.
- 18 of the 30 are within one minute; all 30 are within three minutes.
- Five of those pairs include a 92-second calibration gap at one edge, explicitly recorded in the output.
- Five further events have incomplete contact coverage; two on 30 April 1974 lie in the archive's long gap. These seven are retained as unavailable, not counted as matches.

The complete comparison is `cusp_midpoint_comparison.csv`. Differences for these extended radio sources can include physical source displacement and evolution as well as documentary rounding; we have not interpreted them as measured tracking errors. The paper's three detailed examples contain projected lunar limbs and radio-source disappearance bands. Those bands are likewise not substituted for Earth-center contact markers.

## Other documentary leads and exclusions

The search used text/OCR from the cached 15-PDF Dropbox collection, followed by visual inspection of the relevant complete pages, the existing Weber and Kaiser papers, and the two newly retrieved sources. It is a targeted search, not a claim that every page was read. `document_screening.csv` records the usable evidence and exclusions. Reprinted or compiled versions of the same Earth figure are counted once.

Alexander et al.'s Figure 6 is a useful solar-storm occultation with only an elapsed-minute axis. The cited Fainberg presentation was traced to the 1974 URSI proceedings, PDF page 163 (printed page 155): it describes a storm around **20 February 1974**. This identifies a plausible observing period but not the date and UTC epoch of that particular plotted event, so it supplies no additional absolute timing match. [Primary proceedings](https://usncursi.org/archive/PDFs/1974-10-14.pdf).

The postlaunch attitude report, Figure 2-27 on PDF page 67, identifies day 247 of 1973 with a 15:00 reference epoch. Its caption describes an orbit with shadow, but the graph gives attitude solutions rather than explicit illumination transitions. Calling the whole orbit a shadow orbit does not establish that the particular plotted segment is in shadow. It is not used as a timed contact anchor.

The archive guides contain useful shadow-season dates, but some disagree between versions. Coarse season boundaries and prelaunch predictions are not substitutes for dated contacts. RAE-1 engineering plots are excluded from RAE-2 validation.

## Numerical method and checks

The spacecraft input is the separately rebuilt `feb13_adjacent_days_20261027/rollover_corrected_geometry.npy`, decoded from the original IBM-format RV archive. The midnight bug is therefore corrected before event selection. XYZ remains Moon-centered in kilometers; its axes are transformed directly from FK4/B1950 with the observation epoch into ICRS. The adopted spherical lunar radius is 1,737.4 km.

For the direct Sun, Earth, and Jupiter examples, new or retained JPL Horizons queries provide Moon-centered geometric ICRF body vectors with UTC epochs. The source direction is calculated from the spacecraft, including its finite displacement from the lunar center. No saved Earth-direction field is mistakenly treated as B1950. A separate ray/sphere test agrees with the angular-limb predicate for every finite evaluated sample.

The principal contacts are interpolated only between adjacent valid native records. Intervals require finite plausible positions, positive time separation, radius 2,500–3,100 km, implied speed below 3 km/s, and a gap no larger than 180 seconds. Typical brackets are 15.36 seconds; calibration gaps are flagged explicitly. Long gaps never produce fabricated contacts. Prediction-to-marker matching requires the same contact type within ten minutes; this matching bound is not the acceptance allowance.

An independent built-in ephemeris cross-check changes the direct-example contacts by at most **0.046 seconds for Earth, 0.328 seconds for Jupiter, and 0.001 seconds for the Sun**. The terrestrial catalog uses that built-in ephemeris. These checks verify the celestial-body calculation, not independent spacecraft tracking.

The 21 regression checks passed in 3.34 seconds. They include the existing analytic occultation, frame-rotation, invalid-geometry, and gap tests; the decoder rollover regressions; and new checks that scan rescaling cannot change the inferred time, JPL vectors are not extrapolated, and missing edges cannot pair separate orbits.

## What can be concluded

It is now more reasonable to **continue with FK4/B1950 and regard the February 13 figure markers as suspect** than to reject February 13's numerical coordinates. The adjacent-day prediction already showed that the disputed 47.1-minute interval survives removing the entire day. The Earth and Jupiter figures now reproduce historical timings without tuning, and the additional terrestrial catalog shows broad agreement across many epochs.

However, the requested premise that all other occultations work precisely is not fully met: the January solar plot has several-minute residuals, and the additional terrestrial table describes extended radio sources. These checks do not independently determine a unique equatorial epoch or certify the absolute tracking solution. They also cannot identify whether the February problem originated in the original annotation, a mislabeled date, or the reproduction process. A separate accurately timed optical shadow or an independent spacecraft state remains the decisive test.

The practical conclusion is to **exclude the February 13 printed SS/SR markers from calibration**, retain the numerical trajectory, and preserve the discrepancy in the uncertainty record. Ordinary uniform resizing of a scan cannot explain the difference because its ticks and markers scale together.

## Reproduction

The [analysis directory](C:/Users/Adam/Documents/ChatGPT/RAE2/historical_occultation_validation_20261027) contains source PDFs, rendered pages, fixed figure anchors, all comparison rows, ephemeris requests/responses, tests, the comparison figure, and a SHA-256 manifest.

```powershell
$env:PYTHONPATH='C:/Users/Adam/Documents/ChatGPT/RAE2/.audit_deps'
$env:OPENBLAS_NUM_THREADS='1'
$auditPython='C:/Users/Adam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $auditPython historical_occultation_validation_20261027/validate_events.py
& $auditPython historical_occultation_validation_20261027/build_report_data.py
& $auditPython -m pytest historical_occultation_validation_20261027/test_validation.py coordinate_frame_audit_20261027/test_frame_audit.py geometry_beam_investigation_20261026/test_archive_rollover.py -q
```
