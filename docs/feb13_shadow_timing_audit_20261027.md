# February 13 solar-shadow discrepancy: independent evidence search

Date: 2026-10-27. This extends the [coordinate audit](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/docs/coordinate_frame_audit_20261027.md). No production frame or detection threshold was changed.

**Later FK4 check:** [direct FK4 tests and an antenna-pointing comparison](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/docs/fk4_reassessment_20261027.md) now favor FK4/B1950 as the spacecraft XYZ working interpretation. Full FK4 predicts 08:17:00–09:04:06, a 47.10-minute interval. This strengthens the frame preference while leaving the February discrepancy unresolved. The true-of-date results below remain a comparison case, not a production-frame change.

## Finding

The available evidence does **not yet identify which record is wrong**. It argues against an ordinary equatorial-frame conversion as the cause. The competing possibilities remain a problem with the paper's SS/SR annotations and an event-specific problem in the archived spacecraft trajectory or its association with time. A separate February 13 tracking solution or measured illumination transition is still missing.

The six-minute difference is a **duration discrepancy**, not a uniform clock offset:

| Quantity | Ingress UTC | Egress UTC | Duration |
|---|---|---|---|
| Weber Figure 2, read from original scan | approximately 08:21 | approximately 09:02 | approximately 41 min |
| Native RV XYZ interpreted as true equator/equinox of date, JPL Sun center | 08:17:06.668 | 09:04:11.382 | 47.0786 min |
| Full optical shadow, accounting for solar disc | 08:17:16.642 | 09:04:01.342 | 46.7450 min |
| Any part of solar disc occulted | 08:16:56.697 | 09:04:21.417 | 47.4120 min |

The displayed decimal seconds are computational output, not established absolute accuracy of the spacecraft orbit. The paper readings retain a conservative ±60 seconds per marker, not a statistical confidence interval. Even moving both readings to maximize their interval gives 43 minutes, shorter than the calculated full optical shadow.

![February 13 evidence](C:/Users/Adam/Documents/ChatGPT/RAE2/solar_shadow_audit_20261027/feb13_shadow_evidence.png)

## What the historical documents establish

The search covered the cached 15 Dropbox PDFs (565 pages), the two GTDS/PEST and MAESTRO supplements (265 pages), the full Weber paper, and a newly retrieved NASA catalog. Keyword searches were followed by visual checks of relevant figures. This is a bounded search of the available documents, not a claim that no mission tracking records survive elsewhere.

1. **Weber, Fainberg & Stone (1976), Figure 2, PDF page 2.** The high-resolution original unambiguously reads **13 February 1974**. SS and SR are identified as sunset and sunrise. The nearby discussion distinguishes the radio-noise disappearance from geometric shadow; its reported variable radio offset does not explain why the geometric markers disagree with our geometry. The paper does not identify the tracking solution or housekeeping channel used to place those two markers. [Original paper](C:/Users/Adam/Documents/ChatGPT/RAE2/solar_lf_correction_20261027/weber1976.pdf), DOI [10.1029/GL003i006p00297](https://doi.org/10.1029/GL003i006p00297).

2. **RAE-B Postlaunch Attitude Operations Analysis, NASA 19740024148.** PDF page 65 (printed 2-44) discusses shadow periods, scanner failures, and scarce attitude solutions during shadow. Figure 2-27 on PDF page 67 is labeled day 247, reference time 1500 HR, processed 73/09/05. Day 247 in 1973 is September 4. It is an attitude plot from the first shadow season, not a February 1974 illumination record, and does not give independently measured SS/SR edges. PDF page 65 reports a maximum shadow duration of approximately 48 minutes, which is compatible with a near-central 47-minute passage but does not validate this event. [Report](C:/Users/Adam/Documents/ChatGPT/RAE2/historic_docs_review_20261026/19740024148.pdf).

3. **RAE archive guide, PDF pages 8 and 25.** The receiver thermistor is included in housekeeping every 19.7 minutes; the RV logical record includes an ambient-temperature field. The spacecraft XYZ axes are separately documented as X toward the vernal equinox and Z parallel to Earth's spin axis. This supplies a useful data-layout contract, not an independently timed February shadow. [Archive guide](C:/Users/Adam/Documents/ChatGPT/RAE2/historic_docs_review_20261026/RAE-B_Doc.PDF).

4. **NASA SIDS Report, December 1974, PDF page 88 / printed page 85.** The visually verified recent-orbit epoch is **August 14, 1974**. Its 1052.66–1077.82 km periapsis/apoapsis heights compare with 1052.77–1078.39 km daily extrema in native RV XYZ using a 1737.4 km lunar radius. The listed inclination is 55.340°, versus approximately 55.540° for the native coordinate-plane normal. This is broad consistency, not an exact frame solution: the entry does not specify the equinox, exact epoch time, node, argument of periapsis, or orbital phase. It cannot reconstruct the February event. The source's 221.98-minute period is also broadly consistent with the archive's approximately 222-minute local angular period; the latter is not a fitted Keplerian mean period. [NASA catalog](C:/Users/Adam/Documents/ChatGPT/RAE2/solar_shadow_audit_20261027/19750006723.pdf), [official download](https://ntrs.nasa.gov/api/citations/19750006723/downloads/19750006723.pdf).

5. **Tracking-paper lead, not acquired evidence.** Bryant & Williamson's *Lunar gravity analysis results from Explorer-49*, AIAA 74-810, reports analysis of 234 days of mean Keplerian elements. The [NASA record](https://ntrs.nasa.gov/citations/19740059045) has no downloadable paper. Its abstract does not establish the dates covered or provide a February 13 state. It is a specific lead for locating the underlying tracking series, not a successful independent validation. The related 1975 [lunar moment-of-inertia study](https://ntrs.nasa.gov/citations/19750058040) describes 230 days of Explorer-49 elements but likewise supplies no downloadable state series in its NASA record.

The PEST numerical example remains unsuitable as flight truth: it is a simulated perturbed-gravity test. The MAESTRO coordinate-output options likewise do not establish which option generated the science tape. Broad archival listings of “R.A.E. orbital elements” should not be assumed to refer to Radio Astronomy Explorer; that acronym can refer to the Royal Aircraft Establishment.

## Checks against native data

**Independent Sun ephemeris.** A new NASA/JPL Horizons query supplies 27 geometric, Moon-centered ICRF Sun vectors at five-minute intervals over 07:30–09:40 UTC. Their maximum directional difference from the previously used built-in ephemeris is 0.0139 arcseconds. Recomputed contacts differ by less than 0.001 seconds. JPL validates the Sun calculation here; it does not supply independent Explorer-49 tracking. The finite photosphere changes each edge by approximately 10 seconds, not minutes.

**Raw housekeeping.** The temperature field was independently decoded from the original IBM-format RV binary files for February 9–17. For this event, the same value is repeated across nine logical records covering approximately 18 minutes, with successive sets starting 19.711 minutes apart. The decoded values decline from about 1.76 to 1.48 and then rise to 1.60. They are left in raw decoded units because a verified physical calibration was not located. The actual thermistor sampling phase relative to the associated science records and thermal lag are also unresolved. Neither the broad minimum nor the repeated values provide a minute-resolution illumination edge. A plot of every repeated record as a new temperature measurement would greatly overstate the information available.

**Geometry required by the paper markers.** At the nominal SS/SR times, the calculated Sun-ray distances from the lunar center are approximately 1486 and 1599 km, versus a lunar radius of 1737.4 km. Both times are inside the shadow. Solving for a fixed distant source direction tangent at both archived positions requires at least a **17.21°** departure from the actual solar direction. A 10-second grid allowing ±60 seconds on each paper marker gives minimum required rotations from **13.94° to 19.85°**. This is a diagnostic under a fixed trajectory and radius, not a recovered coordinate correction. It does not rule out a bad trajectory. The paper's ingress time lies in a 92.158-second calibration gap; the interpolated diagnostic is correspondingly approximate. The actual calculated solar crossings each have native 15.36-second brackets.

**Date and clock alternatives.** Among the 16 fully observed nearby ingress/egress pairs retained between January 30 and February 27 with centers within two hours of the figure's clock time, February 13 remains the closest match. February 15 gives approximately 08:22:38–09:09:26 and does not repair the width. This is a coverage-limited check, not exclusion of every possible labeling error. A common clock shift cannot change a 47.08-minute interval into 41 minutes. The earlier B1950/date-frame comparison moves the February edges by only about 5–7 seconds.

## Consequence for the coordinate audit

The February SS/SR pair should remain a **disputed diagnostic**, not a calibration target used to force a new rotation or clock correction. The separate Earth and Jupiter figure checks support the equatorial interpretation within their plotted precision, but are neither exact tracking measurements nor independent February shadow telemetry. The XYZ equinox remains unresolved.

The decisive missing record is one of:

- a dated Explorer-49 tracking state or full orbital elements near **1974-02-13 08:00–09:30 UTC**, with origin, frame, epoch, and time system;
- a contemporaneous eclipse-prediction listing with its tracking solution identified;
- sufficiently frequent Sun-sensor or solar-array-current telemetry whose transition can be calibrated as entry to or exit from lunar shadow.

The first discriminates against our trajectory interpretation; the third supplies a measured illumination edge independent of the radio phenomenon. Another radio dropout or a second file derived from the same orbit export does not provide that independence.

## Reproduction and validation

Inputs, scripts, and results are in [solar_shadow_audit_20261027](C:/Users/Adam/Documents/ChatGPT/RAE2/solar_shadow_audit_20261027). The cached JPL response retains its query metadata. [Source and script hashes](C:/Users/Adam/Documents/ChatGPT/RAE2/solar_shadow_audit_20261027/provenance_manifest.json) support reproduction.

```powershell
$env:PYTHONPATH='C:/Users/Adam/Documents/ChatGPT/RAE2/.audit_deps'
$env:OPENBLAS_NUM_THREADS='1'
$auditPython='C:/Users/Adam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $auditPython solar_shadow_audit_20261027/check_event.py
& $auditPython solar_shadow_audit_20261027/build_evidence.py
& $auditPython -m unittest discover -s solar_shadow_audit_20261027 -p 'test_*.py'
```

Five new tests cover known-source recovery from synthetic contacts, impossible contact geometry, pandas timestamp storage units, invalid trajectory interpolation, and explicitly bounded calibration gaps. These and the eleven existing frame-audit tests passed together: **16 passed**. The new temperature reader excludes empty science blocks and does not interpret their zero time as midnight. An initial exploratory timestamp conversion was corrected before generating the retained results; it did not affect the pre-existing native geometry cache or production pipeline.
