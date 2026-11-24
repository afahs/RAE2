# RAE 2 coordinate frame audit using historical occultations

**Subsequent FK4 reassessment:** the [new antenna-pointing consistency check](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/docs/fk4_reassessment_20261027.md) favors FK4/B1950 spacecraft XYZ over true-of-date by approximately 45-fold in held-out internal residuals across 516 days. FK4/B1950 is now the preferred working interpretation; independent absolute tracking remains unverified. Use the [updated interpretation record](C:/Users/Adam/Documents/ChatGPT/RAE2/fk4_reassessment_20261027/frame_contract.json). The original historical-contact results below are retained as evidence, and the Earth-direction field remains true-of-date.

The historical Sun, Jupiter and Earth examples establish that the archived spacecraft XYZ values are Moon-centered equatorial positions with the documented axis order. They do **not yet uniquely establish the equinox of those positions**. In particular, the very strong evidence that the separate Earth-direction field uses true equator and equinox of date must not be transferred automatically to spacecraft XYZ.

The initial audit's field-by-field interpretation was:

| Field | Interpretation | Evidence and remaining uncertainty |
| --- | --- | --- |
| Spacecraft XYZ | Moon-centered, kilometers, equatorial axes; X toward the vernal equinox, Z toward the north terrestrial spin pole, Y completing the right-handed triad | Tape documentation and dated occultations agree on the origin and axis convention. B1950 versus an equator/equinox of date remains unresolved. |
| Earth unit vector | Moon-centered direction, true equator and equinox of date | Independent JPL vectors give a median residual of 3.81 arcseconds, versus 1,252 arcseconds for B1950. |
| Upper V antenna RA/Dec | B1950; RA interpreted in hours | The tape format explicitly labels these angles 1950.0. That label is not explicitly attached to XYZ. |

No production frame default was changed. The initial machine-readable interpretation is [frame_contract.json](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/frame_contract.json); its equal-weight XYZ status is superseded by the follow-up linked above.

## Dated occultation checks

Three primary-source figures were read visually. The approximate marker readings below carry a conservative one-minute allowance per edge; this is an assessment of plot-reading precision, not a published statistical uncertainty. The markers are geometrical predictions, not independently measured spacecraft tracking positions. Their agreement therefore tests reproduction of the historical geometry rather than proving the original ephemeris was correct.

| Object and date | Historical markers in UTC | XYZ treated as B1950 | XYZ treated as true equator and equinox of date |
| --- | --- | --- | --- |
| Sun, 13 February 1974 | 08:21 to 09:02 | 08:17:00 to 09:04:06 | 08:17:07 to 09:04:11 |
| Earth, 12 December 1973 | 14:58 to 15:33 | 14:58:46 to 15:32:34 | 14:58:42 to 15:32:52 |
| Jupiter, 3 October 1973 | 04:43:15 to 05:06:15 | 04:43:21 to 05:05:30 | 04:43:09 to 05:06:00 |

Sources: Weber, Fainberg and Stone (1976), Figure 2, PDF page 2; Alexander et al. (1975), Figure 5, PDF page 6; Kaiser (1977), Figure 3, PDF page 4. Kaiser's paper was found in the local reference collection, supplementing the Dropbox documents. Its upper and lower plot panels should not be treated as second-precision timing tables.

![Historical contact comparison](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/historical_contacts_comparison.png)

Both B1950 and date-based XYZ reproduce the Earth and Jupiter examples within the plot-reading allowance. J2000 also falls within that allowance, so these three examples alone cannot distinguish nearby equatorial epochs. Mean and true equator of date differ by less than half a second at these particular contacts.

Ecliptic interpretations fail much more clearly: the Earth ingress moves about 15 minutes early and the Jupiter ingress about 19 minutes early. Testing all 48 signed axis permutations gives the documented XYZ order the lowest RMS limb-clearance discrepancy, 2.96 degrees; the next-best permutation gives 13.44 degrees. The residual of the best convention is dominated by the disputed Sun figure. This permutation comparison rules against simple axis swaps/sign errors; it is not a fit of arbitrary rotations or an absolute orbit determination.

Earth emersion in the date-based and J2000 cases lies in a 92.159-second native gap. The displayed time interpolates across that gap and is explicitly marked in the contact table. The other main-event contacts have approximately 15.36-second brackets. The figure comparison permits gaps up to 180 seconds; mission-wide season/contact counts use a stricter 45-second limit.

Full results: [event_contacts.csv](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/event_contacts.csv), [event_anchors.json](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/event_anchors.json), [axis_permutation_check.csv](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/axis_permutation_check.csv).

## The Earth field independently identifies its own frame

The raw saved JPL Horizons response was re-evaluated against the nearest native archive epochs, checking that the requested times match within 0.001 second. It contains geometric ICRF Earth-minus-Moon vectors in kilometers at 64 UT epochs. No old aggregate residual table was used as the measurement.

| Candidate interpretation of the Earth direction | Median difference | 95th percentile |
| --- | ---: | ---: |
| J2000 | 1,262.46 arcsec | 1,313.65 arcsec |
| B1950 | 1,251.64 arcsec | 1,288.47 arcsec |
| Mean equator and equinox of date | 13.06 arcsec | 14.77 arcsec |
| True equator and equinox of date | 3.81 arcsec | 5.04 arcsec |

A separate 2,048-epoch check using Astropy's built-in ephemeris reaches the same conclusion. The Earth field also agrees with the date-based interpretation on each of the three figure dates. The improvement from B1950 is roughly a factor of 329 in median angular residual. This is a coordinate consistency improvement, not a source sensitivity measurement.

![Earth vector comparison](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/earth_field_frame_comparison.png)

The practical error to avoid is assigning one global `B1950` or `J2000` label to every vector on the tape. The Earth direction and the explicitly B1950 antenna angles already require separate metadata. Spacecraft XYZ needs its own unresolved-epoch status until another independent constraint resolves it.

Results: [JPL recheck](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/earth_field_jpl_recheck.csv), [broader independent check](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/earth_field_check.csv).

## Occultation seasons and conflicting historical dates

The documents disagree materially. These are visual readings of the source pages, not OCR-only conclusions:

| Item | Preprint or archive guide | Published instrumentation paper |
| --- | --- | --- |
| Jupiter first season | 26 July 1973 to 10 July 1974, preprint PDF p8 | 13 August 1973 to 19 June 1974, published PDF p2 |
| Saturn first season begins | 1 June 1974, preprint PDF p8 | 2 April 1974, published PDF p2 |

The archive guide's event table, PDF p3 of B27602-000A, lists the second solar-shadow season in January-March **1973**, before launch. Its chronological position implies 1974; the raw printed value and that inference are retained separately. The table's other seasons are 29 July-26 October 1973, 10 June-28 August 1974, 17 November 1974-1 February 1975, and 21 April-9 July 1975.

Native positions were used to calculate actual sampled crossings, with no radio-intensity selection. An initial daily orbit-plane/circular-orbit approximation was only exploratory: using a daily median radius can distort grazing-season boundaries and cannot reliably handle trajectory updates. The final season comparison uses native positions and excludes unphysical intervals.

| Season check | Native XYZ as B1950 | Native XYZ as true of date | Interpretation |
| --- | --- | --- | --- |
| First sampled Jupiter contact | 10 August 1973 | 28 July 1973 | Published August start is closer to B1950; preprint July start is closer to date-based XYZ. Grazing contacts and coverage prevent interpreting the first sampled crossing as an exact season start. |
| Last sampled Jupiter contact in first season | 18 June 1974, 23:23 | 18 June 1974, 19:38 | Both are near the published 19 June end; neither supports the preprint's 10 July end. |
| First solar season starts | 30 July 1973 | 29 July 1973 | Date-based result matches the guide's day. |
| First solar season ends | 17 October 1973 | 16 October 1973 | Both disagree with the guide's 26 October end. This remains a document/trajectory discrepancy. |
| Second solar season starts | 3 January 1974 | 3 January 1974 | Both match. Its March end cannot be tested with this RV archive because of the March-May gap. |
| Third solar season | 10 June-28 August 1974 | 10 June-28 August 1974 | Both reproduce the printed days. |
| Fourth solar season | 17 November 1974-1 February 1975 | 17 November 1974-1 February 1975 | Both reproduce the printed days; archived geometry has a brief interruption near 20 November. |
| Fifth solar season starts | 22 April 1975 | 21 April 1975 | Date-based result matches the guide's day. July end is beyond this archive's coverage. |

Saturn is already occulted in the first available May 1974 data under all four equatorial candidates, so the preprint's June start is inconsistent with these positions. The archive gap prevents directly verifying the published 2 April start. These comparisons provide no coherent reason to force all XYZ into one epoch by selecting only favorable document dates.

The archive spans July 1973-June 1975. A missing crossing during a gap is not evidence of visibility. All season endpoints above are explicitly sampled-contact bounds, not gap-filled estimates of the first or last physically possible orbit. Jupiter's near-grazing July/August geometry makes its apparent season start particularly sensitive to small rotations.

Sources and results: [historical season anchors](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/historical_season_anchors.json), [season comparison](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/season_anchor_comparison.csv), [native contact catalogue](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/native_contacts.csv), [daily coverage](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/native_daily_coverage.csv).

## Why the February Sun figure remains inconsistent

The paper's approximate geometrical interval is 41 minutes. Native XYZ predicts 47.08 minutes in the true-of-date interpretation. Switching to B1950 changes the two contacts by only about 6.7 and 5.3 seconds, respectively. It cannot explain a roughly six-minute duration discrepancy.

Treating the Sun as an extended optical disc changes each contact by approximately 10 seconds. A clock correction shifts both contacts together and cannot remove the duration error: the best common shift still leaves opposite residuals of about three minutes. A UTC/TT convention error has the same limitation.

As another time check, the recorded Earth direction was compared with a time-shifted ephemeris on each event day. Robust median clock-equivalent offsets are -5.5 seconds on the Sun date, -0.7 seconds on the Earth date and -14.4 seconds on the Jupiter date. Historical ephemeris differences and field timing can contribute to these offsets; they are not calibrated clock corrections. Rare corrupted direction rows make a non-robust least-squares estimate misleading, so the per-row estimates and their quantiles are retained. There is no support here for a general multi-minute timestamp correction.

Weber's Figure 1 on 16 January 1974 supplies another solar check: the native predicted occultations broadly align with its coarse plotted bars. Those bars have only a few-minute reading precision and cannot distinguish B1950 from a date-based frame. The February mismatch therefore cannot be taken as proof of a universal solar-frame error.

The February plot must remain an unresolved geometry/figure-provenance check. Its frequency-dependent radio dropouts must not be used to fit a coordinate rotation or clock shift; the paper discusses an extended/local plasma-noise phenomenon, not a simple point-source edge. An independent trajectory state or solar-aspect/shadow telemetry for that event is needed to decide whether its printed markers or the archived spacecraft positions are wrong.

Checks: [finite solar disc](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/solar_disc_check.json), [clock-equivalent residuals](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/earth_clock_equivalent.csv), [January solar contacts](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/weber_figure1_contacts.csv).

## Consequences for point source searches

For mutually matched contacts over this archive, changing spacecraft XYZ from B1950 to true equator and equinox of date gives:

| Source | Median absolute ingress shift | 95th percentile ingress shift | 99th percentile ingress shift |
| --- | ---: | ---: | ---: |
| Fornax A | 6.80 s | 7.77 s | 7.87 s |
| Virgo A | 11.29 s | 43.79 s | 97.84 s |
| Taurus A | 5.91 s | 18.35 s | 46.85 s |

For Fornax after full lower-V extension, median shifts are 6.24 seconds for ingress and 6.71 seconds for egress. Grazing events can be much more sensitive than the median, and the available event sets also change with the frame. Matching uses the same edge, mutual nearest times and a 900-second maximum separation; unmatched events are not silently assigned zero shift. These are geometry differences, not injection-recovery sensitivity gains or detections.

The source pipeline should retain both plausible XYZ epochs and their timing uncertainty. It should use the independently established date-based frame for the Earth direction and the documented B1950 frame for antenna angles. Absolute-coordinate claims should wait for a dated tracking state or a set of independent contacts precise enough to distinguish these hypotheses. The current historical plots alone do not justify choosing one spacecraft epoch and discarding the other.

Result: [point source frame impact](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/point_source_frame_impact.csv).

## Reproducibility and validation

The audit reads the separately decoded IBM RV archive geometry, not the known-invalid IEEE interpretation in the saved burst-receiver tplot files. Native rows are retained when checking adjacency, so a discarded row cannot cause interpolation across a newly created gap. Geometric intervals require plausible radii, finite values, positive elapsed time and implied speed below 3 km/s. The season calculation accepts gaps up to 45 seconds and uses 1,957,435 of 1,999,427 adjacent intervals. Rejections include gaps as well as malformed or discontinuous geometry; that count is not a count of corrupt measurements.

The native archive can contain interleaved positions with plausible radii but impossible motion. The beginning of 16 January 1974 is an example. Without the speed guard, such rows produce many spurious occultation crossings. The independent historical-anchor scripts guard against that failure.

The main ephemeris is Astropy/ERFA built-in geometric body positions. Full-mission calculations interpolate body vectors on a 600-second grid and use daily frame rotations; the three dated examples evaluate body positions and frame rotations at native epochs. The separate Earth-direction test uses saved JPL ICRF vectors. ICRS/J2000 frame bias is negligible compared with the plotted contact precision. Calculations use a spherical Moon of radius 1,737.4 km. Contacts are source-center contacts unless explicitly labeled as solar-disc contacts. Apparent-direction/light-time conventions and finite Earth source extent would matter before making second-level claims from radio edges.

Eleven regression tests pass: rotation orthogonality/handedness and inverses for six frames, an analytic circular-orbit occultation duration, the Moon-direction sign, gap rejection, rejection of plausible-radius trajectory jumps, and invariance under a common coordinate rotation.

Code is in [coordinate_frame_audit_20261027](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027). Run with the bundled Python and `.audit_deps` on `PYTHONPATH`:

```powershell
$env:PYTHONPATH='C:/Users/Adam/Documents/ChatGPT/RAE2/.audit_deps'
$env:OPENBLAS_NUM_THREADS='1'
$auditPython='C:/Users/Adam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $auditPython coordinate_frame_audit_20261027/independent_checks.py
& $auditPython coordinate_frame_audit_20261027/native_seasons.py
& $auditPython coordinate_frame_audit_20261027/frame_impact.py
& $auditPython coordinate_frame_audit_20261027/build_artifacts.py
& $auditPython -m pytest coordinate_frame_audit_20261027/test_frame_audit.py -q
```

Primary-source PDFs, raw input and analysis-script hashes are recorded in [provenance_manifest.json](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/provenance_manifest.json). The earlier daily-plane outputs are exploratory and should not replace the guarded native contact catalogue.

## Primary documents

- [RAE archive tape description](C:/Users/Adam/Documents/ChatGPT/RAE2/historic_docs_review_20261026/RAE-B_Doc.PDF), PDF p25: spacecraft axes and separate 1950.0 antenna-angle label.
- [Archive event table](C:/Users/Adam/Documents/ChatGPT/RAE2/historic_docs_review_20261026/B27602-000A.pdf), PDF p3: five solar-shadow seasons.
- [Instrumentation preprint](C:/Users/Adam/Documents/ChatGPT/RAE2/historic_docs_review_20261026/B23736-000A.pdf), PDF p8: Jupiter and Saturn season dates.
- [Alexander et al. 1975](C:/Users/Adam/Documents/ChatGPT/RAE2/historic_docs_review_20261026/alexander.1975.pdf), PDF pp2 and 6: revised season dates and dated Earth Figure 5.
- [Weber et al. 1976](C:/Users/Adam/Documents/ChatGPT/RAE2/solar_lf_correction_20261027/weber1976.pdf), PDF pp1-2: January solar bars and the disputed February Figure 2.
- [Kaiser 1977](C:/Users/Adam/Documents/ChatGPT/RAE2/coordinate_frame_audit_20261027/kaiser1977.pdf), PDF pp3-4: planetary season chart and dated Jupiter Figure 3.

The earlier GTDS/PEST and MAESTRO supplements explain B1950 state inputs and available output transforms, but do not establish the science-tape export epoch. The numerical PEST example is a simulated perturbed-gravity test, not an independently dated observed spacecraft state suitable for resolving this ambiguity.

## February 13 follow-up

The [focused shadow-timing investigation](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/docs/feb13_shadow_timing_audit_20261027.md) checked engineering shadow plots, raw thermistor housekeeping, an independently published orbit catalog, a fresh JPL Sun ephemeris, finite-disc contacts, and nearby dates. The discrepancy remains unresolved; no independent February 13 illumination transition or tracking state was located. The Figure 2 markers should remain disputed diagnostic inputs, not ground truth for fitting a coordinate correction.
