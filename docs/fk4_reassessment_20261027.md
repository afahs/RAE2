# FK4/B1950 reassessment

**FK4/B1950 is now the preferred working interpretation for spacecraft XYZ.** The original pipeline already uses it. A new check against the separately documented B1950 antenna-pointing field strongly favors B1950 over coordinates of date. This changes the relative weight of the frame evidence, but it does **not** resolve the six-minute February 13 solar-shadow discrepancy or establish absolute tracking accuracy.

The subsequent [independent-record search and withheld-orbit test](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/docs/feb13_independent_records_20261027.md) retains FK4/B1950. Fits excluding the entire eclipse interval reproduce the native contacts within 4.60 seconds. This argues against isolated corruption of that segment; an independent absolute tracking state or optical shadow time remains unavailable.

## What supported FK4 previously

- The existing [pipeline constants](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/rylevonberg/constants.py:68) specify `DEFAULT_FRAME="fk4"` and `DEFAULT_EQUINOX="B1950"`. The [source-vector implementation](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/rylevonberg/frames.py) uses Astropy's full FK4 model. FK4 was therefore the working baseline, not a newly proposed alternative.
- The mission's PEST/GTDS interface explicitly defines selenocentric Cartesian positions in kilometers in the equator of 1950.0; its orbital-element description specifies Earth's mean equator and equinox of 1950.0. See [NASA CR-132908](C:/Users/Adam/Documents/ChatGPT/RAE2/historic_docs_review_20261026/supplemental/19740008428.pdf), PDF pages 14 and 84, printed pages 10 and 80. This supports the B1950 axes of the upstream orbit solution. It does not establish whether the science export retained those axes or used catalog E-terms.
- Earlier Earth radio-edge fits gave B1950 a small advantage with clock delay fixed at zero. The saved global-clock experiment fitted -16 seconds for B1950 and -22 seconds for coordinates of date. On its 11 withheld lunar blocks, normalized fit improvements were 6.865 versus 6.964, with the B1950-minus-date difference interval spanning zero (-0.1983 to +0.0025). That particular signal test did not establish a unique winning frame. These are audited saved results, not a new extraction of all Earth signal windows.

The earlier evidence for a true-of-date frame came primarily from the **separate Moon-centered Earth-direction field**. It should not be transferred automatically to spacecraft XYZ. The date-based solar calculations in the previous report were a comparison hypothesis; they did not replace the existing FK4 production default.

## New evidence: relationship between XYZ and antenna pointing

The [archive guide](C:/Users/Adam/Documents/ChatGPT/RAE2/historic_docs_review_20261026/RAE-B_Doc.PDF), PDF page 25, explicitly labels upper-V right ascension and declination as 1950.0. This provides a second field whose frame can be compared with XYZ without using radio power or the disputed solar markers.

For each candidate XYZ frame, the audit constructs the local radial, along-orbit, and orbit-normal directions from valid adjacent native spacecraft positions. The B1950 pointing direction is transformed into that local basis. A constant pointing direction is fitted separately for each UTC day on even nominal 222-minute orbit blocks and evaluated on the odd blocks. Each retained daily split requires at least 32 samples and coverage of at least six of eight orbital phase bins. The same selected days and samples are used for all candidates.

This uses every 25th valid native interior geometry row: 76,823 sampled rows, **516 eligible days**, and **34,261 held-out samples per interpretation**. No signal amplitude, candidate source, or solar timing marker enters the fit.

| XYZ interpretation | Median across days of held-out median residual | Reversed train/test split |
|---|---:|---:|
| Full FK4, B1950 equinox, observation epoch | **0.004545°** | **0.004529°** |
| FK4 without E-terms, B1950 axes | **0.004523°** | **0.004525°** |
| True equator/equinox of date | 0.205215° | 0.206389° |
| J2000 | 0.410993° | 0.413346° |

B1950 without E-terms has the smaller daily error than true-of-date on **515 of 516 days**, with a median paired error ratio of **45.34**. Full FK4 gives essentially the same conclusion. On February 13 specifically, full FK4 gives 0.003496° versus 0.194911° for true-of-date.

These are **internal consistency residuals**, not measured spacecraft pointing errors. The archived pointing may have been generated from the same orbit solution using an assumed antenna orientation; that would naturally produce a very stable relationship. Even in that case, the result is useful evidence that the two exported fields share B1950 axes, given the pointing field's explicit frame label. It does not independently validate the spacecraft trajectory, real antenna libration, or the precision of the original tracking.

![FK4 evidence](C:/Users/Adam/Documents/ChatGPT/RAE2/fk4_reassessment_20261027/fk4_evidence.png)

## Direct FK4 tests of the disputed solar event

The previous historical-contact audit used an `FK4NoETerms` B1950 basis transformation. That is an appropriate geometric axes hypothesis, but it is not literally identical to full catalog FK4. This reassessment directly evaluates both, rather than assuming they are interchangeable. It also varies observation epoch, tests equinox-of-date FK4, and reproduces the current production body-vector transformation.

Astropy distinguishes the equinox from the observation epoch, and distinguishes full FK4 from FK4 with the E-terms removed. See its primary [FK4 documentation](https://docs.astropy.org/en/stable/api/astropy.coordinates.FK4.html) and [FK4NoETerms documentation](https://docs.astropy.org/en/stable/api/astropy.coordinates.FK4NoETerms.html). Catalog E-terms are direction corrections, not a rigid coordinate-axis rotation, so full FK4 is transformed directly rather than approximated with a three-vector matrix. The full-FK4 position tests are convention stress tests; a Cartesian orbital state ordinarily represents geometric axes without stellar-catalog aberration terms.

All calculations retain the Moon-centered origin. The February Sun comes from the saved independent JPL Horizons vectors. The results below concern the solar center:

| Interpretation | Ingress UTC | Egress UTC | Duration |
|---|---|---|---:|
| Full FK4/B1950, observation epoch | **08:17:00.012** | **09:04:06.102** | **47.10149 min** |
| Full FK4/B1950, epoch B1950 | 08:17:00.013 | 09:04:06.103 | 47.10149 min |
| FK4 without E-terms, epoch B1950 | 08:17:00.011 | 09:04:06.103 | 47.10154 min |
| Current production FK4 body transformation | 08:17:00.009 | 09:04:06.101 | 47.10155 min |
| True equator/equinox of date comparison | 08:17:06.668 | 09:04:11.382 | 47.07856 min |
| Paper Figure 2, approximate readings | 08:21 | 09:02 | approximately 41 min |

The four B1950 FK4 variants (with/without E-terms, default/observation epoch) span less than **0.004 seconds per contact**. The observation-epoch effect is approximately one millisecond here. Those differences cannot explain several minutes. Decimal seconds describe numerical differences between implementations, not the absolute accuracy of the archived orbit. Native crossing brackets are approximately 15.36 seconds.

The same full-FK4 calculation gives **14:58:45.679–15:32:33.864** for the December 12 Earth example and **04:43:21.296–05:05:30.218** for the October 3 Jupiter example (see the retained CSV for precise numerical results). These remain within the earlier conservative ±60-second figure-reading allowances. The figures cannot discriminate between full FK4 and the no-E geometric convention.

At full FK4, the February interval is approximately **6.10 minutes longer** than the paper's nominal interval. A common clock shift cannot change that width. The previous finite-Sun check changed contacts by only approximately ten seconds. The independent February tracking/illumination record is still needed to decide between inaccurate paper annotations and an event-specific archive problem.

## Field-specific decision

| Field | Working interpretation after this reassessment |
|---|---|
| Spacecraft XYZ | **Prefer FK4/B1950 equatorial axes**, Moon-centered kilometers. Keep coordinates of date as a sensitivity alternative; exact catalog-versus-geometric convention is not resolved by the data. |
| Upper-V RA/Dec | **B1950**, with RA in hours, as documented. |
| Moon-centered Earth unit vector | **True equator/equinox of date**. A repeat of the 64-epoch independent JPL check gives 3.807 arcseconds median disagreement, versus approximately 1251.5 arcseconds for full FK4/B1950. |
| February 13 SS/SR markers | Disputed diagnostic; do not force the coordinate system or clock to reproduce them. |

The new [frame interpretation record](C:/Users/Adam/Documents/ChatGPT/RAE2/fk4_reassessment_20261027/frame_contract.json) supersedes the earlier equal-weight treatment of XYZ frame candidates. Production code has not changed: its FK4/B1950 default was already in place. A field-specific Earth-vector correction remains separate from the choice of spacecraft XYZ axes.

## Reproduction

Scripts and machine-readable results are in [fk4_reassessment_20261027](C:/Users/Adam/Documents/ChatGPT/RAE2/fk4_reassessment_20261027). [The manifest](C:/Users/Adam/Documents/ChatGPT/RAE2/fk4_reassessment_20261027/provenance_manifest.json) records input and script hashes. The run used Astropy 8.0.2.

```powershell
$env:PYTHONPATH='C:/Users/Adam/Documents/ChatGPT/RAE2/.audit_deps'
$env:OPENBLAS_NUM_THREADS='1'
$auditPython='C:/Users/Adam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $auditPython fk4_reassessment_20261027/reassess_fk4.py
& $auditPython fk4_reassessment_20261027/attitude_consistency.py
& $auditPython fk4_reassessment_20261027/build_artifacts.py
& $auditPython -m pytest fk4_reassessment_20261027/test_fk4_reassessment.py -q
```

**Four new tests passed.** They check full FK4 against the separately implemented ERFA FK4-to-FK5 algorithm, direct no-E transforms against the old basis matrix, radius preservation and round trips, and a synthetic known antenna direction in an orbital basis with an injected frame mismatch. The independent ERFA comparison differs by at most 0.0018 arcseconds for the test directions. A ray/sphere occultation predicate also agreed with the angular-clearance predicate at every evaluated row in all 33 event/convention combinations.
