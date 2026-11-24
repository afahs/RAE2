# February 13: independent-record search and withheld-orbit check

**Follow-up:** The [adjacent-day test](C:/Users/Adam/Documents/ChatGPT/RAE2/remote_upload_snapshot/docs/feb13_adjacent_days_20261027.md) now excludes all February 13 coordinates, predicts the event from February 12 and 14 only, and tests plot-label alternatives. It also fixes a midnight date-rollover bug in the native tape decoder; the long February 13 eclipse prediction survives that correction. The follow-up supersedes the local-corruption assessment below with the stronger whole-day result.

**FK4/B1950 remains the working interpretation of spacecraft XYZ. The new calculation argues against an isolated corruption of the February 13 eclipse segment, but no independently timed eclipse or absolute tracking state has yet been recovered.** A coherent error affecting the surrounding orbit solution remains possible. The separate Earth-direction field remains true equator/equinox of date.

The solar-center prediction remains **08:17:00–09:04:06 UTC on 13 February 1974**, lasting **47.10 minutes**. Weber et al.'s Figure 2 markers read approximately **08:21–09:02**, or **41 minutes**. The ±60-second allowance used for each paper marker is a conservative reading allowance, not a statistical uncertainty. Numerical seconds below compare calculations; they do not establish absolute spacecraft tracking accuracy.

## Predicting the event without its archived XYZ

The new [orbit test](C:/Users/Adam/Documents/ChatGPT/RAE2/feb13_independent_records_20261027/orbit_holdout.py) excludes **07:30–09:45 UTC** from every fit. It estimates a six-component state using a fixed spherical two-body Moon model and surrounding native XYZ only. It uses valid adjacent records with intervals no longer than 45 seconds, plausible radii and speeds, and every eighth retained training row. Solar contacts use the saved JPL Sun vectors and the direct full FK4/B1950 transformation.

There is a substantial daytime coverage gap. The later fit therefore uses the next available orbit, not data immediately after sunrise. The actual fitted samples are 03:30:14–07:29:50 (106 points) and 17:39:06–21:09:01 (94 points). The combined fit has 199 points because thinning is applied to the combined selection.

| Input used to predict the withheld interval | Ingress UTC | Egress UTC | Duration | Largest contact difference from native XYZ |
|---|---|---|---:|---:|
| Native eclipse XYZ, for comparison | 08:17:00.01 | 09:04:06.10 | 47.101 min | — |
| Earlier orbit only | 08:16:59.18 | 09:04:05.32 | 47.102 min | **0.83 s** |
| Later orbit only | 08:17:04.61 | 09:04:09.89 | 47.088 min | **4.60 s** |
| Both surrounding arcs | 08:17:00.06 | 09:04:05.95 | 47.098 min | **0.15 s** |
| Paper, approximate readings | 08:21 | 09:02 | ~41 min | — |

The maximum withheld position residual is 1.48 km for the earlier fit, 6.75 km for the later fit and 0.63 km for the combined fit. These are consistency residuals against the same archive. They include the effects of omitted lunar harmonics, Earth/Sun tides and radiation pressure, and are **not** estimates of the absolute orbit error.

An isolated bad eclipse segment would generally disagree with a trajectory fitted outside that segment. Here, all three surrounding-data predictions retain the long shadow interval. This substantially weakens that specific explanation. A common orientation error or a coherent incorrect orbit could pass this test, so it cannot certify the paper's markers as wrong.

![Withheld orbit results](C:/Users/Adam/Documents/ChatGPT/RAE2/feb13_independent_records_20261027/orbit_holdout.png)

## What the independent-record search found

**A tracking study spans the disputed date, but does not supply the missing absolute state.** Gapcynski, Blackshear, Tolson and Compton's 1975 paper analyzes 63 Explorer 49 averaged orbital-element sets over 230 days. Its Figure 1 epoch is JD 2441879.5, which converts to 16 July 1973; February 13 is day 212. The nominal 230-day endpoint is 3 March 1974. The article describes elements supplied by W. C. Bryant from a post-flight gravity study and analyzes longitude of node. Its residual plot and gravity coefficients cannot reconstruct orbital phase or independently predict the two eclipse contacts. This is separate published analysis, not proof of independent tracking inputs. [Author-uploaded article text](https://www.researchgate.net/publication/23584660_A_determination_of_the_lunar_moment_of_inertia), [NASA record](https://ntrs.nasa.gov/citations/19750058040).

The article's full text was readable through the web index. PDF acquisition from ADS and the author's linked copy returned HTTP 403; the figure epoch is therefore transcribed from the accessible article text rather than a locally verified PDF image. A 1977 follow-up added another 600 days, according to its [NASA abstract](https://ntrs.nasa.gov/citations/19770048051), but that record supplies no download. The [Bryant and Williamson 1974 record](https://ntrs.nasa.gov/citations/19740059045) describes 234 days of mean elements; its publisher PDF was also unavailable. I have not treated those metadata records as inspected full articles.

**The best direct documentary lead is NSSDC dataset 73-039A-02C.** NASA's 1988 catalog lists 49 reels of hourly burst-receiver plots spanning 12 July 1973–26 April 1977. The description explicitly says that Sun and Earth geometric-occultation intervals are drawn as bars, identified by S and E, with two-minute grid lines and four-minute labels. The relevant request is for both antennas' **08:00 and 09:00 UTC plots on 13 February 1974**, preferably with the adjacent hours and original date headers. This was visually checked on PDF page 107, printed Volume 5B page 37 of [NASA TM-101908](https://ntrs.nasa.gov/citations/19900020097); the local file is named `catalog_1990.pdf` after its accession year.

Those bars could show whether the journal figure differs from routine mission plotting. They are still calculated geometry, not measured optical shadow times, and could share an orbit solution with the science tapes. The catalog is evidence of historical holdings, not confirmation that the reels have been digitized or remain immediately available. No February 13 reel image was found in the accessible sources.

**An optical cross-check must handle known sensor faults.** The existing RAE-B postlaunch report describes eight solid-mode Sun sensors and says IDs 9 or 11 nominally indicate no selected sensor. It then documents valid Sun-angle data associated with ID 9, and delays of 1.35, 3.30 or 3.84 seconds in specific spin-mode investigations. Thus ID 9 alone must not be converted into an eclipse flag. These reported delays are seconds, not the several-minute changes required here; they are not measured February 13 corrections. See [NASA 19740024148](C:/Users/Adam/Documents/ChatGPT/RAE2/historic_docs_review_20261026/19740024148.pdf), PDF pages 108–111 and 140–142, especially Appendix B.

The newly retrieved [MSAD/OABIAS guide](https://ntrs.nasa.gov/citations/19830025650) explicitly models an orbital timing error alongside attitude and sensor biases (PDF page 66, printed 3-30). Its RAE-B discussion concerns PAS adaptations, including the translunar phase; it does not give a February 13 solution. Its reference 15 identifies a more specific software record: **CSC 3000-05300-02TR, RAE-B Attitude Support System Description and Operating Guide, R. Williams et al., June 1974**. The identifier is visually verified on PDF page 198: web OCR incorrectly rendered it as `03TR` in one result. Raw optical measurements and mode/calibration records would be required before adapting this method to the February event.

A further archival locator is **NASA Headquarters SPFL-43-B36-F6, item 5884**, whose indexed finding-aid description includes correspondence about extension and termination of Explorer 49 tracking. The [finding-aid URL](https://www.nasa.gov/wp-content/uploads/2026/02/satellite-subject-files.pdf?emrc=fe68e8) currently returned 404 in direct retrieval. Only the indexed description was available; the correspondence itself was not inspected, and there is no evidence yet that it contains a February state.

## Evidence needed to settle the discrepancy

1. The two hourly microfilm plots could distinguish a journal annotation problem from a discrepancy already present in routine mission plots.
2. A definitive tracking state, or complete orbital elements with epoch, frame, time standard and provenance, could independently predict February 13's contacts. Node residuals alone cannot do this.
3. Raw solid-mode Sun-sensor measurements or solar-array/battery telemetry could time actual illumination. Sensor IDs, stale registers, operating mode, calibration and telemetry sampling delays must be checked. The 19.7-minute ambient thermistor field already available is too coarse for this decision.

The [retrieval specification](C:/Users/Adam/Documents/ChatGPT/RAE2/feb13_independent_records_20261027/records_to_retrieve.txt) records exact identifiers and time windows. It is a local reference, not a sent request.

The present result supports continuing with FK4/B1950 and keeping Figure 2 out of coordinate or clock calibration. **It does not establish which absolute orbit or marker set is correct.** The new finding is that the eclipse interval is consistent with the surrounding archived motion to seconds, making a local XYZ splice or event-only corruption an inadequate explanation.

## Reproduction and validation

[Scripts, contact tables, fit states, residuals and source retrieval logs](C:/Users/Adam/Documents/ChatGPT/RAE2/feb13_independent_records_20261027) are retained with a SHA-256 manifest. Four new tests passed in 19.19 seconds: bidirectional propagation against an independently calculated analytic circular orbit; recovery of a withheld interval corrupted by a 187 km displacement; exclusion of the full event from every training selection; and rejection of unordered training epochs. The analytic propagation test avoids using the same integrator to generate its reference truth.

```powershell
$env:PYTHONPATH='C:/Users/Adam/Documents/ChatGPT/RAE2/.audit_deps'
$env:OPENBLAS_NUM_THREADS='1'
$auditPython='C:/Users/Adam/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $auditPython feb13_independent_records_20261027/orbit_holdout.py
& $auditPython feb13_independent_records_20261027/build_artifacts.py
& $auditPython -m pytest feb13_independent_records_20261027/test_orbit_holdout.py -q
```

Production code was not changed because its existing default already uses FK4/B1950. This investigation does not alter archived XYZ or force either paper contact into the fit.
