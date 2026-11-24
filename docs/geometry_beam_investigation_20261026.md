# Independent geometry, beam, and point-source recovery investigation

26 October 2026. **No defensible Fornax A or Virgo A detection emerged.**
The investigation did identify a coordinate-frame inconsistency, a substantial
beam-provenance mismatch, and an injection-tested way to improve recovery on
the selected backgrounds. Earlier detections, null results, and geometry
assumptions were treated as hypotheses, not validation data.

**Historic-document addendum:** The subsequently supplied Dropbox collection
has now been inventoried. The narrow legacy beam widths closely match the
RAE-1 patterns in Weber et al. (1971); the broader Alexander et al. (1975)
table summarizes calculated RAE-2 patterns, with flight consistency checks.
The wording below has been corrected accordingly. Additional attitude records
strengthen the concern about lower-V asymmetry and pointing-data reliability.
See [the document review](historic_documents_review_20261026.md). These are
provenance/interpretation updates; the source fits have not been rerun.

A [further XYZ provenance search](xyz_provenance_and_historic_findings_20261026.md)
found explicit Moon-centered, 1950.0-equatorial GTDS input states in the RAE-B
mission software documentation. This supports the upstream B1950 hypothesis;
the archive export remains unverified because the software supported several
output frames. Historical ephemeris timing errors also motivate mismatched
timing injections. This documentary finding does not change the fitted results.

All new code and outputs are in
[`geometry_beam_investigation_20261026`](../../geometry_beam_investigation_20261026/).
Production pipeline defaults were not changed: the spacecraft position frame
and the full antenna response remain insufficiently established.

## What was checked independently

The native receiver data were decoded again from
`C:\Users\Adam\Documents\PhD\Diffraction\RV-DATZIP.zip`, rather than relying on
the master CSV or its derived profiles. The 166 DAT members have individual
SHA-256 hashes in `cache/binary_inventory.csv`. The original archive format
document was extracted from `Ryle-Vonberg.zip` as `archive_format.pdf`.

The independent decoder handles big-endian IBM hexadecimal floats and the
fixed record layout: a four-byte physical header plus 35 logical records of
908 bytes, each containing eight frequency dwells. It preserves the two fine
samples independently of coarse validity, imposes no brightness cut, and
excludes every duplicate band/header-time block. This produces 2,000,832 valid
frequency blocks and 1,999,428 distinct geometry epochs. Finite positive fine
outputs number 3,982,085 upper-V and 3,979,098 lower-V across all bands.

The archive documentation describes seven retained coarse samples, with the
first coarse sample omitted because of settling, and a 7.68-second fine
cadence. The decoder uses 1.92-second substeps. The precise sub-dwell phase is
still inherited rather than independently established; that uncertainty is
explicit in `cache/manifest.json`. See the original document, PDF pages 10
and 27, rather than interpreting this reconstruction as a timing calibration.

### Earth coordinates are of date; spacecraft XYZ is still ambiguous

An 8,192-epoch check against independently calculated Earth-minus-Moon
directions favored equatorial coordinates of date. A second check used
[NASA JPL Horizons](https://ssd-api.jpl.nasa.gov/doc/horizons.html) at 64 native
archive epochs, with geometric Earth vectors centered on the Moon:

| Interpretation of archive Earth unit vector | Median angular discrepancy |
|---|---:|
| ICRS/J2000 | 0.35068 degrees |
| B1950 | 0.34765 degrees |
| Mean equator/equinox of date | 0.00360 degrees |
| True equator/equinox of date | 0.00106 degrees, about 3.8 arcseconds |

The response and exact query are preserved in `horizons_anchor.json` and
`jpl_check.json`. This is strong evidence against interpreting the **Earth
unit-vector field** as B1950. It does not establish the frame of every field
on the tape. The archive explicitly labels upper-V RA/Dec as 1950, while its
spacecraft XYZ description gives equatorial axes without an epoch.

Consequently, both B1950 and true-of-date spacecraft-position hypotheses were
tested, keeping the recorded attitude in B1950. Their selected target crossing
times differ by a median +6.34 seconds for Fornax and +3.06 seconds for Virgo;
Virgo's grazing crossings differ by as much as about 83 seconds. A global
frame replacement would be premature without an independent tracking orbit.

Geometry screening rejected 1,405 invalid-radius epochs, including 1,398 zero
positions. Interpolation uses actual timestamps, rejects intervals longer
than 45 seconds or implausible velocities, and interpolates attitude unit
vectors to handle RA wrap. Recorded upper-V pointing lies about 8.96 degrees
from the spacecraft radial direction, so a radial antenna approximation is
not equivalent. Leave-one-epoch-out interpolation gives a median error of
7.9 metres and a 95th percentile of 260 metres. These measure internal
consistency, **not absolute orbit accuracy**; the archived trajectory itself
appears to contain interpolated segments.

### The legacy beam curves do not match the published in-orbit widths

The six digitized beam cuts were copied from
`perlmutter:/global/u2/a/afahs/antennaDigitization/` with their source image and
digitization script. Their first half-power crossings disagree with several
calculated RAE-2 widths in [Alexander et al. (1975), Table 1](https://ntrs.nasa.gov/api/citations/19750010212/downloads/19750010212.pdf):

| MHz | Plane | Legacy FWHM | Published calculated RAE-2 FWHM |
|---:|:---:|---:|---:|
| 1.31 | E / H | 183.8 / 120.6 degrees | 180 / 120 degrees |
| 3.93 | E / H | 24.5 / 53.8 degrees | 80 / 63 degrees |
| 6.55 | E / H | 15.6 / 36.1 degrees | 27 / 55 degrees |

The original 3.93-MHz image itself shows a narrow pattern. This establishes a
provenance/configuration mismatch, not necessarily a digitization mistake.
The subsequent historic-document review found closely matching RAE-1 widths
in Weber et al. (1971), Figure 2/Table 1. Alexander's RAE-2 table is a
calculation, supported by Earth-scan checks on main beam size, rather than a
direct measurement of the full beam. The 60-degree RAE-1 V geometry and
35-degree equivalent upper-V apex described for RAE-2 must be distinguished.
The new test uses smooth beam envelopes constrained by the published widths
and front/back ratio. Averaging E/H cuts is an approximation, not a measured
two-dimensional beam; sidelobes and azimuthal structure remain unresolved.
Lower-V pointing is approximated as opposite upper-V pointing.

![Beam provenance comparison](../../geometry_beam_investigation_20261026/beam_provenance_comparison.png)

## The new source experiment

Fornax and Virgo were tested at 1.31, 3.93, and 6.55 MHz using native fine
outputs. Geometry alone selected at most one event per day, with a fixed
random seed and a cap of 192 days. Fornax uses both antennas after 7 November
1974. No Virgo crossings existed in that interval; before fitting power,
Virgo was extended back to 15 November 1973 and restricted to upper-V, which
was then fully extended. This is not a repeat of the previous full-data search.

Two models fit the source simultaneously with a local intercept and slope.
The joint model adds forward-integrated diffuse nuisance terms: a smooth
Galactic plane and that plane multiplied by a Galactic-center dipole. Their
integrals include the antenna envelope and the spherical Moon visibility
boundary. Doubling the angular quadrature changes the computed features by
less than 4.4e-7 in absolute units in the checked samples. No previous sky
map or claimed source amplitude supplies the answer.

Joint fitting matters: these nuisance terms overlap strongly with the source
template. The first local-fit diagnostics retain roughly half the source
Fisher information, implying about 1.4 times the white-noise amplitude error.
Subtracting a nuisance-only fit before estimating the source would attenuate
the source itself. Fitting both together preserves the modelled source
amplitude, at the cost of increased uncertainty.

An initial clipped, studentized contrast score with separately centered
nearby fields failed to localize some strong injected sources. That failure
is retained in `full_source_injections.csv`; it motivated a second scoring
family, recorded in `EXPERIMENT.md`.

The final coherent comparison evaluates the target and 24 nearby directions
(3- and 7-degree rings) on **identical native samples**, using physical
visibility and beam gain in the target window and both time-control windows.
Windows are 20 minutes long and controls are displaced by 30 minutes. It
jointly projects nuisance terms, combines source-template numerators and
information with a constant amplitude per channel, and recomputes off-time
noise weights after each injection. It does not clip the score or normalize
by the scatter of the source-bearing block estimates.

Complete paired windows leave 87 Fornax days and 73 Virgo days, each spanning
eight lunar blocks. Exact enumeration of 256 block-sign patterns calibrates
the maximum over bands, receivers, frames, and nuisance methods. An additional
factor of four accounts for two sources and two explored scoring families.
Block independence and sign symmetry remain assumptions, so these are
exploratory significance checks. Nearby directions are correlated controls;
their rank is a localization diagnostic, not an independent probability.

| Real source | Best tested combination | Adjusted block-sign p | Target rank of 25 |
|---|---|---:|---:|
| Fornax A | B1950 XYZ, 6.55 MHz lower-V, joint diffuse | 1.0 | 7 |
| Virgo A | True-of-date XYZ, 1.31 MHz upper-V, joint diffuse | 1.0 | 23 |

Neither source passes. The raw coherent statistics are not Gaussian sigma
values and must not be presented as detection significances. Upper/lower
measurements also share backgrounds and are not independent replications.

## Injection recovery: a useful improvement with important limits

Injected source power enters all overlapping target, time-control, and
nearby-sky samples with the source's actual visibility and beam modulation.
Both spacecraft-frame hypotheses were used as injection truth. The full
nuisance fit and control normalization are propagated; this is not an
injection into an already cleaned profile. Exact linear response operators
accelerate the calculation and were checked against direct native refits.

One injection unit is defined separately for each source/band/receiver from
training-control residual noise divided by typical limb gain. Thus the
experiment defines a synthetic test spectrum, **not a common flux in Jy**.
Both nuisance methods receive identical injected power within a channel.

| Source | Joint model: first passing tested amplitude | Linear model: first passing tested amplitude |
|---|---:|---:|
| Fornax A | 12 units under either truth frame | 48 units under either truth frame |
| Virgo A | 16 units under either truth frame | 48 units for B1950; 64 for true-of-date |

A pass requires adjusted p <= 0.05 and target rank 1. The grid is
0, 0.5, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64. These first passing points are
3--4 times lower for the joint model. **They are not calibrated sensitivity
limits or recovery probabilities.** There is one real-background realization
per source; injections use the fitted beam family; the reported score uses
all eight blocks rather than an independent held-out subset. Even-block
controls set the normalization floor and injection scale, not an independent
validation sample.

Two high-amplitude reversals remain: at 64 units, Fornax/true-of-date/linear
falls to spatial rank 2, and Virgo/B1950/joint has adjusted p=0.0625. Changing
control weights and competition among frames/channels can make this recovery
criterion nonmonotonic. The full curves and `injection_reversals.csv` expose
those failures rather than turning the first passing point into a blanket
claim of sensitivity improvement.

![Conditional injection recovery](../../geometry_beam_investigation_20261026/coherent_recovery_comparison.png)

## Validation and next priority

All **18 new regression tests passed**. They cover archive decoding, sample
layout, gap rejection, RA wrap, visibility, beam constraints, quadrature,
joint-fit source preservation, full control contamination, and coherent
localization/refit algebra. **648 direct native injected-window refits**
passed: 432 for the first scoring family and 216 for the coherent family.
Maximum normalized discrepancies were 1.8e-13 and 2.1e-14 respectively. These
are numerical equivalence checks, not 648 independent astronomical trials.
The previous pipeline's 187 tests were not rerun in this investigation.

The most useful next work is independent recovery of spacecraft tracking/frame
metadata and the full in-orbit antenna response, then a fixed-method,
held-out-block recovery experiment with beam, orbit, and timing mismatches.
A background model constrained across multiple orbits may reduce the current
source/background degeneracy. The current geometric source model also omits
diffraction, finite source extent, lunar topography, and absolute flux
calibration. Those omissions should be tested before interpreting a null
result as an astrophysical upper limit or extending to a larger source list.

`EXPERIMENT.md` records changes to the original plan.
`run_reproduce.ps1` gives the full execution order; `provenance_manifest.json`
records hashes and software versions. Numeric findings are available in
`final_source_results.csv`, `tested_injection_thresholds.csv`,
`coherent_coverage.csv`, `beam_width_audit.csv`, and `jpl_check.json`.
