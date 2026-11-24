# Historic Dropbox documents: coverage and implications

26 October 2026. **The collection had not been systematically reviewed before
the latest geometry/beam experiment.** That experiment used the original
Ryle-Vonberg archive-format document and the Alexander et al. (1975)
instrument paper. Prior project notes were treated as leads, not independent
validation. Having a paper on disk does not establish that its contents were
incorporated into the analysis.

The [user-supplied folder](https://www.dropbox.com/scl/fo/84ww3b7nqzer51ldzpnn0/AJ4QggneP-aBAo8NbFX542w?dl=0&e=1&rlkey=fwq1ulh1c3a8p2dxrfyw45yx2)
contains **15 PDFs, 565 PDF pages, 39,889,222 bytes**. All were downloaded and
hashed in `historic_docs_review_20261026/inventory.json`. Relevant pages were
rendered and read visually, including scanned pages with no searchable text.
This is an inventory and targeted review, **not a claim that all 565 pages
have been read**. Page references below refer to PDF page numbers.

## Findings that affect interpretation

**XYZ follow-up:** A deeper search, including OCR of 112 scanned pages and
two supplemental NASA mission-software reports, found an explicit GTDS
interface for Moon-centered spacecraft states in Earth's 1950.0 equatorial
frame. The final conversion into the receiver archive is still untraced.
See [XYZ provenance and additional findings](xyz_provenance_and_historic_findings_20261026.md)
for the tracking-to-state evidence, page references, timing uncertainty and
instrumental periodicities. This strengthens the upstream B1950 evidence
without changing any numerical source result.

### 1. The beam discrepancy is consistent with mixing RAE-1 and RAE-2 patterns

Weber, Alexander, and Stone (1971), pages 3-5, describe **RAE-1**. Figure 2
explicitly labels its six E/H cuts as RAE-1 patterns. Table 1 gives widths of
23 x 52 degrees at 3.93 MHz and 14 x 34 degrees at 6.55 MHz. The legacy
digitizations gave approximately 24.5 x 53.8 and 15.6 x 36.1 degrees: close
agreement. The depicted RAE-1 V apex is 60 degrees.

Alexander et al. (1975), pages 2-3, instead describe the RAE-2 upper V with
an equivalent apex angle of 35 degrees and give the 80 x 63 and 27 x 55
degree calculated widths used in the new experiment. They attribute the
detailed calculations to Sayre (1974), and report that Earth scans generally
support the theoretical main-beam size.

**Implication:** the legacy curves plausibly represent the wrong spacecraft
configuration, rather than bad tracing of the figure. The precise tracing
lineage still needs to be recorded. The replacement envelopes are constrained
by published RAE-2 calculations, not by a measured full two-dimensional beam.
The previous report's term "in-orbit widths" was therefore imprecise and is
corrected. Numerical widths, injection scores, and source fits are unchanged.

### 2. Lower-V asymmetry and attitude reliability need explicit tests

The separate *Analysis of RAE-B Attitude Data* report, pages 6, 8, 15 and 30,
describes abnormal deployment and a damaged lower boom. It reports an
equilibrium near -6, -7, -40 degrees in roll, pitch, yaw, and a failure to
reproduce the observed equilibrium with reasonable simulation parameters.
Its conclusion raises possible attitude-data unreliability because later
deployment changes did not produce the expected attitude change. This is a
documented concern, not proof that every archived attitude value is wrong.

The instrument paper independently describes a lower leg deployed almost
parallel to the local vertical, an asymmetric lower V, and an offset of its
maximum-gain direction. The current opposite-upper-axis approximation does
not encode that asymmetric response. Recovering injected sources with that
same approximation cannot validate it against the actual spacecraft.

**Implication:** lower-V source results require separate shape and pointing
uncertainty tests. The recorded upper-V pointing should also be traced to its
generation method, distinguishing measured attitude from an assumed nominal
orientation. The near-constant angle to the radial direction is a clue to
investigate, not proof of either provenance.

### 3. These records offer routes to validation, not a solved XYZ frame

The 143-page postlaunch attitude-operations report describes actual sensor
performance and the MAPS/RAE-B support software. The camera report describes
a system intended to measure boom-tip angles relative to the spacecraft.
These are useful leads for attitude and physical-geometry reconstruction.
Neither the design specification alone nor the sections checked here establish
the celestial reference epoch of the archived spacecraft XYZ field. Original
tracking products and the reduction program's definitions remain necessary.

The camera report specifies a target boom-angle accuracy of +/-0.35 degrees;
this is a requirement, not demonstrated flight accuracy. The 1973 dynamics
report proposes experiments and should not be mistaken for a record that
those experiments were performed.

### 4. Deployment dates and receiver products must be distinguished

The general RAE-2 archive document, page 2, lists upper-V completion on
14 November 1973 and lower-leg deployments on 11 October and 6 November
1974. The separate attitude report says 13 November for upper completion.
The current selection begins on 15 November and 7 November respectively, so
this one-day discrepancy does not change those cutoffs. The historical event
table also contains at least one apparent year typo, reinforcing the need
to cross-check even primary documentation.

The B27602 guide is for the **burst-receiver** data sets 73-039A-02A/02B;
its sampling/layout must not be substituted for the Ryle-Vonberg format.
The original Ryle-Vonberg document in this Dropbox is byte-for-byte identical
to the format document already used. The Dropbox Alexander journal PDF is
also identical to the local copy in `Diffraction/Papers`.

## Review coverage

| File | Pages | Identified content | Status in this review |
|---|---:|---|---|
| `19710006764.pdf` | 23 | Kaiser, RAE-1 Data Displays | Introduction and coordinate/calibration section checked; RAE-1 context only |
| `19730017648.pdf` | 39 | Proposed gravity-gradient dynamics experiments in lunar orbit using RAE-B | Title and abstract checked; proposals distinguished from flight results |
| `19730021146.pdf` | 40 | Delta-95 / RAE-B Operations Summary | Title, contents, mission and tracking passages screened |
| `19740024148.pdf` | 143 | RAE-B Postlaunch Attitude Operations Analysis | Abstract/contents and keyword screening; detailed sensor/software review remains |
| `alexander.1975.pdf` | 7 | Scientific Instrumentation of the RAE-2 Satellite | Pages 1-3 rendered; geometry, beam Table 1 and surrounding explanation checked |
| `antenna_array.pdf` | 8 | Radio Astronomy Explorer 1500-ft-Long Antenna Array | Title/abstract and mechanical-calibration passages screened; not an in-flight beam map |
| `APL-03-05-Fischell.pdf` | 10 | Gravity Gradient Stabilization | Opening section checked; general background |
| `attitude.pdf` | 30 | Analysis of RAE-B Attitude Data | Relevant deployment, equilibrium and conclusion sections checked; pages 8 and 30 visually confirmed |
| `B23736-000A.pdf` | 34 | Alexander et al. RAE-2 instrumentation preprint, X-693-75-40 | Cover checked; same work as the instrument source already consulted, different document version |
| `B27602-000A.pdf` | 17 | Data User's Guide, sets 73-039A-02A/02B | Cover and introduction checked; burst-receiver scope identified |
| `cameras.pdf` | 67 | RAE-B Antenna Aspect System Final Report | Title and design-requirements section checked, page 6 visually confirmed |
| `earthnoise.pdf` | 80 | 136/400-MHz Earth-station antenna-noise temperature prediction | Title/introduction checked; ground telemetry/tracking noise, not the low-frequency science-receiver sky model |
| `RAE-B_Doc.PDF` | 26 | General RAE-2 experiment and archive description | Pages 1-3 rendered, operational dates checked |
| `RAE2_73039A01B_02B_Doc.pdf` | 28 | Ryle-Vonberg archive-format documentation | SHA-256 identity with the previously used source confirmed; prior relevant-page review retained |
| `weber.1971.pdf` | 13 | The Radio Astronomy Explorer satellite, a low-frequency observatory | Pages 3-5 checked, Figure 2/Table 1 visually confirmed; RAE-1 scope explicit |

## Consequences for the next analysis

The next model should separate RAE-1 historical comparisons from RAE-2
calibration, distinguish upper and damaged lower geometries, and propagate
pointing/shape uncertainty through injections into actual backgrounds. The
Sayre calculations and original boom-aspect/attitude products are the most
direct leads. A publication's reference to them does not mean their full
contents have been recovered here.

No new source detection or stronger sensitivity claim follows from this
document review. The earlier fits remain exploratory. The existing report,
experiment notes, and beam-table label were corrected; numerical source
analyses were not rerun. The original provenance manifest is retained as
`provenance_before_historic_review.json` before the updated manifest is written.
