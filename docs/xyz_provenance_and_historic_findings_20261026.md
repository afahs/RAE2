# Spacecraft XYZ provenance and additional historic findings

26 October 2026. This follow-up searches the 15 supplied PDFs, including
112 scanned pages now made searchable with local OCR, and follows the orbit
software trail into two additional official NASA reports. These supplemental
reports were not in the Dropbox folder. Critical passages and tables were
checked against rendered pages. Search coverage is not equivalent to a full
line-by-line review of every page.

**Main finding:** there is explicit documentary support for a Moon-centered,
Earth-equator-of-1950 spacecraft state at the GTDS interface used by the RAE-B
mission software. The tape layout alone was not the strongest available
evidence. However, the exact reduction step from those orbit products to the
Ryle-Vonberg archive remains untraced; it is not yet justified to label every
archived field B1950 or to switch the spacecraft XYZ to coordinates of date.

## What the documents actually establish

| Stage or product | Documentary evidence | Limit of the evidence |
|---|---|---|
| Tracking measurements | RAE-B Operations Summary, PDF p.26 / printed 2-9: STDN tracking using spacecraft 136-MHz range and range-rate signals, with orbit results produced at GSFC. | This is a launch-operations plan, not the complete history of the later science-archive reductions. |
| Short-arc orbit estimation | PEST supplement, PDF p.84 / printed 80: lunar orbit elements obtained from short-arc determinations using range and Doppler data. | Describes the mission software's measurement interface, not a recovered set of flight solution files. |
| Cartesian state supplied by GTDS | PEST supplement, PDF p.14 / printed 10: time, three position components in km, and three velocity components in km/s; Moon-centered and equator of 1950.0. | The explicit frame applies to this interface. A subsequent export could transform it. |
| Propagation and coordinate conversion | MAESTRO report, PDF pp.13-15 / printed 7-9: numerical trajectory methods, lunar gravity models, planetary ephemerides, and several selectable input/output frames. | No job configuration tying one output option to the receiver archive was recovered. |
| Receiver archive geometry | RAE-B archive description, PDF p.23, Attachment 3: Moon-centered spacecraft XYZ in km, equatorial axes translated to the Moon. Ryle-Vonberg document, PDF p.27, Attachment 4: X toward the vernal equinox, Z parallel to Earth's spin. | These field descriptions do not explicitly date the equinox of spacecraft XYZ. Upper-V RA/Dec is separately labelled 1950.0. |

The defensible description is therefore: ground tracking supports an orbit
fit; the RAE-B software ecosystem exchanges timed Cartesian orbit states and
propagates them; the receiver archives contain spacecraft coordinates at
sample/summary epochs. Identifying the **specific writer, interpolation, and
frame conversion used for the archive** is the remaining step. This is narrower
and more useful than saying the XYZ provenance is simply unknown.

### The strongest explicit state-vector definition

The *Parameter Estimation Supplement to MAESTRO* was developed for the RAE-B
lunar mission. Its section headed **Input Measurements from GTDS**, printed
page 10, describes the position as:

> Cartesian position components (km), selenocentric, Equator of 1950.0

It similarly defines the velocity components, and gives an eight-field input
record containing two epoch fields and six Cartesian components. The logical
unit description calls the Cartesian state **EE50**. Printed page 80 specifies
Earth's mean equator and equinox of 1950.0 for the lunar orbital elements and
identifies range/Doppler short-arc determination as their source.

This is strong evidence for B1950/EE50 **upstream spacecraft states**. It is
not evidence that the Moon-centered Earth unit vector must use that same
frame. The earlier independent comparison still places that Earth field much
closer to coordinates of date. Mixed conventions are possible, and metadata
must be followed field by field.

Source: [NASA CR-132908 / report 73-48](https://ntrs.nasa.gov/citations/19740008428),
August 1973; local file `historic_docs_review_20261026/supplemental/19740008428.pdf`.

### Why MAESTRO does not, by itself, finish the frame identification

MAESTRO receives an initial state from a separate orbit-determination program
(PDF p.124 / printed 118). Its documented trajectory options include Cowell,
Encke, integrations of orbital elements, and averaged-element methods. It
supports DE69 planetary ephemerides and lunar gravity models including an
L1 field and a JPL 15-by-8 field. These are available options; the report does
not prove which options generated any particular archived receiver sample.

Most decisively, its output menu explicitly includes both **mean equator and
equinox of 1950** and **true equator and equinox of date** (PDF p.15 / printed
9). Knowing the upstream frame therefore does not prove the final export
frame. A record of the selected output option, a reduction program listing,
or a matching original GTDS state vector would close the chain.

Source: [NASA CR-132924 / report 73-8](https://ntrs.nasa.gov/citations/19740009427),
March 1973; local file `historic_docs_review_20261026/supplemental/19740009427.pdf`.

### The explicit Brouwer passage is about RAE-1

Kaiser's *Radio Astronomy Explorer-1 Data Displays*, PDF p.5 / printed 4,
states that spacecraft positions were calculated using a Brouwer general
perturbation routine described by Kaiser (1968). It reports agreement with
GSFC Refined World Maps within +/-0.3 degrees in sub-satellite position and
+/-2 km in height. It then defines upper-V pointing from geocentric spacecraft
RA/Dec, an approximation rather than a separate pointing measurement.

Its bibliography (PDF p.22 / printed 21) identifies the underlying reference
as *The Radio Frequency Interference Program for the IBM 360*, November 1968,
NASA/Goddard contract NAS 5-9756-91, verified against the scan. This is a
useful historical lead, but it cannot be
silently substituted for the lunar RAE-2 orbit calculation. Likewise, its
quoted position-error bounds should not be assigned to RAE-2.

### The JPL XYZ in the Earth-noise report is a different quantity

The Earth-station noise prediction report uses a JPL ephemeris tape for the
Sun and Moon (PDF p.23 / printed 19), and Appendix B explains how those
coordinates enter celestial-direction calculations (PDF pp.62-64). It is a
ground tracking/telemetry antenna-noise model. Its x,y,z should not be mistaken
for the RAE-2 spacecraft trajectory. This distinction is important because it
is one of the collection's most explicit descriptions of XYZ generation.

## Other information relevant to source detection

1. **The historical occultation timing budget is not negligible.** Alexander
   et al., journal PDF p.4 / printed 368, quotes worst-case spacecraft
   ephemeris errors of +/-10 seconds and a resulting angular uncertainty of
   about +/-15 arcminutes. This is a historical worst-case estimate, not an
   empirically established Gaussian error for every current record. It is
   nevertheless enough to justify timing-mismatch injections and a bounded,
   trials-corrected timing fit. Accurate interpolation of archived positions
   does not eliminate the underlying orbit error.

2. **Earth occultations provide a non-solar validation route.** The same
   paper distinguishes high-frequency terrestrial emission that disappears
   near the predicted visible-Earth occultation from extended auroral emission
   that does not. Higher-frequency events are promising checks on timing and
   geometry; low-frequency auroral edges cannot simply be treated as the
   Earth's optical limb. This is a calibration experiment for the point-source
   search, not a claim of a new celestial-source detection. Geometry must use
   the spacecraft-to-Earth direction, with both positions in the same frame,
   rather than substituting a Moon-centered Earth unit vector unaltered.

3. **Some periodic pulses are generated by the instruments themselves.** The
   instrumentation preprint's Figure 8 caption (Dropbox B23736, PDF pp.25-26)
   explains 80-second calibration gaps every 20 minutes and weak high-frequency
   burst-receiver pulses every 144 seconds from the Ryle-Vonberg local
   oscillator when the two receivers tune to the same frequency. Such
   cross-receiver agreement is not independent evidence for a sky source.
   The description is specifically about the burst-receiver example.

4. **Eclipse-related boom motion can modulate apparent radio intensity.** The
   Ryle-Vonberg archive document, PDF p.9 / printed 7, describes approximately
   50-minute scissor-mode oscillations during lunar-shadow seasons, changing
   antenna impedance and signal level, especially near 737 and 1310 kHz. It
   distinguishes which antenna was most affected in different shadow seasons.
   This is a reason to test season/antenna-dependent nuisance behaviour,
   rather than assume all repeated orbital variation is sky structure. It
   does not establish the size of this effect in each RV channel.

5. **The ten-minute burst summaries are not native samples or literal extrema.**
   RAE-B archive Attachment 3, PDF p.23, stores time at the start of each
   ten-minute interval. Its reported minimum and maximum use histogram bins
   with at least 2% of samples; the mode is also histogram-derived. Some
   unavailable statistics are represented by zero when no qualifying bin
   exists. These products are unsuitable for resolving an individual sharp
   occultation edge and should not be decoded as ordinary native intensities.
   This does not change the current experiment's use of native RV fine data.

6. **Attitude measurements had documented selection and hardware problems.**
   The postlaunch attitude report, PDF p.98 / printed 3-31, describes a period
   in which one orbit per day was used for attitude determination and useful
   solutions were restricted by Sun/Moon illumination geometry. PDF pp.110-111
   / printed 3-43 to 3-44 discuss scanner-selection anomalies and unreliable
   or reconstructable sensor identifiers. These observations concern specific
   mission phases and do not prove later archive pointing is wrong; they do
   make the provenance and quality flags of any adopted attitude series
   important.

## Implications for the existing investigation

The explicit GTDS EE50 definition increases the documentary support for the
B1950 spacecraft hypothesis. It does **not** justify changing all fields to
B1950, because the Earth-vector test and selectable software output frames
remain relevant. Keep both spacecraft-frame tests until the archive-export
link is verified, and use geometric frame rotations appropriate to state
vectors rather than adding stellar-aberration terms.

The next decisive checks are an original GTDS state/export specification and
high-frequency Earth occultation timing on native data. Subsequent source
injections should perturb orbit timing, pointing and antenna shape; matching
the same assumed template to itself tests numerical recovery but not those
physical uncertainties. No source fits or production defaults were changed
in this documentary investigation.

## Reproducible evidence

- `historic_docs_review_20261026/inventory.json`: the original 15 Dropbox PDFs,
  exact URLs, sizes, hashes and page counts.
- `historic_docs_review_20261026/ocr/`: searchable transcriptions of five
  scanned PDFs. OCR is a search aid, not the authoritative source.
- `historic_docs_review_20261026/ocr_scanned.ps1`: local Windows OCR workflow;
  run with Windows PowerShell 5.1 using `ocr_manifest.json`.
- `historic_docs_review_20261026/supplemental/inventory.json`: two NASA reports,
  separately identified from the Dropbox collection.
- `historic_docs_review_20261026/index_documents.py`: builds a page-addressable
  index of 830 pages in 17 PDFs and a targeted search-hit list. Index coverage
  does not imply every page has been manually reviewed.
- Critical page renders are retained alongside the indexed documents.
