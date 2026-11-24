# Data Requirements

The pipeline expects a local CSV with one row per measurement time and
frequency band. It does not download or ship RAE-2 data.

## Required Columns

Core columns:

- `time`: timestamp parseable by `pandas.to_datetime`
- `frequency_band`: integer RAE-2 band number
- `position_x`, `position_y`, `position_z`: spacecraft position vector in km,
  interpreted as Moon-centered
- `earth_unit_vector_x`, `earth_unit_vector_y`, `earth_unit_vector_z`: Moon to
  Earth unit vector
- `right_ascension`, `declination`: geometry columns preserved during ingest

Default power columns:

- `rv1_coarse`
- `rv2_coarse`

Additional antenna/power columns can be used with `--value-columns`, for
example `rv1_fine rv2_fine`, if the CSV contains them.

## Frequency Bands

The package maps RAE-2 bands to MHz as:

| Band | MHz |
| ---: | --: |
| 1 | 0.45 |
| 2 | 0.70 |
| 3 | 0.90 |
| 4 | 1.31 |
| 5 | 2.20 |
| 6 | 3.93 |
| 7 | 4.70 |
| 8 | 6.55 |
| 9 | 9.18 |

## Geometry Assumptions

- The spacecraft vector is treated as Moon-centered and inertial.
- The default source frame is FK4 with equinox B1950.
- A source is occulted when its spacecraft-frame direction is inside the lunar
  disk centered on `-spacecraft_position`.
- Moving-body positions are generated with Astropy's ephemeris support.

These assumptions should be checked against the provenance of any CSV used for
science-grade results.

## Power quality policy

Ingest now records `quality_policy_version=instrument_limits_v2`. Percentile
high-power and jump flags are advisory: either can describe a real burst or
occultation. Non-finite/nonpositive power, malformed geometry, duplicate times,
and gap flags still invalidate samples. A known instrument saturation limit
can be supplied as `IngestOptions(saturation_power=...)` or
`validate_timeseries(..., saturation_power=...)`; no limit is inferred from
the brightest observed sample. Quantile clipping in `strict_power_mask` is
disabled by default and remains an explicit sensitivity option.

Regenerate cleaned tables from the retained raw power before comparing the
new policy with old results. Old `is_valid` masks cannot recover samples they
previously rejected. Preserve the old tables for comparison.

## Minimal CSV Example

```csv
time,frequency_band,position_x,position_y,position_z,earth_unit_vector_x,earth_unit_vector_y,earth_unit_vector_z,right_ascension,declination,rv1_coarse,rv2_coarse
1974-01-01 00:00:00,1,2000,0,0,1,0,0,0,0,1.0,1.1
1974-01-01 00:01:00,1,2001,0,0,1,0,0,0,0,1.2,1.3
```

