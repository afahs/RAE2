# KaiserMethod Occultation Stacking (RAE-2)

Implements the M. L. Kaiser (1977) *occultation stacking* method for detecting
planetary low-frequency radio emission in RAE-2 total-power measurements. The
pipeline reduces each occultation to a 1-bit sign of the pre/post power change
and stacks many events to form a detection statistic per channel.

This method is **1-bit stacking of before/after averages** with confusion
rejection and candidate listing based on **>= 0.5 dB** drops across **>= 3
consecutive channels**. No flux calibration is performed.

## Inputs

Measurements CSV:
- Required columns: time, channel_id, power (or equivalent).
- Optional: center_frequency_mhz (if missing, a channel map is used).

Events CSV:
- Required columns: event_time, event_type (DISAPPEARANCE or REAPPEARANCE).
- Optional: event_id, planet, and precomputed separation columns.

Separation columns (for confusion rejection):
- `sun_sep_deg`, `earth_sep_deg`, and `*_sep_deg` for other planets.
- Aliases supported: `*_limb_sep_deg`, `*_limb_distance_deg`.

If separations are not provided, you can compute them using SPICE by supplying
kernels and an observer ID (see below).

## Quick Start

```
python RAE2Agent/KaiserMethod/occultation_stack.py \
  --events path/to/predicted_events.csv \
  --planet SATURN \
  --window-min 4 \
  --min-samples 2 \
  --outdir results/
```

### Using the interpolated master file
The existing RAE2 interpolated master file uses `time`, `frequency_band`, and
`rv2_coarse` columns. The defaults in `occultation_stack.py` match this, so a
minimal invocation is:

```
python RAE2Agent/KaiserMethod/occultation_stack.py \
  --data /global/cfs/projectdirs/m4895/RAE2Data/interpolatedRAE2MasterFile.csv \
  --events path/to/predicted_events.csv
```

## Outputs

The CLI writes to `--outdir`:
- `event_channel_deltas.csv`: per-event, per-channel mean-before, mean-after,
  delta, and sign bit.
- `stack_statistics.csv`: `S` per channel (sum of sign bits / sqrt(N)).
- `candidate_events.csv`: runs of >= 3 consecutive channels with +1 sign and
  >= 0.5 dB drop.
- `plots/`: intensity vs time plots per candidate run (and optional dynamic
  spectra).

A detection claim requires positive `S` across multiple adjacent channels and
visual confirmation of step-like changes at the predicted event times.

## Confusion Rejection

Events are discarded if any other body is within:
- **3 deg** of the target direction (Sun/planets)
- **5 deg** for Earth

By default this uses precomputed separation columns. If they are missing, you
can compute separations with SPICE:

```
python RAE2Agent/KaiserMethod/occultation_stack.py \
  --events path/to/predicted_events.csv \
  --spice-kernel path/to/kernels.bsp \
  --spice-observer RAE-2
```

The SPICE computation estimates the limb-direction separation using the
apparent target direction as seen from the spacecraft.

## Frequency-dependent Windows

You can enlarge the averaging window for low frequencies to mitigate
interplanetary scattering:

```
--lowfreq-threshold-mhz 1.0 --lowfreq-window-min 8
```

Or supply a per-frequency configuration CSV with:
`min_freq_mhz,max_freq_mhz,window_min`.

## Antenna Extension Filter

To exclude early data (e.g., before the antenna extension date), use:

```
--antenna-extension-date 1974-11-06
```

This is an alias for `--event-start` and filters predicted events accordingly.

## Example (Synthetic Data)

Synthetic inputs and outputs are included:
- `RAE2Agent/KaiserMethod/examples/synthetic_measurements.csv`
- `RAE2Agent/KaiserMethod/examples/synthetic_events.csv`
- `RAE2Agent/KaiserMethod/examples/outputs/`

These demonstrate the pipeline end-to-end, including candidate plots.
