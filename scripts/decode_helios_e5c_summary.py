#!/usr/bin/env python3
"""Decode the recovered Helios-1 E5c 10-minute summary tape image.

The SPDF file is a recovered, unreviewed IBM-360 binary product.  Each
physical block starts with a two-byte big-endian payload length.  The payload
contains 520-byte logical records, each comprising 130 IBM hexadecimal
single-precision values: date code, milliseconds of day, and eight archived
summary fields for each of sixteen E5c frequency channels.

The decoder deliberately preserves the archived fields without asserting an
absolute antenna-temperature calibration.  Downstream work uses ordinal,
within-channel contrasts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_URL = (
    "https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/"
    "26pt5KHz_3MHz_radio_wave/10min_averaged_summaries/"
    "DATA2_DR004795_DR004795_20080611_083141/dr004795_f00001.phys.1"
)
SOURCE_SHA1_URL = (
    "https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/"
    "26pt5KHz_3MHz_radio_wave/10min_averaged_summaries/"
    "DATA2_DR004795_DR004795_20080611_083141/SHA1SUM"
)
WARNING_URL = (
    "https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/"
    "26pt5KHz_3MHz_radio_wave/10min_averaged_summaries/"
    "AA_WARNING_recovered_but_unreviewed_dataset.txt"
)
METADATA_URL = (
    "https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/"
    "26pt5KHz_3MHz_radio_wave/10min_averaged_summaries/SORA-00003_nmc.html"
)

FREQUENCIES_KHZ = np.array(
    [26.5, 50.0, 65.0, 85.0, 115.0, 150.0, 195.0, 255.0,
     340.0, 445.0, 585.0, 765.0, 1010.0, 1320.0, 2280.0, 3000.0],
    dtype=float,
)

LOGICAL_RECORD_BYTES = 520
WORDS_PER_RECORD = 130
CHANNEL_FIELDS = (
    "channel_id",
    "nonzero_count",
    "two_percent_min_10logt",
    "two_percent_max_10logt",
    "mode_10logt",
    "median_10logt",
    "scaled_log_sigma",
    "linear_sum",
)


@dataclass(frozen=True)
class DecodeAudit:
    physical_blocks: int
    logical_records: int
    unique_date_codes: int
    first_time: str
    last_time: str
    monotonic_time_violations: int
    wrapped_negative_milliseconds_records: int
    invalid_block_lengths: int
    invalid_channel_layout_records: int
    trailing_bytes: int


def ibm32_words_to_float(words: np.ndarray) -> np.ndarray:
    """Convert big-endian IBM hexadecimal 32-bit words to float64."""

    values = np.asarray(words, dtype=np.uint32)
    sign = np.where((values & np.uint32(0x80000000)) != 0, -1.0, 1.0)
    exponent = ((values >> np.uint32(24)) & np.uint32(0x7F)).astype(np.int16) - 64
    fraction = (values & np.uint32(0x00FFFFFF)).astype(np.float64) / float(1 << 24)
    out = sign * fraction * np.power(16.0, exponent.astype(np.float64))
    out[values == 0] = 0.0
    return out


def _date_code_to_midnight(date_code: int) -> datetime:
    text = f"{date_code:06d}"
    return datetime.strptime("19" + text, "%Y%m%d").replace(tzinfo=timezone.utc)


def archived_time(date_value: float, milliseconds_value: float) -> datetime:
    date_code = int(round(float(date_value)))
    milliseconds = int(round(float(milliseconds_value)))
    if date_code < 700101 or date_code > 991231:
        raise ValueError(f"invalid YYMMDD date code {date_code}")
    # Six records in the recovered tape encode the last minutes of a coded day
    # as negative offsets from the following midnight.  Modulo normalization
    # restores their documented day-of-date ordering while the raw field is
    # retained and the anomaly is counted in the decoder audit.
    if -86_400_000 < milliseconds < 0:
        milliseconds += 86_400_000
    if milliseconds < 0 or milliseconds >= 86_401_000:
        raise ValueError(f"invalid milliseconds of day {milliseconds}")
    return _date_code_to_midnight(date_code) + timedelta(milliseconds=milliseconds)


def sha1_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.replace(destination)


def fetch_text(url: str, destination: Path) -> str:
    if not destination.exists():
        download(url, destination)
    return destination.read_text(encoding="utf-8", errors="replace")


def expected_sha1_from_text(text: str, filename: str) -> str | None:
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and fields[1].lstrip("*") == filename:
            candidate = fields[0].lower()
            if len(candidate) == 40 and all(char in "0123456789abcdef" for char in candidate):
                return candidate
    return None


def decode_file(path: Path) -> tuple[pd.DataFrame, DecodeAudit]:
    rows: list[dict[str, object]] = []
    block_count = 0
    record_count = 0
    invalid_lengths = 0
    invalid_layouts = 0
    trailing = 0
    previous_time: datetime | None = None
    monotonic_violations = 0
    wrapped_negative_milliseconds = 0

    with path.open("rb") as stream:
        while True:
            prefix = stream.read(2)
            if not prefix:
                break
            if len(prefix) != 2:
                trailing += len(prefix)
                break
            block_bytes = struct.unpack(">H", prefix)[0]
            if block_bytes <= 0 or block_bytes % LOGICAL_RECORD_BYTES != 0:
                invalid_lengths += 1
                raise ValueError(
                    f"invalid physical block payload length {block_bytes} at byte {stream.tell() - 2}"
                )
            payload = stream.read(block_bytes)
            if len(payload) != block_bytes:
                raise EOFError(
                    f"truncated physical block {block_count + 1}: expected {block_bytes}, got {len(payload)}"
                )
            block_count += 1
            for record_offset in range(0, block_bytes, LOGICAL_RECORD_BYTES):
                raw_record = payload[record_offset:record_offset + LOGICAL_RECORD_BYTES]
                words = np.frombuffer(raw_record, dtype=">u4").astype(np.uint32)
                if words.size != WORDS_PER_RECORD:
                    raise ValueError(f"logical record has {words.size} words, expected {WORDS_PER_RECORD}")
                values = ibm32_words_to_float(words)
                if int(round(float(values[1]))) < 0:
                    wrapped_negative_milliseconds += 1
                stamp = archived_time(values[0], values[1])
                if previous_time is not None and stamp < previous_time:
                    monotonic_violations += 1
                previous_time = stamp
                date_code = int(round(values[0]))
                milliseconds = int(round(values[1]))
                channel_values = values[2:].reshape(16, 8)
                channel_ids = np.rint(channel_values[:, 0]).astype(int)
                valid_layout = np.array_equal(channel_ids, np.arange(1, 17))
                if not valid_layout:
                    invalid_layouts += 1
                for index, fields in enumerate(channel_values):
                    row: dict[str, object] = {
                        "time": stamp,
                        "date_code": date_code,
                        "milliseconds_of_day": milliseconds,
                        "milliseconds_of_day_normalized": milliseconds % 86_400_000,
                        "physical_block": block_count,
                        "logical_record": record_count,
                        "channel_layout_valid": bool(valid_layout),
                        "frequency_khz": float(FREQUENCIES_KHZ[index]),
                        "frequency_mhz": float(FREQUENCIES_KHZ[index] / 1000.0),
                    }
                    for name, value in zip(CHANNEL_FIELDS, fields, strict=True):
                        row[name] = float(value)
                    row["channel_id"] = int(round(float(row["channel_id"])))
                    rows.append(row)
                record_count += 1

        trailing_data = stream.read()
        trailing += len(trailing_data)

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError("decoded table is empty")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    record_times = frame.drop_duplicates("logical_record")["time"].sort_values()
    audit = DecodeAudit(
        physical_blocks=block_count,
        logical_records=record_count,
        unique_date_codes=int(frame["date_code"].nunique()),
        first_time=record_times.iloc[0].isoformat(),
        last_time=record_times.iloc[-1].isoformat(),
        monotonic_time_violations=monotonic_violations,
        wrapped_negative_milliseconds_records=wrapped_negative_milliseconds,
        invalid_block_lengths=invalid_lengths,
        invalid_channel_layout_records=invalid_layouts,
        trailing_bytes=trailing,
    )
    return frame, audit


def channel_quality(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (channel_id, frequency_khz), group in frame.groupby(
        ["channel_id", "frequency_khz"], sort=True
    ):
        upper = pd.to_numeric(group["two_percent_max_10logt"], errors="coerce")
        median = pd.to_numeric(group["median_10logt"], errors="coerce")
        counts = pd.to_numeric(group["nonzero_count"], errors="coerce")
        finite_upper = upper[np.isfinite(upper)]
        modal_fraction = (
            float(finite_upper.value_counts(dropna=False).iloc[0] / len(finite_upper))
            if len(finite_upper)
            else math.nan
        )
        distinct_upper = int(finite_upper.nunique())
        constant = bool(distinct_upper <= 3 or (np.isfinite(modal_fraction) and modal_fraction >= 0.95))
        sparse = bool(
            len(group) == 0
            or float(np.mean(np.isfinite(upper))) < 0.90
            or float(np.nanmedian(counts)) <= 0
        )
        rows.append(
            {
                "channel_id": int(channel_id),
                "frequency_khz": float(frequency_khz),
                "frequency_mhz": float(frequency_khz) / 1000.0,
                "n_records": int(len(group)),
                "finite_upper_fraction": float(np.mean(np.isfinite(upper))),
                "finite_median_fraction": float(np.mean(np.isfinite(median))),
                "median_nonzero_count": float(np.nanmedian(counts)),
                "distinct_upper_values": distinct_upper,
                "upper_modal_fraction": modal_fraction,
                "upper_robust_span": float(np.nanpercentile(upper, 95) - np.nanpercentile(upper, 5)),
                "constant_or_floor_flag": constant,
                "sparse_flag": sparse,
                "structural_flag": bool(~group["channel_layout_valid"].all()),
                "science_eligible_without_event_selection": bool(
                    not constant and not sparse and group["channel_layout_valid"].all()
                ),
            }
        )
    return pd.DataFrame(rows)


def nearest_timestamp(frame: pd.DataFrame, target: str) -> tuple[str, float]:
    target_time = pd.Timestamp(target, tz="UTC")
    times = frame["time"].drop_duplicates().sort_values()
    delta = (times - target_time).abs()
    position = int(np.argmin(delta.to_numpy()))
    nearest = times.iloc[position]
    return nearest.isoformat(), float((nearest - target_time).total_seconds())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--expected-blocks", type=int, default=309)
    parser.add_argument("--expected-records", type=int, default=18_828)
    parser.add_argument("--expected-date-codes", type=int, default=297)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = args.source or args.raw_dir / "dr004795_f00001.phys.1"
    if not source.exists():
        print(f"downloading {args.source_url}", flush=True)
        download(args.source_url, source)

    sha_text = fetch_text(SOURCE_SHA1_URL, args.raw_dir / "SOURCE_SHA1SUM.txt")
    warning_text = fetch_text(WARNING_URL, args.raw_dir / "SPDF_UNREVIEWED_WARNING.txt")
    fetch_text(METADATA_URL, args.raw_dir / "SORA-00003_nmc.html")
    observed_sha1 = sha1_file(source)
    expected_sha1 = expected_sha1_from_text(sha_text, source.name)
    if expected_sha1 is not None and observed_sha1 != expected_sha1:
        raise ValueError(f"SHA1 mismatch: observed {observed_sha1}, expected {expected_sha1}")

    print(f"decoding {source}", flush=True)
    frame, audit = decode_file(source)
    expected = {
        "physical_blocks": args.expected_blocks,
        "logical_records": args.expected_records,
        "unique_date_codes": args.expected_date_codes,
    }
    actual = {
        "physical_blocks": audit.physical_blocks,
        "logical_records": audit.logical_records,
        "unique_date_codes": audit.unique_date_codes,
    }
    if actual != expected:
        raise ValueError(f"decoder invariants disagree: actual={actual}, expected={expected}")
    if (
        audit.monotonic_time_violations
        or audit.invalid_channel_layout_records
        or audit.invalid_block_lengths
        or audit.trailing_bytes
    ):
        raise ValueError(f"structural decoder audit failed: {audit}")

    targets = {
        "episode_54": nearest_timestamp(frame, "1975-01-18 09:24:48"),
        "episode_55": nearest_timestamp(frame, "1975-01-20 10:07:24"),
    }
    expected_targets = {
        "episode_54": ("1975-01-18T09:25:00.080000+00:00", 12.08),
        "episode_55": ("1975-01-20T10:10:23.520000+00:00", 179.52),
    }
    for key, (stamp, delta_s) in targets.items():
        expected_stamp, expected_delta = expected_targets[key]
        if stamp != expected_stamp or not np.isclose(delta_s, expected_delta, atol=0.001):
            raise ValueError(f"{key} timestamp invariant failed: {(stamp, delta_s)}")

    decoded_path = args.out_dir / "helios_e5c_decoded.parquet"
    frame.to_parquet(decoded_path, index=False, compression="zstd")
    quality = channel_quality(frame)
    quality.to_csv(args.out_dir / "helios_channel_quality.csv", index=False)
    audit_payload = {
        **audit.__dict__,
        "source": str(source),
        "source_url": args.source_url,
        "source_bytes": source.stat().st_size,
        "source_sha1": observed_sha1,
        "expected_source_sha1": expected_sha1,
        "sha1_verified": expected_sha1 is not None and observed_sha1 == expected_sha1,
        "spdf_warning_present": bool(warning_text.strip()),
        "spdf_warning_sha256": hashlib.sha256(warning_text.encode("utf-8")).hexdigest(),
        "frequency_khz": FREQUENCIES_KHZ.tolist(),
        "channel_fields": list(CHANNEL_FIELDS),
        "target_timestamp_checks": {
            key: {"nearest_time": value[0], "delta_s": value[1]}
            for key, value in targets.items()
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "helios_e5c_decoder_audit.json").write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(actual, sort_keys=True), flush=True)
    print(f"wrote {decoded_path}", flush=True)


if __name__ == "__main__":
    main()
