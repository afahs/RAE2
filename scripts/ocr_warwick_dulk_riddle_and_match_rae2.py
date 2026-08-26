#!/usr/bin/env python
"""OCR Warwick/Dulk/Riddle UAG-42 tables and compare with RAE-2 bursts.

The OCR output is intentionally provenance-rich.  Scanned table rows are noisy,
especially leading zeros and ones in HHMM fields, so the script writes:

- raw OCR candidate rows;
- OCR rows with parse/repair quality flags;
- a comparison catalog that uses the existing manually checked overlap
  transcription where available;
- direct and shifted-time overlap summaries against the RAE-2 blind burst
  catalog.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, write_json  # noqa: E402
from scripts.plot_warwick_dulk_riddle_catalog_phase_cml import load_catalog as load_manual_catalog  # noqa: E402


DEFAULT_PDF = ROOT / "outputs/jupiter_historical_active_windows_v1/external_data/warwick_dulk_riddle_1975_uag42.pdf"
DEFAULT_MANUAL = ROOT / "configs/jupiter_warwick_dulk_riddle_1975_active_windows.csv"
DEFAULT_RAE2 = ROOT / "outputs/jupiter_upper_v_beam_single_frequency_burst_phase_v1/jupiter_blind_burst_catalog.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_warwick_dulk_riddle_full_ocr_v1"


@dataclass(frozen=True)
class HhmmCandidate:
    value: str
    cost: int


def longest_vertical_run(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape[1], dtype=int)
    for x, col in enumerate(mask.T):
        best = 0
        cur = 0
        for value in col:
            if value:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        out[x] = best
    return out


def detect_table_vertical_lines(image_path: Path) -> list[int]:
    image = Image.open(image_path).convert("L")
    arr = np.asarray(image)
    for pixel_threshold in [80, 100, 130, 160, 190]:
        runs = longest_vertical_run(arr < pixel_threshold)
        for run_threshold in [250, 200, 150, 100]:
            xs = np.where(runs > run_threshold)[0]
            groups: list[tuple[int, int]] = []
            if len(xs):
                start = prev = int(xs[0])
                for value in xs[1:]:
                    x = int(value)
                    if x - prev > 5:
                        groups.append((start, prev))
                        start = x
                    prev = x
                groups.append((start, prev))
            centers = [(lo + hi) // 2 for lo, hi in groups if hi - lo < 30]
            if len(centers) >= 9:
                return centers[:9]
    return []


def run_tesseract_tsv(image_path: Path, out_prefix: Path) -> Path:
    out_tsv = out_prefix.with_suffix(".tsv")
    if out_tsv.exists():
        return out_tsv
    ensure_dir(out_tsv.parent)
    cmd = [
        "tesseract",
        str(image_path),
        str(out_prefix),
        "--psm",
        "6",
        "-l",
        "eng",
        "tsv",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_tsv


def clean_numeric_token(value: object) -> str:
    translation = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "Q": "0",
            "D": "0",
            "G": "0",
            "C": "0",
            "I": "1",
            "l": "1",
            "|": "1",
            "S": "5",
            "s": "5",
            "B": "8",
        }
    )
    return re.sub(r"\D", "", str(value).translate(translation))


def is_valid_hhmm(value: str) -> bool:
    if len(value) != 4 or not value.isdigit():
        return False
    hour = int(value[:2])
    minute = int(value[2:])
    return 0 <= hour <= 29 and 0 <= minute < 60


def _hhmm_minutes(value: str) -> int:
    return int(value[:2]) * 60 + int(value[2:])


def _time_candidate_variants(raw_digits: str) -> list[HhmmCandidate]:
    if not raw_digits:
        return []
    chunks = set()
    if len(raw_digits) <= 4:
        chunks.add(raw_digits.zfill(4))
    else:
        for start in range(0, max(1, len(raw_digits) - 3)):
            chunks.add(raw_digits[start : start + 4])
        chunks.add(raw_digits[:4])
        chunks.add(raw_digits[-4:])

    alternatives = {
        "0": ["0"],
        "1": ["1", "0", "4"],
        "2": ["2"],
        "3": ["3"],
        "4": ["4", "1"],
        "5": ["5"],
        "6": ["6", "0"],
        "7": ["7"],
        "8": ["8"],
        "9": ["9", "0"],
    }
    candidates: dict[str, int] = {}
    for chunk in chunks:
        if len(chunk) != 4 or not chunk.isdigit():
            continue
        choices = [alternatives.get(ch, [ch]) for ch in chunk]
        for variant in product(*choices):
            value = "".join(variant)
            if not is_valid_hhmm(value):
                continue
            cost = sum(a != b for a, b in zip(chunk, value))
            candidates[value] = min(candidates.get(value, 99), cost)
    return [HhmmCandidate(value, cost) for value, cost in sorted(candidates.items(), key=lambda item: (item[1], item[0]))]


def _time_tokens(texts: list[str]) -> list[str]:
    values: list[str] = []
    for text in texts:
        digits = clean_numeric_token(text)
        if not digits:
            continue
        if len(digits) <= 4:
            values.append(digits.zfill(4))
        else:
            for start in range(0, len(digits), 4):
                chunk = digits[start : start + 4]
                if len(chunk) >= 3:
                    values.append(chunk.zfill(4))
    return values


def repair_time_pair(raw_start: str, raw_end: str, obs_start: str | None = None, obs_end: str | None = None) -> tuple[str, str, str, int]:
    start_candidates = _time_candidate_variants(raw_start)
    end_candidates = _time_candidate_variants(raw_end)
    if not start_candidates or not end_candidates:
        return raw_start, raw_end, "unrepairable", 99

    obs_start_min = _hhmm_minutes(obs_start) if obs_start and is_valid_hhmm(obs_start) else None
    obs_end_min = _hhmm_minutes(obs_end) if obs_end and is_valid_hhmm(obs_end) else None
    if obs_start_min is not None and obs_end_min is not None and obs_end_min <= obs_start_min:
        obs_end_min += 24 * 60

    best: tuple[float, str, str, int] | None = None
    for start in start_candidates:
        for end in end_candidates:
            start_min = _hhmm_minutes(start.value)
            end_min = _hhmm_minutes(end.value)
            if end_min <= start_min:
                end_min += 24 * 60
            duration = end_min - start_min
            if duration <= 0 or duration > 8 * 60:
                continue
            score = start.cost + end.cost
            if obs_start_min is not None and obs_end_min is not None:
                if start_min < obs_start_min - 30:
                    score += 5
                if end_min > obs_end_min + 30:
                    score += 5
            if raw_start == start.value and raw_end == end.value:
                score -= 0.25
            candidate = (float(score), start.value, end.value, start.cost + end.cost)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return raw_start, raw_end, "unrepairable", 99
    _, start_value, end_value, cost = best
    status = "valid_raw" if start_value == raw_start and end_value == raw_end else "repaired"
    return start_value, end_value, status, int(cost)


def parse_date_from_column(texts: list[str], year: int | None) -> tuple[int | None, int | None]:
    values = []
    for text in texts:
        for match in re.findall(r"\d+", str(text)):
            value = int(match)
            if value != year and value < 100:
                values.append(value)
    if len(values) >= 2 and 1 <= values[0] <= 12 and 1 <= values[1] <= 31:
        return values[0], values[1]
    return None, None


def parse_year_from_texts(texts: list[str]) -> int | None:
    for text in texts:
        match = re.search(r"197[2-5]", str(text))
        if match:
            return int(match.group(0))
    return None


def parse_page_rows(image_path: Path, tsv_path: Path, pdf_page: int) -> list[dict[str, object]]:
    boundaries = detect_table_vertical_lines(image_path)
    if len(boundaries) < 9:
        return []
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["level"].eq(5) & df["text"].notna()].copy()
    if df.empty:
        return []
    df["cx"] = df["left"] + df["width"] / 2.0
    df["cy"] = df["top"] + df["height"] / 2.0

    current_year = parse_year_from_texts(list(df["text"].astype(str)))

    rows: list[dict[str, object]] = []
    current_month: int | None = None
    current_day: int | None = None
    current_obs_start: str | None = None
    current_obs_end: str | None = None
    line_cols = ["block_num", "par_num", "line_num"]
    for _, group in df.groupby(line_cols, sort=True):
        cy = float(group["cy"].median())
        if cy < 300:
            continue
        columns: list[list[str]] = []
        for idx in range(8):
            lo = boundaries[idx]
            hi = boundaries[idx + 1]
            sub = group[group["cx"].between(lo - 8, hi + 8)].sort_values("left")
            columns.append(list(sub["text"].astype(str)))

        line_year = parse_year_from_texts(columns[0])
        if line_year is not None:
            current_year = line_year

        month, day = parse_date_from_column(columns[0], current_year)
        date_parsed_on_line = month is not None and day is not None
        if date_parsed_on_line:
            current_month = month
            current_day = day

        obs_tokens = _time_tokens(columns[1])
        if len(obs_tokens) >= 2:
            obs_start, obs_end, obs_status, _ = repair_time_pair(obs_tokens[0], obs_tokens[1])
            if obs_status != "unrepairable":
                current_obs_start = obs_start
                current_obs_end = obs_end

        jupiter_tokens = _time_tokens(columns[2])
        if len(jupiter_tokens) < 2:
            continue
        raw_start = jupiter_tokens[0]
        raw_end = jupiter_tokens[1]
        start, end, time_status, time_repair_cost = repair_time_pair(raw_start, raw_end, current_obs_start, current_obs_end)
        intensity_digits = clean_numeric_token(" ".join(columns[3]))
        rows.append(
            {
                "ocr_pdf_page": int(pdf_page),
                "ocr_line_y": cy,
                "year": current_year,
                "month": current_month,
                "day": current_day,
                "date_parsed_on_line": bool(date_parsed_on_line),
                "observing_start_hhmm": current_obs_start,
                "observing_end_hhmm": current_obs_end,
                "raw_event_start_hhmm": raw_start,
                "raw_event_end_hhmm": raw_end,
                "event_start_hhmm": start,
                "event_end_hhmm": end,
                "time_parse_status": time_status,
                "time_repair_cost": int(time_repair_cost),
                "intensity": int(intensity_digits) if intensity_digits.isdigit() else np.nan,
                "burstiness": " ".join(columns[4]).replace("|", "").replace("[", "").strip().upper(),
                "reported_freq_range_mhz": " ".join(columns[5]).replace("|", "").replace("[", "").strip(),
                "jupiter_longitude_range_reported": " ".join(columns[6]).replace("|", "").strip(),
                "io_range_reported": " ".join(columns[7]).replace("|", "").strip(),
                "ocr_raw_line": " | ".join(" ".join(col) for col in columns),
            }
        )
    return rows


def timestamp_from_hhmm(year: int, month: int, day: int, hhmm: str) -> pd.Timestamp:
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])
    day_add, hour = divmod(hour, 24)
    return pd.Timestamp(year=int(year), month=int(month), day=int(day)) + pd.Timedelta(days=day_add, hours=hour, minutes=minute)


def finalize_ocr_catalog(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    valid = (
        out["year"].notna()
        & out["month"].notna()
        & out["day"].notna()
        & out["event_start_hhmm"].map(lambda value: is_valid_hhmm(str(value)))
        & out["event_end_hhmm"].map(lambda value: is_valid_hhmm(str(value)))
        & out["time_parse_status"].ne("unrepairable")
    )
    out["ocr_parse_valid"] = valid
    starts = []
    ends = []
    for row in out.itertuples(index=False):
        if not bool(row.ocr_parse_valid):
            starts.append(pd.NaT)
            ends.append(pd.NaT)
            continue
        start = timestamp_from_hhmm(int(row.year), int(row.month), int(row.day), str(row.event_start_hhmm))
        end = timestamp_from_hhmm(int(row.year), int(row.month), int(row.day), str(row.event_end_hhmm))
        if end <= start:
            end += pd.Timedelta(days=1)
        starts.append(start)
        ends.append(end)
    out["event_start_time"] = starts
    out["event_end_time"] = ends
    out["duration_min"] = (pd.to_datetime(out["event_end_time"]) - pd.to_datetime(out["event_start_time"])).dt.total_seconds() / 60.0
    out["catalog_source"] = "uag42_ocr"
    out["catalog_event_id"] = [f"ocr_{idx + 1:04d}" for idx in range(len(out))]
    return out


def load_manual_overlap(path: Path) -> pd.DataFrame:
    manual = load_manual_catalog(path).copy()
    manual["catalog_source"] = "manual_overlap_transcription"
    manual["catalog_event_id"] = [f"manual_{idx + 1:03d}" for idx in range(len(manual))]
    manual["ocr_parse_valid"] = True
    manual["time_parse_status"] = "manual"
    return manual


def build_analysis_catalog(ocr_catalog: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    valid_ocr = ocr_catalog[ocr_catalog["ocr_parse_valid"].astype(bool)].copy()
    if manual.empty:
        return valid_ocr
    manual_start = pd.to_datetime(manual["event_start_time"]).min() - pd.Timedelta(days=1)
    manual_end = pd.to_datetime(manual["event_end_time"]).max() + pd.Timedelta(days=1)
    non_overlap_ocr = valid_ocr[
        (pd.to_datetime(valid_ocr["event_start_time"]) < manual_start)
        | (pd.to_datetime(valid_ocr["event_start_time"]) > manual_end)
    ].copy()
    common_cols = sorted(set(non_overlap_ocr.columns).union(manual.columns))
    return pd.concat(
        [non_overlap_ocr.reindex(columns=common_cols), manual.reindex(columns=common_cols)],
        ignore_index=True,
        sort=False,
    ).sort_values("event_start_time").reset_index(drop=True)


def load_rae2_bursts(path: Path) -> pd.DataFrame:
    bursts = pd.read_csv(path, parse_dates=["burst_start_time", "burst_end_time", "burst_peak_time"])
    return bursts.sort_values("burst_peak_time").reset_index(drop=True)


def interval_match_summary(wdr: pd.DataFrame, rae: pd.DataFrame, padding_min: float) -> pd.DataFrame:
    rows = []
    for interval in wdr.itertuples(index=False):
        start = pd.Timestamp(interval.event_start_time) - pd.Timedelta(minutes=float(padding_min))
        end = pd.Timestamp(interval.event_end_time) + pd.Timedelta(minutes=float(padding_min))
        matched = rae[rae["burst_peak_time"].between(start, end)].copy()
        if matched.empty:
            nearest_delta_min = np.nan
        else:
            nearest_delta_min = 0.0
        if matched.empty and len(rae):
            peak = rae["burst_peak_time"]
            before = (start - peak).dt.total_seconds() / 60.0
            after = (peak - end).dt.total_seconds() / 60.0
            gap = np.where(peak < start, before, np.where(peak > end, after, 0.0))
            nearest_delta_min = float(np.nanmin(np.abs(gap)))
        base = {
            "catalog_event_id": interval.catalog_event_id,
            "catalog_source": interval.catalog_source,
            "event_start_time": interval.event_start_time,
            "event_end_time": interval.event_end_time,
            "duration_min": getattr(interval, "duration_min", np.nan),
            "padding_min": float(padding_min),
            "n_matched_rae2_bursts": int(len(matched)),
            "n_matched_frequencies": int(matched["frequency_mhz"].nunique()) if len(matched) else 0,
            "nearest_rae2_gap_min": nearest_delta_min,
            "matched_frequency_mhz_list": ",".join(f"{v:.2f}" for v in sorted(matched["frequency_mhz"].unique())) if len(matched) else "",
        }
        for freq, group in matched.groupby("frequency_mhz", sort=True):
            base[f"n_rae2_{float(freq):.2f}_mhz"] = int(len(group))
        rows.append(base)
    return pd.DataFrame(rows)


def count_rae_matches(wdr: pd.DataFrame, rae: pd.DataFrame, padding_min: float) -> dict[str, object]:
    masks = []
    for interval in wdr.itertuples(index=False):
        start = pd.Timestamp(interval.event_start_time) - pd.Timedelta(minutes=float(padding_min))
        end = pd.Timestamp(interval.event_end_time) + pd.Timedelta(minutes=float(padding_min))
        masks.append(rae["burst_peak_time"].between(start, end).to_numpy(dtype=bool))
    if not masks:
        matched = np.zeros(len(rae), dtype=bool)
    else:
        matched = np.logical_or.reduce(masks)
    intervals = interval_match_summary(wdr, rae, padding_min)
    return {
        "padding_min": float(padding_min),
        "n_catalog_intervals": int(len(wdr)),
        "n_catalog_intervals_with_rae2": int((intervals["n_matched_rae2_bursts"] > 0).sum()) if len(intervals) else 0,
        "n_rae2_bursts_matched": int(matched.sum()),
        "fraction_rae2_bursts_matched": float(matched.mean()) if len(matched) else np.nan,
        "n_rae2_bursts_total": int(len(rae)),
    }


def shifted_control_summary(wdr: pd.DataFrame, rae: pd.DataFrame, padding_min: float, shifts_days: list[int]) -> pd.DataFrame:
    rows = []
    real = count_rae_matches(wdr, rae, padding_min)
    real["shift_days"] = 0
    real["is_real_catalog_time"] = True
    rows.append(real)
    for shift in shifts_days:
        shifted = wdr.copy()
        shifted["event_start_time"] = pd.to_datetime(shifted["event_start_time"]) + pd.Timedelta(days=int(shift))
        shifted["event_end_time"] = pd.to_datetime(shifted["event_end_time"]) + pd.Timedelta(days=int(shift))
        row = count_rae_matches(shifted, rae, padding_min)
        row["shift_days"] = int(shift)
        row["is_real_catalog_time"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def frequency_match_summary(wdr: pd.DataFrame, rae: pd.DataFrame, paddings: list[float]) -> pd.DataFrame:
    rows = []
    for padding in paddings:
        masks = []
        for interval in wdr.itertuples(index=False):
            start = pd.Timestamp(interval.event_start_time) - pd.Timedelta(minutes=float(padding))
            end = pd.Timestamp(interval.event_end_time) + pd.Timedelta(minutes=float(padding))
            masks.append(rae["burst_peak_time"].between(start, end).to_numpy(dtype=bool))
        matched = np.logical_or.reduce(masks) if masks else np.zeros(len(rae), dtype=bool)
        for freq, grp in rae.groupby("frequency_mhz", sort=True):
            local = matched[grp.index.to_numpy()]
            rows.append(
                {
                    "padding_min": float(padding),
                    "frequency_mhz": float(freq),
                    "n_rae2_bursts": int(len(grp)),
                    "n_matched_rae2_bursts": int(np.count_nonzero(local)),
                    "matched_fraction": float(np.mean(local)) if len(local) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_timeline(wdr: pd.DataFrame, rae: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    for interval in wdr.itertuples(index=False):
        ax.axvspan(pd.Timestamp(interval.event_start_time), pd.Timestamp(interval.event_end_time), color="#d95f02", alpha=0.12, lw=0)
    ax.scatter(
        rae["burst_peak_time"],
        rae["frequency_mhz"],
        s=14,
        c=rae["peak_local_normalized_power"],
        cmap="viridis",
        alpha=0.72,
        edgecolor="none",
    )
    ax.set_yscale("log")
    ax.set_yticks(sorted(rae["frequency_mhz"].unique()))
    ax.set_yticklabels([f"{v:.2f}" for v in sorted(rae["frequency_mhz"].unique())])
    ax.set_xlabel("UTC time")
    ax.set_ylabel("RAE-2 burst frequency (MHz)")
    ax.set_title("RAE-2 blind bursts against Warwick/Dulk/Riddle reported intervals")
    ax.grid(True, axis="y", color="0.90", lw=0.5)
    fig.tight_layout()
    path = out_dir / "warwick_dulk_riddle_rae2_temporal_alignment.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_frequency_matches(freq_summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    width = 0.24
    paddings = sorted(freq_summary["padding_min"].unique())
    freqs = sorted(freq_summary["frequency_mhz"].unique())
    x = np.arange(len(freqs))
    colors = {0.0: "#4d4d4d", 30.0: "#d95f02", 60.0: "#1b9e77"}
    for idx, padding in enumerate(paddings):
        work = freq_summary[freq_summary["padding_min"].eq(padding)].sort_values("frequency_mhz")
        ax.bar(
            x + (idx - (len(paddings) - 1) / 2) * width,
            work["n_matched_rae2_bursts"],
            width=width,
            label=f"+/- {padding:.0f} min",
            color=colors.get(float(padding)),
            alpha=0.82,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in freqs])
    ax.set_xlabel("RAE-2 frequency (MHz)")
    ax.set_ylabel("RAE-2 burst peaks inside catalog intervals")
    ax.set_title("RAE-2 bursts temporally matching Warwick/Dulk/Riddle intervals")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="0.90", lw=0.5)
    fig.tight_layout()
    path = out_dir / "warwick_dulk_riddle_rae2_overlap_counts_by_frequency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_shift_controls(shift_summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    work = shift_summary.sort_values("shift_days")
    colors = np.where(work["is_real_catalog_time"].astype(bool), "#d95f02", "0.55")
    ax.bar(work["shift_days"], work["n_rae2_bursts_matched"], color=colors, alpha=0.82)
    ax.set_xlabel("shift applied to WDR intervals (days)")
    ax.set_ylabel("matched RAE-2 burst peaks")
    ax.set_title("Shifted-time control for WDR / RAE-2 temporal overlap")
    ax.grid(True, axis="y", color="0.90", lw=0.5)
    fig.tight_layout()
    path = out_dir / "warwick_dulk_riddle_rae2_shifted_time_control.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, paths: list[Path], ocr_catalog: pd.DataFrame, analysis_wdr: pd.DataFrame, rae_overlap: pd.DataFrame, shift_summary: pd.DataFrame) -> Path:
    exact = rae_overlap[rae_overlap["padding_min"].eq(0.0)].iloc[0]
    pad30 = rae_overlap[rae_overlap["padding_min"].eq(30.0)].iloc[0]
    pad60 = rae_overlap[rae_overlap["padding_min"].eq(60.0)].iloc[0]
    controls = shift_summary[~shift_summary["is_real_catalog_time"].astype(bool)]
    lines = [
        "# Warwick/Dulk/Riddle OCR and RAE-2 Burst Alignment",
        "",
        "The UAG-42 table pages were rendered from the local NOAA PDF and OCRed with Tesseract.  The parser uses fixed table-column geometry and repairs common HHMM OCR confusions, but raw OCR rows and parse flags are retained.",
        "",
        "For the RAE-2 alignment check, the existing manually checked Warwick/Dulk/Riddle overlap transcription is used where it overlaps the RAE-2 burst-search interval.  This prevents obvious OCR digit errors from driving the timing comparison.",
        "",
        "## OCR Yield",
        "",
        f"- raw OCR candidate rows: `{len(ocr_catalog)}`",
        f"- parse-valid OCR rows: `{int(ocr_catalog['ocr_parse_valid'].sum())}`",
        f"- OCR year counts: `{ocr_catalog.groupby('year').size().to_dict()}`",
        f"- analysis catalog intervals after manual-overlap replacement: `{len(analysis_wdr)}`",
        "",
        "## RAE-2 Temporal Overlap",
        "",
        f"- exact interval matches: `{int(exact['n_rae2_bursts_matched'])}` RAE-2 burst peaks in `{int(exact['n_catalog_intervals_with_rae2'])}` WDR intervals",
        f"- +/-30 min matches: `{int(pad30['n_rae2_bursts_matched'])}` RAE-2 burst peaks in `{int(pad30['n_catalog_intervals_with_rae2'])}` WDR intervals",
        f"- +/-60 min matches: `{int(pad60['n_rae2_bursts_matched'])}` RAE-2 burst peaks in `{int(pad60['n_catalog_intervals_with_rae2'])}` WDR intervals",
        f"- shifted-control +/-30 min matched-burst range: `{int(controls['n_rae2_bursts_matched'].min())}-{int(controls['n_rae2_bursts_matched'].max())}`",
        "",
        "## Interpretation",
        "",
        "A temporal match is suggestive only if the real catalog intervals exceed shifted-time controls and if matched bursts appear in plausible low-frequency channels.  The catalog is mostly 8-37 MHz, slightly above the RAE-2 bands, so exact frequency agreement is not expected.",
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "warwick_dulk_riddle_ocr_rae2_alignment_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render_pages(pdf_path: Path, image_dir: Path, dpi: int) -> None:
    ensure_dir(image_dir)
    existing = sorted(image_dir.glob("page_*.png"))
    if len(existing) >= 56:
        return
    cmd = [
        "gs",
        "-q",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=png16m",
        f"-r{int(dpi)}",
        f"-sOutputFile={image_dir / 'page_%03d.png'}",
        str(pdf_path),
    ]
    subprocess.run(cmd, check=True)


def run_full_ocr(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = ensure_dir(args.out_dir)
    image_dir = ensure_dir(out_dir / "page_images_all")
    tsv_dir = ensure_dir(out_dir / "ocr_tsv")
    render_pages(args.pdf, image_dir, dpi=int(args.dpi))
    rows: list[dict[str, object]] = []
    for page in range(int(args.first_table_page), int(args.last_table_page) + 1):
        image = image_dir / f"page_{page:03d}.png"
        tsv = run_tesseract_tsv(image, tsv_dir / f"page_{page:03d}")
        rows.extend(parse_page_rows(image, tsv, page))
    raw = pd.DataFrame(rows)
    catalog = finalize_ocr_catalog(raw) if not raw.empty else raw
    return raw, catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--manual-overlap-catalog", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--rae2-burst-catalog", type=Path, default=DEFAULT_RAE2)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--first-table-page", type=int, default=9)
    parser.add_argument("--last-table-page", type=int, default=28)
    parser.add_argument("--match-paddings-min", type=float, nargs="+", default=[0.0, 30.0, 60.0])
    parser.add_argument("--control-shift-days", type=int, nargs="+", default=[-14, -7, -3, -1, 1, 3, 7, 14])
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    write_json(
        out_dir / "warwick_dulk_riddle_ocr_rae2_alignment_config.json",
        {
            "pdf": str(args.pdf),
            "manual_overlap_catalog": str(args.manual_overlap_catalog),
            "rae2_burst_catalog": str(args.rae2_burst_catalog),
            "dpi": int(args.dpi),
            "first_table_page": int(args.first_table_page),
            "last_table_page": int(args.last_table_page),
            "match_paddings_min": [float(v) for v in args.match_paddings_min],
            "control_shift_days": [int(v) for v in args.control_shift_days],
        },
    )

    raw, ocr_catalog = run_full_ocr(args)
    raw_path = out_dir / "warwick_dulk_riddle_ocr_candidate_rows_raw.csv"
    ocr_path = out_dir / "warwick_dulk_riddle_ocr_catalog_with_parse_flags.csv"
    raw.to_csv(raw_path, index=False)
    ocr_catalog.to_csv(ocr_path, index=False)

    manual = load_manual_overlap(args.manual_overlap_catalog)
    analysis_wdr = build_analysis_catalog(ocr_catalog, manual)
    analysis_wdr_path = out_dir / "warwick_dulk_riddle_analysis_catalog_ocr_plus_manual_overlap.csv"
    analysis_wdr.to_csv(analysis_wdr_path, index=False)

    rae = load_rae2_bursts(args.rae2_burst_catalog)
    overlap_start = max(pd.to_datetime(analysis_wdr["event_start_time"]).min(), pd.to_datetime(rae["burst_peak_time"]).min())
    overlap_end = min(pd.to_datetime(analysis_wdr["event_end_time"]).max(), pd.to_datetime(rae["burst_peak_time"]).max())
    wdr_overlap = analysis_wdr[
        pd.to_datetime(analysis_wdr["event_end_time"]).ge(overlap_start)
        & pd.to_datetime(analysis_wdr["event_start_time"]).le(overlap_end)
    ].copy()
    rae_overlap = rae[rae["burst_peak_time"].between(overlap_start, overlap_end)].copy()

    padding_rows = [count_rae_matches(wdr_overlap, rae_overlap, float(pad)) for pad in args.match_paddings_min]
    padding_summary = pd.DataFrame(padding_rows)
    padding_summary_path = out_dir / "warwick_dulk_riddle_rae2_padding_match_summary.csv"
    padding_summary.to_csv(padding_summary_path, index=False)

    interval_summaries = []
    for padding in args.match_paddings_min:
        interval_summaries.append(interval_match_summary(wdr_overlap, rae_overlap, float(padding)))
    interval_summary = pd.concat(interval_summaries, ignore_index=True)
    interval_summary_path = out_dir / "warwick_dulk_riddle_rae2_interval_match_summary.csv"
    interval_summary.to_csv(interval_summary_path, index=False)

    freq_summary = frequency_match_summary(wdr_overlap, rae_overlap, [float(v) for v in args.match_paddings_min])
    freq_summary_path = out_dir / "warwick_dulk_riddle_rae2_frequency_match_summary.csv"
    freq_summary.to_csv(freq_summary_path, index=False)

    shift_summary = shifted_control_summary(wdr_overlap, rae_overlap, 30.0, [int(v) for v in args.control_shift_days])
    shift_summary_path = out_dir / "warwick_dulk_riddle_rae2_shifted_time_control.csv"
    shift_summary.to_csv(shift_summary_path, index=False)

    paths = [
        raw_path,
        ocr_path,
        analysis_wdr_path,
        padding_summary_path,
        interval_summary_path,
        freq_summary_path,
        shift_summary_path,
        plot_timeline(wdr_overlap, rae_overlap, out_dir),
        plot_frequency_matches(freq_summary, out_dir),
        plot_shift_controls(shift_summary, out_dir),
    ]
    report = write_report(out_dir, paths, ocr_catalog, analysis_wdr, padding_summary, shift_summary)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
