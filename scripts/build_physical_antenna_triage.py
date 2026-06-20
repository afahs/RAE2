#!/usr/bin/env python
"""Build physically informed upper/lower V antenna triage from channel screen outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402


def _signed_snr(row: pd.Series) -> float:
    value = pd.to_numeric(row.get("prepost_snr"), errors="coerce")
    if np.isfinite(value):
        return float(value)
    value = pd.to_numeric(row.get("constant_template_snr"), errors="coerce")
    return float(value) if np.isfinite(value) else np.nan


def build_antenna_triage(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in summary.groupby(["source_name", "frequency_mhz", "window_s"], sort=True):
        source, freq, window = keys
        upper = grp[grp["antenna"].astype(str).eq("rv1_coarse")]
        lower = grp[grp["antenna"].astype(str).eq("rv2_coarse")]
        if upper.empty or lower.empty:
            continue
        u = upper.iloc[0]
        l = lower.iloc[0]
        upper_snr = _signed_snr(u)
        lower_snr = _signed_snr(l)
        upper_abs = abs(upper_snr) if np.isfinite(upper_snr) else np.nan
        lower_abs = abs(lower_snr) if np.isfinite(lower_snr) else np.nan
        stronger = "upper_v" if upper_abs > lower_abs else "lower_v"
        ratio = upper_abs / (lower_abs + 1e-12) if np.isfinite(upper_abs) and np.isfinite(lower_abs) else np.nan
        same_sign = bool(np.sign(upper_snr) == np.sign(lower_snr)) if np.isfinite(upper_snr) and np.isfinite(lower_snr) else False
        concern = "ok"
        if source != "earth" and max(upper_abs, lower_abs) >= 5:
            if not same_sign:
                concern = "antenna_sign_conflict"
            elif stronger == "upper_v" and ratio > 1.5:
                concern = "upper_v_dominant_for_lunar_limb_source"
        rows.append(
            {
                "source_name": source,
                "frequency_mhz": float(freq),
                "window_s": float(window),
                "upper_v_signed_snr": upper_snr,
                "lower_v_signed_snr": lower_snr,
                "upper_status": u.get("screen_status", ""),
                "lower_status": l.get("screen_status", ""),
                "stronger_antenna": stronger,
                "upper_to_lower_abs_snr_ratio": ratio,
                "same_signed_response": same_sign,
                "antenna_plausibility_flag": concern,
            }
        )
    return pd.DataFrame.from_records(rows)


def add_source_level_antenna_flags(triage: pd.DataFrame) -> pd.DataFrame:
    """Add repeated upper-V dominance counts without marking unrelated rows."""

    out = triage.copy()
    out["upper_v_dominant_frequency_count"] = 0
    out["upper_v_repeated_dominance_flag"] = False
    for (source, window_s), grp in out.groupby(["source_name", "window_s"], sort=True):
        mask = (
            grp["stronger_antenna"].astype(str).eq("upper_v")
            & grp["same_signed_response"].astype(bool)
            & (pd.to_numeric(grp["upper_to_lower_abs_snr_ratio"], errors="coerce") > 1.2)
            & (pd.to_numeric(grp["upper_v_signed_snr"], errors="coerce").abs() >= 5)
        )
        count = int(mask.sum())
        out.loc[grp.index, "upper_v_dominant_frequency_count"] = count
        if source != "earth" and count >= 2:
            out.loc[grp.index[mask], "upper_v_repeated_dominance_flag"] = True
    return out


def write_report(out_dir: Path, triage: pd.DataFrame) -> None:
    lines = [
        "# Antenna Pair Triage",
        "",
        "This table compares upper V and lower V for the same source/frequency/window.",
        "",
        "Physical heuristic:",
        "",
        "- lower V is preferred for lunar-limb occultation because it points toward the Moon;",
        "- upper-V-only or upper-V-dominant rows are downgraded unless geometry later justifies them;",
        "- same-sign response in both antennas is less concerning than opposite-sign response;",
        "- repeated upper-V dominance across multiple frequencies is concerning;",
        "- Earth is treated as an empirical calibration source, not a candidate.",
        "",
        "## Counts",
        "",
    ]
    counts = triage.groupby(["source_name", "antenna_plausibility_flag"]).size().reset_index(name="n")
    lines.append(_markdown_table(counts) if not counts.empty else "No rows.")
    lines.extend(["", "## Most Relevant Fornax A Rows", ""])
    fornax = triage[triage["source_name"].eq("fornax_a")].copy()
    if not fornax.empty:
        fornax["max_abs_snr"] = fornax[["upper_v_signed_snr", "lower_v_signed_snr"]].abs().max(axis=1)
        cols = [
            "frequency_mhz",
            "window_s",
            "upper_v_signed_snr",
            "lower_v_signed_snr",
            "stronger_antenna",
            "same_signed_response",
            "antenna_plausibility_flag",
        ]
        lines.append(_markdown_table(fornax.sort_values("max_abs_snr", ascending=False)[cols].head(12)))
    (out_dir / "antenna_pair_triage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    work = frame.copy()
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    cols = list(work.columns)
    widths = {col: max(len(str(col)), *(len(str(v)) for v in work[col])) for col in cols}
    lines = [
        "| " + " | ".join(str(col).ljust(widths[col]) for col in cols) + " |",
        "| " + " | ".join("-" * widths[col] for col in cols) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(ROOT / "outputs/no_trendline_channel_screen_v1/no_trendline_channel_summary.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/physically_informed_triage_v1"))
    args = parser.parse_args()
    summary = read_table(args.summary, low_memory=False)
    out_dir = ensure_dir(args.out_dir)
    triage = add_source_level_antenna_flags(build_antenna_triage(summary))
    triage.to_csv(out_dir / "antenna_pair_triage.csv", index=False)
    write_json(
        out_dir / "antenna_pair_triage_config.json",
        {"summary": str(args.summary), "software_versions": software_versions()},
    )
    write_report(out_dir, triage)
    print(out_dir / "antenna_pair_triage.csv")
    print(out_dir / "antenna_pair_triage_report.md")


if __name__ == "__main__":
    main()
