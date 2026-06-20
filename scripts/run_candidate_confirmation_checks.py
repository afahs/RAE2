#!/usr/bin/env python
"""Confirmation-oriented checks for Sun, Jupiter, and Fornax A candidates."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rylevonberg.ingest import IngestOptions, ingest_csv
from rylevonberg.quality import event_window_quality
from rylevonberg.stacking import aligned_profiles, stack_profiles
from rylevonberg.util import ensure_dir, write_json


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CandidateSet:
    source_name: str
    label: str
    events_path: Path
    clean_path: Path | None
    use_raw_full_dataset: bool
    channels: tuple[tuple[int, str], ...]
    windows: tuple[float, ...]


def _log(message: str) -> None:
    print(f"[candidate-checks] {message}", flush=True)


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _load_clean(args: argparse.Namespace, candidate: CandidateSet) -> pd.DataFrame:
    if candidate.use_raw_full_dataset:
        _log(f"ingesting full dataset for {candidate.label}: {args.data}")
        clean, _report = ingest_csv(
            args.data,
            IngestOptions(
                start_time=None,
                end_time=None,
                value_columns=("rv1_coarse", "rv2_coarse"),
                gap_factor=args.gap_factor,
                artifact_sigma=args.artifact_sigma,
            ),
        )
        return clean
    if candidate.clean_path is None:
        raise ValueError(f"{candidate.label} requires either a clean CSV or full-dataset ingest")
    _log(f"reading cleaned table for {candidate.label}: {candidate.clean_path}")
    return _read(candidate.clean_path, parse_dates=["time"])


def _template_aligned_peak(profiles: pd.DataFrame) -> tuple[float, float]:
    if profiles.empty:
        return np.nan, np.nan
    tmp = profiles.copy()
    tmp["template_aligned_value"] = tmp["profile_value"].astype(float) * tmp["template"].astype(float)
    binned = tmp.groupby("t_bin_sec", dropna=False)["template_aligned_value"].mean().reset_index()
    if binned.empty:
        return np.nan, np.nan
    idx = binned["template_aligned_value"].abs().idxmax()
    row = binned.loc[idx]
    return float(row["t_bin_sec"]), float(row["template_aligned_value"])


def _month_block_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return pd.DataFrame()
    prof = profiles.copy()
    prof["month_block"] = pd.to_datetime(prof["predicted_event_time"]).dt.strftime("%Y-%m")
    _stack, summary = stack_profiles(
        prof,
        by=["source_name", "frequency_band", "antenna", "month_block"],
        n_bootstrap=0,
    )
    return summary


def _profile_table(profiles: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if profiles.empty:
        return pd.DataFrame()
    prof = profiles.copy()
    prof["template_aligned_value"] = prof["profile_value"].astype(float) * prof["template"].astype(float)
    rows = []
    for keys, grp in prof.groupby([*by, "t_bin_sec"], dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip([*by, "t_bin_sec"], keys))
        rows.append(
            {
                **meta,
                "n_samples": int(len(grp)),
                "n_events": int(grp["event_id"].nunique()),
                "mean_profile": float(grp["profile_value"].mean()),
                "median_profile": float(grp["profile_value"].median()),
                "mean_template_aligned": float(grp["template_aligned_value"].mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_profile(profile: pd.DataFrame, out: Path, title: str) -> None:
    if profile.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for key, grp in profile.groupby("event_type", dropna=False, sort=True) if "event_type" in profile.columns else [("combined", profile)]:
        ax.plot(grp["t_bin_sec"], grp["mean_template_aligned"], label=str(key), linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Seconds from predicted event")
    ax.set_ylabel("Template-aligned normalized residual")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _analyze_channel(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    source_name: str,
    freq: int,
    antenna: str,
    window: float,
    out_dir: Path,
    bin_seconds: float,
    n_bootstrap: int,
    baseline_mode: str,
    sideband_exclusion_seconds: float,
) -> dict[str, object]:
    sub = events[
        events["source_name"].astype(str).eq(source_name)
        & events["frequency_band"].astype(int).eq(int(freq))
        & events["antenna"].astype(str).eq(str(antenna))
    ].copy()
    if sub.empty:
        return {
            "source_name": source_name,
            "frequency_band": freq,
            "antenna": antenna,
            "window_s": window,
            "n_predicted_events": 0,
            "status": "not_evaluated",
            "decision_reason": "no predicted events",
        }

    _log(f"{source_name} band {freq} {antenna} window {int(window)}s: extracting profiles for {len(sub)} events")
    profiles = aligned_profiles(
        clean,
        sub,
        window_seconds=window,
        bin_seconds=bin_seconds,
        normalize=True,
        baseline_mode=baseline_mode,
        sideband_exclusion_seconds=sideband_exclusion_seconds,
    )
    stem = f"{source_name}_band{int(freq)}_{antenna}_{int(window)}s"
    profiles_path = out_dir / f"{stem}_binned_profile.csv"
    combined_profile = _profile_table(profiles, ["source_name", "frequency_band", "antenna", "event_type"])
    combined_profile.to_csv(profiles_path, index=False)
    _plot_profile(combined_profile, out_dir / f"{stem}_event_type_profile.png", f"{source_name} band {freq} {antenna} {int(window)}s")

    _stack, summary = stack_profiles(profiles, by=["source_name", "frequency_band", "antenna"], n_bootstrap=n_bootstrap)
    _etype_stack, event_type_summary = stack_profiles(
        profiles,
        by=["source_name", "frequency_band", "antenna", "event_type"],
        n_bootstrap=n_bootstrap,
    )
    month_summary = _month_block_summary(profiles)
    quality = event_window_quality(clean, sub, window)
    event_type_summary.to_csv(out_dir / f"{stem}_event_type_summary.csv", index=False)
    month_summary.to_csv(out_dir / f"{stem}_month_block_summary.csv", index=False)
    quality.to_csv(out_dir / f"{stem}_quality.csv", index=False)

    row = summary.iloc[0].to_dict() if not summary.empty else {}
    peak_offset, peak_value = _template_aligned_peak(profiles)
    clean_fraction = float(quality["primary_quality_failure"].fillna("").astype(str).eq("").mean()) if not quality.empty else np.nan
    primary_failure = ""
    if not quality.empty:
        failures = quality["primary_quality_failure"].fillna("").astype(str)
        failures = failures[failures.ne("")]
        primary_failure = str(failures.value_counts().idxmax()) if not failures.empty else ""

    etype_snrs = {}
    for _, erow in event_type_summary.iterrows():
        etype_snrs[str(erow["event_type"])] = float(erow["stacked_snr"])
    dis_snr = etype_snrs.get("disappearance", np.nan)
    rep_snr = etype_snrs.get("reappearance", np.nan)
    same_sign_event_types = bool(np.sign(dis_snr) == np.sign(rep_snr)) if np.isfinite(dis_snr) and np.isfinite(rep_snr) else False

    month_snrs = pd.to_numeric(month_summary.get("stacked_snr", pd.Series(dtype=float)), errors="coerce").dropna()
    full_snr = float(row.get("stacked_snr", np.nan))
    full_amp = float(row.get("stacked_amplitude", np.nan))
    month_amp = pd.to_numeric(month_summary.get("stacked_amplitude", pd.Series(dtype=float)), errors="coerce").dropna()
    month_sign_fraction = float((np.sign(month_amp) == np.sign(full_amp)).mean()) if np.isfinite(full_amp) and not month_amp.empty else np.nan
    max_month_abs_snr = float(month_snrs.abs().max()) if not month_snrs.empty else np.nan
    max_month_snr_fraction = float(max_month_abs_snr / abs(full_snr)) if np.isfinite(max_month_abs_snr) and np.isfinite(full_snr) and abs(full_snr) > 0 else np.nan

    status = "unresolved"
    reasons = []
    if abs(full_snr) < 3:
        status = "not_detected"
        reasons.append("stacked SNR below 3")
    if np.isfinite(peak_offset) and abs(peak_offset) > 120:
        reasons.append(f"template-aligned peak offset {peak_offset:.0f}s")
    if np.isfinite(month_sign_fraction) and month_sign_fraction < 0.6:
        reasons.append(f"month sign fraction {month_sign_fraction:.2f}")
    if np.isfinite(max_month_snr_fraction) and max_month_snr_fraction > 0.8:
        reasons.append(f"one month can approach full-stack SNR fraction {max_month_snr_fraction:.2f}")
    if np.isfinite(clean_fraction) and clean_fraction < 0.5:
        reasons.append(f"low clean fraction {clean_fraction:.2f}")
    if same_sign_event_types and abs(full_snr) >= 3 and not reasons:
        status = "repeatable_candidate"
        reasons.append("event types have same sign and diagnostics are acceptable")
    elif abs(full_snr) >= 3 and not reasons:
        status = "candidate_needs_event_type_review"
        reasons.append("stack significant but event-type sign is not jointly established")
    elif status != "not_detected":
        status = "diagnostic_caution"

    return {
        "source_name": source_name,
        "frequency_band": int(freq),
        "frequency_mhz": float(row.get("frequency_mhz", np.nan)),
        "antenna": antenna,
        "window_s": float(window),
        "n_predicted_events": int(len(sub)),
        "n_profile_events": int(row.get("n_events", 0)),
        "stacked_amplitude": full_amp,
        "stacked_snr": full_snr,
        "bootstrap_std": float(row.get("bootstrap_std", np.nan)),
        "jackknife_std": float(row.get("jackknife_std", np.nan)),
        "disappearance_snr": dis_snr,
        "reappearance_snr": rep_snr,
        "event_type_same_sign": same_sign_event_types,
        "template_aligned_peak_offset_s": peak_offset,
        "template_aligned_peak_value": peak_value,
        "n_month_blocks": int(month_summary["month_block"].nunique()) if "month_block" in month_summary else 0,
        "month_sign_fraction": month_sign_fraction,
        "max_month_abs_snr": max_month_abs_snr,
        "max_month_snr_fraction": max_month_snr_fraction,
        "clean_fraction": clean_fraction,
        "primary_quality_failure": primary_failure,
        "status": status,
        "decision_reason": "; ".join(reasons),
    }


def _candidate_sets(args: argparse.Namespace) -> list[CandidateSet]:
    sun_scored = _read(ROOT / "outputs/sun_whole_dataset_validation_highcontrols_bands4_8/summary/sun_whole_dataset_scored_stacks.csv")
    sun_channels = tuple(
        (int(row["frequency_band"]), str(row["antenna"]))
        for _, row in sun_scored[sun_scored["status"].eq("candidate")].iterrows()
    )
    if not sun_channels:
        sun_channels = ((6, "rv2_coarse"), (8, "rv2_coarse"))

    return [
        CandidateSet(
            source_name="sun",
            label="Sun high-control candidates",
            events_path=ROOT / "outputs/sun_whole_dataset_validation_highcontrols_bands4_8/02_events/sun_predicted_events.csv",
            clean_path=None,
            use_raw_full_dataset=True,
            channels=sun_channels,
            windows=tuple(args.windows),
        ),
        CandidateSet(
            source_name="jupiter",
            label="Jupiter post-November candidates",
            events_path=ROOT / "outputs/control_survey_jupiter_postnov1974_v1/02_events/predicted_events.csv",
            clean_path=ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv",
            use_raw_full_dataset=False,
            channels=((4, "rv2_coarse"),),
            windows=tuple(args.windows),
        ),
        CandidateSet(
            source_name="fornax_a",
            label="Fornax A post-November candidates",
            events_path=ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv",
            clean_path=ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv",
            use_raw_full_dataset=False,
            channels=((9, "rv1_coarse"), (1, "rv1_coarse")),
            windows=tuple(args.windows),
        ),
    ]


def _write_report(summary: pd.DataFrame, out_root: Path) -> None:
    lines = [
        "# Candidate Confirmation Checks",
        "",
        "These checks test whether candidate stacks look source-like beyond a single aggregate SNR.",
        "",
        "Diagnostics include event-type split stacks, template-aligned peak offset, month-block repeatability, clean fraction, and independent window sizes.",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Interpretation Rules",
        "",
        "- `repeatable_candidate`: stack is significant, event types have the same sign, peak timing is near the prediction, and month/quality diagnostics are acceptable.",
        "- `candidate_needs_event_type_review`: stack is significant but the disappearance/reappearance split needs review.",
        "- `diagnostic_caution`: at least one timing, month-block, or quality diagnostic is concerning.",
        "- `not_detected`: aggregate stack SNR is below 3 for that channel/window.",
        "",
        "These labels are confirmation diagnostics, not final discovery claims.",
    ]
    (out_root / "candidate_confirmation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.environ.get("RAE2_MASTER_CSV", "data/interpolatedRAE2MasterFile.csv"))
    parser.add_argument("--output-root", default="outputs/candidate_confirmation_checks")
    parser.add_argument("--windows", nargs="+", type=float, default=[300.0, 600.0, 900.0])
    parser.add_argument("--bin-seconds", type=float, default=30.0)
    parser.add_argument("--n-bootstrap", type=int, default=40)
    parser.add_argument("--gap-factor", type=float, default=3.0)
    parser.add_argument("--artifact-sigma", type=float, default=12.0)
    parser.add_argument("--baseline-mode", default="sideband_linear", choices=["linear_all", "constant_all", "sideband_linear", "sideband_constant", "joint_step_linear", "pre_event_anchor"])
    parser.add_argument("--sideband-exclusion-seconds", type=float, default=120.0)
    args = parser.parse_args()

    out_root = ensure_dir(ROOT / args.output_root)
    all_rows = []
    for candidate in _candidate_sets(args):
        source_dir = ensure_dir(out_root / candidate.source_name)
        clean = _load_clean(args, candidate)
        events = _read(candidate.events_path, parse_dates=["predicted_event_time"])
        for freq, antenna in candidate.channels:
            for window in candidate.windows:
                row = _analyze_channel(
                    clean=clean,
                    events=events,
                    source_name=candidate.source_name,
                    freq=freq,
                    antenna=antenna,
                    window=float(window),
                    out_dir=source_dir,
                    bin_seconds=float(args.bin_seconds),
                    n_bootstrap=int(args.n_bootstrap),
                    baseline_mode=args.baseline_mode,
                    sideband_exclusion_seconds=args.sideband_exclusion_seconds,
                )
                all_rows.append(row)
        del clean

    summary = pd.DataFrame.from_records(all_rows).sort_values(["source_name", "frequency_band", "antenna", "window_s"])
    summary.to_csv(out_root / "candidate_confirmation_summary.csv", index=False)
    _write_report(summary, out_root)
    write_json(out_root / "run_config.json", vars(args))
    print(out_root / "candidate_confirmation_report.md")
    print(out_root / "candidate_confirmation_summary.csv")


if __name__ == "__main__":
    main()
