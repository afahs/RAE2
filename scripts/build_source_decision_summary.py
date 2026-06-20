#!/usr/bin/env python
"""Build a compact source-level decision table from confidence diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False)


def _sign(value: float) -> str:
    if not np.isfinite(value) or value == 0:
        return "zero_or_nan"
    return "positive" if value > 0 else "negative"


def _pick_primary_contrast(source: str, tables: list[pd.DataFrame]) -> pd.Series | None:
    df = pd.concat([t for t in tables if not t.empty], ignore_index=True) if tables else pd.DataFrame()
    if df.empty:
        return None
    sub = df[df["source_name"].astype(str).eq(source) & pd.to_numeric(df["time_shift_s"], errors="coerce").eq(0.0)].copy()
    if sub.empty:
        return None
    # Prefer z-score normalization for the decision row because it is scale-free
    # but still a direct pre/post statistic. Use raw/fractional only as fallback.
    sub["normalize_rank"] = sub["normalize"].map({"zscore": 0, "fractional": 1, "raw": 2}).fillna(9)
    sub["abs_primary_snr"] = pd.to_numeric(sub["event_bootstrap_contrast_snr"], errors="coerce").abs()
    return sub.sort_values(["normalize_rank", "abs_primary_snr"], ascending=[True, False]).iloc[0]


def _pick_template_fit(source: str, tables: list[pd.DataFrame]) -> pd.Series | None:
    df = pd.concat([t for t in tables if not t.empty], ignore_index=True) if tables else pd.DataFrame()
    if df.empty:
        return None
    sub = df[df["source_name"].astype(str).eq(source) & df["stack_group"].astype(str).eq("combined")].copy()
    if sub.empty:
        return None
    sub["variant_rank"] = sub["variant"].map(
        {
            "raw_zscore_no_baseline": 0,
            "sideband_constant_subtracted": 1,
            "raw_fractional_no_baseline": 2,
            "sideband_linear_subtracted": 3,
        }
    ).fillna(9)
    sub["model_rank"] = sub["stack_model"].map({"linear_plus_step": 0, "constant_plus_step": 1}).fillna(9)
    # Prefer the simplest scale-free/no-drift-subtraction variant; within that,
    # use the stronger evidence row. This avoids cherry-picking across variants.
    score = pd.to_numeric(sub.get("event_bootstrap_snr", sub["stack_fit_snr"]), errors="coerce").abs()
    sub["abs_template_score"] = score.fillna(pd.to_numeric(sub["stack_fit_snr"], errors="coerce").abs())
    return sub.sort_values(["variant_rank", "model_rank", "abs_template_score"], ascending=[True, True, False]).iloc[0]


def _month_stability(primary: pd.Series | None, template: pd.Series | None) -> tuple[str, str]:
    if primary is None:
        return "not_evaluated", "no primary contrast row"
    psnr = float(primary.get("event_bootstrap_contrast_snr", np.nan))
    pmin = float(primary.get("month_median_min", np.nan))
    pmax = float(primary.get("month_median_max", np.nan))
    leave_abs = float(template.get("leave_one_month_min_abs_snr", np.nan)) if template is not None else np.nan
    reasons = []
    stable_sign = False
    if psnr > 0:
        stable_sign = np.isfinite(pmin) and pmin > 0
        if not stable_sign:
            reasons.append("month medians include zero/opposite sign")
    elif psnr < 0:
        stable_sign = np.isfinite(pmax) and pmax <= 0
        if not stable_sign:
            reasons.append("month medians include zero/opposite sign")
    if np.isfinite(leave_abs) and leave_abs < 3:
        reasons.append(f"leave-one-month template SNR drops below 3 ({leave_abs:.2f})")
    if not reasons and stable_sign:
        return "stable", "month sign and leave-one-month checks pass"
    if stable_sign:
        return "partly_stable", "; ".join(reasons)
    return "unstable_or_clustered", "; ".join(reasons) if reasons else "month stability not demonstrated"


def _classify(source: str, primary: pd.Series | None, template: pd.Series | None) -> tuple[str, str, dict[str, object]]:
    if primary is None:
        return "not_evaluated", "missing primary contrast row", {}
    primary_snr = float(primary.get("event_bootstrap_contrast_snr", np.nan))
    primary_sign = _sign(primary_snr)
    p_shift = float(primary.get("time_shift_empirical_p", np.nan))
    template_snr = float(template.get("event_bootstrap_snr", np.nan)) if template is not None else np.nan
    if not np.isfinite(template_snr) and template is not None:
        template_snr = float(template.get("stack_fit_snr", np.nan))
    template_sign = _sign(template_snr)
    timing_offset = float(template.get("best_timing_offset_s", np.nan)) if template is not None else np.nan
    sign_agree = bool(primary_sign in {"positive", "negative"} and primary_sign == template_sign)
    stability, stability_reason = _month_stability(primary, template)
    diagnostics = {
        "primary_sign": primary_sign,
        "template_sign": template_sign,
        "sign_agreement": sign_agree,
        "month_stability": stability,
        "month_stability_reason": stability_reason,
    }

    if source == "earth":
        if primary_snr >= 10 and sign_agree and p_shift <= 0.05 and stability in {"stable", "partly_stable"}:
            return "positive_control_confirmed", "Earth passes simple contrast, template sign, shifted-time, and month checks", diagnostics
        return "positive_control_problem", "Earth failed one or more expected positive-control checks", diagnostics

    if primary_snr <= -3 and template_snr <= -3 and sign_agree:
        return (
            "anti_template_unresolved",
            "simple contrast and stack template agree, but both have the opposite sign from positive-source occultation",
            diagnostics,
        )
    if primary_snr >= 5 and template_snr >= 5 and sign_agree and p_shift <= 0.05 and stability == "stable" and abs(timing_offset) <= 120:
        return "strong_candidate", "primary contrast and stack template agree with stable positive-source sign", diagnostics
    if primary_snr >= 3 and template_snr >= 3 and sign_agree and p_shift <= 0.1:
        if stability == "stable":
            return "candidate", "positive-source sign is present but not strong enough for confirmation", diagnostics
        return "episodic_candidate", "positive-source sign is present but month stability is weak or clustered", diagnostics
    if abs(primary_snr) >= 3 and (not sign_agree or not np.isfinite(template_snr)):
        return "unresolved", "primary contrast is present but template agreement is missing", diagnostics
    return "not_detected", "no robust positive-source signal in the simplified decision tree", diagnostics


def _row_for_source(source: str, primary: pd.Series | None, template: pd.Series | None) -> dict[str, object]:
    status, reason, diagnostics = _classify(source, primary, template)
    row = {
        "source_name": source,
        "decision_class": status,
        "decision_reason": reason,
        "primary_frequency_mhz": np.nan,
        "primary_antenna": "",
        "primary_window_s": np.nan,
        "primary_normalize": "",
        "primary_median_contrast": np.nan,
        "primary_contrast_snr": np.nan,
        "primary_positive_fraction": np.nan,
        "primary_shift_control_p": np.nan,
        "primary_n_events": np.nan,
        "template_variant": "",
        "template_model": "",
        "template_snr": np.nan,
        "template_amplitude": np.nan,
        "template_timing_offset_s": np.nan,
        "template_delta_bic": np.nan,
        "template_leave_one_month_min_abs_snr": np.nan,
        "primary_sign": diagnostics.get("primary_sign", ""),
        "template_sign": diagnostics.get("template_sign", ""),
        "sign_agreement": diagnostics.get("sign_agreement", False),
        "month_stability": diagnostics.get("month_stability", ""),
        "month_stability_reason": diagnostics.get("month_stability_reason", ""),
    }
    if primary is not None:
        row.update(
            {
                "primary_frequency_mhz": float(primary.get("frequency_mhz", np.nan)),
                "primary_antenna": primary.get("antenna", ""),
                "primary_window_s": float(primary.get("window_s", np.nan)),
                "primary_normalize": primary.get("normalize", ""),
                "primary_median_contrast": float(primary.get("median_contrast", np.nan)),
                "primary_contrast_snr": float(primary.get("event_bootstrap_contrast_snr", np.nan)),
                "primary_positive_fraction": float(primary.get("positive_fraction", np.nan)),
                "primary_shift_control_p": float(primary.get("time_shift_empirical_p", np.nan)),
                "primary_n_events": int(primary.get("n_events", 0)),
            }
        )
    if template is not None:
        tsnr = float(template.get("event_bootstrap_snr", np.nan))
        if not np.isfinite(tsnr):
            tsnr = float(template.get("stack_fit_snr", np.nan))
        row.update(
            {
                "template_variant": template.get("variant", ""),
                "template_model": template.get("stack_model", ""),
                "template_snr": tsnr,
                "template_amplitude": float(template.get("amplitude", np.nan)),
                "template_timing_offset_s": float(template.get("best_timing_offset_s", np.nan)),
                "template_delta_bic": float(template.get("delta_bic", np.nan)),
                "template_leave_one_month_min_abs_snr": float(template.get("leave_one_month_min_abs_snr", np.nan)),
            }
        )
    return row


def _write_report(out_dir: Path, table: pd.DataFrame) -> None:
    lines = [
        "# Source Decision Summary",
        "",
        "This is the simplified decision layer. It uses a decision tree rather than a long checklist.",
        "",
        "Primary statistic: direct signed pre/post contrast with event-bootstrap SNR.",
        "Secondary check: stack-first template fit sign, timing, and stability.",
        "",
        "Positive sign means the expected positive-source occultation behavior: disappearance drops and reappearance rises.",
        "",
        "## Decisions",
        "",
        "```\n"
        + table[
            [
                "source_name",
                "decision_class",
                "primary_contrast_snr",
                "template_snr",
                "primary_sign",
                "template_sign",
                "sign_agreement",
                "primary_shift_control_p",
                "month_stability",
                "decision_reason",
            ]
        ].to_string(index=False)
        + "\n```",
        "",
        "## Interpretation",
        "",
        "- Earth should pass as a positive control. If it fails, the pipeline is not trustworthy.",
        "- A non-Earth source needs positive primary contrast and positive template agreement before being called a candidate.",
        "- A strong negative result is not negative emission; it is anti-template behavior and should be investigated as geometry, contamination, antenna response, or calibration.",
        "- Month-clustered positive behavior is treated as episodic, not a stable continuum detection.",
    ]
    (out_dir / "source_decision_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/source_decision_summary_v1")
    parser.add_argument("--sources", nargs="+", default=["earth", "sun", "jupiter"])
    parser.add_argument("--contrast-summary", nargs="+", default=[
        "outputs/pipeline_confidence_audit_earth_v1/all_simple_contrast_summary.csv",
        "outputs/pipeline_confidence_audit_v2/all_simple_contrast_summary.csv",
    ])
    parser.add_argument("--template-summary", nargs="+", default=[
        "outputs/stack_first_fit_diagnostics_lower_v_v2/all_stack_first_fit_summary.csv",
        "outputs/refined_stackfit_sun_earth_excluded_v1/all_stack_first_fit_summary.csv",
        "outputs/refined_stackfit_jupiter_episodic_v1/all_stack_first_fit_summary.csv",
    ])
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    contrast_tables = [_read(ROOT / path) for path in args.contrast_summary]
    template_tables = [_read(ROOT / path) for path in args.template_summary]
    rows = []
    for source in args.sources:
        primary = _pick_primary_contrast(source, contrast_tables)
        template = _pick_template_fit(source, template_tables)
        rows.append(_row_for_source(source, primary, template))
    table = pd.DataFrame.from_records(rows)
    table.to_csv(out_dir / "source_decision_summary.csv", index=False)
    _write_report(out_dir, table)
    print(out_dir / "source_decision_summary.md")
    print(out_dir / "source_decision_summary.csv")


if __name__ == "__main__":
    main()
