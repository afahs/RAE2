#!/usr/bin/env python
"""Run Bayesian contrast diagnostics for Earth validation and Sun analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.bayes_contrast import (  # noqa: E402
    draw_posterior,
    fit_bayesian_contrast_model,
    summarize_coefficients,
    summarize_draws,
)
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402


EARTH_SELECTED = ROOT / "outputs/pipeline_confidence_audit_earth_v1/all_simple_event_contrasts.csv"
SUN_SELECTED = ROOT / "outputs/pipeline_confidence_audit_v2/all_simple_event_contrasts.csv"
SUN_ALL_CHANNEL = ROOT / "outputs/solar_detection_blocker_diagnostics_v1/sun_all_channel_event_contrasts.csv"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return read_table(path, low_memory=False)


def _filter_contrasts(
    frame: pd.DataFrame,
    source: str,
    window_s: float,
    normalize: str,
    time_shift_s: float = 0.0,
) -> pd.DataFrame:
    work = frame.copy()
    if "source_name" in work.columns:
        work = work[work["source_name"].astype(str).str.lower().eq(source.lower())]
    work = work[
        np.isclose(pd.to_numeric(work["window_s"], errors="coerce"), float(window_s))
        & np.isclose(pd.to_numeric(work["time_shift_s"], errors="coerce"), float(time_shift_s))
        & work["normalize"].astype(str).eq(str(normalize))
    ].copy()
    work["contrast"] = pd.to_numeric(work["contrast"], errors="coerce")
    work = work[np.isfinite(work["contrast"])].reset_index(drop=True)
    if "month_block" not in work.columns:
        work["month_block"] = pd.to_datetime(work["predicted_event_time"]).dt.strftime("%Y-%m")
    return work


def _add_channel(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["frequency_band"] = pd.to_numeric(work["frequency_band"], errors="coerce").astype("Int64")
    work["frequency_mhz"] = pd.to_numeric(work["frequency_mhz"], errors="coerce")
    work["channel"] = (
        work["frequency_mhz"].map(lambda x: f"{float(x):.2f} MHz")
        + " "
        + work["antenna"].astype(str).map({"rv1_coarse": "upper V", "rv2_coarse": "lower V"}).fillna(work["antenna"].astype(str))
    )
    return work


def _decision(summary: dict[str, float], source: str) -> str:
    p_pos = summary["p_gt_0"]
    p_neg = summary["p_lt_0"]
    if source.lower() == "earth" and p_pos >= 0.99 and summary["ci_2p5"] > 0:
        return "positive_control_validated"
    if p_pos >= 0.975 and summary["ci_2p5"] > 0:
        return "positive_source_like"
    if p_neg >= 0.975 and summary["ci_97p5"] < 0:
        return "anti_template_behavior"
    return "not_resolved"


def _run_model(
    label: str,
    frame: pd.DataFrame,
    categorical_columns: list[str],
    out_dir: Path,
    n_draws: int,
    seed: int,
) -> dict[str, object]:
    fit = fit_bayesian_contrast_model(frame, categorical_columns=categorical_columns)
    draws = draw_posterior(fit, n_draws=n_draws, seed=seed)
    coef_summary = summarize_coefficients(draws)
    global_summary = summarize_draws(draws["global_amplitude"])
    coef_summary.insert(0, "model", label)
    coef_summary.to_csv(out_dir / f"{label}_coefficient_summary.csv", index=False)
    draws[["global_amplitude"]].to_csv(out_dir / f"{label}_global_posterior_draws.csv", index=False)
    return {
        "label": label,
        "fit": fit,
        "draws": draws,
        "coefficient_summary": coef_summary,
        "global_summary": global_summary,
    }


def _plot_global_posteriors(results: list[dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {
        "earth_selected": "#2ca02c",
        "sun_selected": "#d62728",
        "sun_all_channel": "#9467bd",
        "sun_lower_v_all_channel": "#ff7f0e",
    }
    for result in results:
        label = str(result["label"])
        draws = result["draws"]["global_amplitude"].to_numpy(dtype=float)
        ax.hist(draws, bins=80, density=True, histtype="step", linewidth=2.0, color=colors.get(label), label=label)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Posterior global occultation contrast amplitude")
    ax.set_ylabel("Posterior density")
    ax.set_title("Bayesian contrast posterior: positive means source-like occultation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_sun_channel_posteriors(draws: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for col in draws.columns:
        if not col.startswith("channel="):
            continue
        channel = col.split("=", 1)[1]
        quantity = draws["global_amplitude"] + draws[col]
        rows.append({"channel": channel, **summarize_draws(quantity)})
    summary = pd.DataFrame.from_records(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values("posterior_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.34 * len(summary))))
    y = np.arange(len(summary))
    ax.errorbar(
        summary["posterior_mean"],
        y,
        xerr=[
            summary["posterior_mean"] - summary["ci_2p5"],
            summary["ci_97p5"] - summary["posterior_mean"],
        ],
        fmt="o",
        color="#4c78a8",
        ecolor="#9ecae9",
        capsize=2,
    )
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(summary["channel"])
    ax.set_xlabel("Posterior channel contrast amplitude")
    ax.set_title("Sun all-channel posterior amplitudes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return summary


def _write_report(
    out_dir: Path,
    model_rows: pd.DataFrame,
    channel_summary: pd.DataFrame,
    config: dict[str, object],
) -> None:
    earth = model_rows[model_rows["model"].eq("earth_selected")].iloc[0]
    sun = model_rows[model_rows["model"].eq("sun_selected")].iloc[0]
    sun_all = model_rows[model_rows["model"].eq("sun_all_channel")].iloc[0]
    sun_lower = model_rows[model_rows["model"].eq("sun_lower_v_all_channel")].iloc[0]
    positive_channels = channel_summary[channel_summary["p_gt_0"] >= 0.975] if not channel_summary.empty else pd.DataFrame()
    lines = [
        "# Bayesian Contrast Model Report",
        "",
        "This run adds a lightweight Bayesian layer to the existing signed event-contrast tables. "
        "The model uses only source contrast terms, event-type/month/channel deviation terms, and a Gaussian prior. "
        "It intentionally excludes contaminant terms and does not use an antenna-weighted forward model.",
        "",
        "Positive amplitude means the median pre/post contrast has the expected occultation sign: "
        "disappearance is `pre - post`, reappearance is `post - pre`.",
        "",
        "## Configuration",
        "",
        f"- window: {config['window_s']} s",
        f"- normalize: {config['normalize']}",
        f"- time shift: {config['time_shift_s']} s",
        f"- posterior draws: {config['n_draws']}",
        "",
        "## Global Posterior Results",
        "",
        "| model | n observations | posterior mean | 95% CI | P(amplitude > 0) | P(amplitude < 0) | decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in model_rows.iterrows():
        lines.append(
            f"| {row['model']} | {int(row['n_observations'])} | {row['posterior_mean']:.3f} | "
            f"[{row['ci_2p5']:.3f}, {row['ci_97p5']:.3f}] | {row['p_gt_0']:.4f} | "
            f"{row['p_lt_0']:.4f} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Earth validates the contrast model if its posterior is decisively positive. "
            f"In this run Earth is `{earth['decision']}` with P(A > 0) = {earth['p_gt_0']:.4f}.",
            f"- The selected Sun channel is `{sun['decision']}` with P(A > 0) = {sun['p_gt_0']:.4f} "
            f"and P(A < 0) = {sun['p_lt_0']:.4f}. This does not support a positive solar occultation detection.",
            f"- The Sun all-channel diagnostic is `{sun_all['decision']}`. "
            "Because all channels share events, this is a channel-pattern diagnostic rather than independent evidence.",
            f"- The Sun lower-V-only all-channel diagnostic is `{sun_lower['decision']}` with "
            f"P(A > 0) = {sun_lower['p_gt_0']:.4f}. This is the more relevant antenna split for lunar occultation by the Moon.",
            "",
        ]
    )
    if not positive_channels.empty:
        lines.extend(
            [
                "## Positive Sun Channel Exceptions",
                "",
                "These channels have posterior P(A > 0) >= 0.975 in the all-channel diagnostic and should be inspected separately:",
                "",
                "| channel | posterior mean | 95% CI | P(A > 0) |",
                "|---|---:|---:|---:|",
            ]
        )
        for _, row in positive_channels.iterrows():
            lines.append(
                f"| {row['channel']} | {row['posterior_mean']:.3f} | "
                f"[{row['ci_2p5']:.3f}, {row['ci_97p5']:.3f}] | {row['p_gt_0']:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Outputs",
            "",
            "- `model_global_summary.csv`",
            "- `sun_all_channel_posterior_channel_summary.csv`",
            "- `global_amplitude_posteriors.png`",
            "- `sun_all_channel_posterior_channels.png`",
            "",
        ]
    )
    (out_dir / "bayesian_contrast_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/bayesian_contrast_models_v1"))
    parser.add_argument("--window-s", type=float, default=300.0)
    parser.add_argument("--time-shift-s", type=float, default=0.0)
    parser.add_argument("--normalize", default="zscore")
    parser.add_argument("--n-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    config = {
        "window_s": float(args.window_s),
        "time_shift_s": float(args.time_shift_s),
        "normalize": str(args.normalize),
        "n_draws": int(args.n_draws),
        "seed": int(args.seed),
        "software_versions": software_versions(),
        "inputs": {
            "earth_selected": str(EARTH_SELECTED),
            "sun_selected": str(SUN_SELECTED),
            "sun_all_channel": str(SUN_ALL_CHANNEL),
        },
        "excluded_model_terms": ["contaminant_terms", "antenna_weighted_forward_model"],
    }
    write_json(out_dir / "run_config.json", config)

    earth = _filter_contrasts(_read(EARTH_SELECTED), "earth", args.window_s, args.normalize, args.time_shift_s)
    sun_selected = _filter_contrasts(_read(SUN_SELECTED), "sun", args.window_s, args.normalize, args.time_shift_s)
    sun_all = _add_channel(_filter_contrasts(_read(SUN_ALL_CHANNEL), "sun", args.window_s, args.normalize, args.time_shift_s))
    sun_lower = sun_all[sun_all["antenna"].astype(str).eq("rv2_coarse")].copy()

    if earth.empty:
        raise RuntimeError("Earth selected contrast table is empty after filtering")
    if sun_selected.empty:
        raise RuntimeError("Sun selected contrast table is empty after filtering")
    if sun_all.empty:
        raise RuntimeError("Sun all-channel contrast table is empty after filtering")
    if sun_lower.empty:
        raise RuntimeError("Sun lower-V all-channel contrast table is empty after filtering")

    earth.to_csv(out_dir / "earth_selected_model_input.csv", index=False)
    sun_selected.to_csv(out_dir / "sun_selected_model_input.csv", index=False)
    sun_all.to_csv(out_dir / "sun_all_channel_model_input.csv", index=False)
    sun_lower.to_csv(out_dir / "sun_lower_v_all_channel_model_input.csv", index=False)

    results = [
        _run_model("earth_selected", earth, ["event_type", "month_block"], out_dir, args.n_draws, args.seed),
        _run_model("sun_selected", sun_selected, ["event_type", "month_block"], out_dir, args.n_draws, args.seed + 1),
        _run_model("sun_all_channel", sun_all, ["event_type", "month_block", "channel"], out_dir, args.n_draws, args.seed + 2),
        _run_model(
            "sun_lower_v_all_channel",
            sun_lower,
            ["event_type", "month_block", "channel"],
            out_dir,
            args.n_draws,
            args.seed + 3,
        ),
    ]

    model_rows = []
    for result in results:
        summary = result["global_summary"]
        label = str(result["label"])
        source = "earth" if label.startswith("earth") else "sun"
        fit = result["fit"]
        model_rows.append(
            {
                "model": label,
                "source": source,
                "n_observations": fit.n_observations,
                "n_parameters": fit.n_parameters,
                "residual_sigma": fit.residual_sigma,
                **summary,
                "decision": _decision(summary, source),
            }
        )
    model_summary = pd.DataFrame.from_records(model_rows)
    model_summary.to_csv(out_dir / "model_global_summary.csv", index=False)

    all_channel_result = next(result for result in results if result["label"] == "sun_all_channel")
    channel_summary = _plot_sun_channel_posteriors(
        all_channel_result["draws"],
        out_dir / "sun_all_channel_posterior_channels.png",
    )
    channel_summary.to_csv(out_dir / "sun_all_channel_posterior_channel_summary.csv", index=False)
    _plot_global_posteriors(results, out_dir / "global_amplitude_posteriors.png")
    _write_report(out_dir, model_summary, channel_summary, config)

    print(model_summary.to_string(index=False))
    print(f"Wrote Bayesian contrast outputs to {out_dir}")


if __name__ == "__main__":
    main()
