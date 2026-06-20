#!/usr/bin/env python
"""Generate verification plots for Bayesian contrast-model outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "outputs/bayesian_contrast_models_v1"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return read_table(path, low_memory=False)


def _ant_label(value: str) -> str:
    return {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}.get(str(value), str(value))


def _clip(values: pd.Series, limit: float = 12.0) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return np.clip(arr, -limit, limit)


def _boxplot(ax: plt.Axes, groups: list[np.ndarray], labels: list[str], title: str, ylabel: str) -> None:
    ax.boxplot(
        groups,
        tick_labels=labels,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.6},
        boxprops={"color": "#4c78a8"},
        whiskerprops={"color": "#4c78a8"},
        capprops={"color": "#4c78a8"},
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)


def plot_selected_event_distributions(out_dir: Path, earth: pd.DataFrame, sun: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bins = np.linspace(-8, 8, 65)
    for ax, frame, title, color in [
        (axes[0], earth, "Earth selected lower V channel", "#2ca02c"),
        (axes[1], sun, "Sun selected lower V channel", "#d62728"),
    ]:
        vals = _clip(frame["contrast"], 8)
        ax.hist(vals, bins=bins, color=color, alpha=0.75)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.axvline(np.nanmedian(vals), color="white", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Signed event contrast, clipped to +/-8")
    axes[0].set_ylabel("Number of event contrasts")
    fig.suptitle("Observed event contrasts used by the Bayesian model")
    fig.tight_layout()
    path = out_dir / "verification_selected_event_contrast_histograms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_event_type_split(out_dir: Path, earth: pd.DataFrame, sun: pd.DataFrame) -> Path:
    groups = []
    labels = []
    for source, frame in [("Earth", earth), ("Sun", sun)]:
        for event_type in ["disappearance", "reappearance"]:
            vals = _clip(frame.loc[frame["event_type"].astype(str).eq(event_type), "contrast"], 10)
            groups.append(vals)
            labels.append(f"{source}\n{event_type[:5]}.")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    _boxplot(
        ax,
        groups,
        labels,
        "Event-type split: disappearance and reappearance should have the same positive sign",
        "Signed event contrast",
    )
    path = out_dir / "verification_event_type_split.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_month_stability(out_dir: Path, earth: pd.DataFrame, sun: pd.DataFrame) -> Path:
    rows = []
    for label, frame in [("Earth selected", earth), ("Sun selected", sun)]:
        month = frame.groupby("month_block", sort=True)["contrast"].agg(["median", "count"]).reset_index()
        month["source"] = label
        rows.append(month)
    data = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for label, color in [("Earth selected", "#2ca02c"), ("Sun selected", "#d62728")]:
        sub = data[data["source"].eq(label)]
        ax.plot(sub["month_block"], sub["median"], marker="o", linewidth=1.8, label=label, color=color)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Month stability of selected-channel event contrasts")
    ax.set_ylabel("Median signed event contrast")
    ax.set_xlabel("Month")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "verification_month_stability_selected.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sun_channel_observed_and_posterior(out_dir: Path, sun_all: pd.DataFrame, channel_summary: pd.DataFrame) -> Path:
    observed = (
        sun_all.groupby(["frequency_mhz", "antenna"], sort=True)["contrast"]
        .agg(observed_median="median", n="count")
        .reset_index()
    )
    observed["channel"] = observed["frequency_mhz"].map(lambda x: f"{float(x):.2f} MHz") + " " + observed["antenna"].map(_ant_label)
    merged = observed.merge(channel_summary, on="channel", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    colors = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}
    for antenna, sub in merged.groupby("antenna", sort=True):
        label = _ant_label(antenna)
        axes[0].plot(sub["frequency_mhz"], sub["observed_median"], marker="o", linewidth=1.5, label=label, color=colors.get(antenna))
        axes[1].errorbar(
            sub["frequency_mhz"],
            sub["posterior_mean"],
            yerr=[sub["posterior_mean"] - sub["ci_2p5"], sub["ci_97p5"] - sub["posterior_mean"]],
            marker="o",
            linewidth=1.2,
            capsize=2,
            label=label,
            color=colors.get(antenna),
        )
    for ax in axes:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Frequency (MHz)")
    axes[0].set_title("Observed median Sun contrast by channel")
    axes[0].set_ylabel("Median signed contrast")
    axes[1].set_title("Bayesian posterior channel amplitude")
    axes[1].set_ylabel("Posterior amplitude with 95% interval")
    axes[1].legend(frameon=False)
    fig.suptitle("Sun channel diagnostic: upper-V 0.45 MHz drives the all-channel positive result")
    fig.tight_layout()
    path = out_dir / "verification_sun_channel_observed_posterior.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_posterior_ci_ladder(out_dir: Path, model_summary: pd.DataFrame) -> Path:
    order = ["earth_selected", "sun_selected", "sun_lower_v_all_channel", "sun_all_channel"]
    data = model_summary.set_index("model").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(data))
    ax.errorbar(
        data["posterior_mean"],
        y,
        xerr=[data["posterior_mean"] - data["ci_2p5"], data["ci_97p5"] - data["posterior_mean"]],
        fmt="o",
        color="#333333",
        ecolor="#9ecae9",
        capsize=3,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(data["model"])
    ax.set_xlabel("Posterior global contrast amplitude")
    ax.set_title("Decision check: credible intervals relative to zero")
    fig.tight_layout()
    path = out_dir / "verification_posterior_ci_ladder.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, paths: list[Path], model_summary: pd.DataFrame, channel_summary: pd.DataFrame) -> Path:
    strongest = channel_summary.sort_values("posterior_mean", ascending=False).head(1).iloc[0]
    earth = model_summary[model_summary["model"].eq("earth_selected")].iloc[0]
    sun = model_summary[model_summary["model"].eq("sun_selected")].iloc[0]
    sun_lower = model_summary[model_summary["model"].eq("sun_lower_v_all_channel")].iloc[0]
    report = [
        "# Bayesian Verification Plots",
        "",
        "These plots verify the Bayesian contrast model against the actual event-contrast measurements. "
        "A positive contrast means the event has the expected occultation sign.",
        "",
        "## How to interpret the plots",
        "",
        "- `verification_selected_event_contrast_histograms.png`: compares the raw event-contrast distribution for the Earth positive control and the selected Sun lower-V channel. A real positive occultation population should be shifted to the right of zero.",
        "- `verification_event_type_split.png`: checks whether disappearance and reappearance have the same signed behavior after applying the expected sign convention. A source-like signal should be positive for both.",
        "- `verification_month_stability_selected.png`: checks whether the signal is stable across months. A signal confined to one month is weaker evidence than one that persists across date blocks.",
        "- `verification_sun_channel_observed_posterior.png`: compares observed channel medians with Bayesian channel amplitudes. This identifies whether the all-channel result is broad lower-V support or a single-channel exception.",
        "- `verification_posterior_ci_ladder.png`: shows the main posterior intervals relative to zero. Intervals entirely right of zero are positive source-like; entirely left are anti-template.",
        "",
        "## Current evidence",
        "",
        f"- Earth remains a strong positive-control validation: posterior mean {earth['posterior_mean']:.3f}, 95% CI [{earth['ci_2p5']:.3f}, {earth['ci_97p5']:.3f}], P(A > 0) = {earth['p_gt_0']:.4f}.",
        f"- The selected Sun lower-V channel is anti-template: posterior mean {sun['posterior_mean']:.3f}, 95% CI [{sun['ci_2p5']:.3f}, {sun['ci_97p5']:.3f}], P(A < 0) = {sun['p_lt_0']:.4f}.",
        f"- Sun lower-V all-channel is also anti-template: posterior mean {sun_lower['posterior_mean']:.3f}, 95% CI [{sun_lower['ci_2p5']:.3f}, {sun_lower['ci_97p5']:.3f}], P(A < 0) = {sun_lower['p_lt_0']:.4f}.",
        f"- The strongest positive Sun channel is {strongest['channel']} with posterior mean {strongest['posterior_mean']:.3f}; because this is upper V, it should be treated as a systematics/antenna diagnostic before being treated as solar evidence.",
        "",
        "## New options suggested by these diagnostics",
        "",
        "1. Promote lower-V-only solar scoring to the primary solar decision path; keep upper-V channels as controls unless a geometry-specific reason justifies otherwise.",
        "2. Add a channel-consistency criterion: require a solar claim to be supported by multiple neighboring lower-V channels, not only one upper-V channel.",
        "3. Run the same Bayesian contrast model on time-shift controls and off-source controls, producing posterior probabilities for real-vs-control superiority rather than only real-vs-zero.",
        "4. Add a month-block hierarchical shrinkage variant or month-block bootstrap for the Bayesian contrast layer, because month stability is central to separating source behavior from telemetry/date-block effects.",
        "5. Test shorter sideband contrast windows for the Sun, since broad windows can wash out a compact solar limb response if the baseline is drifting.",
        "6. For the Sun, split quiet-Sun-like intervals from burst-like/high-variance intervals before stacking; the current all-event average may mix physically different regimes.",
        "",
        "## Generated files",
        "",
    ]
    report.extend(f"- `{path.name}`" for path in paths)
    report_path = out_dir / "bayesian_verification_plot_guide.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_IN))
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    earth = _read(in_dir / "earth_selected_model_input.csv")
    sun = _read(in_dir / "sun_selected_model_input.csv")
    sun_all = _read(in_dir / "sun_all_channel_model_input.csv")
    model_summary = _read(in_dir / "model_global_summary.csv")
    channel_summary = _read(in_dir / "sun_all_channel_posterior_channel_summary.csv")

    paths = [
        plot_selected_event_distributions(in_dir, earth, sun),
        plot_event_type_split(in_dir, earth, sun),
        plot_month_stability(in_dir, earth, sun),
        plot_sun_channel_observed_and_posterior(in_dir, sun_all, channel_summary),
        plot_posterior_ci_ladder(in_dir, model_summary),
    ]
    report = write_report(in_dir, paths, model_summary, channel_summary)
    print(f"Wrote {len(paths)} verification plots")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
