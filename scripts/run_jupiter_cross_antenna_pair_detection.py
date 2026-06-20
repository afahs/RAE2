#!/usr/bin/env python
"""Cross-antenna Jupiter burst-pair detection.

This script treats a candidate as stronger when upper V and lower V both show a
same-frequency local-z burst within a short time tolerance. It uses the event
catalog from ``run_jupiter_event_burst_search.py`` and the daily selector
denominators from the same run, then tests whether pair-event rates rise in
expected-active Jupiter selectors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.run_jupiter_expected_active_selector_analysis import SELECTOR_LABEL  # noqa: E402
from scripts.run_jupiter_literature_controls import bootstrap_mean_ci, sign_flip_p  # noqa: E402


DEFAULT_EVENTS = ROOT / "outputs/jupiter_event_burst_search_v1/jupiter_burst_event_catalog.csv"
DEFAULT_DAILY = ROOT / "outputs/jupiter_event_burst_search_v1/jupiter_burst_event_selector_daily_rates.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_cross_antenna_pair_detection_v1"
SELECTOR_ORDER = [
    "literature_io_phase_windows",
    "maser_top25",
    "maser_top10",
    "maser_top05",
    "io_windows_and_maser_top10",
    "historical_wdr_reported_windows",
]


def selector_flag_names(events: pd.DataFrame) -> list[str]:
    flags = []
    for col in events.columns:
        if col.endswith("_at_peak") and not col.startswith("has_"):
            flags.append(col[: -len("_at_peak")])
    return [name for name in SELECTOR_ORDER if name in flags]


def build_cross_antenna_pairs(events: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    work = events.copy()
    work["burst_peak_time"] = pd.to_datetime(work["burst_peak_time"])
    selectors = selector_flag_names(work)
    rows: list[dict[str, object]] = []
    for freq, grp in work.groupby("frequency_mhz", sort=True):
        upper = grp[grp["antenna"].eq("rv1_coarse")].sort_values("burst_peak_time").reset_index(drop=True)
        lower = grp[grp["antenna"].eq("rv2_coarse")].sort_values("burst_peak_time").reset_index(drop=True)
        if upper.empty or lower.empty:
            continue
        lower_times = lower["burst_peak_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        used_lower: set[int] = set()
        tol_ns = int(float(tolerance_s) * 1e9)
        for _, up in upper.iterrows():
            up_ns = pd.Timestamp(up["burst_peak_time"]).value
            lo = int(np.searchsorted(lower_times, up_ns - tol_ns, side="left"))
            hi = int(np.searchsorted(lower_times, up_ns + tol_ns, side="right"))
            choices = [idx for idx in range(lo, hi) if idx not in used_lower]
            if not choices:
                continue
            best = min(choices, key=lambda idx: abs(int(lower_times[idx]) - up_ns))
            used_lower.add(best)
            low = lower.loc[best]
            low_ns = pd.Timestamp(low["burst_peak_time"]).value
            pair_time = pd.to_datetime((up_ns + low_ns) // 2)
            row = {
                "pair_id": len(rows),
                "frequency_band": int(up["frequency_band"]),
                "frequency_mhz": float(freq),
                "date": pair_time.floor("D"),
                "pair_peak_time": pair_time,
                "upper_event_id": int(up["event_id"]),
                "lower_event_id": int(low["event_id"]),
                "peak_time_delta_s": float(abs(up_ns - low_ns) / 1e9),
                "upper_peak_local_z": float(up["peak_local_z"]),
                "lower_peak_local_z": float(low["peak_local_z"]),
                "min_peak_local_z": float(min(float(up["peak_local_z"]), float(low["peak_local_z"]))),
                "mean_peak_local_z": float(np.mean([float(up["peak_local_z"]), float(low["peak_local_z"])])),
                "io_phase_spice_deg": float(np.mean([float(up["io_phase_spice_deg"]), float(low["io_phase_spice_deg"])])),
                "jupiter_cml_spice_deg": float(np.mean([float(up["jupiter_cml_spice_deg"]), float(low["jupiter_cml_spice_deg"])])),
                "maser_zarka_io_score": float(np.mean([float(up["maser_zarka_io_score"]), float(low["maser_zarka_io_score"])])),
            }
            for selector in selectors:
                col = f"{selector}_at_peak"
                row[f"{selector}_pair"] = bool(up[col]) and bool(low[col])
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def daily_pair_denominators(daily: pd.DataFrame, min_selected_samples: int, min_control_samples: int) -> pd.DataFrame:
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"])
    if "event_set" in work.columns:
        work = work[work["event_set"].eq("all_bursts")].copy()
    keys = ["selector", "date", "frequency_band", "frequency_mhz"]
    den = (
        work.groupby(keys, sort=True)
        .agg(
            selected_pair_samples=("selected_n_samples", "min"),
            control_pair_samples=("control_n_samples", "min"),
        )
        .reset_index()
    )
    return den[
        (den["selected_pair_samples"] >= int(min_selected_samples))
        & (den["control_pair_samples"] >= int(min_control_samples))
    ].copy()


def daily_pair_rate_points(
    pairs: pd.DataFrame,
    denominators: pd.DataFrame,
    selector: str,
) -> pd.DataFrame:
    if denominators.empty:
        return pd.DataFrame()
    flag_col = f"{selector}_pair"
    if pairs.empty or flag_col not in pairs.columns:
        selected_counts = pd.DataFrame(columns=["date", "frequency_band", "frequency_mhz", "selected_n_pairs"])
        control_counts = pd.DataFrame(columns=["date", "frequency_band", "frequency_mhz", "control_n_pairs"])
    else:
        selected = pairs[pairs[flag_col].astype(bool)].copy()
        control = pairs[~pairs[flag_col].astype(bool)].copy()
        keys = ["date", "frequency_band", "frequency_mhz"]
        selected_counts = (
            selected.groupby(keys, sort=True).size().reset_index(name="selected_n_pairs")
            if not selected.empty
            else pd.DataFrame(columns=keys + ["selected_n_pairs"])
        )
        control_counts = (
            control.groupby(keys, sort=True).size().reset_index(name="control_n_pairs")
            if not control.empty
            else pd.DataFrame(columns=keys + ["control_n_pairs"])
        )
    den = denominators[denominators["selector"].eq(selector)].copy()
    out = den.merge(selected_counts, on=["date", "frequency_band", "frequency_mhz"], how="left")
    out = out.merge(control_counts, on=["date", "frequency_band", "frequency_mhz"], how="left")
    out[["selected_n_pairs", "control_n_pairs"]] = out[["selected_n_pairs", "control_n_pairs"]].fillna(0).astype(int)
    out["selected_pair_rate_per_1000_samples"] = out["selected_n_pairs"] / out["selected_pair_samples"] * 1000.0
    out["control_pair_rate_per_1000_samples"] = out["control_n_pairs"] / out["control_pair_samples"] * 1000.0
    out["selected_minus_control_pair_rate_per_1000_samples"] = (
        out["selected_pair_rate_per_1000_samples"] - out["control_pair_rate_per_1000_samples"]
    )
    out["selector_label"] = SELECTOR_LABEL.get(selector, selector)
    return out


def summarize_pair_rates(
    daily: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (selector, band, freq), grp in daily.groupby(["selector", "frequency_band", "frequency_mhz"], sort=True):
        diff = grp["selected_minus_control_pair_rate_per_1000_samples"].to_numpy(dtype=float)
        mean, lo, hi = bootstrap_mean_ci(diff, rng, n_boot)
        p_two, p_pos = sign_flip_p(diff, rng, n_perm)
        rows.append(
            {
                "selector": selector,
                "selector_label": SELECTOR_LABEL.get(str(selector), str(selector)),
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "n_paired_days": int(len(grp)),
                "selected_total_pairs": int(grp["selected_n_pairs"].sum()),
                "control_total_pairs": int(grp["control_n_pairs"].sum()),
                "selected_total_pair_samples": int(grp["selected_pair_samples"].sum()),
                "control_total_pair_samples": int(grp["control_pair_samples"].sum()),
                "selected_pair_rate_per_1000_samples_mean": float(grp["selected_pair_rate_per_1000_samples"].mean()),
                "control_pair_rate_per_1000_samples_mean": float(grp["control_pair_rate_per_1000_samples"].mean()),
                "mean_selected_minus_control_pair_rate_per_1000_samples": mean,
                "boot_lo_selected_minus_control_pair_rate_per_1000_samples": lo,
                "boot_hi_selected_minus_control_pair_rate_per_1000_samples": hi,
                "positive_day_fraction_pair_rate": float((diff > 0).mean()) if len(diff) else np.nan,
                "signflip_p_two_sided_pair_rate": p_two,
                "signflip_p_positive_pair_rate": p_pos,
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity_summary(
    pairs: pd.DataFrame,
    denominators: pd.DataFrame,
    selectors: list[str],
    thresholds: list[float],
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        sub = pairs[pairs["min_peak_local_z"].ge(float(threshold))].copy()
        daily_tables = []
        for selector in selectors:
            points = daily_pair_rate_points(sub, denominators, selector)
            if not points.empty:
                daily_tables.append(points)
        if not daily_tables:
            continue
        summary = summarize_pair_rates(
            pd.concat(daily_tables, ignore_index=True),
            rng=rng,
            n_boot=n_boot,
            n_perm=n_perm,
        )
        if not summary.empty:
            summary["min_pair_z_threshold"] = float(threshold)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_pair_rate_spectrum(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    if summary.empty:
        return None
    selectors = [s for s in SELECTOR_ORDER if s in set(summary["selector"])]
    fig, axes = plt.subplots(len(selectors), 1, figsize=(9.0, max(2.0 * len(selectors), 5.0)), sharex=True)
    axes = np.atleast_1d(axes)
    metric = "mean_selected_minus_control_pair_rate_per_1000_samples"
    lo_col = "boot_lo_selected_minus_control_pair_rate_per_1000_samples"
    hi_col = "boot_hi_selected_minus_control_pair_rate_per_1000_samples"
    for ax, selector in zip(axes, selectors):
        grp = summary[summary["selector"].eq(selector)].sort_values("frequency_mhz")
        y = grp[metric].to_numpy(dtype=float)
        lo = grp[lo_col].to_numpy(dtype=float)
        hi = grp[hi_col].to_numpy(dtype=float)
        err = np.vstack([y - lo, hi - y])
        err[~np.isfinite(err)] = 0.0
        ax.errorbar(grp["frequency_mhz"], y, yerr=err, marker="o", lw=1.35, capsize=2.5, color="#4c78a8")
        sig = pd.to_numeric(grp["signflip_p_positive_pair_rate"], errors="coerce").to_numpy(dtype=float) < 0.05
        if np.any(sig):
            ax.scatter(
                grp.loc[sig, "frequency_mhz"],
                grp.loc[sig, metric],
                s=72,
                facecolors="none",
                edgecolors="black",
                linewidths=1.1,
                zorder=5,
            )
        ax.axhline(0, color="0.35", lw=0.85)
        ax.set_xscale("log")
        ax.set_xticks([FREQUENCY_MAP_MHZ[1], FREQUENCY_MAP_MHZ[6]])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_ylabel("pairs / 1000 samples")
        ax.set_title(SELECTOR_LABEL.get(selector, selector), loc="left", fontsize=10)
        ax.grid(True, color="0.9", lw=0.45)
    axes[-1].set_xlabel("frequency (MHz)")
    fig.suptitle("Cross-antenna Jupiter burst-pair excess in expected-active selectors")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / "jupiter_cross_antenna_pair_rate_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_pair_phase_scatter(pairs: pd.DataFrame, out_dir: Path) -> Path | None:
    if pairs.empty:
        return None
    freqs = sorted(pairs["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(1, len(freqs), figsize=(6.2 * len(freqs), 5.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    finite = pairs["min_peak_local_z"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    vmax = float(np.nanpercentile(finite, 95)) if finite.size else 10.0
    norm = Normalize(vmin=3.0, vmax=max(5.0, vmax))
    for ax, freq in zip(axes, freqs):
        sub = pairs[pairs["frequency_mhz"].eq(freq)].copy()
        sc = ax.scatter(
            sub["jupiter_cml_spice_deg"],
            sub["io_phase_spice_deg"],
            c=np.clip(sub["min_peak_local_z"], norm.vmin, norm.vmax),
            s=14,
            cmap="viridis",
            norm=norm,
            alpha=0.72,
            linewidths=0,
        )
        for lo, hi in [(80, 100), (235, 260)]:
            ax.axhspan(lo, hi, color="#d95f02", alpha=0.10, lw=0)
        ax.set_title(f"{freq:.2f} MHz cross-antenna pairs")
        ax.set_xlabel("Jupiter System III CML proxy (deg)")
        ax.grid(True, color="0.9", lw=0.45)
    axes[0].set_ylabel("Io phase (deg)")
    cbar = fig.colorbar(sc, ax=axes, pad=0.015)
    cbar.set_label("min upper/lower peak local z")
    fig.suptitle("Cross-antenna burst pairs in Io-CML space")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "jupiter_cross_antenna_pair_io_cml_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_threshold_sensitivity(threshold_summary: pd.DataFrame, out_dir: Path) -> Path | None:
    if threshold_summary.empty:
        return None
    selectors = [s for s in SELECTOR_ORDER if s in set(threshold_summary["selector"])]
    fig, axes = plt.subplots(len(selectors), 1, figsize=(9.5, max(2.0 * len(selectors), 5.0)), sharex=True)
    axes = np.atleast_1d(axes)
    metric = "mean_selected_minus_control_pair_rate_per_1000_samples"
    colors = {0.45: "#4c78a8", 3.93: "#d95f02"}
    for ax, selector in zip(axes, selectors):
        sub = threshold_summary[threshold_summary["selector"].eq(selector)].copy()
        for freq, grp in sub.groupby("frequency_mhz", sort=True):
            grp = grp.sort_values("min_pair_z_threshold")
            ax.plot(
                grp["min_pair_z_threshold"],
                grp[metric],
                marker="o",
                lw=1.25,
                color=colors.get(round(float(freq), 2), "0.2"),
                label=f"{float(freq):.2f} MHz",
            )
        ax.axhline(0, color="0.35", lw=0.85)
        ax.set_ylabel("pairs / 1000")
        ax.set_title(SELECTOR_LABEL.get(selector, selector), loc="left", fontsize=10)
        ax.grid(True, color="0.9", lw=0.45)
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("minimum paired upper/lower peak local z")
    fig.suptitle("Cross-antenna pair detection threshold sensitivity")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / "jupiter_cross_antenna_pair_threshold_sensitivity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    pairs: pd.DataFrame,
    summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
) -> Path:
    rank_cols = [
        "selector_label",
        "frequency_mhz",
        "n_paired_days",
        "selected_total_pairs",
        "control_total_pairs",
        "mean_selected_minus_control_pair_rate_per_1000_samples",
        "boot_lo_selected_minus_control_pair_rate_per_1000_samples",
        "boot_hi_selected_minus_control_pair_rate_per_1000_samples",
        "signflip_p_positive_pair_rate",
    ]
    top = summary.sort_values("mean_selected_minus_control_pair_rate_per_1000_samples", ascending=False) if not summary.empty else summary
    threshold_top = (
        threshold_summary[
            threshold_summary["selector"].isin(["maser_top10", "io_windows_and_maser_top10"])
            & threshold_summary["frequency_mhz"].isin([0.45, 3.93])
        ]
        .sort_values(["selector", "frequency_mhz", "min_pair_z_threshold"])
        if not threshold_summary.empty
        else threshold_summary
    )
    pair_counts = (
        pairs.groupby("frequency_mhz", sort=True)
        .size()
        .reset_index(name="n_cross_antenna_pairs")
        .to_string(index=False)
        if not pairs.empty
        else "(none)"
    )
    lines = [
        "# Jupiter Cross-Antenna Burst-Pair Detection",
        "",
        "This run pairs upper-V and lower-V burst candidates at the same frequency within the configured tolerance and counts each pair once. Pair rates in expected-active selectors are compared with same-day controls using the pair-opportunity denominator from the event-search daily tables.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Pair Counts",
        "",
        pair_counts,
        "",
        "## Strongest Positive Pair-Rate Excesses",
        "",
        top[rank_cols].head(18).to_string(index=False) if not top.empty else "(none)",
        "",
        "## Threshold Sensitivity Highlights",
        "",
        threshold_top[
            [
                "min_pair_z_threshold",
                "selector_label",
                "frequency_mhz",
                "selected_total_pairs",
                "control_total_pairs",
                "mean_selected_minus_control_pair_rate_per_1000_samples",
                "signflip_p_positive_pair_rate",
            ]
        ].to_string(index=False)
        if not threshold_top.empty
        else "(none)",
        "",
        "## Strongest Negative Pair-Rate Excesses",
        "",
        top.tail(12).sort_values("mean_selected_minus_control_pair_rate_per_1000_samples")[rank_cols].to_string(index=False)
        if not top.empty
        else "(none)",
        "",
        "## Interpretation Notes",
        "",
        "- A cross-antenna pair is a stronger candidate than a one-antenna event because both Ryle-Vonberg V channels show a same-frequency local-z excursion at nearly the same time.",
        "- The denominator is conservative: per day/frequency/selector it uses the smaller available sample count across upper and lower V.",
        "- A convincing Jupiter result should concentrate in Io/MASER-active selectors and preferably persist at neighboring frequencies; a result only at 0.45 MHz is suggestive but still needs artifact controls.",
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_cross_antenna_pair_detection_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pair-tolerance-s", type=float, default=30.0)
    parser.add_argument("--min-selected-samples-per-day", type=int, default=5)
    parser.add_argument("--min-control-samples-per-day", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--permutations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--threshold-grid", type=float, nargs="+", default=[3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0])
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    rng = np.random.default_rng(int(args.seed))
    config = {
        "events": str(args.events),
        "daily": str(args.daily),
        "pair_tolerance_s": float(args.pair_tolerance_s),
        "min_selected_samples_per_day": int(args.min_selected_samples_per_day),
        "min_control_samples_per_day": int(args.min_control_samples_per_day),
        "bootstrap_samples": int(args.bootstrap_samples),
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "threshold_grid": [float(v) for v in args.threshold_grid],
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    events = pd.read_csv(args.events, parse_dates=["burst_peak_time", "date"], low_memory=False)
    daily = pd.read_csv(args.daily, parse_dates=["date"], low_memory=False)
    pairs = build_cross_antenna_pairs(events, tolerance_s=float(args.pair_tolerance_s))
    denominators = daily_pair_denominators(
        daily,
        min_selected_samples=int(args.min_selected_samples_per_day),
        min_control_samples=int(args.min_control_samples_per_day),
    )
    selector_names = selector_flag_names(events)
    daily_tables = []
    for selector in selector_names:
        points = daily_pair_rate_points(pairs, denominators, selector)
        if not points.empty:
            daily_tables.append(points)
    daily_all = pd.concat(daily_tables, ignore_index=True) if daily_tables else pd.DataFrame()
    summary = summarize_pair_rates(
        daily_all,
        rng=rng,
        n_boot=int(args.bootstrap_samples),
        n_perm=int(args.permutations),
    )
    threshold_summary = threshold_sensitivity_summary(
        pairs,
        denominators,
        selector_names,
        thresholds=[float(v) for v in args.threshold_grid],
        rng=rng,
        n_boot=int(args.bootstrap_samples),
        n_perm=int(args.permutations),
    )

    paths: list[Path] = []
    pair_path = out_dir / "jupiter_cross_antenna_burst_pairs.csv"
    daily_path = out_dir / "jupiter_cross_antenna_pair_daily_rates.csv"
    summary_path = out_dir / "jupiter_cross_antenna_pair_summary.csv"
    threshold_summary_path = out_dir / "jupiter_cross_antenna_pair_threshold_sensitivity.csv"
    pairs.to_csv(pair_path, index=False)
    daily_all.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    threshold_summary.to_csv(threshold_summary_path, index=False)
    paths.extend([pair_path, daily_path, summary_path, threshold_summary_path])
    for maybe_path in [
        plot_pair_rate_spectrum(summary, out_dir),
        plot_pair_phase_scatter(pairs, out_dir),
        plot_threshold_sensitivity(threshold_summary, out_dir),
    ]:
        if maybe_path is not None:
            paths.append(maybe_path)
    report_path = write_report(
        out_dir,
        pairs=pairs,
        summary=summary,
        threshold_summary=threshold_summary,
        paths=paths,
        config=config,
    )
    print(f"Wrote {pair_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
