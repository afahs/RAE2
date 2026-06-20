#!/usr/bin/env python
"""Build event-level fit inspection plots using real raw data."""

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
from scipy.interpolate import UnivariateSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.detection import baseline_matrix, event_template
from rylevonberg.util import datetime_ns, robust_sigma


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
MODEL_COLORS = {
    "joint_linear": "#d62728",
    "robust_joint_huber": "#9467bd",
    "sideband_linear": "#2ca02c",
    "prepost_median": "#8c564b",
}
MODEL_LABELS = {
    "joint_linear": "joint linear",
    "robust_joint_huber": "robust Huber",
    "sideband_linear": "sideband linear",
    "prepost_median": "pre/post median",
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _antenna_label(antenna: str) -> str:
    return ANT_LABEL.get(str(antenna), str(antenna))


def _source_title(source: str) -> str:
    return source.capitalize()


def _fit_linear(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        w = np.sqrt(weights)
        beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return beta, y - X @ beta


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, event_time: pd.Timestamp, window_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = event_time.value
    half_ns = int(window_s * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    tr = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & (np.abs(tr) <= window_s)
    if "is_valid" in local.columns:
        valid &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(valid) < 8:
        return None
    tr = tr[valid]
    y = y[valid]
    order = np.argsort(tr)
    return tr[order], y[order]


def _robust_joint(tr: np.ndarray, y: np.ndarray, tmpl: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    X = np.column_stack([baseline_matrix(tr, 1), tmpl])
    weights = np.ones(len(y), dtype=float)
    beta, resid = _fit_linear(X, y, weights)
    for _ in range(8):
        sigma = robust_sigma(resid)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        cutoff = 1.5 * sigma
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(resid), cutoff))
        beta, resid = _fit_linear(X, y, weights)
    model = X @ beta
    return model, resid, float(beta[-1])


def _model_curve(
    tr: np.ndarray,
    y: np.ndarray,
    event_type: str,
    timing_offset_s: float,
    method: str,
    exclusion_s: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    tmpl = event_template(tr, event_type, timing_offset_sec=timing_offset_s)
    if method == "joint_linear":
        X = np.column_stack([baseline_matrix(tr, 1), tmpl])
        beta, resid = _fit_linear(X, y)
        return X @ beta, resid, float(beta[-1])
    if method == "robust_joint_huber":
        return _robust_joint(tr, y, tmpl)
    if method == "sideband_linear":
        fit_mask = np.abs(tr) >= float(exclusion_s)
        if np.count_nonzero(fit_mask) < 6:
            fit_mask = np.ones(len(y), dtype=bool)
        B_fit = baseline_matrix(tr[fit_mask], 1)
        beta, _ = _fit_linear(B_fit, y[fit_mask])
        baseline = baseline_matrix(tr, 1) @ beta
        resid0 = y - baseline
        den = float(np.dot(tmpl, tmpl))
        amp = float(np.dot(resid0, tmpl) / den) if den > 0 else np.nan
        model = baseline + amp * tmpl
        return model, y - model, amp
    if method == "prepost_median":
        pre_mask = tr <= -float(exclusion_s)
        post_mask = tr >= float(exclusion_s)
        if np.count_nonzero(pre_mask) < 3 or np.count_nonzero(post_mask) < 3:
            baseline = np.full_like(y, np.nanmedian(y))
            return baseline, y - baseline, np.nan
        pre_med = float(np.nanmedian(y[pre_mask]))
        post_med = float(np.nanmedian(y[post_mask]))
        model = np.where(tr < 0.0, pre_med, post_med)
        amp = (post_med - pre_med) if str(event_type) == "reappearance" else (pre_med - post_med)
        return model, y - model, float(amp)
    raise ValueError(method)


def _select_events(fits: pd.DataFrame, source: str) -> pd.DataFrame:
    quality_method = "prepost_median" if source in {"earth", "sun"} else "robust_joint_huber"
    q = fits[fits["method"].eq(quality_method)].copy()
    if q.empty:
        q = fits.copy()
    q["abs_event_snr"] = pd.to_numeric(q["event_snr"], errors="coerce").abs()
    q["abs_delta_bic"] = pd.to_numeric(q["delta_bic"], errors="coerce")
    selected = []
    # Strong examples by event type.
    selected.append(q.sort_values(["event_type", "abs_event_snr"], ascending=[True, False]).groupby("event_type").head(2))
    # Poor/model-questionable examples.
    selected.append(q.sort_values(["event_type", "abs_event_snr"], ascending=[True, True]).groupby("event_type").head(1))
    # Large BIC improvement examples if different from SNR choices.
    selected.append(q.sort_values(["event_type", "delta_bic"], ascending=[True, False]).groupby("event_type").head(1))
    out = pd.concat(selected, ignore_index=True).drop_duplicates("event_id")
    return out.head(8).reset_index(drop=True)


def _plot_event(
    tr: np.ndarray,
    y: np.ndarray,
    event: pd.Series,
    methods: list[str],
    source_summary: pd.Series,
    out_path: Path,
    exclusion_s: float,
) -> None:
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8.6, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(tr / 60.0, y, ".", ms=3.5, color="0.35", label="raw power")
    residuals = {}
    for method in methods:
        model, resid, amp = _model_curve(
            tr,
            y,
            str(event["event_type"]),
            float(source_summary["best_timing_offset_s"]),
            method,
            exclusion_s,
        )
        residuals[method] = resid
        ax.plot(tr / 60.0, model, lw=1.8, color=MODEL_COLORS.get(method), label=f"{MODEL_LABELS.get(method, method)} A={amp:.2g}")
    main_method = "prepost_median" if str(event["source_name"]) in {"earth", "sun"} else "robust_joint_huber"
    resid = residuals.get(main_method)
    if resid is None:
        resid = next(iter(residuals.values()))
    axr.axhline(0.0, color="black", lw=0.8)
    axr.plot(tr / 60.0, resid, ".", ms=3, color="0.25")
    ax.axvline(0.0, color="black", lw=1.0, ls="--", label="predicted")
    if float(source_summary["best_timing_offset_s"]) != 0.0:
        ax.axvline(float(source_summary["best_timing_offset_s"]) / 60.0, color="crimson", lw=1.0, ls=":", label="best dt")
    title = (
        f"{_source_title(str(event['source_name']))} {event['event_type']} "
        f"{pd.Timestamp(event['predicted_event_time']).date()} "
        f"{float(source_summary['target_frequency_mhz']):.2f} MHz {_antenna_label(source_summary['target_antenna'])}"
    )
    ax.set_title(title)
    ax.set_ylabel("Raw power")
    ax.legend(fontsize=7, ncol=2)
    axr.set_xlabel("Relative time from predicted event (min)")
    axr.set_ylabel("resid.")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_montage(image_paths: list[Path], out_path: Path) -> None:
    import matplotlib.image as mpimg

    if not image_paths:
        return
    ncols = 2
    nrows = int(np.ceil(len(image_paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, image_paths):
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
    for ax in axes[len(image_paths) :]:
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _build_source(
    source: str,
    clean: pd.DataFrame,
    events: pd.DataFrame,
    source_summary: pd.Series,
    fits: pd.DataFrame,
    out_dir: Path,
    exclusion_s: float,
) -> list[Path]:
    band = int(source_summary["target_band"])
    antenna = str(source_summary["target_antenna"])
    window_s = float(source_summary["target_window_s"])
    group = clean[
        clean["frequency_band"].astype(int).eq(band)
        & clean["antenna"].astype(str).eq(antenna)
    ].sort_values("time").reset_index(drop=True)
    group_ns = datetime_ns(group["time"])
    selected = _select_events(fits, source)
    selected.to_csv(out_dir / source / f"{source}_fit_inspection_event_index.csv", index=False)
    paths = []
    methods = ["joint_linear", "robust_joint_huber", "sideband_linear", "prepost_median"]
    for i, row in selected.iterrows():
        local = _event_window(group, group_ns, pd.Timestamp(row["predicted_event_time"]), window_s)
        if local is None:
            continue
        tr, y = local
        out_path = out_dir / source / f"{source}_fit_event_{i:02d}_{row['event_type']}.png"
        _plot_event(tr, y, row, methods, source_summary, out_path, exclusion_s)
        paths.append(out_path)
    montage = out_dir / source / f"{source}_fit_event_montage.png"
    _plot_montage(paths, montage)
    return [montage, *paths]


def _lower_v_summary_rows(survey: Path, summary: pd.DataFrame, timing_offset_from_original: bool = True) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        source = str(row["source_name"])
        scan = _read(survey / source / "initial_channel_scan.csv")
        if scan.empty:
            continue
        score_col = "robust_stack_snr" if "robust_stack_snr" in scan.columns else "stacked_snr"
        lower = scan[scan["antenna"].astype(str).eq("rv2_coarse")].copy()
        if lower.empty:
            continue
        best = lower.assign(abs_score=pd.to_numeric(lower[score_col], errors="coerce").abs()).sort_values("abs_score", ascending=False).iloc[0]
        timing_offset = float(row["best_timing_offset_s"]) if timing_offset_from_original else 0.0
        timing = _read(survey / source / "timing_scan.csv")
        if not timing.empty:
            matching = timing[
                timing["frequency_band"].astype(int).eq(int(best["frequency_band"]))
                & timing["antenna"].astype(str).eq("rv2_coarse")
                & pd.to_numeric(timing["window_s"], errors="coerce").eq(float(best["window_s"]))
            ].copy()
            if not matching.empty:
                tscore = "robust_stack_snr" if "robust_stack_snr" in matching.columns else "stacked_snr"
                timing_offset = float(
                    matching.assign(abs_score=pd.to_numeric(matching[tscore], errors="coerce").abs())
                    .sort_values("abs_score", ascending=False)
                    .iloc[0]["timing_offset_s"]
                )
        payload = row.to_dict()
        payload.update(
            {
                "target_band": int(best["frequency_band"]),
                "target_frequency_mhz": float(best["frequency_mhz"]),
                "target_antenna": "rv2_coarse",
                "target_window_s": float(best["window_s"]),
                "best_timing_offset_s": timing_offset,
                "lower_v_selection_score": float(best[score_col]),
                "lower_v_selection_score_column": score_col,
            }
        )
        rows.append(payload)
    return pd.DataFrame.from_records(rows)


def _write_readme(out_dir: Path, paths_by_source: dict[str, list[Path]], force_lower_v: bool = False) -> None:
    lines = [
        "# Fit Inspection Plots",
        "",
        "These figures show real raw power samples with fitted model curves overlaid.",
        "Each event plot compares joint linear, robust Huber, sideband-linear, and raw pre/post median models.",
        "The lower panel shows residuals for the source's primary inspection method.",
        "",
    ]
    if force_lower_v:
        lines.extend([
            "This run forces `rv2_coarse` / lower V for every source. Selected channels are listed in `lower_v_selected_channels.csv`.",
            "",
        ])
    for source, paths in paths_by_source.items():
        lines.extend([f"## {_source_title(source)}", ""])
        for path in paths:
            lines.append(f"- [{path.name}]({path.relative_to(out_dir)})")
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-root", default="outputs/planetary_confirmation_survey_science_baseline_v2")
    parser.add_argument("--playground-root", default="outputs/baseline_model_playground_fitquality_v1")
    parser.add_argument("--output-dir", default="outputs/fit_inspection_plots_fitquality_v1")
    parser.add_argument("--sideband-exclusion-seconds", type=float, default=120.0)
    parser.add_argument("--force-lower-v", action="store_true", help="Inspect best rv2_coarse channel for each source instead of best overall channel.")
    args = parser.parse_args()

    survey = ROOT / args.survey_root
    playground = ROOT / args.playground_root
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = _read(CLEAN, parse_dates=["time"])
    events = _read(survey / "events" / "all_planet_predicted_events.csv", parse_dates=["predicted_event_time"])
    summary = _read(survey / "planetary_confirmation_summary.csv")
    if args.force_lower_v:
        summary = _lower_v_summary_rows(survey, summary)
        summary.to_csv(out_dir / "lower_v_selected_channels.csv", index=False)
    paths_by_source: dict[str, list[Path]] = {}
    for _, source_summary in summary.iterrows():
        source = str(source_summary["source_name"])
        (out_dir / source).mkdir(parents=True, exist_ok=True)
        fits = _read(playground / source / f"{source}_baseline_model_event_fits.csv", parse_dates=["predicted_event_time"])
        paths_by_source[source] = _build_source(source, clean, events, source_summary, fits, out_dir, args.sideband_exclusion_seconds)
    _write_readme(out_dir, paths_by_source, force_lower_v=args.force_lower_v)
    print(out_dir / "README.md")


if __name__ == "__main__":
    main()
