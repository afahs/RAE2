"""Helper utilities converted from ingressEgressHistograms.ipynb.

These routines were promoted into a module so they can be reused by a CLI
runner, as requested. Where required, we now translate frequency bands into
MHz via RAEAnglesUtilities.bandToFreq to satisfy the instruction about
plotting frequencies in MHz.
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm.auto import tqdm

import RAEAnglesUtilities as rae
from astropy.coordinates import SkyCoord

DEFAULT_DATA_PATH = Path("/global/cfs/projectdirs/m4895/RAE2Data/interpolatedRAE2MasterFile.csv")


def load_occultation_dataframe(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    start: Union[str, pd.Timestamp] = "1974-01-01 14:00",
    end: Union[str, pd.Timestamp] = "1975-12-31 16:00",
) -> pd.DataFrame:
    """Load and time-slice the master CSV used in the notebook."""

    df = pd.read_csv(data_path)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    return df.loc[start_ts:end_ts].copy()


def append_if_valid_median_std(signal: Iterable[float], median_list: list, std_list: list) -> None:
    """Append sigma-clipped medians/stds when the values are finite."""

    median = np.nanmedian(signal)
    std = np.nanstd(signal)
    if not (np.isnan(median) or np.isnan(std)):
        median_list.append(median)
        std_list.append(std)


def append_if_valid_pair(
    pre_signal: Iterable[float],
    post_signal: Iterable[float],
    pre_median_list: list,
    pre_std_list: list,
    post_median_list: list,
    post_std_list: list,
) -> None:
    """Record paired ingress/egress statistics when both halves are usable."""

    pre_median = np.nanmedian(pre_signal)
    post_median = np.nanmedian(post_signal)
    pre_std = np.nanstd(pre_signal)
    post_std = np.nanstd(post_signal)

    if all(np.isfinite([pre_median, post_median, pre_std, post_std])):
        pre_median_list.append(pre_median)
        pre_std_list.append(pre_std)
        post_median_list.append(post_median)
        post_std_list.append(post_std)


def occultationStatisticsIngressEgressPairs(
    data: pd.DataFrame,
    col: Optional[str] = "isVis",
    window: pd.Timedelta = pd.Timedelta(minutes=2),
    antenn: str = "rv2_coarse",
    *,
    progress: bool = True,
    exclude_mask: Optional[pd.Series] = None,
    visibility_override: Optional[pd.Series] = None,
    return_counts: bool = False,
) -> Union[
    Dict[Union[int, float], Dict[str, Dict[str, list]]],
    Tuple[Dict[Union[int, float], Dict[str, Dict[str, list]]], Dict[str, int]],
]:
    """Mirror the notebook logic while adding optional progress + diagnostics."""

    freqs = sorted(data["frequency_band"].unique())
    stats: Dict[Union[int, float], Dict[str, Dict[str, list]]] = {}

    if visibility_override is not None:
        vis_series_full = visibility_override.reindex(data.index)
        if vis_series_full.isnull().any():
            raise ValueError("visibility_override index must align with dataframe index")
        vis_series_full = vis_series_full.astype(bool)
    else:
        if col is None:
            raise ValueError("Either 'col' or 'visibility_override' must be provided")
        vis_series_full = data[col].astype(bool)

    occultation_changes = vis_series_full.astype(int).diff()
    start_times = data.index[occultation_changes == -1]
    end_times = data.index[occultation_changes == 1]

    if len(end_times) > 0 and (len(start_times) == 0 or start_times[0] > end_times[0]):
        end_times = end_times[1:]

    if len(start_times) > len(end_times):
        start_times = start_times[:-1]

    pair_list: List[Tuple[pd.Timestamp, pd.Timestamp]] = [
        (s, e)
        for s, e in zip(start_times, end_times)
        if (e - s) >= pd.Timedelta(minutes=2)
    ]

    total_pairs = len(pair_list)
    guard_filtered_pairs = 0
    usable_pairs: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    exclude_mask_full: Optional[pd.Series] = None
    if exclude_mask is not None:
        exclude_mask_full = (
            exclude_mask.reindex(data.index, fill_value=False).astype(bool)
        )

    for start, end in pair_list:
        if exclude_mask_full is not None:
            start_window = start - window
            end_window = end + window
            mask_slice = exclude_mask_full.loc[
                (exclude_mask_full.index >= start_window)
                & (exclude_mask_full.index <= end_window)
            ]
            if mask_slice.any():
                guard_filtered_pairs += 1
                continue
        usable_pairs.append((start, end))

    iterator = tqdm(freqs, desc="Processing frequencies", disable=not progress)
    missing_data_events = 0
    freq_pair_attempts = 0

    for freq in iterator:
        freq_data = data[data["frequency_band"] == freq]
        ingress_stats = {"preMedian": [], "preStd": [], "postMedian": [], "postStd": []}
        egress_stats = {"preMedian": [], "preStd": [], "postMedian": [], "postStd": []}

        vis_series = freq_data[col].astype(bool) if col in freq_data else vis_series_full.loc[freq_data.index]

        for start, end in usable_pairs:
            freq_pair_attempts += 1
            pre_ingress = freq_data.loc[
                (freq_data.index >= start - window)
                & (freq_data.index < start)
                & vis_series
            ]

            post_ingress = freq_data.loc[
                (freq_data.index >= start)
                & (freq_data.index < start + window)
                & (~vis_series)
            ]

            pre_egress = freq_data.loc[
                (freq_data.index > end - window)
                & (freq_data.index <= end)
                & (~vis_series)
            ]

            post_egress = freq_data.loc[
                (freq_data.index > end)
                & (freq_data.index <= end + window)
                & vis_series
            ]

            if pre_ingress.empty or post_ingress.empty:
                print(f"Skipping empty ingress period: {start} to {end}")
                missing_data_events += 1
                continue
            if post_egress.empty or pre_egress.empty:
                print(f"Skipping empty non occult: {start} to {end}")
                missing_data_events += 1
                continue

            pre_ingress_sig = rae.sigmaClip(pre_ingress[antenn], n=5)
            post_ingress_sig = rae.sigmaClip(post_ingress[antenn], n=5)
            pre_egress_sig = rae.sigmaClip(pre_egress[antenn], n=5)
            post_egress_sig = rae.sigmaClip(post_egress[antenn], n=5)

            append_if_valid_pair(
                pre_ingress_sig,
                post_ingress_sig,
                ingress_stats["preMedian"],
                ingress_stats["preStd"],
                ingress_stats["postMedian"],
                ingress_stats["postStd"],
            )
            append_if_valid_pair(
                pre_egress_sig,
                post_egress_sig,
                egress_stats["preMedian"],
                egress_stats["preStd"],
                egress_stats["postMedian"],
                egress_stats["postStd"],
            )

        stats[freq] = {"ingress": ingress_stats, "egress": egress_stats}

    if return_counts:
        filter_counts = {
            "total_pairs": int(total_pairs),
            "guard_filtered_pairs": int(guard_filtered_pairs),
            "usable_pairs": int(len(usable_pairs)),
            "freq_pair_attempts": int(freq_pair_attempts),
            "missing_data_events": int(missing_data_events),
            "filtered_candidates": int(guard_filtered_pairs + missing_data_events),
        }
        return stats, filter_counts

    return stats


def convert_band_to_freq_mhz(freq_label: Union[int, float, str]) -> Optional[float]:
    """Convert a band identifier to MHz per the explicit instruction."""

    try:
        band = int(round(float(freq_label)))
    except (TypeError, ValueError):
        return None

    freq_values = np.atleast_1d(rae.bandToFreq(band))
    freq_mhz = float(freq_values[0])
    return freq_mhz if freq_mhz >= 0 else None


def has_histogram_content(freq_stats: Dict[str, Dict[str, Sequence[float]]]) -> bool:
    """Check if there are any paired measurements to plot."""

    for label in ("ingress", "egress"):
        pre = freq_stats[label]["preMedian"]
        post = freq_stats[label]["postMedian"]
        if len(pre) > 0 and len(post) > 0:
            return True
    return False


def compute_static_source_visibility(
    df: pd.DataFrame,
    source_name: str,
    coord: SkyCoord,
) -> Tuple[pd.Series, pd.Series]:
    """Return angle/visibility series without mutating df (parallel-safe request)."""

    angle_col = f"{source_name}Angle"
    vis_col = f"{source_name}Vis"
    source_angle = [coord.ra, coord.dec]
    angle_values = rae.raeAngFromSource(df, source_angle)
    angle_series = pd.Series(angle_values, index=df.index, name=angle_col, dtype=float)
    visibility = rae.isVisible(df, angle_series)
    vis_series = pd.Series(visibility, index=df.index, name=vis_col, dtype=bool)
    return angle_series, vis_series


def _ensure_parent_dir(save_path: Union[str, Path]) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)


SOLAR_SYSTEM_BODIES: Tuple[str, ...] = (
    "Sun",
    "Earth",
    "Moon",
    "Mercury",
    "Venus",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto",
)


def add_solar_system_visibility(
    df: pd.DataFrame,
    body_name: str,
    *,
    angle_suffix: str = "Angle",
    vis_suffix: str = "Vis",
) -> str:
    """Attach solar-system visibility columns using dynamic ephemerides."""

    angle_series, vis_series = compute_solar_system_visibility(
        df,
        body_name,
        angle_suffix=angle_suffix,
        vis_suffix=vis_suffix,
    )
    df[angle_series.name] = angle_series
    df[vis_series.name] = vis_series
    return vis_series.name


def compute_solar_system_visibility(
    df: pd.DataFrame,
    body_name: str,
    *,
    angle_suffix: str = "Angle",
    vis_suffix: str = "Vis",
) -> Tuple[pd.Series, pd.Series]:
    """Return solar-system angle/visibility without mutating df (parallel-safe request)."""

    sorted_df = df.sort_index()
    angles = rae.solarSystemAngles(sorted_df, body_name)
    if len(angles) != len(sorted_df):
        raise ValueError(
            f"solarSystemAngles returned {len(angles)} samples for {body_name}, but the dataframe has {len(sorted_df)} rows"
        )

    angle_col = f"{body_name}{angle_suffix}"
    vis_col = f"{body_name}{vis_suffix}"
    angle_series = pd.Series(angles, index=sorted_df.index, name=angle_col, dtype=float).reindex(df.index)
    visibility = rae.isVisible(df, angle_series)
    vis_series = pd.Series(visibility, index=df.index, name=vis_col, dtype=bool)
    return angle_series, vis_series


def plotIngressEgressHistograms(
    data: Dict[str, Dict[str, Sequence[float]]],
    *,
    use_std_weights: bool = False,
    min_bin_percentage: Optional[Union[float, Sequence[Optional[float]]]] = None,
    palette: Tuple[str, str] = ("tab:blue", "tab:orange"),
    suptitle: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Figure:
    """Replicate the notebook plotting with optional saving/closing."""

    labels = ["ingress", "egress"]
    n_panels = len(labels)

    if isinstance(min_bin_percentage, (int, float)) or min_bin_percentage is None:
        min_bin_percentage = [min_bin_percentage] * n_panels
    elif len(min_bin_percentage) != n_panels:
        raise ValueError("min_bin_percentage must have one entry per label")

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    for ax, label, pct in zip(axes, labels, min_bin_percentage):
        pre_median = np.asarray(data[label]["preMedian"], dtype=float)
        post_median = np.asarray(data[label]["postMedian"], dtype=float)
        pre_std = np.asarray(data[label]["preStd"], dtype=float)
        post_std = np.asarray(data[label]["postStd"], dtype=float)

        if pre_median.size == 0 and post_median.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        if pct is not None:
            if pre_median.size > 0:
                threshold_pre = np.percentile(pre_median, 100 * (1 - pct))
                keep_pre = pre_median <= threshold_pre
                pre_median = pre_median[keep_pre]
                pre_std = pre_std[keep_pre]
            if post_median.size > 0:
                threshold_post = np.percentile(post_median, 100 * (1 - pct))
                keep_post = post_median <= threshold_post
                post_median = post_median[keep_post]
                post_std = post_std[keep_post]

        pre_weights = 1 / (pre_std + 1e-9) if use_std_weights and pre_std.size else None
        post_weights = 1 / (post_std + 1e-9) if use_std_weights and post_std.size else None

        all_medians = np.concatenate([pre_median, post_median])
        if all_medians.size == 0:
            ax.text(0.5, 0.5, "Filtered out", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        bins = np.histogram_bin_edges(all_medians, bins=40)
        pre_counts, _ = np.histogram(pre_median, bins=bins, weights=pre_weights, density=True)
        post_counts, _ = np.histogram(post_median, bins=bins, weights=post_weights, density=True)

        if pre_counts.size > 0 and pre_counts.max() > 0:
            pre_counts /= pre_counts.max()
        if post_counts.size > 0 and post_counts.max() > 0:
            post_counts /= post_counts.max()

        width = np.diff(bins)
        ax.bar(
            bins[:-1],
            pre_counts,
            width=width,
            align="edge",
            alpha=0.65,
            color=palette[0],
            edgecolor="black",
            label="Pre",
        )
        ax.bar(
            bins[:-1],
            post_counts,
            width=width,
            align="edge",
            alpha=0.65,
            color=palette[1],
            edgecolor="black",
            label="Post",
        )

        ax.set_title(f"{label.capitalize()} histogram")
        ax.set_xlabel("Median signal")
        ax.set_ylabel("Normalized count")
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        ax.xaxis.get_offset_text().set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)

    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plotIngressEgressDiffHistograms(
    data: Dict[str, Dict[str, Sequence[float]]],
    *,
    use_std_weights: bool = False,
    min_bin_percentage: Optional[Union[float, Sequence[Optional[float]]]] = None,
    palette: Tuple[str, ...] = ("tab:green",),
    suptitle: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    n_bins: int = 40,
    show: bool = True,
) -> Figure:
    """Plot post - pre histograms with optional weighting."""

    labels = ["ingress", "egress"]
    n_panels = len(labels)

    if isinstance(min_bin_percentage, (int, float)) or min_bin_percentage is None:
        min_bin_percentage = [min_bin_percentage] * n_panels
    elif len(min_bin_percentage) != n_panels:
        raise ValueError("min_bin_percentage must have one entry per label")

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    filtered_diffs = {}
    filtered_pre_std = {}
    filtered_post_std = {}
    for label, pct in zip(labels, min_bin_percentage):
        pre_med = np.asarray(data[label]["preMedian"], dtype=float)
        post_med = np.asarray(data[label]["postMedian"], dtype=float)
        pre_std = np.asarray(data[label]["preStd"], dtype=float)
        post_std = np.asarray(data[label]["postStd"], dtype=float)

        n = min(len(pre_med), len(post_med), len(pre_std), len(post_std))
        if n == 0:
            filtered_diffs[label] = np.array([])
            filtered_pre_std[label] = np.array([])
            filtered_post_std[label] = np.array([])
            continue

        pre_med = pre_med[:n]
        post_med = post_med[:n]
        pre_std = pre_std[:n]
        post_std = post_std[:n]

        if pct is not None and n > 0:
            max_meds = np.maximum(pre_med, post_med)
            thr = np.percentile(max_meds, 100 * (1 - pct))
            keep = max_meds <= thr
            pre_med = pre_med[keep]
            post_med = post_med[keep]
            pre_std = pre_std[keep]
            post_std = post_std[keep]

        diff = post_med - pre_med
        filtered_diffs[label] = diff
        filtered_pre_std[label] = pre_std
        filtered_post_std[label] = post_std

    all_diff = np.hstack([filtered_diffs[label] for label in labels if filtered_diffs[label].size > 0])
    if all_diff.size == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        if suptitle:
            fig.suptitle(suptitle, fontsize=16, y=1.02)
        if save_path:
            _ensure_parent_dir(save_path)
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    max_range = np.max(np.abs(all_diff))
    bin_edges = np.linspace(-max_range, max_range, n_bins + 1)

    for ax, label in zip(axes, labels):
        diff = filtered_diffs[label]
        if diff.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        pre_s = filtered_pre_std[label]
        post_s = filtered_post_std[label]
        weights = (1 / (np.sqrt(pre_s**2 + post_s**2) + 1e-9)) if use_std_weights else None

        counts, _ = np.histogram(diff, bins=bin_edges, weights=weights, density=True)
        if counts.size > 0 and counts.max() > 0:
            counts /= counts.max()

        ax.bar(
            bin_edges[:-1],
            counts,
            width=np.diff(bin_edges),
            align="edge",
            alpha=0.75,
            color=palette[0],
            edgecolor="black",
            label=f"{label.capitalize()} Diff",
        )
        ax.set_title(f"{label.capitalize()} (Post - Pre) Histogram")
        ax.set_xlabel("Post - Pre median difference")
        ax.set_ylabel("Normalized count")
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.axvline(0, color="gray", linestyle="--", lw=1)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        ax.legend(frameon=False, fontsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)
    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def compute_ingress_egress_objective_over_freqs(
    data_per_freq: Union[Dict, Sequence],
    *,
    use_std_weights: bool = False,
    min_bin_percentage: Optional[Union[float, Sequence[Optional[float]], Sequence]] = None,
    aggregate: str = "sum",
) -> Dict[str, Sequence]:
    """Aggregate the objective metric while converting bands to MHz."""

    if isinstance(data_per_freq, dict):
        items = list(data_per_freq.items())
        try:
            items.sort(key=lambda kv: float(kv[0]))
        except Exception:
            items.sort(key=lambda kv: str(kv[0]))
    else:
        items = list(enumerate(data_per_freq))
    n_freq = len(items)

    labels = ("ingress", "egress")

    def _to_pair(p):
        if p is None or isinstance(p, (int, float)):
            return (p, p)
        if isinstance(p, dict):
            return (p.get("ingress", None), p.get("egress", None))
        if isinstance(p, (list, tuple)) and len(p) == 2:
            return (p[0], p[1])
        raise ValueError(
            "Each min_bin_percentage element must be scalar/None, "
            "a 2-seq (ingress, egress), or a dict with those keys."
        )

    if isinstance(min_bin_percentage, (int, float)) or min_bin_percentage is None or (
        isinstance(min_bin_percentage, (list, tuple))
        and len(min_bin_percentage) == 2
        and all(isinstance(x, (int, float)) or x is None for x in min_bin_percentage)
    ) or isinstance(min_bin_percentage, dict):
        p_ing, p_egr = _to_pair(min_bin_percentage)
        pct_per_freq = [(p_ing, p_egr)] * n_freq
    else:
        if not isinstance(min_bin_percentage, (list, tuple)) or len(min_bin_percentage) != n_freq:
            raise ValueError(
                "min_bin_percentage must be scalar/None, 2-seq/dict, or a list "
                "matching the number of frequency bins."
            )
        pct_per_freq = [_to_pair(p) for p in min_bin_percentage]

    def _filter_once(data, p_ing, p_egr):
        diffs = {}
        preS = {}
        postS = {}
        for label, p in zip(labels, (p_ing, p_egr)):
            pre_med = np.asarray(data[label]["preMedian"])
            post_med = np.asarray(data[label]["postMedian"])
            pre_std = np.asarray(data[label]["preStd"])
            post_std = np.asarray(data[label]["postStd"])

            n = min(len(pre_med), len(post_med), len(pre_std), len(post_std))
            if n == 0:
                diffs[label] = np.array([])
                preS[label] = np.array([])
                postS[label] = np.array([])
                continue

            pre_med = pre_med[:n]
            post_med = post_med[:n]
            pre_std = pre_std[:n]
            post_std = post_std[:n]

            if p is not None and n > 0:
                max_meds = np.maximum(pre_med, post_med)
                thr = np.percentile(max_meds, 100 * (1 - p))
                keep = max_meds <= thr
                pre_med = pre_med[keep]
                post_med = post_med[keep]
                pre_std = pre_std[keep]
                post_std = post_std[keep]

            diffs[label] = post_med - pre_med
            preS[label] = pre_std
            postS[label] = post_std
        return diffs, preS, postS

    def _agg_stat(x, w, kind):
        if x.size == 0:
            return np.nan
        if kind == "sum":
            return np.sum(w * x) if w is not None else np.sum(x)
        if kind == "mean":
            return (np.sum(w * x) / np.sum(w)) if w is not None else np.mean(x)
        if kind == "weighted_sum":
            return np.sum(w * x)
        if kind == "weighted_mean":
            sw = np.sum(w)
            return np.sum(w * x) / sw if sw > 0 else np.nan
        raise ValueError("aggregate must be one of {'sum','mean','weighted_sum','weighted_mean'}")

    def _jackknife(x, w, kind):
        n = len(x)
        if n == 0:
            return np.nan, np.nan
        theta_full = _agg_stat(x, w, kind)
        if n == 1:
            return theta_full, np.nan
        thetas = np.empty(n, dtype=float)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            mask[i] = False
            thetas[i] = _agg_stat(x[mask], (w[mask] if w is not None else None), kind)
            mask[i] = True
        theta_bar = thetas.mean()
        se = np.sqrt((n - 1) * np.mean((thetas - theta_bar) ** 2))
        return theta_full, se

    rows = []
    for (freq_label, data), (p_ing, p_egr) in zip(items, pct_per_freq):
        diffs, preS, postS = _filter_once(data, p_ing, p_egr)

        out = {}
        for label in labels:
            x = diffs[label]
            w = None
            if use_std_weights and x.size > 0:
                w = 1.0 / (np.sqrt(preS[label] ** 2 + postS[label] ** 2) + 1e-9)

            kind = aggregate
            if aggregate in ("sum", "mean") and use_std_weights:
                kind = "weighted_sum" if aggregate == "sum" else "weighted_mean"

            theta, se = _jackknife(x, w, kind)
            out[label] = {
                "metric": float(theta) if np.isfinite(theta) else np.nan,
                "stderr": float(se) if np.isfinite(se) else np.nan,
                "n_used": int(x.size),
            }

        diff_parts = [arr for arr in (diffs["ingress"], diffs["egress"]) if arr.size]
        x_comb = np.concatenate(diff_parts) if diff_parts else np.array([])

        if use_std_weights:
            w_ing = (
                1.0 / (np.sqrt(preS["ingress"] ** 2 + postS["ingress"] ** 2) + 1e-9)
                if diffs["ingress"].size
                else np.array([])
            )
            w_egr = (
                1.0 / (np.sqrt(preS["egress"] ** 2 + postS["egress"] ** 2) + 1e-9)
                if diffs["egress"].size
                else np.array([])
            )
            weight_parts = [w for w in (w_ing, w_egr) if w.size]
            w_comb = np.concatenate(weight_parts) if weight_parts else None
        else:
            w_comb = None

        kind = aggregate
        if aggregate in ("sum", "mean") and use_std_weights:
            kind = "weighted_sum" if aggregate == "sum" else "weighted_mean"

        theta_c, se_c = _jackknife(x_comb, w_comb, kind)
        out["combined"] = {
            "metric": float(theta_c) if np.isfinite(theta_c) else np.nan,
            "stderr": float(se_c) if np.isfinite(se_c) else np.nan,
            "n_used": int(x_comb.size),
        }

        freq_mhz = convert_band_to_freq_mhz(freq_label)
        freq_value = float(freq_mhz) if freq_mhz is not None else freq_label
        tick_label = f"{freq_mhz:g}" if freq_mhz is not None else str(freq_label)

        row = {"freq": freq_value, "freq_label": tick_label, "freq_band": freq_label}
        row.update(out)
        rows.append(row)

    return {"schema": ["label", "metric", "stderr", "n_used"], "rows": rows}


def plot_objective_vs_frequency(
    results: Dict[str, Sequence],
    *,
    ingress_color: str = "tab:blue",
    egress_color: str = "tab:orange",
    diff_color: str = "tab:blue",
    sum_color: str = "tab:orange",
    title: str = "Objective metric vs. Frequency",
    xlabel: str = "Frequency (MHz)",
    ylabel: str = "Metric (± jackknife SE)",
    marker: str = "o",
    capsize: float = 3,
    ingress_alpha: float = 1.0,
    egress_alpha: float = 1.0,
    combo_alpha: float = 0.2,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Tuple[Figure, Axes]:
    """Plot ingress/egress plus their ± combinations with quadrature errors."""

    rows = results.get("rows", [])
    if not rows:
        raise ValueError("No rows found in results. Did you pass the correct object?")

    freqs = [row.get("freq") for row in rows]
    tick_labels = [row.get("freq_label", str(row.get("freq"))) for row in rows]

    rows_sorted = rows
    try:
        floats = np.array(freqs, dtype=float)
        if not np.all(np.isfinite(floats)):
            raise ValueError
        order = np.argsort(floats)
        xs = floats[order]
        rows_sorted = [rows[i] for i in order]
        tick_labels = [tick_labels[i] for i in order]
    except (TypeError, ValueError):
        xs = np.arange(len(rows))
        rows_sorted = rows

    def _series(label: str):
        m = np.array([r[label]["metric"] for r in rows_sorted], dtype=float)
        s = np.array([r[label]["stderr"] for r in rows_sorted], dtype=float)
        return m, s

    m_in, s_in = _series("ingress")
    m_eg, s_eg = _series("egress")

    diff_values = m_in - m_eg
    sum_values = m_in + m_eg
    combined_error = np.sqrt(s_in**2 + s_eg**2)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(
        xs,
        m_in,
        yerr=s_in,
        fmt=marker,
        capsize=capsize,
        label="Ingress",
        color=ingress_color,
        alpha=ingress_alpha,
        linestyle="none",
    )
    ax.plot(xs, m_in, "-", color=ingress_color, alpha=ingress_alpha)

    ax.errorbar(
        xs,
        m_eg,
        yerr=s_eg,
        fmt=marker,
        capsize=capsize,
        label="Egress",
        color=egress_color,
        alpha=egress_alpha,
        linestyle="none",
    )
    ax.plot(xs, m_eg, "-", color=egress_color, alpha=egress_alpha)

    ax.errorbar(
        xs,
        diff_values,
        yerr=combined_error,
        fmt=marker,
        capsize=capsize,
        label="Ingress - Egress",
        color=diff_color,
        alpha=combo_alpha,
        linestyle="none",
    )
    ax.plot(xs, diff_values, "-", color=diff_color, alpha=combo_alpha)

    ax.errorbar(
        xs,
        sum_values,
        yerr=combined_error,
        fmt=marker,
        capsize=capsize,
        label="Ingress + Egress",
        color=sum_color,
        alpha=combo_alpha,
        linestyle="none",
    )
    ax.plot(xs, sum_values, "-", color=sum_color, alpha=combo_alpha)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if tick_labels:
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels)

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    if save_path:
        _ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


__all__ = [
    "DEFAULT_DATA_PATH",
    "append_if_valid_median_std",
    "append_if_valid_pair",
    "occultationStatisticsIngressEgressPairs",
    "convert_band_to_freq_mhz",
    "has_histogram_content",
    "load_occultation_dataframe",
    "plotIngressEgressHistograms",
    "plotIngressEgressDiffHistograms",
    "compute_ingress_egress_objective_over_freqs",
    "plot_objective_vs_frequency",
    "SOLAR_SYSTEM_BODIES",
    "add_solar_system_visibility",
]
