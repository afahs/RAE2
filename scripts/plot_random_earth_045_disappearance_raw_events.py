#!/usr/bin/env python
"""Plot random occultation raw-power event windows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import datetime_ns, ensure_dir, write_json, software_versions  # noqa: E402


DEFAULT_EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/earth_predicted_events.csv"
DEFAULT_CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
DEFAULT_OUT = ROOT / "outputs/random_earth_disappearance_raw_events_v1"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, event_time: pd.Timestamp, window_s: float) -> pd.DataFrame:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(window_s * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return pd.DataFrame()
    local = group.iloc[lo:hi].copy()
    local["t_rel_sec"] = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    return local[np.abs(local["t_rel_sec"]) <= window_s].copy()


def _usable_events(events: pd.DataFrame, clean_groups: dict[str, tuple[pd.DataFrame, np.ndarray]], window_s: float) -> pd.DataFrame:
    rows = []
    for event_id, ev_group in events.groupby("event_id", sort=True):
        counts = {}
        for antenna, payload in clean_groups.items():
            group, group_ns = payload
            ev_ant = ev_group[ev_group["antenna"].astype(str).eq(antenna)]
            if ev_ant.empty:
                counts[f"n_{antenna}"] = 0
                continue
            local = _event_window(group, group_ns, pd.Timestamp(ev_ant["predicted_event_time"].iloc[0]), window_s)
            counts[f"n_{antenna}"] = int(len(local))
        if max(counts.values()) > 0:
            row = ev_group.iloc[0].to_dict()
            row.update(counts)
            rows.append(row)
    return pd.DataFrame(rows)


def _format_freq(freq_mhz: float) -> str:
    return f"{freq_mhz:.2f}".replace(".", "p")


def _source_label(source: str) -> str:
    return {
        "earth": "Earth",
        "sun": "Sun",
        "jupiter": "Jupiter",
        "fornax_a": "Fornax-A",
        "cyg_a": "Cyg-A",
        "cas_a": "Cas-A",
        "tau_a": "Tau-A",
    }.get(source.lower(), source)


def _plot(
    selected: pd.DataFrame,
    clean_groups: dict[str, tuple[pd.DataFrame, np.ndarray]],
    window_s: float,
    frequency_mhz: float,
    source: str,
    event_type: str,
    out: Path,
) -> Path:
    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(11, max(8, 1.75 * n)), sharex=True)
    if n == 1:
        axes = [axes]
    colors = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}
    for ax, ev in zip(axes, selected.itertuples(index=False)):
        for antenna, payload in clean_groups.items():
            group, group_ns = payload
            local = _event_window(group, group_ns, pd.Timestamp(ev.predicted_event_time), window_s)
            if local.empty:
                continue
            ax.plot(
                local["t_rel_sec"].to_numpy(dtype=float) / 60.0,
                local["power"].to_numpy(dtype=float),
                ".-",
                color=colors[antenna],
                markersize=3,
                linewidth=0.7,
                alpha=0.82,
                label=ANT_LABEL.get(antenna, antenna),
            )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel("raw power")
        ax.set_title(
            f"event_id {int(ev.event_id)} | {pd.Timestamp(ev.predicted_event_time)} UTC",
            fontsize=9,
            loc="left",
        )
        ax.grid(True, color="0.90", linewidth=0.5)
    axes[-1].set_xlabel(f"minutes from predicted {event_type}")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)
    fig.suptitle(
        f"{len(selected)} random {_source_label(source)} {event_type} events at {frequency_mhz:.2f} MHz, "
        f"raw power, +/-{window_s / 60:.0f} min",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    ant_suffix = "both_v" if len(clean_groups) > 1 else next(iter(clean_groups)).replace("_coarse", "")
    path = out / (
        f"{source}_{_format_freq(frequency_mhz)}mhz_{len(selected)}_random_{event_type}_"
        f"{ant_suffix}_raw_power_pm{int(window_s/60)}min.png"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--clean", default=str(DEFAULT_CLEAN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--n-events", type=int, default=10)
    parser.add_argument("--window-s", type=float, default=600.0)
    parser.add_argument("--frequency-mhz", type=float, default=0.45)
    parser.add_argument("--source", default="earth")
    parser.add_argument("--event-type", default="disappearance", choices=["disappearance", "reappearance"])
    parser.add_argument(
        "--antenna",
        choices=["rv1_coarse", "rv2_coarse", "both"],
        default="both",
        help="Antenna to plot. Use rv2_coarse for lower V only.",
    )
    args = parser.parse_args()

    out = ensure_dir(Path(args.out_dir))
    source = str(args.source).lower()
    event_type = str(args.event_type)
    events = _read(Path(args.events), parse_dates=["predicted_event_time"])
    events = events[
        events["source_name"].astype(str).str.lower().eq(source)
        & events["event_type"].astype(str).eq(event_type)
        & np.isclose(events["frequency_mhz"].astype(float), args.frequency_mhz)
    ].copy()
    if args.antenna != "both":
        events = events[events["antenna"].astype(str).eq(args.antenna)].copy()

    clean = _read(
        Path(args.clean),
        usecols=["time", "frequency_mhz", "frequency_band", "antenna", "power"],
        parse_dates=["time"],
    )
    clean = clean[np.isclose(clean["frequency_mhz"].astype(float), args.frequency_mhz)].copy()
    antennas = ["rv1_coarse", "rv2_coarse"] if args.antenna == "both" else [args.antenna]
    clean = clean[clean["antenna"].isin(antennas)].copy()
    clean_groups = {}
    for antenna, grp in clean.groupby("antenna", sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        clean_groups[str(antenna)] = (g, datetime_ns(g["time"]))

    usable = _usable_events(events, clean_groups, args.window_s)
    if usable.empty:
        raise RuntimeError(
            f"No {source} {args.frequency_mhz:.2f} MHz {event_type} events with samples inside the requested window."
        )
    selected = usable.sample(n=min(args.n_events, len(usable)), random_state=args.seed).sort_values("predicted_event_time")
    freq_tag = _format_freq(args.frequency_mhz)
    ant_tag = args.antenna if args.antenna != "both" else "both_v"
    selected_path = out / f"selected_{source}_{freq_tag}mhz_{event_type}_{ant_tag}_events.csv"
    usable_path = out / f"usable_{source}_{freq_tag}mhz_{event_type}_{ant_tag}_events.csv"
    selected.to_csv(selected_path, index=False)
    usable.to_csv(usable_path, index=False)
    plot_path = _plot(selected, clean_groups, args.window_s, args.frequency_mhz, source, event_type, out)
    write_json(
        out / "run_config.json",
        {
            "events": str(Path(args.events)),
            "clean": str(Path(args.clean)),
            "seed": args.seed,
            "n_requested_events": args.n_events,
            "n_usable_events": int(len(usable)),
            "window_s": args.window_s,
            "frequency_mhz": args.frequency_mhz,
            "antenna": args.antenna,
            "source": source,
            "event_type": event_type,
            "software_versions": software_versions(),
        },
    )
    report = [
        f"# Random {_source_label(source)} {args.frequency_mhz:.2f} MHz {event_type.title()} Raw Events",
        "",
        f"Selected `{len(selected)}` random usable {_source_label(source)} {event_type} events at {args.frequency_mhz:.2f} MHz using seed `{args.seed}`.",
        f"Antenna selection: `{args.antenna}`.",
        f"Usable event count before random selection: `{len(usable)}`.",
        "",
        "No normalization, detrending, or fitting is applied. These are raw power samples from the cleaned time-series table.",
        "",
        f"Plot: `{plot_path}`",
        f"Selected events: `{selected_path}`",
        f"Usable event table: `{usable_path}`",
    ]
    report_path = out / f"random_{source}_{freq_tag}mhz_{event_type}_{ant_tag}_raw_events_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
