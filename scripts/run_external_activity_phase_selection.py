#!/usr/bin/env python
"""External activity/phase-gated lower-V occultation tests.

This run selects events using information external to the RAE-2 Ryle-Vonberg
power values:

- Sun: daily total sunspot number from SILSO.
- Jupiter: approximate System-III CML and Io phase, then Io-A/B/C style gates.

The raw detection statistic is still the event-level pre/post log contrast
computed by `run_conditional_event_detection.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.run_prepost_rank_detection import (  # noqa: E402
    _binom_sf_one_sided,
    _cliffs_delta,
    _mannwhitney_greater,
)


DEFAULT_FEATURES = (
    ROOT
    / "outputs/lower_v_conditional_event_detection_sun_jupiter_v1/"
    / "conditional_selected_event_features.csv"
)
DEFAULT_OUT = ROOT / "outputs/lower_v_external_activity_phase_selection_v1"
SILSO_DAILY_URL = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.txt"

SOURCE_LABEL = {"sun": "Sun", "jupiter": "Jupiter", "earth": "Earth"}
MODE_COLOR = {
    "all_events": "#4d4d4d",
    "sunspot_top25": "#d95f02",
    "sunspot_top10": "#b2182b",
    "jupiter_cml_io_gate": "#1b9e77",
    "jupiter_io_phase_gate": "#7570b3",
}


def download_silso_daily(out_dir: Path, refresh: bool = False) -> pd.DataFrame:
    path = out_dir / "silso_daily_sunspot_number.csv"
    raw_path = out_dir / "SN_d_tot_V2.0.txt"
    if refresh or not path.exists():
        ensure_dir(out_dir)
        with urllib.request.urlopen(SILSO_DAILY_URL, timeout=60) as response:
            raw = response.read()
        raw_path.write_bytes(raw)
        rows = []
        for line in raw.decode("ascii", errors="ignore").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            year, month, day = map(int, parts[:3])
            rows.append(
                {
                    "date": pd.Timestamp(year=year, month=month, day=day),
                    "decimal_year": float(parts[3]),
                    "sunspot_number": float(parts[4]),
                    "sunspot_std": float(parts[5]),
                    "n_observations": int(float(parts[6])),
                    "provisional_flag": str(parts[7]) if len(parts) > 7 else "",
                }
            )
        df = pd.DataFrame(rows)
        df = df[df["sunspot_number"] >= 0].copy()
        df.to_csv(path, index=False)
    return read_table(path, parse_dates=["date"])


def _wrap360(x: np.ndarray | float) -> np.ndarray | float:
    return np.mod(x, 360.0)


def cml_system_iii_deg(jd: np.ndarray) -> np.ndarray:
    """Approximate Jovian System-III CML from Project Pluto/Meeus-style formula."""
    jd = np.asarray(jd, dtype=float)
    jup_mean = (jd - 2455636.938) * 360.0 / 4332.89709
    eqn_center = 5.55 * np.sin(np.deg2rad(jup_mean))
    angle = (jd - 2451870.628) * 360.0 / 398.884 - eqn_center
    correction = (
        11.0 * np.sin(np.deg2rad(angle))
        + 5.0 * np.cos(np.deg2rad(angle))
        - 1.25 * np.cos(np.deg2rad(jup_mean))
        - eqn_center
    )
    return _wrap360(138.41 + 870.4535567 * jd + correction)


def io_phase_approx_deg(jd: np.ndarray) -> np.ndarray:
    """Approximate Io orbital phase using Io mean longitude since J2000.

    This is intentionally marked approximate. It is suitable for broad phase
    folding and first-pass event selection, not publication-grade ephemerides.
    """
    jd = np.asarray(jd, dtype=float)
    d = jd - 2451545.0
    return _wrap360(106.07719 + 203.488955790 * d)


def _between_circular(values: pd.Series | np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if lo <= hi:
        return (v >= lo) & (v <= hi)
    return (v >= lo) | (v <= hi)


def annotate_external(features: pd.DataFrame, sunspot: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["predicted_event_time"] = pd.to_datetime(out["predicted_event_time"])
    out["event_date"] = out["predicted_event_time"].dt.floor("D")
    sunspot_map = sunspot[["date", "sunspot_number"]].rename(columns={"date": "event_date"})
    out = out.merge(sunspot_map, on="event_date", how="left")

    times = Time(out["predicted_event_time"].dt.tz_localize(None).dt.to_pydatetime(), scale="utc")
    jd = np.asarray(times.jd, dtype=float)
    out["jd_utc"] = jd
    out["jupiter_cml_iii_deg"] = cml_system_iii_deg(jd)
    out["io_phase_approx_deg"] = io_phase_approx_deg(jd)

    cml = out["jupiter_cml_iii_deg"]
    io = out["io_phase_approx_deg"]
    # Theoretical Io-A/B/C boxes reported in the CML/Io phase literature.
    io_a = _between_circular(cml, 180, 300) & _between_circular(io, 180, 260)
    io_b = _between_circular(cml, 15, 240) & _between_circular(io, 40, 110)
    io_c = _between_circular(cml, 60, 280) & _between_circular(io, 200, 260)
    out["jupiter_io_a_gate"] = io_a
    out["jupiter_io_b_gate"] = io_b
    out["jupiter_io_c_gate"] = io_c
    out["jupiter_cml_io_gate"] = io_a | io_b | io_c
    out["jupiter_io_phase_gate"] = _between_circular(io, 90, 120) | _between_circular(io, 240, 270)

    sun_real = out[out["analysis_source"].astype(str).eq("sun") & out["control_family"].astype(str).eq("real")]
    valid_sunspot = pd.to_numeric(sun_real["sunspot_number"], errors="coerce").dropna()
    if valid_sunspot.empty:
        q75 = np.nan
        q90 = np.nan
    else:
        q75 = float(valid_sunspot.quantile(0.75))
        q90 = float(valid_sunspot.quantile(0.90))
    out["sunspot_real_q75_threshold"] = q75
    out["sunspot_real_q90_threshold"] = q90
    out["sunspot_top25"] = pd.to_numeric(out["sunspot_number"], errors="coerce") >= q75 if np.isfinite(q75) else False
    out["sunspot_top10"] = pd.to_numeric(out["sunspot_number"], errors="coerce") >= q90 if np.isfinite(q90) else False
    return out


def mode_mask(df: pd.DataFrame, source: str, mode: str) -> pd.Series:
    src = df["analysis_source"].astype(str).eq(source)
    if mode == "all_events":
        return src
    if mode == "sunspot_top25":
        return src & df["sunspot_top25"].astype(bool)
    if mode == "sunspot_top10":
        return src & df["sunspot_top10"].astype(bool)
    if mode == "jupiter_cml_io_gate":
        return src & df["jupiter_cml_io_gate"].astype(bool)
    if mode == "jupiter_io_phase_gate":
        return src & df["jupiter_io_phase_gate"].astype(bool)
    raise ValueError(mode)


def summarize_one(features: pd.DataFrame, source: str, mode: str) -> pd.DataFrame:
    sub = features[mode_mask(features, source, mode)].copy()
    rows = []
    real = sub[sub["control_family"].astype(str).eq("real")]
    controls = sub[~sub["control_family"].astype(str).eq("real")]
    for keys, grp in real.groupby(["frequency_band", "frequency_mhz"], sort=True):
        band, freq = keys
        vals = pd.to_numeric(grp["source_like_log_contrast"], errors="coerce").dropna().to_numpy(dtype=float)
        same_controls = controls[controls["frequency_band"].astype(int).eq(int(band))]
        cvals = pd.to_numeric(same_controls["source_like_log_contrast"], errors="coerce").dropna().to_numpy(dtype=float)
        group_meds = same_controls.groupby(["control_family", "control_id"], sort=True)["source_like_log_contrast"].median().dropna().to_numpy(dtype=float)
        n = int(len(vals))
        k = int(np.count_nonzero(vals > 0))
        med = float(np.nanmedian(vals)) if n else np.nan
        empirical = float((1 + np.count_nonzero(group_meds >= med)) / (1 + len(group_meds))) if len(group_meds) else np.nan
        rows.append(
            {
                "source": source,
                "mode": mode,
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "n_real_events": n,
                "real_sign_fraction": float(k / n) if n else np.nan,
                "real_median_log_contrast": med,
                "real_median_fractional_contrast": float(np.exp(med) - 1.0) if np.isfinite(med) else np.nan,
                "one_sided_sign_p": _binom_sf_one_sided(k, n),
                "mannwhitney_real_gt_controls_p": _mannwhitney_greater(vals, cvals),
                "cliffs_delta_real_vs_controls": _cliffs_delta(vals, cvals),
                "control_group_empirical_p_median_ge_real": empirical,
                "control_group_median_log_contrast": float(np.nanmedian(group_meds)) if len(group_meds) else np.nan,
                "control_group_q25_log_contrast": float(np.nanquantile(group_meds, 0.25)) if len(group_meds) else np.nan,
                "control_group_q75_log_contrast": float(np.nanquantile(group_meds, 0.75)) if len(group_meds) else np.nan,
                "n_control_events": int(len(cvals)),
                "n_control_groups": int(len(group_meds)),
            }
        )
    return pd.DataFrame(rows)


def classify(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    classes = []
    for _, row in out.iterrows():
        n = int(row.get("n_real_events", 0))
        med = float(row.get("real_median_log_contrast", np.nan))
        sign_frac = float(row.get("real_sign_fraction", np.nan))
        mw_p = float(row.get("mannwhitney_real_gt_controls_p", np.nan))
        emp = float(row.get("control_group_empirical_p_median_ge_real", np.nan))
        delta = float(row.get("cliffs_delta_real_vs_controls", np.nan))
        if n < 8:
            cls = "too_few_external_gate_events"
        elif med > 0 and sign_frac >= 0.65 and mw_p <= 0.01 and emp <= 0.10 and delta >= 0.15:
            cls = "external_gate_candidate"
        elif med > 0 and sign_frac >= 0.60 and mw_p <= 0.05:
            cls = "positive_but_control_or_count_limited"
        elif med < 0 and sign_frac <= 0.40:
            cls = "anti_template"
        else:
            cls = "not_detected"
        classes.append(cls)
    out["external_gate_evidence_class"] = classes
    return out


def summarize_external(features: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for mode in ["all_events", "sunspot_top25", "sunspot_top10"]:
        pieces.append(summarize_one(features, "sun", mode))
    for mode in ["all_events", "jupiter_cml_io_gate", "jupiter_io_phase_gate"]:
        pieces.append(summarize_one(features, "jupiter", mode))
    return classify(pd.concat([p for p in pieces if not p.empty], ignore_index=True))


def plot_external_spectra(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for source, src in summary.groupby("source", sort=True):
        fig, ax = plt.subplots(figsize=(10.5, 5.0))
        ax.axhline(0, color="0.65", lw=0.8)
        for mode, grp in src.groupby("mode", sort=False):
            grp = grp.sort_values("frequency_mhz")
            ax.plot(
                grp["frequency_mhz"],
                grp["real_median_log_contrast"],
                marker="o",
                lw=1.5,
                color=MODE_COLOR.get(mode, None),
                label=mode,
            )
            for _, row in grp.iterrows():
                if str(row.get("external_gate_evidence_class")) == "external_gate_candidate":
                    ax.scatter(float(row["frequency_mhz"]), float(row["real_median_log_contrast"]), s=120, facecolor="none", edgecolor="black", lw=1.5)
        ax.set_xscale("log")
        ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("median source-like log contrast")
        ax.set_title(f"{SOURCE_LABEL.get(source, source)} external-gated lower-V test")
        ax.grid(True, color="0.9", lw=0.5)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{source}_external_gate_spectrum.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def select_visual_cases(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, src in summary.groupby("source", sort=True):
        cand = src[src["mode"].ne("all_events")].copy()
        cand["_score"] = (
            cand["real_median_log_contrast"].fillna(-999)
            + cand["real_sign_fraction"].fillna(0)
            - cand["control_group_empirical_p_median_ge_real"].fillna(1)
            + 0.5 * (cand["external_gate_evidence_class"].eq("external_gate_candidate")).astype(float)
        )
        rows.append(cand.sort_values("_score", ascending=False).head(3))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_case_distributions(features: pd.DataFrame, selected: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for source, cases in selected.groupby("source", sort=True):
        fig, axes = plt.subplots(len(cases), 1, figsize=(9.5, 3.1 * len(cases)))
        if len(cases) == 1:
            axes = [axes]
        for ax, (_, case) in zip(axes, cases.iterrows()):
            mode = str(case["mode"])
            freq = float(case["frequency_mhz"])
            sub = features[mode_mask(features, source, mode) & np.isclose(pd.to_numeric(features["frequency_mhz"], errors="coerce"), freq)].copy()
            real = sub[sub["control_family"].astype(str).eq("real")]["source_like_log_contrast"].dropna().to_numpy(dtype=float)
            ctrl = sub[~sub["control_family"].astype(str).eq("real")]["source_like_log_contrast"].dropna().to_numpy(dtype=float)
            vals = np.r_[real, ctrl]
            if len(vals) == 0:
                continue
            lo = float(np.nanpercentile(vals, 1))
            hi = float(np.nanpercentile(vals, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = -1.0, 1.0
            bins = np.linspace(lo, hi, 45)
            ax.hist(real, bins=bins, density=True, histtype="step", color="black", lw=2.0, label=f"real n={len(real)}")
            ax.hist(ctrl, bins=bins, density=True, histtype="step", color="#7570b3", lw=1.5, label=f"controls n={len(ctrl)}")
            ax.axvline(0, color="0.55", lw=0.8)
            ax.axvline(float(case["real_median_log_contrast"]), color="black", ls="--", lw=1.2)
            ax.set_title(
                f"{SOURCE_LABEL.get(source, source)} {freq:.2f} MHz {mode}: "
                f"{case['external_gate_evidence_class']}, sign={float(case['real_sign_fraction']):.2f}, "
                f"emp p={float(case['control_group_empirical_p_median_ge_real']):.3g}"
            )
            ax.set_xlabel("source-like log contrast")
            ax.set_ylabel("density")
            ax.grid(True, color="0.92", lw=0.5)
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{source}_external_gate_selected_distributions.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_jupiter_phase(features: pd.DataFrame, out_dir: Path) -> Path:
    real = features[features["analysis_source"].astype(str).eq("jupiter") & features["control_family"].astype(str).eq("real")].copy()
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    if not real.empty:
        band = real[real["frequency_mhz"].eq(0.45)].copy()
        if band.empty:
            band = real.copy()
        sc = ax.scatter(
            band["jupiter_cml_iii_deg"],
            band["io_phase_approx_deg"],
            c=band["source_like_log_contrast"],
            cmap="coolwarm",
            s=45,
            edgecolor="black",
            linewidth=0.25,
        )
        fig.colorbar(sc, ax=ax, label="source-like log contrast")
    # Io-A/B/C broad boxes.
    boxes = [
        ("Io-A", 180, 300, 180, 260, "#1b9e77"),
        ("Io-B", 15, 240, 40, 110, "#d95f02"),
        ("Io-C", 60, 280, 200, 260, "#7570b3"),
    ]
    for label, x0, x1, y0, y1, color in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=1.5))
        ax.text(x0 + 4, y1 - 6, label, color=color, fontsize=9, va="top")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xlabel("approx. System-III CML (deg)")
    ax.set_ylabel("approx. Io phase (deg)")
    ax.set_title("Jupiter real lower-V events in approximate CML/Io phase plane")
    ax.grid(True, color="0.9", lw=0.5)
    fig.tight_layout()
    path = out_dir / "jupiter_cml_io_phase_real_events.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sunspot_timeline(features: pd.DataFrame, sunspot: pd.DataFrame, out_dir: Path) -> Path:
    real = features[features["analysis_source"].astype(str).eq("sun") & features["control_family"].astype(str).eq("real")].copy()
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    date_min = real["event_date"].min() - pd.Timedelta(days=30)
    date_max = real["event_date"].max() + pd.Timedelta(days=30)
    sp = sunspot[(sunspot["date"] >= date_min) & (sunspot["date"] <= date_max)].copy()
    ax.plot(sp["date"], sp["sunspot_number"], color="0.35", lw=1.0, label="daily sunspot number")
    daily = real.groupby("event_date")["source_like_log_contrast"].median().reset_index()
    ax2 = ax.twinx()
    ax2.scatter(daily["event_date"], daily["source_like_log_contrast"], s=18, color="#d95f02", alpha=0.75, label="Sun event median contrast")
    q75 = float(real["sunspot_real_q75_threshold"].dropna().iloc[0]) if real["sunspot_real_q75_threshold"].notna().any() else np.nan
    q90 = float(real["sunspot_real_q90_threshold"].dropna().iloc[0]) if real["sunspot_real_q90_threshold"].notna().any() else np.nan
    if np.isfinite(q75):
        ax.axhline(q75, color="#d95f02", ls="--", lw=0.9, label="real-event q75")
    if np.isfinite(q90):
        ax.axhline(q90, color="#b2182b", ls=":", lw=1.0, label="real-event q90")
    ax.set_ylabel("daily sunspot number")
    ax2.set_ylabel("median source-like log contrast")
    ax.set_title("Solar external activity gate from SILSO daily sunspot number")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(frameon=False, loc="upper left")
    ax2.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = out_dir / "sun_silso_sunspot_timeline.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, summary: pd.DataFrame, selected: pd.DataFrame, paths: list[Path], config: dict[str, object]) -> Path:
    cols = [
        "source",
        "mode",
        "frequency_mhz",
        "external_gate_evidence_class",
        "n_real_events",
        "real_sign_fraction",
        "real_median_log_contrast",
        "mannwhitney_real_gt_controls_p",
        "control_group_empirical_p_median_ge_real",
    ]
    lines = [
        "# External Activity / Phase Selection Test",
        "",
        "This run selects events using information external to the RAE-2 power values.",
        "",
        "Solar gate:",
        "",
        "- Daily total sunspot number from SILSO.",
        "- `sunspot_top25`: dates at or above the 75th percentile among real Sun event dates.",
        "- `sunspot_top10`: dates at or above the 90th percentile among real Sun event dates.",
        "",
        "Jupiter gate:",
        "",
        "- Approximate System-III CML from the Project Pluto/Meeus-style formula.",
        "- Approximate Io phase from Io mean longitude folding.",
        "- Broad Io-A/B/C CML-Io boxes are used as a first-pass external geometry gate.",
        "",
        "These are selection gates only; the detection statistic remains raw lower-V pre/post log contrast.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Selected Visual Cases",
        "",
        selected[cols].to_string(index=False) if not selected.empty else "(none)",
        "",
        "## Summary Table",
        "",
        summary[cols].sort_values(["source", "mode", "frequency_mhz"]).to_string(index=False),
        "",
        "## Plots",
        "",
        *[f"- `{p}`" for p in paths],
        "",
        "## Caveats",
        "",
        "- The Io phase calculation is approximate; this is enough for a first-pass fold, but not final ephemeris-grade Jupiter radio prediction.",
        "- Sunspot number is a broadband solar activity proxy, not a direct low-frequency solar burst catalog.",
        "- A source claim still requires real-control separation in the distribution plots, not only positive selected-event contrast.",
    ]
    path = out_dir / "external_activity_phase_selection_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh-sunspot", action="store_true")
    parser.add_argument(
        "--save-large-tables",
        action="store_true",
        help="Save the full externally annotated event table. This is usually tens of MB.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    external_dir = ensure_dir(out_dir / "external_data")
    sunspot = download_silso_daily(external_dir, refresh=bool(args.refresh_sunspot))
    features = read_table(args.features, parse_dates=["predicted_event_time"], low_memory=False)
    annotated = annotate_external(features, sunspot)
    if args.save_large_tables:
        annotated.to_csv(out_dir / "external_annotated_event_features.csv", index=False)
    summary = summarize_external(annotated)
    summary.to_csv(out_dir / "external_gate_detection_summary.csv", index=False)
    selected = select_visual_cases(summary)
    selected.to_csv(out_dir / "external_gate_selected_visual_cases.csv", index=False)
    if not selected.empty:
        selected_rows = []
        for _, case in selected.iterrows():
            source = str(case["source"])
            mode = str(case["mode"])
            freq = float(case["frequency_mhz"])
            sub = annotated[
                mode_mask(annotated, source, mode)
                & np.isclose(pd.to_numeric(annotated["frequency_mhz"], errors="coerce"), freq)
            ].copy()
            sub["selected_external_mode"] = mode
            selected_rows.append(sub)
        pd.concat(selected_rows, ignore_index=True).to_csv(
            out_dir / "external_gate_selected_event_features.csv",
            index=False,
        )

    paths: list[Path] = []
    paths.extend(plot_external_spectra(summary, out_dir))
    paths.extend(plot_case_distributions(annotated, selected, out_dir))
    paths.append(plot_jupiter_phase(annotated, out_dir))
    paths.append(plot_sunspot_timeline(annotated, sunspot, out_dir))

    config = {
        "features": str(args.features),
        "sunspot_url": SILSO_DAILY_URL,
        "jupiter_cml_formula": "Project Pluto simple CML III formula",
        "jupiter_io_phase": "approximate Io mean longitude fold since J2000",
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    report = write_report(out_dir, summary, selected, paths, config)
    print(report)


if __name__ == "__main__":
    main()
