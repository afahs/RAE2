#!/usr/bin/env python
"""Low-frequency external selectors for Sun and Jupiter.

This run improves on the earlier broad activity gates:

- Sun: use NOAA/NCEI RSTN spectral listings, specifically the dekameter
  10-29 MHz event fields, rather than only fixed-frequency reports.
- Jupiter: use published MASER/PADC Jovian probability/occurrence map image
  products sampled at SPICE-derived CML/Io coordinates, rather than only
  hand-written CML/Io boxes.

The underlying RAE-2 statistic remains the lower-V raw pre/post log contrast
already produced by the conditional-event pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import urllib.request

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import rgb_to_hsv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.run_prepost_rank_detection import (  # noqa: E402
    _binom_sf_one_sided,
    _cliffs_delta,
    _mannwhitney_greater,
)
from scripts.run_solar_burst_spice_jupiter_visibility import (  # noqa: E402
    ANTENNA,
    annotate_jupiter_spice,
    count_bursts_near,
)


DEFAULT_FEATURES = (
    ROOT
    / "outputs/lower_v_conditional_event_detection_sun_jupiter_v1/"
    / "conditional_selected_event_features.csv"
)
DEFAULT_OUT = ROOT / "outputs/lower_v_lowfreq_external_selectors_v1"

SPECTRAL_URL_TEMPLATE = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/"
    "solar-radio/radio-bursts/reports/spectral-listings/{year}/SPECTRAL.{yy}"
)

MASER_MAPS = {
    "maser_zarka_full": {
        "url": "https://maser.obspm.fr/doi/10.25935/1zbg-ek27/content/zarka-2018a-fig-1a.png",
        "description": "Zarka et al. 2018a full-catalog Io-CML probability map, NDA 1990-2015",
        "kind": "color_probability",
    },
    "maser_zarka_io": {
        "url": "https://maser.obspm.fr/doi/10.25935/1zbg-ek27/content/zarka-2018a-fig-3a.png",
        "description": "Zarka et al. 2018a Io-controlled Io-CML probability map, NDA 1990-2015",
        "kind": "color_probability",
    },
    "maser_leblanc_1978": {
        "url": "https://maser.obspm.fr/doi/10.25935/1zbg-ek27/content/leblanc-1993-fig-2.png",
        "description": "Leblanc et al. 1993 Io-CML occurrence map, NDA 1978-1988",
        "kind": "dark_occurrence",
    },
}

MODE_COLOR = {
    "all_events": "#4d4d4d",
    "dek_near_1h": "#b2182b",
    "dek_near_6h": "#d95f02",
    "dek_typeiii_near_6h": "#e6ab02",
    "dek_high_day": "#7570b3",
    "maser_zarka_full_top25": "#1b9e77",
    "maser_zarka_full_top10": "#006d2c",
    "maser_zarka_io_top25": "#7570b3",
    "maser_zarka_io_top10": "#54278f",
    "maser_leblanc_1978_top25": "#e7298a",
}

SOURCE_LABEL = {"sun": "Sun", "jupiter": "Jupiter"}


def download(url: str, path: Path, refresh: bool = False) -> Path:
    if refresh or not path.exists():
        ensure_dir(path.parent)
        with urllib.request.urlopen(url, timeout=180) as response:
            path.write_bytes(response.read())
    return path


def parse_spectral_time(token: str) -> float | None:
    """Parse NOAA spectral HHMM.t fields into decimal hours."""
    s = re.sub(r"[^0-9]", "", str(token or ""))
    if len(s) < 4:
        return None
    hour = int(s[:2])
    minute = int(s[2:4])
    if hour > 23 or minute > 59:
        return None
    frac_min = int(s[4]) / 10.0 if len(s) >= 5 else 0.0
    return hour + (minute + frac_min) / 60.0


def _time_from_field(date: pd.Timestamp, token: str) -> pd.Timestamp | pd.NaT:
    hour = parse_spectral_time(token)
    if hour is None:
        return pd.NaT
    return date + pd.to_timedelta(hour, unit="h")


def parse_noaa_spectral_line(line: str) -> dict[str, object] | None:
    if len(line) < 70 or not re.match(r"^[A-Z0-9]{4}\d{6}", line):
        return None
    station = line[0:4].strip()
    yy = int(line[4:6])
    year = 1900 + yy if yy >= 50 else 2000 + yy
    month = int(line[6:8])
    day = int(line[8:10])
    date = pd.Timestamp(year=year, month=month, day=day)
    dek_start = _time_from_field(date, line[57:62])
    dek_end = _time_from_field(date, line[63:68])
    if pd.notna(dek_start) and pd.notna(dek_end) and dek_end < dek_start:
        dek_end = dek_end + pd.Timedelta(days=1)
    intensity_raw = line[69:70].strip()
    intensity = int(intensity_raw) if intensity_raw.isdigit() else np.nan
    spectral_type = line[70:100].strip()
    return {
        "station": station,
        "date": date,
        "observation_start_time": _time_from_field(date, line[10:14]),
        "observation_end_time": _time_from_field(date, line[14:18]),
        "dekameter_start_time": dek_start,
        "dekameter_end_time": dek_end,
        "dekameter_intensity": intensity,
        "spectral_type": spectral_type,
        "is_type_iii": "III" in spectral_type.upper(),
        "is_type_v": "V" in spectral_type.upper(),
        "raw_line": line.rstrip("\n"),
    }


def download_parse_noaa_spectral(start_year: int, end_year: int, out_dir: Path, refresh: bool = False) -> pd.DataFrame:
    rows = []
    raw_dir = ensure_dir(out_dir / "external_data" / "noaa_spectral")
    for year in range(int(start_year), int(end_year) + 1):
        yy = f"{year % 100:02d}"
        url = SPECTRAL_URL_TEMPLATE.format(year=year, yy=yy)
        path = raw_dir / f"SPECTRAL.{yy}"
        try:
            download(url, path, refresh=refresh)
        except Exception as exc:
            print(f"WARNING: could not download {url}: {exc}", file=sys.stderr)
            continue
        for line in path.read_text(encoding="latin1", errors="replace").splitlines():
            row = parse_noaa_spectral_line(line)
            if row is not None:
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[pd.notna(out["dekameter_start_time"])].copy()
        out = out.sort_values("dekameter_start_time").reset_index(drop=True)
        out.to_csv(out_dir / "noaa_spectral_dekameter_events_parsed.csv", index=False)
    return out


def _daily_counts(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["event_date", f"{suffix}_daily_count", f"{suffix}_daily_max_intensity"])
    return (
        df.groupby("date", sort=True)
        .agg(
            **{
                f"{suffix}_daily_count": ("dekameter_start_time", "size"),
                f"{suffix}_daily_max_intensity": ("dekameter_intensity", "max"),
            }
        )
        .reset_index()
        .rename(columns={"date": "event_date"})
    )


def annotate_solar_spectral(features: pd.DataFrame, spectral: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["predicted_event_time"] = pd.to_datetime(out["predicted_event_time"])
    out["event_date"] = out["predicted_event_time"].dt.floor("D")
    dek = spectral.copy()
    typeiii = dek[dek["is_type_iii"].astype(bool)].copy() if not dek.empty else dek
    if dek.empty:
        out["dek_burst_count_1h"] = 0
        out["dek_burst_count_6h"] = 0
        out["dek_typeiii_count_6h"] = 0
    else:
        out["dek_burst_count_1h"] = count_bursts_near(out["predicted_event_time"], dek["dekameter_start_time"], 1.0)
        out["dek_burst_count_6h"] = count_bursts_near(out["predicted_event_time"], dek["dekameter_start_time"], 6.0)
        out["dek_typeiii_count_6h"] = count_bursts_near(out["predicted_event_time"], typeiii["dekameter_start_time"], 6.0)
    daily = _daily_counts(dek, "dek")
    typeiii_daily = _daily_counts(typeiii, "dek_typeiii")
    out = out.merge(daily, on="event_date", how="left")
    out = out.merge(typeiii_daily, on="event_date", how="left")
    for col in [
        "dek_daily_count",
        "dek_daily_max_intensity",
        "dek_typeiii_daily_count",
        "dek_typeiii_daily_max_intensity",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna(0)
        else:
            out[col] = 0
    sun_real = out[out["analysis_source"].astype(str).eq("sun") & out["control_family"].astype(str).eq("real")]
    q75 = float(sun_real["dek_daily_count"].quantile(0.75)) if not sun_real.empty else np.inf
    out["dek_high_day_threshold"] = q75
    out["dek_near_1h"] = out["dek_burst_count_1h"].astype(int) > 0
    out["dek_near_6h"] = out["dek_burst_count_6h"].astype(int) > 0
    out["dek_typeiii_near_6h"] = out["dek_typeiii_count_6h"].astype(int) > 0
    out["dek_high_day"] = out["dek_daily_count"].astype(float) >= q75
    return out


@dataclass
class ProbabilityMap:
    name: str
    path: Path
    description: str
    kind: str
    score_image: np.ndarray


def load_probability_map(name: str, meta: dict[str, str], out_dir: Path, refresh: bool = False) -> ProbabilityMap:
    path = download(meta["url"], out_dir / "external_data" / "maser_probability_maps" / Path(meta["url"]).name, refresh=refresh)
    img = mpimg.imread(path)
    rgb = np.asarray(img[..., :3], dtype=float)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    if meta.get("kind") == "dark_occurrence":
        score = 1.0 - np.mean(rgb, axis=2)
    else:
        hsv = rgb_to_hsv(rgb)
        score = hsv[..., 1] * hsv[..., 2]
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.nanpercentile(score, [1, 99.5])
    if hi > lo:
        score = np.clip((score - lo) / (hi - lo), 0.0, 1.0)
    return ProbabilityMap(
        name=name,
        path=path,
        description=str(meta["description"]),
        kind=str(meta.get("kind", "")),
        score_image=score,
    )


def sample_map_score(pmap: ProbabilityMap, cml_deg: pd.Series, phase_deg: pd.Series) -> np.ndarray:
    image = pmap.score_image
    h, w = image.shape
    cml = np.asarray(cml_deg, dtype=float)
    phase = np.asarray(phase_deg, dtype=float)
    valid = np.isfinite(cml) & np.isfinite(phase)
    x = np.full(len(cml), np.nan, dtype=float)
    y = np.full(len(phase), np.nan, dtype=float)
    x[valid] = (cml[valid] % 360.0) / 360.0 * (w - 1)
    y[valid] = (1.0 - (phase[valid] % 360.0) / 360.0) * (h - 1)
    xi = np.zeros(len(x), dtype=int)
    yi = np.zeros(len(y), dtype=int)
    xi[valid] = np.clip(np.rint(x[valid]).astype(int), 0, w - 1)
    yi[valid] = np.clip(np.rint(y[valid]).astype(int), 0, h - 1)
    out = np.full(len(xi), np.nan, dtype=float)
    out[valid] = image[yi[valid], xi[valid]]
    return out


def annotate_jupiter_probability_maps(features: pd.DataFrame, out_dir: Path, refresh: bool = False) -> tuple[pd.DataFrame, list[ProbabilityMap]]:
    out = annotate_jupiter_spice(features, out_dir, refresh_kernels=refresh)
    maps = [load_probability_map(name, meta, out_dir, refresh=refresh) for name, meta in MASER_MAPS.items()]
    for pmap in maps:
        normal = sample_map_score(pmap, out["jupiter_cml_spice_deg"], out["io_phase_spice_deg"])
        reversed_phase = sample_map_score(pmap, out["jupiter_cml_spice_deg"], out["io_phase_spice_reverse_deg"])
        stacked = np.vstack([normal, reversed_phase])
        score = np.full(stacked.shape[1], np.nan, dtype=float)
        finite = np.isfinite(stacked).any(axis=0)
        score[finite] = np.nanmax(stacked[:, finite], axis=0)
        out[f"{pmap.name}_score"] = score
        real_jup = out[out["analysis_source"].astype(str).eq("jupiter") & out["control_family"].astype(str).eq("real")]
        vals = pd.to_numeric(real_jup[f"{pmap.name}_score"], errors="coerce").dropna()
        q75 = float(vals.quantile(0.75)) if not vals.empty else np.inf
        q90 = float(vals.quantile(0.90)) if not vals.empty else np.inf
        out[f"{pmap.name}_top25_threshold"] = q75
        out[f"{pmap.name}_top10_threshold"] = q90
        out[f"{pmap.name}_top25"] = pd.to_numeric(out[f"{pmap.name}_score"], errors="coerce") >= q75
        out[f"{pmap.name}_top10"] = pd.to_numeric(out[f"{pmap.name}_score"], errors="coerce") >= q90
    return out, maps


def mode_mask(df: pd.DataFrame, source: str, mode: str) -> pd.Series:
    src = df["analysis_source"].astype(str).eq(source)
    if mode == "all_events":
        return src
    if mode in {"dek_near_1h", "dek_near_6h", "dek_typeiii_near_6h", "dek_high_day"}:
        return src & df[mode].astype(bool)
    if mode.endswith("_top25") or mode.endswith("_top10"):
        return src & df[mode].astype(bool)
    raise ValueError(mode)


def summarize_modes(features: pd.DataFrame, source_modes: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for source, modes in source_modes.items():
        for mode in modes:
            sub = features[mode_mask(features, source, mode)].copy()
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
                emp = float((1 + np.count_nonzero(group_meds >= med)) / (1 + len(group_meds))) if len(group_meds) else np.nan
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
                        "control_group_empirical_p_median_ge_real": emp,
                        "control_group_median_log_contrast": float(np.nanmedian(group_meds)) if len(group_meds) else np.nan,
                        "n_control_events": int(len(cvals)),
                        "n_control_groups": int(len(group_meds)),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    evidence = []
    for _, row in out.iterrows():
        n = int(row["n_real_events"])
        med = float(row["real_median_log_contrast"])
        sign = float(row["real_sign_fraction"])
        mw = float(row["mannwhitney_real_gt_controls_p"])
        emp = float(row["control_group_empirical_p_median_ge_real"])
        delta = float(row["cliffs_delta_real_vs_controls"])
        if n < 8:
            label = "too_few_selected_events"
        elif med > 0 and sign >= 0.65 and mw <= 0.01 and emp <= 0.10 and delta >= 0.15:
            label = "externally_selected_candidate"
        elif med > 0 and sign >= 0.60 and mw <= 0.05:
            label = "positive_but_control_limited"
        elif med < 0 and sign <= 0.40:
            label = "anti_template"
        else:
            label = "not_detected"
        evidence.append(label)
    out["evidence_class"] = evidence
    return out


def select_visual_cases(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, src in summary.groupby("source", sort=True):
        cand = src[src["mode"].ne("all_events")].copy()
        cand["_score"] = (
            cand["real_median_log_contrast"].fillna(-999)
            + cand["real_sign_fraction"].fillna(0)
            - cand["control_group_empirical_p_median_ge_real"].fillna(1)
            + 0.5 * cand["evidence_class"].eq("externally_selected_candidate").astype(float)
        )
        rows.append(cand.sort_values("_score", ascending=False).head(5))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_mode_spectra(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for source, src in summary.groupby("source", sort=True):
        fig, ax = plt.subplots(figsize=(11.0, 5.2))
        ax.axhline(0, color="0.65", lw=0.8)
        for mode, grp in src.groupby("mode", sort=False):
            grp = grp.sort_values("frequency_mhz")
            ax.plot(
                grp["frequency_mhz"],
                grp["real_median_log_contrast"],
                marker="o",
                lw=1.5,
                label=mode,
                color=MODE_COLOR.get(str(mode), None),
            )
        ax.set_xscale("log")
        ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("median source-like log contrast")
        ax.set_title(f"{SOURCE_LABEL.get(source, source)} low-frequency external selector spectrum")
        ax.grid(True, color="0.9", lw=0.5)
        ax.legend(frameon=False, fontsize=7, ncols=2)
        fig.tight_layout()
        path = out_dir / f"{source}_lowfreq_external_selector_spectrum.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_selected_distributions(features: pd.DataFrame, selected: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    if selected.empty:
        return paths
    for source, cases in selected.groupby("source", sort=True):
        fig, axes = plt.subplots(len(cases), 1, figsize=(9.7, 3.0 * len(cases)))
        if len(cases) == 1:
            axes = [axes]
        for ax, (_, case) in zip(axes, cases.iterrows()):
            mode = str(case["mode"])
            freq = float(case["frequency_mhz"])
            sub = features[
                mode_mask(features, source, mode)
                & np.isclose(pd.to_numeric(features["frequency_mhz"], errors="coerce"), freq)
            ].copy()
            real = sub[sub["control_family"].astype(str).eq("real")]["source_like_log_contrast"].dropna().to_numpy(dtype=float)
            ctrl = sub[~sub["control_family"].astype(str).eq("real")]["source_like_log_contrast"].dropna().to_numpy(dtype=float)
            vals = np.r_[real, ctrl]
            if len(vals) == 0:
                continue
            lo, hi = np.nanpercentile(vals, [1, 99])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = -1.0, 1.0
            bins = np.linspace(float(lo), float(hi), 45)
            ax.hist(real, bins=bins, density=True, histtype="step", color="black", lw=2.0, label=f"real n={len(real)}")
            ax.hist(ctrl, bins=bins, density=True, histtype="step", color="#7570b3", lw=1.5, label=f"controls n={len(ctrl)}")
            ax.axvline(0, color="0.55", lw=0.8)
            ax.axvline(float(case["real_median_log_contrast"]), color="black", ls="--", lw=1.2)
            ax.set_title(
                f"{SOURCE_LABEL.get(source, source)} {freq:.2f} MHz {mode}: "
                f"{case['evidence_class']}, sign={float(case['real_sign_fraction']):.2f}, "
                f"emp p={float(case['control_group_empirical_p_median_ge_real']):.3g}"
            )
            ax.set_xlabel("source-like log contrast")
            ax.set_ylabel("density")
            ax.grid(True, color="0.92", lw=0.5)
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{source}_lowfreq_external_selected_distributions.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_jupiter_maps(features: pd.DataFrame, maps: list[ProbabilityMap], out_dir: Path) -> list[Path]:
    paths = []
    real = features[features["analysis_source"].astype(str).eq("jupiter") & features["control_family"].astype(str).eq("real")].copy()
    for pmap in maps:
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        ax.imshow(pmap.score_image, origin="upper", extent=[0, 360, 0, 360], cmap="magma", aspect="equal")
        if not real.empty:
            sc = ax.scatter(
                real["jupiter_cml_spice_deg"],
                real["io_phase_spice_deg"],
                c=real["source_like_log_contrast"],
                cmap="coolwarm",
                s=42,
                edgecolor="white",
                linewidth=0.35,
            )
            fig.colorbar(sc, ax=ax, label="RAE-2 source-like log contrast")
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_xlabel("SPICE System-III CML proxy (deg)")
        ax.set_ylabel("SPICE Io phase (deg)")
        ax.set_title(f"{pmap.name}: sampled MASER map with RAE-2 events")
        ax.grid(True, color="white", alpha=0.25, lw=0.5)
        fig.tight_layout()
        path = out_dir / f"{pmap.name}_rae2_event_overlay.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_solar_spectral_timeline(features: pd.DataFrame, spectral: pd.DataFrame, out_dir: Path) -> Path:
    sun = features[features["analysis_source"].astype(str).eq("sun") & features["control_family"].astype(str).eq("real")].copy()
    daily_contrast = sun.groupby("event_date")["source_like_log_contrast"].median().reset_index()
    daily = spectral.groupby("date").size().rename("dekameter_event_count").reset_index() if not spectral.empty else pd.DataFrame(columns=["date", "dekameter_event_count"])
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.bar(daily["date"], daily["dekameter_event_count"], width=1.0, color="0.72", label="NOAA spectral dekameter events/day")
    ax2 = ax.twinx()
    ax2.scatter(daily_contrast["event_date"], daily_contrast["source_like_log_contrast"], s=18, color="#b2182b", alpha=0.75, label="Sun event median contrast")
    ax.set_ylabel("10-29 MHz spectral events/day")
    ax2.set_ylabel("median source-like log contrast")
    ax.set_title("NOAA dekameter spectral events against RAE-2 Sun event dates")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(frameon=False, loc="upper left")
    ax2.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = out_dir / "sun_noaa_dekameter_spectral_timeline.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    spectral: pd.DataFrame,
    maps: list[ProbabilityMap],
    paths: list[Path],
    config: dict[str, object],
) -> Path:
    cols = [
        "source",
        "mode",
        "frequency_mhz",
        "evidence_class",
        "n_real_events",
        "real_sign_fraction",
        "real_median_log_contrast",
        "mannwhitney_real_gt_controls_p",
        "control_group_empirical_p_median_ge_real",
    ]
    lines = [
        "# Low-Frequency External Selector Test",
        "",
        "This run tests two more physically targeted external selectors.",
        "",
        "## Solar Selector",
        "",
        "- Data source: NOAA/NCEI RSTN spectral listings.",
        "- Selection uses the dekameter event fields, documented as 10-29 MHz.",
        "- This is closer to RAE-2 than the fixed-frequency RSTN reports, but still above the RAE-2 0.45-9.18 MHz channels.",
        f"- Parsed dekameter spectral events: `{len(spectral)}`.",
        "",
        "## Jupiter Selector",
        "",
        "- Data source: MASER/PADC collection of published Jovian probability and occurrence maps.",
        "- Maps are sampled as images at SPICE-derived CML/Io coordinates.",
        "- The map-image score is a selector only; it is not an absolute probability calibration.",
        "",
        "Maps used:",
        "",
        *[f"- `{m.name}`: {m.description}; file `{m.path}`" for m in maps],
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Selected Visual Cases",
        "",
        selected[cols].to_string(index=False) if not selected.empty else "(none)",
        "",
        "## Summary",
        "",
        summary[cols].sort_values(["source", "mode", "frequency_mhz"]).to_string(index=False) if not summary.empty else "(none)",
        "",
        "## Plots",
        "",
        *[f"- `{p}`" for p in paths],
        "",
        "## Interpretation",
        "",
        "- A useful source selector should move real events away from time/off-source controls in the distribution plots.",
        "- Solar dekameter selections are not direct RAE-2-band detections unless real/control separation appears.",
        "- Jupiter map selections test known Earth-observed DAM occurrence geometry; RAE-2 frequencies below 10 MHz may include different Jovian emission regimes.",
    ]
    path = out_dir / "low_frequency_external_selector_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh-external", action="store_true")
    parser.add_argument("--save-large-tables", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    features = read_table(args.features, parse_dates=["predicted_event_time"], low_memory=False)
    features = features[features["antenna"].astype(str).eq(ANTENNA)].copy()
    min_year = int(pd.to_datetime(features["predicted_event_time"]).dt.year.min())
    max_year = int(pd.to_datetime(features["predicted_event_time"]).dt.year.max())

    spectral = download_parse_noaa_spectral(min_year, max_year, out_dir, refresh=bool(args.refresh_external))
    annotated = annotate_solar_spectral(features, spectral)
    annotated, maps = annotate_jupiter_probability_maps(annotated, out_dir, refresh=bool(args.refresh_external))

    source_modes = {
        "sun": ["all_events", "dek_near_1h", "dek_near_6h", "dek_typeiii_near_6h", "dek_high_day"],
        "jupiter": [
            "all_events",
            "maser_zarka_full_top25",
            "maser_zarka_full_top10",
            "maser_zarka_io_top25",
            "maser_zarka_io_top10",
            "maser_leblanc_1978_top25",
        ],
    }
    summary = summarize_modes(annotated, source_modes)
    summary.to_csv(out_dir / "low_frequency_external_selector_summary.csv", index=False)
    selected = select_visual_cases(summary)
    selected.to_csv(out_dir / "low_frequency_external_selected_visual_cases.csv", index=False)

    if args.save_large_tables:
        annotated[annotated["analysis_source"].isin(["sun", "jupiter"])].to_csv(
            out_dir / "low_frequency_external_annotated_event_features.csv",
            index=False,
        )
    elif not selected.empty:
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
            out_dir / "low_frequency_external_selected_event_features.csv",
            index=False,
        )

    paths: list[Path] = []
    paths.extend(plot_mode_spectra(summary, out_dir))
    paths.extend(plot_selected_distributions(annotated, selected, out_dir))
    paths.extend(plot_jupiter_maps(annotated, maps, out_dir))
    paths.append(plot_solar_spectral_timeline(annotated, spectral, out_dir))

    config = {
        "features": str(args.features),
        "antenna": ANTENNA,
        "spectral_url_template": SPECTRAL_URL_TEMPLATE,
        "maser_maps": {name: meta["url"] for name, meta in MASER_MAPS.items()},
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    report = write_report(out_dir, summary, selected, spectral, maps, paths, config)
    print(report)


if __name__ == "__main__":
    main()
