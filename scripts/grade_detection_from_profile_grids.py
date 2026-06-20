#!/usr/bin/env python
"""Assign detection grades from lower-V profile-grid morphology.

This script intentionally avoids the old single SNR-style decision.  It grades
each source/frequency channel using the actual stacked profile-grid curves:

- disappearance should step downward;
- reappearance should step upward;
- the strongest split should occur near the predicted event;
- real profiles should separate from randomized/time-shift/off-source controls;
- neighboring frequencies are treated as supporting evidence, not controls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, write_json  # noqa: E402


DEFAULT_PROFILE_SUMMARY = (
    ROOT
    / "outputs/lower_v_hierarchical_event_offset_model_v1/"
    / "hierarchical_event_offset_adjusted_profile_summary.csv"
)
DEFAULT_SOURCE_SUMMARY = (
    ROOT
    / "outputs/lower_v_hierarchical_event_offset_model_v1/"
    / "hierarchical_event_offset_source_summary.csv"
)
DEFAULT_OUT = ROOT / "outputs/lower_v_profile_grid_detection_grades_v1"

SOURCE_LABEL = {"earth": "Earth", "sun": "Sun", "fornax_a": "Fornax-A"}
GRADE_ORDER = {
    "A_positive_control": 5,
    "B_strong_candidate": 4,
    "C_candidate": 3,
    "D_unresolved": 2,
    "E_likely_systematic": 1,
    "F_not_detected": 0,
}
GRADE_COLORS = {
    "A_positive_control": "#2166ac",
    "B_strong_candidate": "#1a9850",
    "C_candidate": "#91cf60",
    "D_unresolved": "#fee08b",
    "E_likely_systematic": "#f46d43",
    "F_not_detected": "#bdbdbd",
}


@dataclass(frozen=True)
class GradeConfig:
    side_window_min_s: float = 300.0
    split_scan_limit_s: float = 600.0
    strong_max_abs_split_offset_s: float = 300.0
    candidate_max_abs_split_offset_s: float = 600.0
    strong_profile_p_source_like: float = 0.10
    candidate_profile_p_source_like: float = 0.25
    strong_control_escape_fraction: float = 0.30
    candidate_control_escape_fraction: float = 0.15
    min_event_type_balance: float = 0.25
    min_events_per_event_type: int = 40


def _read(path: Path) -> pd.DataFrame:
    return read_table(path, low_memory=False)


def _median_side(y: np.ndarray, t: np.ndarray, mask: np.ndarray) -> float:
    vals = y[mask & np.isfinite(y) & np.isfinite(t)]
    return float(np.nanmedian(vals)) if vals.size else np.nan


def _event_type_contrast(frame: pd.DataFrame, event_type: str, cfg: GradeConfig) -> dict[str, float]:
    sub = frame[frame["event_type"].astype(str).eq(event_type)].sort_values("t_bin_sec")
    t = pd.to_numeric(sub["t_bin_sec"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sub["median_adjusted_z_power"], errors="coerce").to_numpy(dtype=float)
    n_events = pd.to_numeric(sub.get("n_events", np.nan), errors="coerce").max()

    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]
    if t.size < 6:
        return {
            f"{event_type}_contrast": np.nan,
            f"{event_type}_pre_median": np.nan,
            f"{event_type}_post_median": np.nan,
            f"{event_type}_best_split_offset_s": np.nan,
            f"{event_type}_n_events": float(n_events) if np.isfinite(n_events) else 0.0,
        }

    pre_mask = t <= -float(cfg.side_window_min_s)
    post_mask = t >= float(cfg.side_window_min_s)
    if np.count_nonzero(pre_mask) < 2 or np.count_nonzero(post_mask) < 2:
        pre_mask = t < 0
        post_mask = t > 0

    pre = _median_side(y, t, pre_mask)
    post = _median_side(y, t, post_mask)
    if event_type == "disappearance":
        contrast = pre - post
    else:
        contrast = post - pre

    best_contrast = -np.inf
    best_split = np.nan
    candidates = np.unique(t[np.abs(t) <= float(cfg.split_scan_limit_s)])
    for split in candidates:
        left = t < split
        right = t > split
        if np.count_nonzero(left) < 3 or np.count_nonzero(right) < 3:
            continue
        left_med = float(np.nanmedian(y[left]))
        right_med = float(np.nanmedian(y[right]))
        val = left_med - right_med if event_type == "disappearance" else right_med - left_med
        if np.isfinite(val) and val > best_contrast:
            best_contrast = val
            best_split = float(split)

    return {
        f"{event_type}_contrast": float(contrast) if np.isfinite(contrast) else np.nan,
        f"{event_type}_pre_median": pre,
        f"{event_type}_post_median": post,
        f"{event_type}_best_split_offset_s": best_split,
        f"{event_type}_n_events": float(n_events) if np.isfinite(n_events) else 0.0,
    }


def compute_curve_features(profiles: pd.DataFrame, cfg: GradeConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["analysis_source", "control_family", "control_id", "frequency_band", "frequency_mhz"]
    for keys, grp in profiles.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        row.update(_event_type_contrast(grp, "disappearance", cfg))
        row.update(_event_type_contrast(grp, "reappearance", cfg))
        d = row.get("disappearance_contrast", np.nan)
        r = row.get("reappearance_contrast", np.nan)
        contrasts = np.asarray([d, r], dtype=float)
        finite = contrasts[np.isfinite(contrasts)]
        row["combined_source_like_contrast"] = float(np.nanmean(finite)) if finite.size else np.nan
        row["min_event_type_contrast"] = float(np.nanmin(finite)) if finite.size else np.nan
        row["both_event_types_source_like"] = bool(np.isfinite(d) and np.isfinite(r) and d > 0 and r > 0)
        row["one_event_type_source_like"] = bool((np.isfinite(d) and d > 0) or (np.isfinite(r) and r > 0))
        abs_contrasts = np.abs(contrasts[np.isfinite(contrasts)])
        if abs_contrasts.size == 2 and np.nanmax(abs_contrasts) > 0:
            row["event_type_balance"] = float(np.nanmin(abs_contrasts) / np.nanmax(abs_contrasts))
        else:
            row["event_type_balance"] = np.nan
        offsets = np.asarray(
            [
                row.get("disappearance_best_split_offset_s", np.nan),
                row.get("reappearance_best_split_offset_s", np.nan),
            ],
            dtype=float,
        )
        offsets = offsets[np.isfinite(offsets)]
        row["max_abs_best_split_offset_s"] = float(np.nanmax(np.abs(offsets))) if offsets.size else np.nan
        row["min_events_per_event_type"] = float(
            np.nanmin(
                [
                    row.get("disappearance_n_events", np.nan),
                    row.get("reappearance_n_events", np.nan),
                ]
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_control_escape_fraction(profiles: pd.DataFrame, cfg: GradeConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    real = profiles[profiles["control_family"].astype(str).eq("real")].copy()
    controls = profiles[~profiles["control_family"].astype(str).eq("real")].copy()
    key_cols = ["analysis_source", "frequency_band", "frequency_mhz"]

    for keys, real_grp in real.groupby(key_cols, dropna=False, sort=True):
        source, band, freq = keys
        ctrl = controls[
            controls["analysis_source"].astype(str).eq(str(source))
            & controls["frequency_band"].eq(band)
            & np.isclose(pd.to_numeric(controls["frequency_mhz"], errors="coerce"), float(freq))
        ]
        row = {"analysis_source": source, "frequency_band": band, "frequency_mhz": float(freq)}
        if ctrl.empty:
            row.update(
                {
                    "profile_control_escape_fraction": np.nan,
                    "profile_control_escape_bins": 0,
                    "profile_control_escape_eligible_bins": 0,
                }
            )
            rows.append(row)
            continue

        ctrl_band = (
            ctrl.groupby(["event_type", "t_bin_sec"], dropna=False)["median_adjusted_z_power"]
            .quantile([0.25, 0.75])
            .unstack()
            .reset_index()
            .rename(columns={0.25: "control_q25", 0.75: "control_q75"})
        )
        merged = real_grp.merge(ctrl_band, on=["event_type", "t_bin_sec"], how="left")
        merged["t_bin_sec"] = pd.to_numeric(merged["t_bin_sec"], errors="coerce")
        merged["median_adjusted_z_power"] = pd.to_numeric(merged["median_adjusted_z_power"], errors="coerce")
        outer = merged[np.abs(merged["t_bin_sec"]) >= float(cfg.side_window_min_s)].copy()
        outer = outer[np.isfinite(outer["median_adjusted_z_power"]) & np.isfinite(outer["control_q25"]) & np.isfinite(outer["control_q75"])]

        if outer.empty:
            success = 0
            total = 0
        else:
            et = outer["event_type"].astype(str)
            t = outer["t_bin_sec"].to_numpy(dtype=float)
            y = outer["median_adjusted_z_power"].to_numpy(dtype=float)
            q25 = outer["control_q25"].to_numpy(dtype=float)
            q75 = outer["control_q75"].to_numpy(dtype=float)
            source_like = np.zeros(len(outer), dtype=bool)
            source_like |= et.eq("disappearance").to_numpy() & (t <= -cfg.side_window_min_s) & (y > q75)
            source_like |= et.eq("disappearance").to_numpy() & (t >= cfg.side_window_min_s) & (y < q25)
            source_like |= et.eq("reappearance").to_numpy() & (t <= -cfg.side_window_min_s) & (y < q25)
            source_like |= et.eq("reappearance").to_numpy() & (t >= cfg.side_window_min_s) & (y > q75)
            success = int(np.count_nonzero(source_like))
            total = int(len(source_like))
        row.update(
            {
                "profile_control_escape_fraction": float(success / total) if total else np.nan,
                "profile_control_escape_bins": success,
                "profile_control_escape_eligible_bins": total,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def assemble_real_grade_inputs(features: pd.DataFrame, source_summary: pd.DataFrame, profiles: pd.DataFrame, cfg: GradeConfig) -> pd.DataFrame:
    real = features[features["control_family"].astype(str).eq("real")].copy()
    controls = features[~features["control_family"].astype(str).eq("real")].copy()

    control_rows: list[dict[str, object]] = []
    for keys, row in real.groupby(["analysis_source", "frequency_band", "frequency_mhz"], dropna=False, sort=True):
        source, band, freq = keys
        real_row = row.iloc[0]
        ctrl = controls[
            controls["analysis_source"].astype(str).eq(str(source))
            & controls["frequency_band"].eq(band)
            & np.isclose(pd.to_numeric(controls["frequency_mhz"], errors="coerce"), float(freq))
        ].copy()
        real_contrast = float(real_row["combined_source_like_contrast"])
        ctrl_contrast = pd.to_numeric(ctrl["combined_source_like_contrast"], errors="coerce").dropna().to_numpy(dtype=float)
        if ctrl_contrast.size:
            p_source = (1 + int(np.count_nonzero(ctrl_contrast >= real_contrast))) / (1 + int(ctrl_contrast.size))
            p_abs = (1 + int(np.count_nonzero(np.abs(ctrl_contrast) >= abs(real_contrast)))) / (1 + int(ctrl_contrast.size))
            q75 = float(np.nanpercentile(ctrl_contrast, 75))
            abs_q75 = float(np.nanpercentile(np.abs(ctrl_contrast), 75))
            max_ctrl = float(np.nanmax(ctrl_contrast))
        else:
            p_source = np.nan
            p_abs = np.nan
            q75 = np.nan
            abs_q75 = np.nan
            max_ctrl = np.nan
        control_rows.append(
            {
                "analysis_source": source,
                "frequency_band": band,
                "frequency_mhz": float(freq),
                "n_profile_controls": int(ctrl_contrast.size),
                "profile_p_source_like": p_source,
                "profile_p_abs_contrast": p_abs,
                "control_q75_source_like_contrast": q75,
                "control_abs_q75_source_like_contrast": abs_q75,
                "control_max_source_like_contrast": max_ctrl,
            }
        )
    control_df = pd.DataFrame(control_rows)

    escape = compute_control_escape_fraction(profiles, cfg)
    out = real.merge(control_df, on=["analysis_source", "frequency_band", "frequency_mhz"], how="left")
    out = out.merge(escape, on=["analysis_source", "frequency_band", "frequency_mhz"], how="left")

    keep_summary = [
        "analysis_source",
        "frequency_band",
        "frequency_mhz",
        "real_fit_quality_flag",
        "real_model_to_profile_q95_ratio",
        "real_n_events",
        "real_amplitude",
        "real_uncertainty",
        "empirical_p_abs_amp_ge_real",
    ]
    summary_cols = [c for c in keep_summary if c in source_summary.columns]
    out = out.merge(source_summary[summary_cols], on=["analysis_source", "frequency_band", "frequency_mhz"], how="left")
    out["fit_quality_ok"] = out["real_fit_quality_flag"].fillna("").astype(str).eq("ok")
    out["enough_events"] = pd.to_numeric(out["min_events_per_event_type"], errors="coerce") >= cfg.min_events_per_event_type
    return out


def add_neighbor_support(table: pd.DataFrame, cfg: GradeConfig) -> pd.DataFrame:
    out = table.copy()
    base = (
        out["fit_quality_ok"]
        & out["enough_events"]
        & out["both_event_types_source_like"]
        & (pd.to_numeric(out["profile_p_source_like"], errors="coerce") <= cfg.candidate_profile_p_source_like)
        & (pd.to_numeric(out["max_abs_best_split_offset_s"], errors="coerce") <= cfg.candidate_max_abs_split_offset_s)
    )
    out["_base_source_like_channel"] = base
    out["neighbor_source_like_support"] = False
    for source, grp in out.groupby("analysis_source", sort=True):
        idxs = list(grp.sort_values("frequency_mhz").index)
        for pos, idx in enumerate(idxs):
            neighbors = []
            if pos > 0:
                neighbors.append(idxs[pos - 1])
            if pos < len(idxs) - 1:
                neighbors.append(idxs[pos + 1])
            out.loc[idx, "neighbor_source_like_support"] = bool(out.loc[neighbors, "_base_source_like_channel"].any()) if neighbors else False
    out = out.drop(columns=["_base_source_like_channel"])
    return out


def _reason(parts: list[str]) -> str:
    return "; ".join(parts)


def assign_grades(table: pd.DataFrame, cfg: GradeConfig) -> pd.DataFrame:
    out = table.copy()
    grades: list[str] = []
    reasons: list[str] = []
    for _, row in out.iterrows():
        source = str(row["analysis_source"])
        fit_ok = bool(row.get("fit_quality_ok", False))
        enough = bool(row.get("enough_events", False))
        both = bool(row.get("both_event_types_source_like", False))
        one = bool(row.get("one_event_type_source_like", False))
        balance = float(row.get("event_type_balance", np.nan))
        p_source = float(row.get("profile_p_source_like", np.nan))
        escape = float(row.get("profile_control_escape_fraction", np.nan))
        offset = float(row.get("max_abs_best_split_offset_s", np.nan))
        contrast = float(row.get("combined_source_like_contrast", np.nan))
        min_contrast = float(row.get("min_event_type_contrast", np.nan))
        ctrl_abs_q75 = float(row.get("control_abs_q75_source_like_contrast", np.nan))
        neighbor = bool(row.get("neighbor_source_like_support", False))

        issues: list[str] = []
        if not fit_ok:
            issues.append("fit-quality flag is not ok")
        if not enough:
            issues.append("too few events after profile filtering")
        if not both:
            issues.append("disappearance/reappearance are not both source-like")
        if np.isfinite(offset) and offset > cfg.candidate_max_abs_split_offset_s:
            issues.append("strongest profile split is far from predicted event")
        if np.isfinite(balance) and balance < cfg.min_event_type_balance:
            issues.append("disappearance/reappearance amplitudes are poorly balanced")

        strong = (
            fit_ok
            and enough
            and both
            and np.isfinite(min_contrast)
            and min_contrast > 0
            and np.isfinite(balance)
            and balance >= cfg.min_event_type_balance
            and np.isfinite(p_source)
            and p_source <= cfg.strong_profile_p_source_like
            and np.isfinite(escape)
            and escape >= cfg.strong_control_escape_fraction
            and np.isfinite(offset)
            and offset <= cfg.strong_max_abs_split_offset_s
        )
        candidate = (
            fit_ok
            and enough
            and one
            and np.isfinite(contrast)
            and contrast > 0
            and np.isfinite(p_source)
            and p_source <= cfg.candidate_profile_p_source_like
            and np.isfinite(escape)
            and escape >= cfg.candidate_control_escape_fraction
            and np.isfinite(offset)
            and offset <= cfg.candidate_max_abs_split_offset_s
        )

        if source == "earth" and strong:
            grade = "A_positive_control"
            reason = _reason(["Earth positive control passes profile-shape rule", f"profile p={p_source:.3g}", f"escape fraction={escape:.2f}"])
        elif source != "earth" and strong and neighbor:
            grade = "B_strong_candidate"
            reason = _reason(["profile-shape rule passes", "neighboring-frequency support present", f"profile p={p_source:.3g}"])
        elif candidate and (source == "earth" or neighbor or both):
            grade = "C_candidate"
            reason = _reason(["source-like profile present but not all strong criteria pass", f"profile p={p_source:.3g}", f"escape fraction={escape:.2f}"])
        elif not fit_ok:
            grade = "E_likely_systematic"
            reason = _reason(issues)
        elif np.isfinite(contrast) and contrast < 0 and np.isfinite(ctrl_abs_q75) and abs(contrast) >= 0.75 * ctrl_abs_q75:
            grade = "E_likely_systematic"
            reason = _reason(["combined profile is anti-template relative to source expectation", *issues])
        elif np.isfinite(contrast) and abs(contrast) < 0.5 * ctrl_abs_q75:
            grade = "F_not_detected"
            reason = _reason(["profile contrast is small compared with controls", *issues])
        else:
            grade = "D_unresolved"
            reason = _reason(["profile evidence is mixed or control-comparable", *issues])
        grades.append(grade)
        reasons.append(reason)

    out["profile_detection_grade"] = grades
    out["profile_detection_reason"] = reasons
    return out


def plot_grade_matrix(table: pd.DataFrame, out_dir: Path) -> Path:
    sources = [s for s in ["earth", "sun", "fornax_a"] if s in set(table["analysis_source"].astype(str))]
    freqs = list(FREQUENCY_MAP_MHZ.values())
    fig, axes = plt.subplots(len(sources), 1, figsize=(11.5, 1.7 * len(sources) + 1.2), sharex=True)
    if len(sources) == 1:
        axes = [axes]
    for ax, source in zip(axes, sources):
        sub = table[table["analysis_source"].astype(str).eq(source)].copy()
        for freq in freqs:
            row = sub[np.isclose(pd.to_numeric(sub["frequency_mhz"], errors="coerce"), float(freq))]
            if row.empty:
                ax.scatter(freq, 0, marker="s", s=220, color="white", edgecolor="0.8")
                continue
            r = row.iloc[0]
            grade = str(r["profile_detection_grade"])
            color = GRADE_COLORS.get(grade, "white")
            ax.scatter(freq, 0, marker="s", s=420, color=color, edgecolor="black", linewidth=0.5)
            contrast = float(r.get("combined_source_like_contrast", np.nan))
            p_source = float(r.get("profile_p_source_like", np.nan))
            text = f"{grade[0]}\nC={contrast:.2g}\np={p_source:.2g}"
            ax.text(freq, 0, text, ha="center", va="center", fontsize=7)
        ax.set_xscale("log")
        ax.set_ylim(-0.7, 0.7)
        ax.set_yticks([])
        ax.set_ylabel(SOURCE_LABEL.get(source, source), rotation=0, ha="right", va="center", labelpad=52)
        ax.grid(True, axis="x", color="0.9", lw=0.5)
    axes[-1].set_xticks(freqs)
    axes[-1].set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    axes[-1].set_xlabel("frequency (MHz)")
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=9, markerfacecolor=color, markeredgecolor="black", label=grade)
        for grade, color in GRADE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Profile-grid detection grades: lower V only")
    fig.tight_layout(rect=[0, 0.13, 1, 0.93])
    path = out_dir / "profile_grid_detection_grade_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    work = df.copy()
    work = work.fillna("")
    headers = [str(c) for c in work.columns]
    rows = [[str(v) for v in row] for row in work.to_numpy(dtype=object)]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_report(table: pd.DataFrame, cfg: GradeConfig, matrix_path: Path, out_dir: Path) -> Path:
    cols = [
        "analysis_source",
        "frequency_mhz",
        "profile_detection_grade",
        "combined_source_like_contrast",
        "disappearance_contrast",
        "reappearance_contrast",
        "profile_p_source_like",
        "profile_control_escape_fraction",
        "max_abs_best_split_offset_s",
        "neighbor_source_like_support",
        "profile_detection_reason",
    ]
    present = [c for c in cols if c in table.columns]
    compact = table[present].sort_values(["analysis_source", "frequency_mhz"]).copy()
    for c in [
        "combined_source_like_contrast",
        "disappearance_contrast",
        "reappearance_contrast",
        "profile_p_source_like",
        "profile_control_escape_fraction",
        "max_abs_best_split_offset_s",
    ]:
        if c in compact.columns:
            compact[c] = pd.to_numeric(compact[c], errors="coerce").round(3)

    lines = [
        "# Profile-Grid Detection Grade Rule",
        "",
        "This rule grades each lower-V source/frequency channel from the stacked profile-grid morphology.",
        "It does not use the old row-level SNR as the deciding quantity.",
        "",
        "## Rule Inputs",
        "",
        "- Disappearance contrast: median profile before event minus median profile after event.",
        "- Reappearance contrast: median profile after event minus median profile before event.",
        "- Combined source-like contrast: average of disappearance and reappearance contrasts.",
        "- Timing offset: split time that maximizes the source-like before/after contrast.",
        "- Control separation: fraction of outer profile bins that sit outside the control IQR in the source-like direction.",
        "- Profile empirical p-value: `(1 + controls with contrast >= real contrast) / (1 + number of controls)`.",
        "- Neighboring-frequency support: at least one adjacent frequency also has source-like profile evidence.",
        "",
        "## Thresholds",
        "",
        f"- Outer pre/post profile regions start at |dt| >= {cfg.side_window_min_s:.0f} s.",
        f"- Strong timing: strongest split within {cfg.strong_max_abs_split_offset_s:.0f} s of prediction.",
        f"- Candidate timing: strongest split within {cfg.candidate_max_abs_split_offset_s:.0f} s of prediction.",
        f"- Strong profile p-value: <= {cfg.strong_profile_p_source_like:.2f}.",
        f"- Candidate profile p-value: <= {cfg.candidate_profile_p_source_like:.2f}.",
        f"- Strong control-escape fraction: >= {cfg.strong_control_escape_fraction:.2f}.",
        f"- Candidate control-escape fraction: >= {cfg.candidate_control_escape_fraction:.2f}.",
        f"- Minimum event-type balance: >= {cfg.min_event_type_balance:.2f}.",
        f"- Minimum events per event type: >= {cfg.min_events_per_event_type}.",
        "",
        "## Grade Definitions",
        "",
        "- `A_positive_control`: Earth-only; strong profile rule passes.",
        "- `B_strong_candidate`: non-Earth; strong profile rule passes and neighboring-frequency support exists.",
        "- `C_candidate`: source-like profile present but one or more strong criteria are missing.",
        "- `D_unresolved`: profile evidence is mixed or comparable to controls.",
        "- `E_likely_systematic`: fit is not profile-grade usable, or the profile is strongly anti-template.",
        "- `F_not_detected`: profile contrast is small compared with controls.",
        "",
        "## Grade Matrix",
        "",
        f"- `{matrix_path}`",
        "",
        "## Channel Grades",
        "",
        _markdown_table(compact),
        "",
        "## Interpretation",
        "",
        "These grades should be used as a triage layer for the profile grids. A source claim still requires looking at the corresponding profile-grid panels and checking that the real curve, not only the scalar summary, has the expected morphology.",
        "Neighboring-frequency support is treated as positive evidence, not as a wrong-frequency control.",
    ]
    path = out_dir / "profile_grid_detection_grade_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-summary", type=Path, default=DEFAULT_PROFILE_SUMMARY)
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    cfg = GradeConfig()
    write_json(out_dir / "profile_detection_grade_config.json", asdict(cfg))

    profiles = _read(args.profile_summary)
    source_summary = _read(args.source_summary)
    features = compute_curve_features(profiles, cfg)
    real = assemble_real_grade_inputs(features, source_summary, profiles, cfg)
    real = add_neighbor_support(real, cfg)
    graded = assign_grades(real, cfg)

    features.to_csv(out_dir / "profile_grid_curve_features_all_controls.csv", index=False)
    graded.to_csv(out_dir / "profile_grid_detection_grades.csv", index=False)
    matrix = plot_grade_matrix(graded, out_dir)
    report = write_report(graded, cfg, matrix, out_dir)
    print(report)


if __name__ == "__main__":
    main()
