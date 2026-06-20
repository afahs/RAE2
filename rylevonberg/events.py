"""Occultation event prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import EARTH_UNIT_COLUMNS, SPACECRAFT_COLUMNS, add_frequency_mhz_column, frequency_mhz_for_band
from .frames import body_unit_vectors_from_moon, fixed_source_unit_vector, offset_unit_vectors_tangent, repeated_unit_vector
from .geometry import earth_direction_from_spacecraft, find_limb_transitions, moon_angular_radius_deg, moon_center_direction, moon_limb_angle_deg
from .util import datetime_ns


def spacecraft_positions(df: pd.DataFrame) -> np.ndarray:
    return df[SPACECRAFT_COLUMNS].to_numpy(dtype=float)


def _append_flag_text(flags: str, flag: str) -> str:
    flags = "" if pd.isna(flags) else str(flags)
    return flag if not flags else f"{flags};{flag}"


def _interpolate_transition_value(times: pd.DatetimeIndex, values: np.ndarray, transition: pd.Series) -> float:
    pre_idx = int(transition["pre_idx"])
    post_idx = int(transition["post_idx"])
    if pre_idx < 0 or post_idx >= len(times):
        return np.nan
    t0 = times[pre_idx].value
    t1 = times[post_idx].value
    te = pd.Timestamp(transition["predicted_event_time"]).value
    v0 = float(values[pre_idx])
    v1 = float(values[post_idx])
    if t1 == t0:
        return v1
    frac = float((te - t0) / (t1 - t0))
    return v0 + frac * (v1 - v0)


def contaminant_limb_angles(
    base_df: pd.DataFrame,
    contaminant_sources_df: pd.DataFrame,
    target_frame: str = "fk4",
    equinox: str = "B1950",
    ephemeris: str = "builtin",
) -> dict[str, np.ndarray]:
    """Compute lunar-limb angles for contaminant sources on the prediction grid."""
    if contaminant_sources_df is None or contaminant_sources_df.empty:
        return {}
    times = pd.DatetimeIndex(base_df["time"])
    sc = spacecraft_positions(base_df)
    out: dict[str, np.ndarray] = {}
    for _, contaminant in contaminant_sources_df.iterrows():
        name = str(contaminant["source_name"])
        vec = source_vectors_for_rows(contaminant, times, base_df, target_frame, equinox, ephemeris)
        out[name] = moon_limb_angle_deg(sc, vec)
    return out


def source_vectors_for_rows(
    source: pd.Series,
    times: pd.DatetimeIndex,
    data_df: pd.DataFrame,
    target_frame: str = "fk4",
    equinox: str = "B1950",
    ephemeris: str = "builtin",
) -> np.ndarray:
    kind = str(source.get("kind", "fixed")).lower()
    if kind == "fixed":
        vec = fixed_source_unit_vector(
            float(source["ra_deg"]),
            float(source["dec_deg"]),
            source_frame=str(source.get("frame", target_frame) or target_frame),
            target_frame=target_frame,
            equinox=equinox,
        )
        return repeated_unit_vector(vec, len(times))
    if kind == "body":
        body = str(source.get("body_name", "") or source["source_name"]).lower()
        return body_unit_vectors_from_moon(body, times, target_frame=target_frame, equinox=equinox, ephemeris=ephemeris)
    if kind == "body_offset":
        body = str(source.get("body_name", "") or source["source_name"]).lower()
        real = body_unit_vectors_from_moon(body, times, target_frame=target_frame, equinox=equinox, ephemeris=ephemeris)
        return offset_unit_vectors_tangent(
            real,
            east_offset_deg=float(source.get("offset_east_deg", 0.0) or 0.0),
            north_offset_deg=float(source.get("offset_north_deg", 0.0) or 0.0),
        )
    if kind == "earth":
        return earth_direction_from_spacecraft(spacecraft_positions(data_df), data_df[EARTH_UNIT_COLUMNS].to_numpy(dtype=float))
    raise ValueError(f"Unsupported source kind: {kind}")


def predict_source_events(
    data_df: pd.DataFrame,
    source: pd.Series,
    target_frame: str = "fk4",
    equinox: str = "B1950",
    ephemeris: str = "builtin",
    max_gap_seconds: float | None = None,
    prediction_cadence_seconds: float | None = None,
    frequencies: list[int] | None = None,
    antennas: list[str] | None = None,
    limb_exclusion_sources_df: pd.DataFrame | None = None,
    limb_exclusion_deg: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict disappearance/reappearance events for one source."""
    base = data_df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if prediction_cadence_seconds and prediction_cadence_seconds > 0 and len(base) > 2:
        times_full = pd.DatetimeIndex(base["time"])
        elapsed = (datetime_ns(times_full) - pd.Timestamp(times_full[0]).value).astype(float) / 1e9
        bucket = np.floor(elapsed / float(prediction_cadence_seconds)).astype(np.int64)
        keep = np.r_[True, bucket[1:] != bucket[:-1]]
        keep[-1] = True
        base = base.loc[keep].reset_index(drop=True)
    times = pd.DatetimeIndex(base["time"])
    sc = spacecraft_positions(base)
    target = source_vectors_for_rows(source, times, base, target_frame, equinox, ephemeris)
    limb = moon_limb_angle_deg(sc, target)
    state = pd.DataFrame(
        {
            "time": times,
            "source_name": source["source_name"],
            "limb_angle_deg": limb,
            "moon_angular_radius_deg": moon_angular_radius_deg(sc),
            "visible_by_moon": limb >= 0.0,
        }
    )
    transitions = find_limb_transitions(times, limb, max_gap_seconds=max_gap_seconds)
    if transitions.empty:
        return transitions, state

    center = moon_center_direction(sc)
    exclusion_limb = contaminant_limb_angles(
        base,
        limb_exclusion_sources_df,
        target_frame=target_frame,
        equinox=equinox,
        ephemeris=ephemeris,
    )
    rows = []
    freq_values = frequencies or [None]
    ant_values = antennas or [None]
    for event_id, ev in transitions.reset_index(drop=True).iterrows():
        idx = int(ev["post_idx"])
        exclusion_values = {
            name: _interpolate_transition_value(times, values, ev)
            for name, values in exclusion_limb.items()
        }
        finite_abs = {name: abs(value) for name, value in exclusion_values.items() if np.isfinite(value)}
        nearest_source = min(finite_abs, key=finite_abs.get) if finite_abs else ""
        nearest_abs = float(finite_abs[nearest_source]) if nearest_source else np.nan
        excluded = bool(
            limb_exclusion_deg is not None
            and np.isfinite(nearest_abs)
            and nearest_abs <= float(limb_exclusion_deg)
        )
        if excluded:
            continue
        exclusion_metadata = ";".join(f"{name}:{value:.6g}" for name, value in sorted(exclusion_values.items()))
        for freq in freq_values:
            for ant in ant_values:
                quality_flags = "" if np.isfinite(ev["pre_limb_angle_deg"]) else "bad_limb_angle"
                if limb_exclusion_deg is not None and not exclusion_values:
                    quality_flags = _append_flag_text(quality_flags, "limb_exclusion_not_evaluated")
                rows.append(
                    {
                        "event_id": int(event_id),
                        "source_name": source["source_name"],
                        "source_ra_deg": source.get("ra_deg", np.nan),
                        "source_dec_deg": source.get("dec_deg", np.nan),
                        "frame": source.get("frame", target_frame),
                        "event_type": ev["event_type"],
                        "predicted_event_time": ev["predicted_event_time"],
                        "frequency_band": freq,
                        "frequency_mhz": frequency_mhz_for_band(freq),
                        "antenna": ant,
                        "limb_angle_deg": 0.0,
                        "pre_limb_angle_deg": ev["pre_limb_angle_deg"],
                        "post_limb_angle_deg": ev["post_limb_angle_deg"],
                        "moon_center_x": float(center[idx, 0]),
                        "moon_center_y": float(center[idx, 1]),
                        "moon_center_z": float(center[idx, 2]),
                        "moon_angular_radius_deg": float(state.loc[idx, "moon_angular_radius_deg"]),
                        "gap_seconds": ev["gap_seconds"],
                        "limb_exclusion_deg": limb_exclusion_deg,
                        "limb_exclusion_nearest_source": nearest_source,
                        "limb_exclusion_nearest_abs_deg": nearest_abs,
                        "limb_exclusion_source_angles_deg": exclusion_metadata,
                        "quality_flags": quality_flags,
                    }
                )
    return add_frequency_mhz_column(pd.DataFrame.from_records(rows)), state


def predict_events(
    data_df: pd.DataFrame,
    sources_df: pd.DataFrame,
    target_frame: str = "fk4",
    equinox: str = "B1950",
    ephemeris: str = "builtin",
    max_gap_seconds: float | None = None,
    prediction_cadence_seconds: float | None = None,
    frequencies: list[int] | None = None,
    antennas: list[str] | None = None,
    limb_exclusion_sources_df: pd.DataFrame | None = None,
    limb_exclusion_deg: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_tables = []
    state_tables = []
    for _, source in sources_df.iterrows():
        events, state = predict_source_events(
            data_df,
            source,
            target_frame=target_frame,
            equinox=equinox,
            ephemeris=ephemeris,
            max_gap_seconds=max_gap_seconds,
            prediction_cadence_seconds=prediction_cadence_seconds,
            frequencies=frequencies,
            antennas=antennas,
            limb_exclusion_sources_df=limb_exclusion_sources_df,
            limb_exclusion_deg=limb_exclusion_deg,
        )
        if not events.empty:
            event_tables.append(events)
        state_tables.append(state)
    all_events = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    all_states = pd.concat(state_tables, ignore_index=True) if state_tables else pd.DataFrame()
    return all_events, all_states
