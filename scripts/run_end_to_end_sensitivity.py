#!/usr/bin/env python
"""Paired raw-sample injection/recovery with independent calibration days.

Read the mission master once, retain externally selected off-time windows,
inject before validation, and run the old/new public stack entry points.
Source coordinates/event predictions are held fixed. This evaluates detection
sensitivity on real backgrounds, not a new detection of the named sources.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rylevonberg.util import datetime_ns, robust_sigma

LEVELS = (0., .25, .5, 1., 2., 4., 8.)
WINDOW = 600.


def load_api(root, alias):
    spec = importlib.util.spec_from_file_location(alias, Path(root) / "__init__.py",
                                                 submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return {name: importlib.import_module(f"{alias}.{name}")
            for name in ("ingest", "stacking", "occultogram")}


def select_windows(input_root, maximum, seed):
    paths = {"earth": "planet_events_science_baseline.parquet", "sun": "sun_events.parquet",
             "fornax_a": "bright_source_events.parquet", "jupiter": "planet_events_science_baseline.parquet"}
    frames = []
    rng = np.random.default_rng(seed)
    for source, filename in paths.items():
        events = pd.read_parquet(input_root / filename)
        events = events[events.source_name.eq(source)].drop_duplicates("predicted_event_time").copy()
        events["predicted_event_time"] = pd.to_datetime(events.predicted_event_time)
        events["day"] = events.predicted_event_time.dt.strftime("%Y-%m-%d")
        # One window per source/day; choose without inspecting measured power.
        events = events.iloc[rng.permutation(len(events))].drop_duplicates("day").head(maximum)
        events = events.sort_values("predicted_event_time").reset_index(drop=True)
        order = rng.permutation(len(events))
        events["split"] = "evaluation"
        events.loc[order[:len(events) // 2], "split"] = "calibration"
        events["shift_seconds"] = np.where(np.arange(len(events)) % 2, -1200., 1200.)
        events["center"] = events.predicted_event_time + pd.to_timedelta(events.shift_seconds, unit="s")
        events["window_id"] = [f"{source}_{i}" for i in range(len(events))]
        frames.append(events[["source_name", "window_id", "day", "split", "center", "event_type",
                              "predicted_event_time", "shift_seconds"]])
    return pd.concat(frames, ignore_index=True)


def read_backgrounds(master, windows):
    # Membership in the union of fixed windows, using the nearest center.
    centers = np.sort(datetime_ns(windows.center))
    columns = ["time", "frequency_band", "rv1_coarse", "rv2_coarse", "position_x", "position_y",
               "position_z", "earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z",
               "right_ascension", "declination"]
    chunks, total = [], 0
    for chunk in pd.read_csv(master, usecols=columns, chunksize=250000, low_memory=False):
        chunk["time"] = pd.to_datetime(chunk.time, errors="coerce")
        ns = datetime_ns(chunk.time)
        index = np.searchsorted(centers, ns)
        before = centers[np.maximum(index - 1, 0)]
        after = centers[np.minimum(index, len(centers) - 1)]
        keep = chunk.time.notna().to_numpy() & (np.minimum(np.abs(ns - before), np.abs(ns - after)) <= WINDOW * 1e9)
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
        total += len(chunk)
        print(f"master rows scanned={total:,}; retained={sum(map(len, chunks)):,}", flush=True)
    if not chunks:
        raise RuntimeError("No master samples overlap the prespecified windows")
    return pd.concat(chunks, ignore_index=True), total


def background_scale(frame, center):
    rel = (datetime_ns(frame.time) - pd.Timestamp(center).value) / 1e9
    values = frame.power.to_numpy(float)
    finite = np.isfinite(values) & (values > 0)
    if np.sum(finite & (rel < 0)) < 4 or np.sum(finite & (rel >= 0)) < 4:
        return np.nan
    design = np.column_stack([np.ones(np.sum(finite)), rel[finite] / WINDOW])
    residual = values[finite] - design @ np.linalg.lstsq(design, values[finite], rcond=None)[0]
    sigma = robust_sigma(residual)
    return sigma if np.isfinite(sigma) and sigma > 1e-10 * max(np.median(values[finite]), 1e-10) else np.nan


def injected_power(frame, center, event_type, shape, amplitude):
    out = frame.copy()
    rel = (datetime_ns(out.time) - pd.Timestamp(center).value) / 1e9
    visible = (rel < 0) if event_type == "disappearance" else (rel >= 0)
    signal = visible.astype(float)
    if shape == "burst":
        signal *= np.abs(rel) <= 180.
    out["power"] = out.power.to_numpy(float) + amplitude * signal
    return out


def measure(api, data, event, old):
    # Recompute quality after injection; never reuse pre-injection is_valid.
    validated, _ = api["ingest"].validate_timeseries(data)
    if old:
        profiles = api["stacking"].aligned_profiles(validated, event, window_seconds=WINDOW)
        _, summary = api["stacking"].stack_profiles(profiles, n_bootstrap=0)
    else:
        summary = api["stacking"].fast_stack_summary(validated, event, WINDOW)
    score = float(summary.stacked_snr.iloc[0]) if len(summary) else np.nan
    # An unnormalized amplitude transfer diagnostic using the same extraction.
    profiles = api["stacking"].aligned_profiles(validated, event, window_seconds=WINDOW, normalize=False)
    if len(profiles):
        p, t = profiles.profile_value.to_numpy(float), profiles.template.to_numpy(float)
        amplitude = float(p @ t / (t @ t))
    else:
        amplitude = np.nan
    return score, amplitude, float(validated.is_valid.mean())


def channel_job(payload):
    source, band, antenna, raw, windows, old_root, run_before = payload
    old_api = load_api(old_root, "legacy_recovery_rv")
    new_api = load_api(ROOT / "rylevonberg", "fixed_recovery_rv")
    raw = raw.rename(columns={antenna: "power"}).copy()
    raw["antenna"] = antenna
    raw = raw.sort_values("time").reset_index(drop=True)
    time_ns = datetime_ns(raw.time)
    records, excluded = [], []
    for window in windows.itertuples(index=False):
        ns = pd.Timestamp(window.center).value
        lo, hi = np.searchsorted(time_ns, [ns - int(WINDOW * 1e9), ns + int(WINDOW * 1e9)])
        frame = raw.iloc[lo:hi].reset_index(drop=True)
        scale = background_scale(frame, window.center)
        if not np.isfinite(scale):
            excluded.append({"source": source, "band": band, "antenna": antenna,
                             "window_id": window.window_id, "reason": "insufficient_support_or_zero_background_scale"})
            continue
        event = pd.DataFrame([dict(event_id=0, source_name=source, frequency_band=band, antenna=antenna,
                                   event_type=window.event_type, predicted_event_time=window.center)])
        pipelines = (("before", old_api), ("after", new_api)) if run_before else (("after", new_api),)
        for pipeline, api in pipelines:
            base = measure(api, frame, event, pipeline == "before")
            base_meta = dict(source=source, band=int(band), antenna=antenna, window_id=window.window_id,
                             day=window.day, split=window.split, pipeline=pipeline,
                             event_type=window.event_type, background_scale=scale, n_samples=len(frame))
            records.append({**base_meta, "shape": "null", "level": 0., "score": base[0],
                            "recovered_increment": 0., "valid_fraction": base[2]})
            if window.split == "calibration":
                continue
            for shape in ("step", "burst"):
                for level in LEVELS[1:]:
                    power = injected_power(frame, window.center, window.event_type, shape, level * scale)
                    score, amplitude, valid = measure(api, power, event, pipeline == "before")
                    records.append({**base_meta, "shape": shape, "level": level, "score": score,
                                    "recovered_increment": (amplitude - base[1]) / (level * scale),
                                    "valid_fraction": valid})
    print(f"completed {source} band={band} {antenna}: {len(records)} measurements", flush=True)
    return records, excluded


def summarize(measurements):
    rows = []
    keys = ["source", "band", "antenna", "pipeline"]
    for key, group in measurements.groupby(keys):
        null = group[group.split.eq("calibration")].score.dropna().to_numpy(float)
        # A finite held-out calibration sample: report its size and empirical
        # test false-positive rate rather than assuming nominal 5% is achieved.
        threshold = float(np.quantile(null, .95, method="higher")) if len(null) >= 10 else np.nan
        evaluation = group[group.split.eq("evaluation")]
        for (shape, level), trial in evaluation.groupby(["shape", "level"]):
            detected = trial.score.gt(threshold) & trial.score.notna() if np.isfinite(threshold) else pd.Series(False, index=trial.index)
            rows.append(dict(zip(keys, key)) | {
                "shape": shape, "level": float(level), "n_calibration": len(null),
                "threshold_95": threshold, "n_evaluation": len(trial),
                "recovery_fraction": float(detected.mean()) if np.isfinite(threshold) else np.nan,
                "n_missing_scores": int(trial.score.isna().sum()),
                "median_amplitude_transfer": float(trial.recovered_increment.median()),
                "median_valid_fraction": float(trial.valid_fraction.median()),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--before-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows-per-source", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--reuse-before-results", type=Path,
                        help="Reuse unchanged baseline measurements from an earlier run with identical selected windows.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    windows = select_windows(args.event_root, args.windows_per_source, args.seed)
    reused = pd.DataFrame()
    if args.reuse_before_results:
        original_windows = pd.read_csv(args.reuse_before_results / "prespecified_windows.csv")
        # CSV rendering gives a deterministic comparison across timestamp units.
        if original_windows.to_csv(index=False) != pd.read_csv(io.StringIO(windows.to_csv(index=False))).to_csv(index=False):
            raise ValueError("Cannot reuse baseline: selected windows changed")
        old_manifest = json.loads((args.reuse_before_results / "run_manifest.json").read_text())
        if old_manifest["master"] != str(args.master) or old_manifest["master_bytes"] != args.master.stat().st_size:
            raise ValueError("Cannot reuse baseline: master path or size changed")
        reused = pd.read_parquet(args.reuse_before_results / "paired_measurements.parquet")
        reused = reused[reused.pipeline.eq("before")]
    windows.to_csv(args.output / "prespecified_windows.csv", index=False)
    raw, scanned = read_backgrounds(args.master, windows)
    jobs = []
    for source, selected in windows.groupby("source_name"):
        for band, group in raw.groupby("frequency_band"):
            for antenna in ("rv1_coarse", "rv2_coarse"):
                columns = [c for c in group if not c.startswith("rv") or c == antenna]
                jobs.append((source, band, antenna, group[columns], selected, args.before_package, reused.empty))
    records, excluded = reused.to_dict("records"), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result, failures in pool.map(channel_job, jobs):
            records.extend(result)
            excluded.extend(failures)
    measurements = pd.DataFrame(records)
    measurements.to_parquet(args.output / "paired_measurements.parquet", index=False)
    pd.DataFrame(excluded).to_csv(args.output / "excluded_windows.csv", index=False)
    summary = summarize(measurements)
    summary.to_csv(args.output / "channel_recovery.csv", index=False)
    manifest = {"master": str(args.master), "master_bytes": args.master.stat().st_size,
                "scanned_master_rows": scanned, "retained_raw_rows": len(raw),
                "windows_per_source": windows.groupby("source_name").size().to_dict(),
                "levels_background_sigma": LEVELS, "window_seconds": WINDOW,
                "seed": args.seed, "measurements": len(measurements), "excluded_channel_windows": len(excluded),
                "numpy": np.__version__, "pandas": pd.__version__,
                "reused_baseline_results": str(args.reuse_before_results) if args.reuse_before_results else None,
                "scope": "raw master power -> injection -> revalidation -> strict sample mask -> default detrending -> stack score",
                "limitations": ["window-local quality statistics; full-mission thresholds were not recomputed",
                                "fixed catalog geometry; event prediction not changed",
                                "off-time backgrounds may contain astrophysical or instrumental structure",
                                "burst injections evaluated by the continuum stack detector",
                                "per-channel exploratory sensitivity, not source-discovery p-values"]}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
