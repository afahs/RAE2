"""Regenerate the retained occultogram event tables with the fixed geometry.

Writes runtime_inputs/occultogram_v2/ beside the untouched v1 tables and a
comparison of each new table against its v1 counterpart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/global/u2/a/afahs/RAE2/RyleVonberg")
sys.path.insert(0, str(ROOT))
from rylevonberg.constants import GEOMETRY_VERSION, MOON_RADIUS_KM  # noqa: E402
from rylevonberg.events import predict_events  # noqa: E402
from rylevonberg.table_io import write_table  # noqa: E402
from rylevonberg.util import software_versions  # noqa: E402

V1 = ROOT / "runtime_inputs/occultogram_v1"
V2 = ROOT / "runtime_inputs/occultogram_v2"
CADENCE_S = 300.0
MAX_GAP_S = 600.0
FREQS = list(range(1, 10))
ANTS = ["rv1_coarse", "rv2_coarse"]

config = pd.read_csv(ROOT / "configs/bright_sources.csv")


def src(name: str) -> dict:
    row = config[config.source_name == name]
    if not row.empty:
        return row.iloc[0].to_dict()
    # Planets and B1950 catalogue sources not in bright_sources.csv.
    extra = {
        "3c_273": {"kind": "fixed", "ra_deg": 186.63855070362368, "dec_deg": 2.328729956947217, "frame": "fk4"},
        "3c_295": {"kind": "fixed", "ra_deg": 212.38958600312117, "dec_deg": 52.43697551841423, "frame": "fk4"},
    }
    if name in extra:
        return {"source_name": name, "body_name": "", **extra[name]}
    return {"source_name": name, "kind": "body", "body_name": name, "ra_deg": np.nan, "dec_deg": np.nan, "frame": "fk4"}


TABLES = {
    "sun_events": {"sources": ["sun"], "exclude": ["earth"], "exclude_deg": 3.0},
    "planet_events_science_baseline": {"sources": ["earth", "jupiter", "sun"]},
    "bright_source_events": {"sources": ["cas_a", "cyg_a", "fornax_a"]},
    "outer_planet_events": {"sources": ["earth", "jupiter", "mars", "mercury", "neptune", "saturn", "uranus", "venus"]},
    "virgo_events": {"sources": ["3c_273", "3c_295", "vir_a"]},
}

print("loading geometry epochs", flush=True)
geo = pd.read_parquet(V1 / "cleaned_timeseries.parquet", columns=["time", "position_x", "position_y", "position_z"])
geo = geo.drop_duplicates("time").sort_values("time").reset_index(drop=True)
radius = np.linalg.norm(geo[["position_x", "position_y", "position_z"]].to_numpy(float), axis=1)
print(f"epochs={len(geo):,} invalid(inside Moon)={int(np.sum(radius <= MOON_RADIUS_KM)):,}", flush=True)
V2.mkdir(exist_ok=True)


def compare(old: pd.DataFrame, new: pd.DataFrame, tol_s: float = 300.0) -> list[dict]:
    rows = []
    key = ["source_name", "predicted_event_time", "event_type"]
    o = old.drop_duplicates(key)
    n = new.drop_duplicates(key)
    span_end = pd.Timestamp(o.predicted_event_time.max()) + pd.Timedelta(hours=1)
    n_in = n[pd.to_datetime(n.predicted_event_time) <= span_end]
    for (s, et), og in o.groupby(["source_name", "event_type"]):
        ng = n_in[(n_in.source_name == s) & (n_in.event_type == et)]
        ot = np.sort(pd.DatetimeIndex(og.predicted_event_time).as_unit("ns").asi8)
        nt = np.sort(pd.DatetimeIndex(ng.predicted_event_time).as_unit("ns").asi8)
        dt = []
        matched_new = set()
        for t in ot:
            if nt.size == 0:
                continue
            i = int(np.clip(np.searchsorted(nt, t), 1, nt.size - 1)) if nt.size > 1 else 0
            cand = [j for j in {i - 1, i} if 0 <= j < nt.size]
            j = min(cand, key=lambda k: abs(nt[k] - t))
            if abs(nt[j] - t) / 1e9 <= tol_s:
                dt.append((nt[j] - t) / 1e9)
                matched_new.add(j)
        dt = np.asarray(dt)
        rows.append({
            "source_name": s, "event_type": et,
            "n_v1": int(ot.size), "n_v2_same_span": int(nt.size), "n_matched": int(dt.size),
            "n_v1_only": int(ot.size - dt.size), "n_v2_only": int(nt.size - len(matched_new)),
            "median_dt_s": float(np.median(dt)) if dt.size else np.nan,
            "median_abs_dt_s": float(np.median(np.abs(dt))) if dt.size else np.nan,
            "p95_abs_dt_s": float(np.percentile(np.abs(dt), 95)) if dt.size else np.nan,
        })
    return rows


manifest = {"geometry_version": GEOMETRY_VERSION, "prediction_cadence_seconds": CADENCE_S,
            "max_gap_seconds": MAX_GAP_S, "frequencies": FREQS, "antennas": ANTS,
            "geometry_source": str(V1 / "cleaned_timeseries.parquet"),
            "frame": "fk4", "equinox": "B1950", "ephemeris": "builtin",
            "versions": software_versions(), "tables": {}}
all_cmp = []
for name, spec in TABLES.items():
    sources = pd.DataFrame([src(s) for s in spec["sources"]])
    excl = pd.DataFrame([src(s) for s in spec.get("exclude", [])]) if spec.get("exclude") else None
    print(f"predicting {name}: {spec['sources']}", flush=True)
    events, _ = predict_events(
        geo, sources, target_frame="fk4", equinox="B1950", ephemeris="builtin",
        max_gap_seconds=MAX_GAP_S, prediction_cadence_seconds=CADENCE_S,
        frequencies=FREQS, antennas=ANTS,
        limb_exclusion_sources_df=excl, limb_exclusion_deg=spec.get("exclude_deg"),
    )
    path = V2 / f"{name}.parquet"
    write_table(events, path, index=False)
    old = pd.read_parquet(V1 / f"{name}.parquet")
    cmp_rows = compare(old, events)
    for r in cmp_rows:
        r["table"] = name
    all_cmp += cmp_rows
    manifest["tables"][name] = {"path": str(path), "rows": int(len(events)),
                                "events": int(len(events.drop_duplicates(["source_name", "predicted_event_time", "event_type"]))),
                                "sources": sources.to_dict("records"),
                                "limb_exclusion": {"sources": spec.get("exclude"), "deg": spec.get("exclude_deg")}}

cmp_df = pd.DataFrame(all_cmp)[["table", "source_name", "event_type", "n_v1", "n_v2_same_span", "n_matched", "n_v1_only", "n_v2_only", "median_dt_s", "median_abs_dt_s", "p95_abs_dt_s"]]
cmp_df.to_csv(V2 / "v1_vs_v2_event_comparison.csv", index=False)
(V2 / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
pd.set_option("display.width", 220)
print(cmp_df.round(2).to_string(index=False))
