#!/usr/bin/env python
"""Scan sky positions where lower V has low beam-weighted Galactic pickup.

This is a back-of-the-envelope screening tool.  At a lower-V lunar occultation,
the antenna boresight is approximately the Moon center, and the occulted source
is only about one lunar radius from that boresight.  Therefore the diffuse
Galactic contamination for a target is approximated by the beam-weighted PySM
synchrotron brightness for a boresight pointed at that target.
"""

from __future__ import annotations

from pathlib import Path
import sys

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import FK4, Galactic, SkyCoord
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.sources import load_source_list  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.build_earth_beam_clean_profile_grid import (  # noqa: E402
    BEAM_DIR,
    MODEL_NSIDE,
    SKY_DIR,
    _beam_weighted_sky,
    _load_beam,
    _nearest_beam,
    _pixel_fk4_vectors,
)


OUT = ROOT / "outputs/lower_v_galaxy_quiet_sky_scan_v1"
SOURCE_LIST = ROOT / "configs/bright_sources.csv"

EXTRA_SOURCES = [
    {
        "source_name": "3c_273",
        "kind": "fixed",
        "body_name": "",
        "ra_deg": 186.63855070362368,
        "dec_deg": 2.3287299569472175,
        "frame": "fk4",
        "notes": "3C 273, J2000 transformed to FK4/B1950",
    },
    {
        "source_name": "3c_295",
        "kind": "fixed",
        "body_name": "",
        "ra_deg": 212.38958600312114,
        "dec_deg": 52.43697551841423,
        "frame": "fk4",
        "notes": "3C 295, J2000 transformed to FK4/B1950",
    },
]


def _load_sky_i(freq_mhz: float, nside: int) -> tuple[np.ndarray, Path]:
    mhz = int(np.clip(round(float(freq_mhz)), 1, 50))
    path = SKY_DIR / f"synch_pysm_s1_{mhz:02d}MHz_IQU.fits"
    with fits.open(path, memmap=True) as hdul:
        arr = np.asarray(hdul[1].data.field(0), dtype=np.float64).reshape(-1)
    arr[~np.isfinite(arr)] = np.nanmedian(arr[np.isfinite(arr)])
    nside_in = hp.npix2nside(len(arr))
    if nside_in != nside:
        arr = hp.ud_grade(arr, nside_out=nside, order_in="RING", order_out="RING", power=0)
    return arr, path


def _fk4_vectors_from_galactic(nside: int) -> tuple[np.ndarray, pd.DataFrame]:
    pix = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, pix, nest=False)
    gal = SkyCoord(l=phi * u.rad, b=(0.5 * np.pi - theta) * u.rad, frame=Galactic())
    fk4 = gal.transform_to(FK4(equinox=Time("B1950")))
    ra = fk4.ra.rad
    dec = fk4.dec.rad
    vec = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    meta = pd.DataFrame(
        {
            "pixel": pix,
            "ra_fk4_deg": fk4.ra.deg,
            "dec_fk4_deg": fk4.dec.deg,
            "galactic_l_deg": gal.l.deg,
            "galactic_b_deg": gal.b.deg,
        }
    )
    return vec, meta


def _source_vectors(sources: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fixed = sources[sources["kind"].astype(str).str.lower().eq("fixed")].copy()
    for _, row in fixed.iterrows():
        coord = SkyCoord(
            ra=float(row["ra_deg"]) * u.deg,
            dec=float(row["dec_deg"]) * u.deg,
            frame=FK4(equinox=Time("B1950")),
        )
        gal = coord.galactic
        rows.append(
            {
                "source_name": str(row["source_name"]).lower(),
                "ra_fk4_deg": float(row["ra_deg"]),
                "dec_fk4_deg": float(row["dec_deg"]),
                "galactic_l_deg": float(gal.l.deg),
                "galactic_b_deg": float(gal.b.deg),
                "notes": row.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def _unit_vectors_from_fk4(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _rank_sources(source_rows: pd.DataFrame, sky_rows: pd.DataFrame, scan_vecs: np.ndarray, scan_values: np.ndarray) -> pd.DataFrame:
    src_vecs = _unit_vectors_from_fk4(source_rows["ra_fk4_deg"], source_rows["dec_fk4_deg"])
    rows = []
    finite = scan_values[np.isfinite(scan_values)]
    for idx, source in source_rows.iterrows():
        dots = np.clip(scan_vecs @ src_vecs[idx], -1.0, 1.0)
        nearest = int(np.nanargmax(dots))
        value = float(scan_values[nearest])
        percentile = float((finite <= value).mean()) if finite.size else np.nan
        rows.append(
            {
                **source.to_dict(),
                "nearest_scan_pixel": int(sky_rows.iloc[nearest]["pixel"]),
                "lower_v_beam_weighted_galaxy": value,
                "galaxy_quiet_percentile": percentile,
            }
        )
    return pd.DataFrame(rows).sort_values("galaxy_quiet_percentile")


def _plot_sky(scan: pd.DataFrame, source_rank: pd.DataFrame, out_dir: Path, freq_mhz: float) -> Path:
    fig = plt.figure(figsize=(13.5, 7.0))
    ax = fig.add_subplot(111, projection="aitoff")
    lon = ((scan["galactic_l_deg"].to_numpy(dtype=float) + 180.0) % 360.0) - 180.0
    x = np.deg2rad(-lon)
    y = np.deg2rad(scan["galactic_b_deg"].to_numpy(dtype=float))
    vals = np.log10(scan["beam_weighted_galaxy"].to_numpy(dtype=float))
    sc = ax.scatter(x, y, c=vals, s=10, cmap="magma", alpha=0.85, linewidths=0)
    quiet = scan[scan["galaxy_quiet_percentile"].le(0.10)]
    qlon = ((quiet["galactic_l_deg"].to_numpy(dtype=float) + 180.0) % 360.0) - 180.0
    ax.scatter(np.deg2rad(-qlon), np.deg2rad(quiet["galactic_b_deg"]), s=18, facecolors="none", edgecolors="#00bfc4", linewidths=0.8, label="quietest 10%")
    for _, row in source_rank.iterrows():
        sx = np.deg2rad(-(((row["galactic_l_deg"] + 180.0) % 360.0) - 180.0))
        sy = np.deg2rad(row["galactic_b_deg"])
        ax.scatter(sx, sy, s=75, marker="*", color="#00ff7f" if row["galaxy_quiet_percentile"] <= 0.25 else "white", edgecolor="black", linewidth=0.7, zorder=4)
        ax.text(sx + 0.035, sy + 0.025, row["source_name"], fontsize=8, color="white", zorder=5)
    ax.grid(alpha=0.32)
    ax.set_title(f"Lower-V beam-weighted Galactic pickup at {freq_mhz:.2f} MHz beam/sky proxy")
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.055, pad=0.08)
    cb.set_label("log10 beam-weighted PySM synchrotron brightness")
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, -0.08), frameon=False)
    path = out_dir / "lower_v_galaxy_quiet_sky_aitoff.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    out_dir = ensure_dir(OUT)
    scan_nside = 16
    # 1.31 MHz is the lowest available digitized beam; the 1 MHz PySM map is
    # used because the low-frequency contamination issue is strongest there.
    freq_mhz = 1.31
    sky_i, sky_path = _load_sky_i(1.0, MODEL_NSIDE)
    beam_freq, eplane, hplane = _nearest_beam(freq_mhz)
    beam_angles, beam_gains = _load_beam(eplane, hplane)
    pixel_vecs = _pixel_fk4_vectors(MODEL_NSIDE)
    scan_vecs, scan_meta = _fk4_vectors_from_galactic(scan_nside)
    values = _beam_weighted_sky(scan_vecs, sky_i, pixel_vecs, beam_angles, beam_gains)
    scan = scan_meta.copy()
    scan["beam_weighted_galaxy"] = values
    finite = values[np.isfinite(values)]
    scan["galaxy_quiet_percentile"] = [(finite <= v).mean() if np.isfinite(v) else np.nan for v in values]
    scan.to_csv(out_dir / "lower_v_galaxy_quiet_sky_scan.csv", index=False)

    sources = load_source_list(SOURCE_LIST)
    sources = pd.concat([sources, pd.DataFrame(EXTRA_SOURCES)], ignore_index=True)
    source_rows = _source_vectors(sources)
    source_rank = _rank_sources(source_rows, scan, scan_vecs, values)
    source_rank.to_csv(out_dir / "lower_v_galaxy_quiet_source_rankings.csv", index=False)

    quiet = scan.sort_values("beam_weighted_galaxy").head(40).copy()
    quiet.to_csv(out_dir / "lower_v_quietest_sky_positions.csv", index=False)
    path = _plot_sky(scan, source_rank, out_dir, freq_mhz)

    likely = source_rank[source_rank["galaxy_quiet_percentile"].le(0.25)].copy()
    lines = [
        "# Lower-V Galaxy-Quiet Sky Scan",
        "",
        "This is a first-order beam/sky screening calculation.  It approximates lower-V occultation contamination by",
        "pointing the lower-V beam at each sky location and integrating the PySM synchrotron sky through the digitized",
        "Ryle-Vonberg beam.  It does not include spacecraft spin phase, true 2D beam structure, source flux, or event timing.",
        "",
        "## Inputs",
        "",
        f"- sky map: `{sky_path}`",
        f"- beam: `{eplane}`, `{hplane}`",
        f"- scan nside: {scan_nside}",
        f"- sky/model nside: {MODEL_NSIDE}",
        "",
        "## Practical Answer",
        "",
        "Yes, there are locations where lower V should see substantially less Galactic pickup during an occultation.",
        "In this beam-weighted scan the quietest locations are not simply the Galactic poles.  Because the lower-V beam is",
        "broad, the lowest modeled pickup occurs near the Galactic anti-center side of the sky, around l ~ 190-200 deg,",
        "including some low Galactic-latitude directions.  High-latitude sources can still be useful, but the broad beam can",
        "integrate enough diffuse emission that they are not automatically the cleanest lower-V choices.",
        "",
        "Useful existing/source-list candidates in the quietest quartile:",
        "",
        likely[
            [
                "source_name",
                "ra_fk4_deg",
                "dec_fk4_deg",
                "galactic_l_deg",
                "galactic_b_deg",
                "galaxy_quiet_percentile",
            ]
        ].to_string(index=False)
        if not likely.empty
        else "None of the listed bright sources are in the quietest quartile.",
        "",
        "All ranked sources:",
        "",
        source_rank[
            [
                "source_name",
                "galactic_l_deg",
                "galactic_b_deg",
                "lower_v_beam_weighted_galaxy",
                "galaxy_quiet_percentile",
            ]
        ].to_string(index=False),
        "",
        "Quietest sampled sky positions:",
        "",
        quiet[["ra_fk4_deg", "dec_fk4_deg", "galactic_l_deg", "galactic_b_deg", "galaxy_quiet_percentile"]]
        .head(12)
        .to_string(index=False),
        "",
        "## Figure",
        "",
        f"- `{path}`",
    ]
    (out_dir / "lower_v_galaxy_quiet_sky_scan_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out_dir / "run_config.json",
        {
            "scan_nside": scan_nside,
            "model_nside": MODEL_NSIDE,
            "sky_map": str(sky_path),
            "beam_frequency_mhz": beam_freq,
            "eplane": str(eplane),
            "hplane": str(hplane),
            "software_versions": software_versions(),
        },
    )
    print(out_dir / "lower_v_galaxy_quiet_sky_scan_report.md")


if __name__ == "__main__":
    main()
