"""Constants and default column names for the Ryle-Vonberg pipeline."""

from __future__ import annotations

MOON_RADIUS_KM = 1737.4
EARTH_RADIUS_KM = 6378.137
EARTH_MOON_DISTANCE_KM = 384400.0

TIME_COLUMN = "time"
FREQUENCY_COLUMN = "frequency_band"
SPACECRAFT_COLUMNS = ["position_x", "position_y", "position_z"]
EARTH_UNIT_COLUMNS = ["earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z"]
GEOMETRY_COLUMNS = [*SPACECRAFT_COLUMNS, *EARTH_UNIT_COLUMNS, "right_ascension", "declination"]
DEFAULT_VALUE_COLUMNS = ["rv1_coarse", "rv2_coarse"]

ANTENNA_METADATA = {
    "rv1_coarse": {
        "receiver": "upper_v",
        "moon_pointing": "away_from_moon",
        "expected_directionality": "upper V is directionally biased away from the lunar disk",
    },
    "rv2_coarse": {
        "receiver": "lower_v",
        "moon_pointing": "toward_moon",
        "expected_directionality": "lower V is directionally biased toward the lunar disk",
    },
    "rv1_fine": {
        "receiver": "upper_v",
        "moon_pointing": "away_from_moon",
        "expected_directionality": "upper V is directionally biased away from the lunar disk",
    },
    "rv2_fine": {
        "receiver": "lower_v",
        "moon_pointing": "toward_moon",
        "expected_directionality": "lower V is directionally biased toward the lunar disk",
    },
}

FREQUENCY_MAP_MHZ = {
    1: 0.45,
    2: 0.70,
    3: 0.90,
    4: 1.31,
    5: 2.20,
    6: 3.93,
    7: 4.70,
    8: 6.55,
    9: 9.18,
}


def frequency_mhz_for_band(band: object) -> float | None:
    """Return RAE-2 Ryle-Vonberg center frequency in MHz for a band number."""
    try:
        return FREQUENCY_MAP_MHZ.get(int(band))
    except (TypeError, ValueError):
        return None


def add_frequency_mhz_column(df):
    """Attach `frequency_mhz` when a table has `frequency_band`."""
    if df is None or getattr(df, "empty", False) or FREQUENCY_COLUMN not in df.columns:
        return df
    out = df.copy()
    out["frequency_mhz"] = out[FREQUENCY_COLUMN].map(frequency_mhz_for_band)
    return out

DEFAULT_FRAME = "fk4"
DEFAULT_EQUINOX = "B1950"
