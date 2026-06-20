from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_lower_v_occultation_contrast_maps import (
    beam_visible_matrix,
    build_galactic_grid,
    build_smoothing_laplacian,
    design_row_from_model,
    lower_v_beam_axes,
    parse_frequencies,
    solve_relative_map,
)


def test_design_row_couples_ingress_and_opposite_egress() -> None:
    pre = np.array(
        [
            [1.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
        ]
    )
    post = np.array(
        [
            [0.0, 1.0, 0.5],
            [0.0, 1.0, 0.5],
        ]
    )
    row = design_row_from_model(pre, post)
    assert np.allclose(row, [-1.0, 1.0, 0.0])


def test_smoothing_laplacian_wraps_longitude_neighbors() -> None:
    lap = build_smoothing_laplacian(n_lat=2, n_lon=4)
    assert lap[0, 3] < 0.0
    assert lap[3, 0] < 0.0
    assert lap[0, 4] < 0.0


def test_solve_relative_map_enforces_mean_zero() -> None:
    design = np.array(
        [
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    truth = np.array([-1.0, 0.0, 1.0])
    observed = design @ truth
    weights = np.ones(len(observed))
    lap = np.zeros((3, 3), dtype=float)
    solution, stderr, residuals, stats = solve_relative_map(
        design,
        observed,
        weights,
        lap,
        smooth_lambda=0.0,
        ridge_lambda=1e-9,
        mean_constraint_weight=1000.0,
    )
    assert np.isclose(np.mean(solution), 0.0, atol=1e-10)
    assert np.allclose(solution, truth, atol=1e-6)
    assert np.allclose(residuals, 0.0, atol=1e-6)
    assert np.all(np.isfinite(stderr))
    assert stats["rank"] == 3.0


def test_grid_and_frequency_expansion() -> None:
    grid = build_galactic_grid(60.0)
    assert len(grid) == 18
    assert set(grid["lon_index"]) == set(range(6))
    assert set(grid["lat_index"]) == set(range(3))
    assert parse_frequencies("all") == list(range(1, 10))
    assert parse_frequencies("1,3,3,9") == [1, 3, 9]
    assert parse_frequencies("0.45,3.93") == [1, 6]


def test_lower_v_beam_axis_can_offset_toward_velocity_or_anti_velocity() -> None:
    rows = pd.DataFrame(
        {
            "time": pd.to_datetime(["1975-01-01T00:00:00", "1975-01-01T00:00:10", "1975-01-01T00:00:20"]),
            "position_x": [1000.0, 1000.0, 1000.0],
            "position_y": [-10.0, 0.0, 10.0],
            "position_z": [0.0, 0.0, 0.0],
        }
    )
    moon_axes = np.tile(np.array([-1.0, 0.0, 0.0]), (3, 1))

    anti_velocity = lower_v_beam_axes(rows, moon_axes, 90.0, "anti_velocity")
    velocity = lower_v_beam_axes(rows, moon_axes, 90.0, "velocity")

    assert np.allclose(anti_velocity[1], [0.0, -1.0, 0.0], atol=1e-12)
    assert np.allclose(velocity[1], [0.0, 1.0, 0.0], atol=1e-12)


def test_beam_offset_does_not_shift_lunar_visibility_disk() -> None:
    rows = pd.DataFrame(
        {
            "time": pd.to_datetime(["1975-01-01T00:00:00", "1975-01-01T00:00:10", "1975-01-01T00:00:20"]),
            "position_x": [10_000.0, 10_000.0, 10_000.0],
            "position_y": [-10.0, 0.0, 10.0],
            "position_z": [0.0, 0.0, 0.0],
        }
    )
    pixels = np.array(
        [
            [-1.0, 0.0, 0.0],  # Moon center: should be occulted.
            [0.0, -1.0, 0.0],  # Anti-velocity side: should be visible and high gain after offset.
        ]
    )
    angles = np.array([0.0, 90.0, 180.0, 270.0])
    gains = np.array([1.0, 0.0, 0.0, 0.0])

    model = beam_visible_matrix(
        rows,
        pixels,
        angles,
        gains,
        beam_offset_deg=90.0,
        beam_offset_direction="anti_velocity",
        chunk_size=3,
    )

    assert np.allclose(model[:, 0], 0.0)
    assert np.allclose(model[1, 1], 1.0)


def test_ring_only_model_ignores_beam_gain_but_keeps_visibility() -> None:
    rows = pd.DataFrame(
        {
            "time": pd.to_datetime(["1975-01-01T00:00:00"]),
            "position_x": [10_000.0],
            "position_y": [0.0],
            "position_z": [0.0],
        }
    )
    pixels = np.array(
        [
            [-1.0, 0.0, 0.0],  # Moon center: occulted.
            [0.0, 1.0, 0.0],  # Visible limb-side sky.
        ]
    )
    angles = np.array([0.0, 180.0])
    gains = np.array([0.0, 0.0])

    model = beam_visible_matrix(
        rows,
        pixels,
        angles,
        gains,
        beam_offset_deg=90.0,
        beam_offset_direction="velocity",
        model_mode="ring_only",
    )

    assert np.allclose(model[0], [0.0, 1.0])
