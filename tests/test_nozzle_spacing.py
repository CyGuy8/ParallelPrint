from __future__ import annotations

import numpy as np

from app import (
    _format_nozzle_spacing_status,
    _resolve_nozzle_layout,
)


def _part(idx: int, bounds: tuple[tuple[float, float, float], tuple[float, float, float]]) -> dict:
    return {
        "idx": idx,
        "color": "#000000",
        "parsed": {"bounds": bounds, "moves": [{"kind": "print"}], "point_count": 1},
    }


def test_calculated_nozzle_layout_uses_requested_part_gap() -> None:
    parts = [
        _part(1, ((0.0, 0.0, 0.0), (10.0, 20.0, 1.0))),
        _part(2, ((0.0, 0.0, 0.0), (5.0, 6.0, 1.0))),
        _part(3, ((-2.0, -1.0, 0.0), (2.0, 3.0, 1.0))),
    ]

    offsets, spacings = _resolve_nozzle_layout(
        parts,
        same_spacing=False,
        part_gap_12_x=2.5,
        part_gap_12_y=-1.0,
        part_gap_23_x=8.0,
        part_gap_23_y=4.0,
    )

    np.testing.assert_allclose(offsets[1], (0.0, 0.0))
    np.testing.assert_allclose(offsets[2], (12.5, -1.0))
    np.testing.assert_allclose(offsets[3], (24.5, 3.0))
    assert spacings[0] == {"from": 1, "to": 2, "dx": 12.5, "dy": -1.0}
    assert spacings[1] == {"from": 2, "to": 3, "dx": 12.0, "dy": 4.0}


def test_calculated_nozzle_layout_allows_negative_x_overlap() -> None:
    parts = [
        _part(1, ((0.0, 0.0, 0.0), (10.0, 10.0, 1.0))),
        _part(2, ((0.0, 0.0, 0.0), (10.0, 10.0, 1.0))),
    ]

    offsets, spacings = _resolve_nozzle_layout(
        parts,
        same_spacing=True,
        part_gap_12_x=-2.0,
        part_gap_12_y=3.0,
        part_gap_23_x=0.0,
        part_gap_23_y=0.0,
    )

    np.testing.assert_allclose(offsets[1], (0.0, 0.0))
    np.testing.assert_allclose(offsets[2], (8.0, 3.0))
    assert spacings == [{"from": 1, "to": 2, "dx": 8.0, "dy": 3.0}]


def test_same_spacing_reuses_first_pair_values_for_second_pair() -> None:
    parts = [
        _part(1, ((0.0, 0.0, 0.0), (10.0, 10.0, 1.0))),
        _part(2, ((0.0, 0.0, 0.0), (4.0, 10.0, 1.0))),
        _part(3, ((0.0, 0.0, 0.0), (6.0, 10.0, 1.0))),
    ]

    offsets, spacings = _resolve_nozzle_layout(
        parts,
        same_spacing=True,
        part_gap_12_x=3.0,
        part_gap_12_y=2.0,
        part_gap_23_x=20.0,
        part_gap_23_y=10.0,
    )

    np.testing.assert_allclose(offsets[1], (0.0, 0.0))
    np.testing.assert_allclose(offsets[2], (13.0, 2.0))
    np.testing.assert_allclose(offsets[3], (20.0, 4.0))
    assert spacings[0] == {"from": 1, "to": 2, "dx": 13.0, "dy": 2.0}
    assert spacings[1] == {"from": 2, "to": 3, "dx": 7.0, "dy": 2.0}


def test_spacing_status_lists_all_nozzle_pair_distances() -> None:
    parts = [
        _part(1, ((0.0, 0.0, 0.0), (10.0, 10.0, 1.0))),
        _part(2, ((0.0, 0.0, 0.0), (4.0, 10.0, 1.0))),
        _part(3, ((0.0, 0.0, 0.0), (6.0, 10.0, 1.0))),
    ]
    offsets, spacings = _resolve_nozzle_layout(
        parts,
        same_spacing=True,
        part_gap_12_x=3.0,
        part_gap_12_y=2.0,
        part_gap_23_x=20.0,
        part_gap_23_y=10.0,
    )

    status = _format_nozzle_spacing_status(parts, offsets, spacings)

    assert "Nozzle 1: X 0.00 mm, Y 0.00 mm." in status
    assert "Nozzle 1 -> 2: Delta X 13.00 mm, Delta Y 2.00 mm" in status
    assert "Nozzle 1 -> 3: Delta X 20.00 mm, Delta Y 4.00 mm" in status
    assert "Nozzle 2 -> 3: Delta X 7.00 mm, Delta Y 2.00 mm" in status
