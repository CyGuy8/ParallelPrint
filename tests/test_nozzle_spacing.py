from __future__ import annotations

import numpy as np

from app import (
    ADVANCED_NOZZLE_SPACING_HEADERS,
    SCALE_MODE_UNIFORM_FACTOR,
    SHAPE_SETTINGS_HEADERS,
    SIMPLE_NOZZLE_SPACING_HEADERS,
    _apply_shape_settings,
    delete_shape_from_settings,
    _format_nozzle_spacing_status,
    _shape_settings_rows,
    _spacing_args_from_table,
    _spacing_table_update,
    normalize_shape_dimensions_for_mode,
    _resolve_nozzle_layout,
)


def _part(
    idx: int,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    nozzle: int | None = None,
) -> dict:
    return {
        "idx": idx,
        "nozzle": nozzle if nozzle is not None else idx,
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
    np.testing.assert_allclose(offsets[3], (27.5, 3.0))
    assert spacings[0] == {"from": 1, "to": 2, "dx": 12.5, "dy": -1.0}
    assert spacings[1] == {"from": 2, "to": 3, "dx": 15.0, "dy": 4.0}


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


def test_nozzle_layout_groups_shapes_that_share_a_nozzle() -> None:
    parts = [
        _part(1, ((0.0, 0.0, 0.0), (10.0, 10.0, 1.0)), nozzle=1),
        _part(2, ((-1.0, -2.0, 0.0), (12.0, 4.0, 1.0)), nozzle=1),
        _part(3, ((0.0, 0.0, 0.0), (6.0, 5.0, 1.0)), nozzle=2),
    ]

    offsets, spacings = _resolve_nozzle_layout(
        parts,
        same_spacing=True,
        part_gap_12_x=5.0,
        part_gap_12_y=1.5,
        part_gap_23_x=0.0,
        part_gap_23_y=0.0,
    )

    assert sorted(offsets) == [1, 2]
    np.testing.assert_allclose(offsets[1], (0.0, 0.0))
    np.testing.assert_allclose(offsets[2], (17.0, 1.5))
    assert spacings == [{"from": 1, "to": 2, "dx": 17.0, "dy": 1.5}]


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


def test_individual_spacing_has_no_fixed_shape_limit() -> None:
    parts = [
        _part(1, ((0.0, 0.0, 0.0), (10.0, 10.0, 1.0))),
        _part(2, ((0.0, 0.0, 0.0), (4.0, 10.0, 1.0))),
        _part(3, ((0.0, 0.0, 0.0), (6.0, 10.0, 1.0))),
        _part(4, ((0.0, 0.0, 0.0), (2.0, 10.0, 1.0))),
        _part(5, ((0.0, 0.0, 0.0), (3.0, 10.0, 1.0))),
        _part(6, ((0.0, 0.0, 0.0), (4.0, 10.0, 1.0))),
        _part(7, ((0.0, 0.0, 0.0), (5.0, 10.0, 1.0))),
    ]

    offsets, spacings = _resolve_nozzle_layout(
        parts,
        False,
        1.0,
        0.0,
        2.0,
        3.0,
        4.0,
        -1.0,
        6.0,
        2.0,
        1.0,
        1.0,
        -3.0,
        0.5,
    )

    np.testing.assert_allclose(offsets[1], (0.0, 0.0))
    np.testing.assert_allclose(offsets[2], (11.0, 0.0))
    np.testing.assert_allclose(offsets[3], (17.0, 3.0))
    np.testing.assert_allclose(offsets[4], (27.0, 2.0))
    np.testing.assert_allclose(offsets[7], (40.0, 5.5))
    assert spacings[2] == {"from": 3, "to": 4, "dx": 10.0, "dy": -1.0}
    assert spacings[5] == {"from": 6, "to": 7, "dx": 1.0, "dy": 0.5}


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


def test_keep_proportions_uses_most_recently_edited_dimension() -> None:
    records = [
        {
            "idx": 1,
            "name": "wide_part",
            "stl_path": "wide_part.stl",
            "original_x": 10.0,
            "original_y": 20.0,
            "original_z": 5.0,
            "target_x": 10.0,
            "target_y": 20.0,
            "target_z": 5.0,
            "pressure": 25.0,
            "valve": 4,
            "port": 1,
            "color": "#000000",
            "last_scaled_axis": "target_x",
        }
    ]
    settings = _shape_settings_rows(records)
    settings[0][3] = 50.0

    updated_records, updated_settings = normalize_shape_dimensions_for_mode(
        records,
        settings,
        SCALE_MODE_UNIFORM_FACTOR,
    )

    assert updated_records[0]["last_scaled_axis"] == "target_y"
    assert updated_settings[0][2:5] == [25.0, 50.0, 12.5]


def test_shape_settings_round_trip_contour_tracing_column() -> None:
    records = [
        {
            "idx": 1,
            "name": "circle",
            "stl_path": "circle.stl",
            "target_x": 10.0,
            "target_y": 11.0,
            "target_z": 12.0,
            "pressure": 25.0,
            "valve": 4,
            "nozzle": 1,
            "port": 1,
            "color": "#111111",
            "contour_tracing": False,
        }
    ]

    rows = _shape_settings_rows(records)
    assert SHAPE_SETTINGS_HEADERS[6:9] == ["Valve", "Nozzle", "Port"]
    assert SHAPE_SETTINGS_HEADERS[-2:] == ["Contour Tracing", "Delete"]
    assert rows[0][6:9] == [4, 1, 1]
    assert rows[0][-2:] == [False, "Delete"]

    rows[0][7] = 3
    rows[0][8] = 2
    rows[0][-2] = True
    updated = _apply_shape_settings(records, rows)

    assert updated[0]["nozzle"] == 3
    assert updated[0]["port"] == 2
    assert updated[0]["contour_tracing"] is True


def test_simple_spacing_table_uses_one_shared_spacing_row() -> None:
    records = [
        {"idx": 1, "name": "first"},
        {"idx": 2, "name": "second"},
        {"idx": 3, "name": "third"},
    ]

    update = _spacing_table_update(records, [["Shape 1", "Shape 2", 7.0, -1.5]], False)
    gap12x, gap12y, gap23x, gap23y, extra = _spacing_args_from_table(update["value"], False)

    assert update["headers"] == SIMPLE_NOZZLE_SPACING_HEADERS
    assert update["value"] == [["Same spacing", "All neighboring nozzles", 7.0, -1.5]]
    assert (gap12x, gap12y, gap23x, gap23y, extra) == (7.0, -1.5, 7.0, -1.5, [])


def test_advanced_spacing_table_uses_named_nozzle_pairs() -> None:
    records = [
        {"idx": 1, "name": "first"},
        {"idx": 2, "name": "second"},
        {"idx": 3, "name": "third"},
    ]

    update = _spacing_table_update(
        records,
        [["Same spacing", "All neighboring shapes", 4.0, 0.5], ["ignored", "ignored", 8.0, 1.0]],
        True,
    )
    gap12x, gap12y, gap23x, gap23y, extra = _spacing_args_from_table(update["value"], True)

    assert update["headers"] == ADVANCED_NOZZLE_SPACING_HEADERS
    assert update["value"][0][:2] == ["Nozzle 1: Shape 1", "Nozzle 2: Shape 2"]
    assert update["value"][1][:2] == ["Nozzle 2: Shape 2", "Nozzle 3: Shape 3"]
    assert (gap12x, gap12y, gap23x, gap23y, extra) == (4.0, 0.5, 8.0, 1.0, [])


def test_advanced_spacing_table_collapses_duplicate_nozzles() -> None:
    records = [
        {"idx": 1, "name": "first", "nozzle": 1},
        {"idx": 2, "name": "second", "nozzle": 1},
        {"idx": 3, "name": "third", "nozzle": 2},
    ]

    update = _spacing_table_update(records, [["ignored", "ignored", 9.0, 2.5]], True)

    assert update["headers"] == ADVANCED_NOZZLE_SPACING_HEADERS
    assert update["value"] == [["Nozzle 1: Shape 1, Shape 2", "Nozzle 2: Shape 3", 9.0, 2.5]]


def test_delete_shape_reindexes_without_losing_shape_data() -> None:
    class Event:
        index = (1, len(SHAPE_SETTINGS_HEADERS) - 1)

    records = [
        {"idx": 1, "name": "first", "stl_path": "first.stl", "target_x": 10.0, "target_y": 11.0, "target_z": 12.0, "pressure": 25, "valve": 4, "port": 1, "color": "#111111"},
        {"idx": 2, "name": "middle", "stl_path": "middle.stl", "target_x": 20.0, "target_y": 21.0, "target_z": 22.0, "pressure": 30, "valve": 5, "port": 2, "color": "#222222"},
        {"idx": 3, "name": "last", "stl_path": "last.stl", "target_x": 30.0, "target_y": 31.0, "target_z": 32.0, "pressure": 35, "valve": 6, "port": 3, "color": "#333333"},
    ]

    outputs = delete_shape_from_settings(records, _shape_settings_rows(records), None, False, 0.0, Event())
    updated_records = outputs[1]
    updated_settings = outputs[2]

    assert [record["idx"] for record in updated_records] == [1, 2]
    assert [record["stl_path"] for record in updated_records] == ["first.stl", "last.stl"]
    assert [record["target_x"] for record in updated_records] == [10.0, 30.0]
    assert updated_settings[1][0] == 2
    assert updated_settings[1][1] == "last"
    assert updated_settings[1][2:5] == [30.0, 31.0, 32.0]


def test_delete_shape_cooldown_blocks_immediate_second_delete() -> None:
    class Event:
        index = (1, len(SHAPE_SETTINGS_HEADERS) - 1)

    records = [
        {"idx": 1, "name": "first", "stl_path": "first.stl", "target_x": 10.0, "target_y": 11.0, "target_z": 12.0, "pressure": 25, "valve": 4, "port": 1, "color": "#111111"},
        {"idx": 2, "name": "middle", "stl_path": "middle.stl", "target_x": 20.0, "target_y": 21.0, "target_z": 22.0, "pressure": 30, "valve": 5, "port": 2, "color": "#222222"},
        {"idx": 3, "name": "last", "stl_path": "last.stl", "target_x": 30.0, "target_y": 31.0, "target_z": 32.0, "pressure": 35, "valve": 6, "port": 3, "color": "#333333"},
    ]

    first_outputs = delete_shape_from_settings(records, _shape_settings_rows(records), None, False, 0.0, Event())
    second_outputs = delete_shape_from_settings(
        first_outputs[1],
        first_outputs[2],
        None,
        False,
        first_outputs[-1],
        Event(),
    )

    assert [record["name"] for record in second_outputs[1]] == ["first", "last"]
