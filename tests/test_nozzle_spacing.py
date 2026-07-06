from __future__ import annotations

import numpy as np

from app import (
    ADVANCED_NOZZLE_SPACING_HEADERS,
    SCALE_MODE_UNIFORM_FACTOR,
    SHAPE_SETTINGS_HEADERS,
    _auto_align_grid_spacing_rows,
    _apply_shape_settings,
    _contour_tracing_sources,
    delete_shape_from_settings,
    _format_nozzle_spacing_status,
    _grid_spacing_rows,
    _parts_from_records,
    _records_from_files,
    _shape_settings_rows,
    normalize_shape_dimensions_for_mode,
    _resolve_nozzle_grid_layout,
)
from vector_toolpath import (
    RASTER_PATTERN_CIRCLE_SPIRAL,
    RASTER_PATTERN_RECTANGULAR_SPIRAL,
    RASTER_PATTERN_SAME_DIRECTION,
    RASTER_PATTERN_Y_DIRECTION,
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


def test_spacing_status_lists_all_nozzle_pair_distances() -> None:
    parts = [
        _part(1, ((0.0, 0.0, 0.0), (10.0, 10.0, 1.0))),
        _part(2, ((0.0, 0.0, 0.0), (4.0, 10.0, 1.0))),
        _part(3, ((0.0, 0.0, 0.0), (6.0, 10.0, 1.0))),
    ]
    offsets = {1: (0.0, 0.0), 2: (13.0, 2.0), 3: (20.0, 4.0)}
    spacings = [
        {"from": 1, "to": 2, "dx": 13.0, "dy": 2.0},
        {"from": 2, "to": 3, "dx": 7.0, "dy": 2.0},
    ]

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


def test_shape_settings_round_trip_infill_column() -> None:
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
    infill_pos = SHAPE_SETTINGS_HEADERS.index("Infill %")
    assert rows[0][infill_pos] == 100.0

    rows[0][infill_pos] = 50
    updated = _apply_shape_settings(records, rows)
    assert updated[0]["infill"] == 50.0

    # Out-of-range values clamp to 0..100.
    rows[0][infill_pos] = 250
    assert _apply_shape_settings(records, rows)[0]["infill"] == 100.0
    rows[0][infill_pos] = -10
    assert _apply_shape_settings(records, rows)[0]["infill"] == 0.0


def test_contour_tracing_sources_use_sliced_layer_stacks() -> None:
    from shapely.geometry import MultiPolygon, box

    from stl_slicer import LayerStack

    stack = LayerStack(
        layers=[MultiPolygon([box(0.0, 0.0, 1.0, 1.0)])],
        z_values=[0.5],
        bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        layer_height=1.0,
        name="traced",
    )
    sources = _contour_tracing_sources(
        [
            {"idx": 1, "contour_tracing": False, "layer_stack": stack},
            {"idx": 2, "contour_tracing": True, "layer_stack": stack},
            {"idx": 3, "contour_tracing": True, "layer_stack": None},
        ]
    )

    assert len(sources) == 1
    assert sources[0].owner_idx == 2
    assert sources[0].stack is stack


def test_repeated_sample_path_gets_next_unused_nozzle() -> None:
    records = _records_from_files(
        ["sample.stl", "sample.stl"],
        previous_records=[{"idx": 1, "name": "sample", "stl_path": "sample.stl", "nozzle": 1}],
    )

    assert [record["nozzle"] for record in records] == [1, 2]


def test_sample_reload_appends_next_nozzle_set() -> None:
    sample_paths = ["sample_a.stl", "sample_b.stl", "sample_c.stl"]
    first_load = _records_from_files(sample_paths)
    first_resync = _records_from_files(sample_paths, first_load)
    second_load = _records_from_files([*sample_paths, *sample_paths], first_load)

    assert [record["nozzle"] for record in first_load] == [1, 2, 3]
    assert [record["nozzle"] for record in first_resync] == [1, 2, 3]
    assert [record["nozzle"] for record in second_load] == [1, 2, 3, 4, 5, 6]


def _split_piece_records(
    tmp_path,
    columns: int = 2,
    rows: int = 2,
    use_reference_motion: bool = False,
    overlapping_layers: bool = False,
    raster_pattern: str | None = None,
) -> list[dict]:
    """Split a rectangle stack into grid pieces and generate real G-code."""
    from shapely.geometry import MultiPolygon, box

    from stl_slicer import LayerStack
    from vector_gcode import generate_vector_gcode
    from vector_toolpath import build_reference_stack, split_layer_stack_grid

    layer = MultiPolygon([box(0.0, 0.0, 8.0, 6.0)])
    stack = LayerStack(
        layers=[layer, layer],
        z_values=[0.5, 1.5],
        bounds=((0.0, 0.0, 0.0), (8.0, 6.0, 2.0)),
        layer_height=1.0,
        name="source",
    )
    pieces = split_layer_stack_grid(
        stack,
        columns=columns,
        rows=rows,
        overlapping_layers=overlapping_layers,
        overlap=1.0 if overlapping_layers else 0.0,
    )
    reference = build_reference_stack(pieces) if use_reference_motion else None

    records: list[dict] = []
    for index, piece in enumerate(pieces):
        gcode_path = generate_vector_gcode(
            piece,
            shape_name=f"piece_{index}",
            pressure=25,
            valve=4 + index,
            port=1,
            fil_width=1.0,
            motion=reference,
            raster_pattern=raster_pattern,
            output_dir=tmp_path / f"piece_{index}",
        )
        records.append(
            {
                "idx": index + 1,
                "name": piece.name,
                "nozzle": index + 1,
                "color": "#000000",
                "gcode_path": str(gcode_path),
                "split_group_id": "split-a",
                "split_index": index,
                "split_row": index // columns + 1,
                "split_col": index % columns + 1,
                "split_rows": rows,
                "split_columns": columns,
            }
        )
    return records


def _assert_aligned_offsets_reassemble(records: list[dict], spacing_rows, columns, rows) -> None:
    """Offsets from the aligned table must equal each piece's world anchor + const."""
    parts, _messages = _parts_from_records(records)
    offsets, _spacings = _resolve_nozzle_grid_layout(
        parts,
        columns,
        rows,
        column_spacing=5.0,
        row_spacing=5.0,
        use_individual_spacing=True,
        spacing_table=spacing_rows,
    )
    anchors = {part["nozzle"]: part["parsed"]["path_origin"] for part in parts}
    constants = [
        (offsets[nozzle][0] - anchors[nozzle][0], offsets[nozzle][1] - anchors[nozzle][1])
        for nozzle in sorted(offsets)
    ]
    for constant in constants[1:]:
        assert abs(constant[0] - constants[0][0]) < 5e-4
        assert abs(constant[1] - constants[0][1]) < 5e-4


def test_auto_align_reassembles_grid_split_pieces(tmp_path) -> None:
    records = _split_piece_records(tmp_path, columns=2, rows=2)

    spacing_rows, column_count, row_count, aligned_count, missing_count = (
        _auto_align_grid_spacing_rows(records, 2, 2, 5.0, 5.0)
    )

    assert (column_count, row_count) == (2, 2)
    assert aligned_count == 3
    assert missing_count == 0
    _assert_aligned_offsets_reassemble(records, spacing_rows, 2, 2)


def test_auto_align_handles_reference_motion_and_overlap(tmp_path) -> None:
    records = _split_piece_records(
        tmp_path,
        columns=2,
        rows=2,
        use_reference_motion=True,
        overlapping_layers=True,
    )

    spacing_rows, _cc, _rc, aligned_count, missing_count = _auto_align_grid_spacing_rows(
        records, 2, 2, 5.0, 5.0
    )

    assert aligned_count == 3
    assert missing_count == 0
    _assert_aligned_offsets_reassemble(records, spacing_rows, 2, 2)


def test_auto_align_handles_y_raster_pattern(tmp_path) -> None:
    records = _split_piece_records(
        tmp_path,
        columns=2,
        rows=1,
        raster_pattern=RASTER_PATTERN_Y_DIRECTION,
    )

    spacing_rows, _cc, _rc, aligned_count, missing_count = _auto_align_grid_spacing_rows(
        records, 2, 1, 5.0, 5.0
    )

    assert aligned_count == 1
    assert missing_count == 0
    _assert_aligned_offsets_reassemble(records, spacing_rows, 2, 1)


def test_split_pieces_share_parent_scan_grid_so_column_gaps_equal_fil(tmp_path) -> None:
    """User scenario: 4-column split + Y raster -> every aligned gap is one fil width.

    The parent width (7.9) is deliberately not a multiple of the fil width or
    the cell width, so per-piece centred grids would drift out of phase at
    every seam. With the shared parent scan grid the assembled columns run at
    an exact fil pitch, so the world gap between adjacent pieces' toolpath
    boxes is exactly one fil width.
    """
    from shapely.geometry import MultiPolygon, box

    from stl_slicer import LayerStack
    from vector_gcode import generate_vector_gcode
    from vector_toolpath import RASTER_PATTERN_Y_DIRECTION, split_layer_stack_grid

    fil = 0.8
    layer = MultiPolygon([box(0.0, 0.0, 7.9, 4.0)])
    stack = LayerStack(
        layers=[layer, layer],
        z_values=[0.5, 1.5],
        bounds=((0.0, 0.0, 0.0), (7.9, 4.0, 2.0)),
        layer_height=1.0,
        name="wide",
    )
    pieces = split_layer_stack_grid(stack, columns=4, rows=1)
    records = []
    for index, piece in enumerate(pieces):
        gcode_path = generate_vector_gcode(
            piece,
            shape_name=f"col_{index}",
            pressure=25,
            valve=4 + index,
            port=1,
            fil_width=fil,
            raster_pattern=RASTER_PATTERN_Y_DIRECTION,
            output_dir=tmp_path / f"col_{index}",
        )
        records.append(
            {
                "idx": index + 1,
                "name": piece.name,
                "nozzle": index + 1,
                "color": "#000000",
                "gcode_path": str(gcode_path),
                "split_group_id": "split-a",
                "split_index": index,
                "split_row": 1,
                "split_col": index + 1,
                "split_rows": 1,
                "split_columns": 4,
            }
        )

    spacing_rows, _cc, _rc, aligned_count, missing_count = _auto_align_grid_spacing_rows(
        records, 4, 1, 5.0, 5.0
    )

    assert aligned_count == 3
    assert missing_count == 0
    # Adjacent columns continue the parent grid: every gap is exactly one fil.
    assert [row[2] for row in spacing_rows] == [fil, fil, fil]
    _assert_aligned_offsets_reassemble(records, spacing_rows, 4, 1)


def test_auto_align_handles_spiral_patterns(tmp_path) -> None:
    for pattern in (RASTER_PATTERN_RECTANGULAR_SPIRAL, RASTER_PATTERN_CIRCLE_SPIRAL):
        records = _split_piece_records(
            tmp_path / pattern.replace(" ", "_"),
            columns=2,
            rows=1,
            raster_pattern=pattern,
        )

        spacing_rows, _cc, _rc, aligned_count, missing_count = (
            _auto_align_grid_spacing_rows(records, 2, 1, 5.0, 5.0)
        )

        assert aligned_count == 1, pattern
        assert missing_count == 0, pattern
        _assert_aligned_offsets_reassemble(records, spacing_rows, 2, 1)


def test_auto_align_reports_missing_gcode_for_split_siblings(tmp_path) -> None:
    records = [
        {"idx": 1, "name": "first", "nozzle": 1, "split_group_id": "split-a", "split_index": 0},
        {"idx": 2, "name": "second", "nozzle": 2, "split_group_id": "split-a", "split_index": 1},
    ]

    rows, _column_count, _row_count, aligned_count, missing_count = (
        _auto_align_grid_spacing_rows(records, 2, 1, 10.0, 3.0)
    )

    assert aligned_count == 0
    assert missing_count == 1
    assert rows == [["Nozzle 1: Shape 1", "Nozzle 2: Shape 2", 10.0, 0.0]]


def test_auto_align_grid_spacing_skips_unsplit_records() -> None:
    records = [
        {"idx": 1, "name": "first", "nozzle": 1},
        {"idx": 2, "name": "second", "nozzle": 2},
    ]

    rows, _column_count, _row_count, aligned_count, missing_count = (
        _auto_align_grid_spacing_rows(records, 2, 1, 10.0, 3.0)
    )

    assert aligned_count == 0
    assert missing_count == 0
    assert rows == [["Nozzle 1: Shape 1", "Nozzle 2: Shape 2", 10.0, 0.0]]


def test_grid_spacing_rows_follow_row_major_pattern() -> None:
    records = [
        {"idx": 1, "name": "first", "nozzle": 1},
        {"idx": 2, "name": "second", "nozzle": 2},
        {"idx": 3, "name": "third", "nozzle": 3},
        {"idx": 4, "name": "fourth", "nozzle": 4},
    ]

    rows, column_count, row_count = _grid_spacing_rows(records, columns=2, rows=2, column_spacing=10.0, row_spacing=3.0)

    assert column_count == 2
    assert row_count == 2
    assert rows == [
        ["Nozzle 1: Shape 1", "Nozzle 2: Shape 2", 10.0, 0.0],
        ["Nozzle 2: Shape 2", "Nozzle 3: Shape 3", 0.0, 3.0],
        ["Nozzle 3: Shape 3", "Nozzle 4: Shape 4", 10.0, 0.0],
    ]


def test_grid_spacing_rows_preserve_existing_advanced_values() -> None:
    records = [
        {"idx": 1, "name": "first", "nozzle": 1},
        {"idx": 2, "name": "second", "nozzle": 2},
        {"idx": 3, "name": "third", "nozzle": 3},
        {"idx": 4, "name": "fourth", "nozzle": 4},
    ]

    rows, _column_count, _row_count = _grid_spacing_rows(
        records,
        columns=2,
        rows=2,
        column_spacing=10.0,
        row_spacing=3.0,
        existing_table=[
            ["Nozzle 1", "Nozzle 2", 1.5, 0.25],
            ["Nozzle 2", "Nozzle 3", 2.0, 4.5],
        ],
    )

    assert rows == [
        ["Nozzle 1: Shape 1", "Nozzle 2: Shape 2", 1.5, 0.25],
        ["Nozzle 2: Shape 2", "Nozzle 3: Shape 3", 2.0, 4.5],
        ["Nozzle 3: Shape 3", "Nozzle 4: Shape 4", 10.0, 0.0],
    ]


def test_nozzle_grid_layout_places_nozzles_by_rows_and_columns() -> None:
    parts = [
        _part(1, ((0.0, 0.0, 0.0), (10.0, 20.0, 1.0))),
        _part(2, ((0.0, 0.0, 0.0), (10.0, 20.0, 1.0))),
        _part(3, ((0.0, 0.0, 0.0), (10.0, 20.0, 1.0))),
        _part(4, ((0.0, 0.0, 0.0), (10.0, 20.0, 1.0))),
    ]

    offsets, spacings = _resolve_nozzle_grid_layout(parts, columns=2, rows=2, column_spacing=2.0, row_spacing=3.0)

    np.testing.assert_allclose(offsets[1], (0.0, 0.0))
    np.testing.assert_allclose(offsets[2], (12.0, 0.0))
    np.testing.assert_allclose(offsets[3], (0.0, 23.0))
    np.testing.assert_allclose(offsets[4], (12.0, 23.0))
    assert spacings == [
        {"from": 1, "to": 2, "dx": 12.0, "dy": 0.0},
        {"from": 2, "to": 3, "dx": -12.0, "dy": 23.0},
        {"from": 3, "to": 4, "dx": 12.0, "dy": 0.0},
    ]


def test_advanced_nozzle_grid_layout_uses_per_connection_gaps() -> None:
    parts = [
        _part(1, ((0.0, 0.0, 0.0), (10.0, 20.0, 1.0))),
        _part(2, ((0.0, 0.0, 0.0), (10.0, 20.0, 1.0))),
        _part(3, ((0.0, 0.0, 0.0), (10.0, 20.0, 1.0))),
        _part(4, ((0.0, 0.0, 0.0), (10.0, 20.0, 1.0))),
    ]

    offsets, spacings = _resolve_nozzle_grid_layout(
        parts,
        columns=2,
        rows=2,
        column_spacing=2.0,
        row_spacing=3.0,
        use_individual_spacing=True,
        spacing_table=[
            ["Nozzle 1", "Nozzle 2", 2.0, 0.0],
            ["Nozzle 2", "Nozzle 3", 4.0, 3.0],
            ["Nozzle 3", "Nozzle 4", 6.0, 0.0],
        ],
    )

    np.testing.assert_allclose(offsets[1], (0.0, 0.0))
    np.testing.assert_allclose(offsets[2], (12.0, 0.0))
    np.testing.assert_allclose(offsets[3], (4.0, 23.0))
    np.testing.assert_allclose(offsets[4], (20.0, 23.0))
    assert spacings == [
        {"from": 1, "to": 2, "dx": 12.0, "dy": 0.0},
        {"from": 2, "to": 3, "dx": -8.0, "dy": 23.0},
        {"from": 3, "to": 4, "dx": 16.0, "dy": 0.0},
    ]


def test_delete_shape_reindexes_without_losing_shape_data() -> None:
    class Event:
        index = (1, len(SHAPE_SETTINGS_HEADERS) - 1)

    records = [
        {"idx": 1, "name": "first", "stl_path": "first.stl", "target_x": 10.0, "target_y": 11.0, "target_z": 12.0, "pressure": 25, "valve": 4, "port": 1, "color": "#111111"},
        {"idx": 2, "name": "middle", "stl_path": "middle.stl", "target_x": 20.0, "target_y": 21.0, "target_z": 22.0, "pressure": 30, "valve": 5, "port": 2, "color": "#222222"},
        {"idx": 3, "name": "last", "stl_path": "last.stl", "target_x": 30.0, "target_y": 31.0, "target_z": 32.0, "pressure": 35, "valve": 6, "port": 3, "color": "#333333"},
    ]

    outputs = delete_shape_from_settings(records, _shape_settings_rows(records), 0.0, Event())
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

    first_outputs = delete_shape_from_settings(records, _shape_settings_rows(records), 0.0, Event())
    second_outputs = delete_shape_from_settings(
        first_outputs[1],
        first_outputs[2],
        first_outputs[-1],
        Event(),
    )

    assert [record["name"] for record in second_outputs[1]] == ["first", "last"]
