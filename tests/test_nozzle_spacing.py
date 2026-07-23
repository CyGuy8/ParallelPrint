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
    assert SHAPE_SETTINGS_HEADERS[-3:] == ["Contour Tracing", "Lead In", "Delete"]
    contour_pos = SHAPE_SETTINGS_HEADERS.index("Contour Tracing")
    lead_in_pos = SHAPE_SETTINGS_HEADERS.index("Lead In")
    assert rows[0][6:9] == [4, 1, 1]
    assert rows[0][contour_pos] is False
    assert rows[0][lead_in_pos] is False  # lead-in is opt-in per shape
    assert rows[0][-1] == "Delete"

    rows[0][7] = 3
    rows[0][8] = 2
    rows[0][contour_pos] = True
    rows[0][lead_in_pos] = True
    updated = _apply_shape_settings(records, rows)

    assert updated[0]["nozzle"] == 3
    assert updated[0]["port"] == 2
    assert updated[0]["contour_tracing"] is True
    assert updated[0]["lead_in"] is True


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


def test_color_cell_renders_an_in_table_dropdown() -> None:
    from app import PARALLEL_COLOR_CHOICES

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
            "color": "#ff7f0e",
            "contour_tracing": False,
        }
    ]

    rows = _shape_settings_rows(records)
    color_pos = SHAPE_SETTINGS_HEADERS.index("Color")
    cell = rows[0][color_pos]

    # The cell is a select carrying the record idx (as a class token — data
    # attributes get sanitized out of markdown cells), wrapped in the
    # pp-color-cell span the head script's pointer-event isolation keys on,
    # with every palette color as an option and the current one selected.
    assert cell.startswith('<span class="pp-color-cell">')
    assert '<select class="pp-color-select pp-idx-1"' in cell
    for name, hex_value in PARALLEL_COLOR_CHOICES:
        assert f'value="{hex_value}"' in cell
        assert f">{name}<" in cell
    assert '<option value="#ff7f0e" selected>Orange</option>' in cell
    assert cell.count(" selected") == 1

    # Round-tripping the table through _apply_shape_settings keeps the color
    # (the html cell never parses as a color value).
    assert _apply_shape_settings(records, rows)[0]["color"] == "#ff7f0e"


def test_apply_color_selection_updates_the_right_record() -> None:
    from app import apply_color_selection

    records = [
        {"idx": 1, "name": "a", "stl_path": "a.stl", "color": "#ff7f0e",
         "target_x": 1.0, "target_y": 1.0, "target_z": 1.0,
         "pressure": 25.0, "valve": 4, "nozzle": 1, "port": 1},
        {"idx": 2, "name": "b", "stl_path": "b.stl", "color": "#1f77b4",
         "target_x": 1.0, "target_y": 1.0, "target_z": 1.0,
         "pressure": 25.0, "valve": 5, "nozzle": 2, "port": 1},
    ]

    updated, rows = apply_color_selection(records, None, "2|#ffe119")
    assert updated[0]["color"] == "#ff7f0e"
    assert updated[1]["color"] == "#ffe119"  # Yellow applied to shape 2
    assert 'value="#ffe119" selected' in rows[1][SHAPE_SETTINGS_HEADERS.index("Color")]

    # White is available; junk payloads change nothing.
    assert apply_color_selection(records, None, "1|#ffffff")[0][0]["color"] == "#ffffff"
    assert apply_color_selection(records, None, "1|#123456")[0][0]["color"] == "#ff7f0e"
    assert apply_color_selection(records, None, "garbage")[0][0]["color"] == "#ff7f0e"
    assert apply_color_selection(records, None, None)[0][0]["color"] == "#ff7f0e"


def test_lead_in_assembly_extension_covers_the_split_extent() -> None:
    from shapely.geometry import MultiPolygon, box

    from app import _lead_in_assembly_extension
    from stl_slicer import LayerStack
    from vector_toolpath import split_layer_stack_grid

    layer = MultiPolygon([box(0.0, 0.0, 9.0, 4.0)])
    stack = LayerStack(
        layers=[layer],
        z_values=[0.5],
        bounds=((0.0, 0.0, 0.0), (9.0, 4.0, 1.0)),
        layer_height=1.0,
        name="bar",
    )
    pieces = split_layer_stack_grid(stack, 3, 1, grid=1.0)
    records = [
        {
            "idx": index + 1,
            "layer_stack": piece,
            "split_group_id": "g",
            "split_columns": 3,
            "split_rows": 1,
        }
        for index, piece in enumerate(pieces)
    ]

    # Purging along the split axis must clear (count-1) cells; there is only
    # one row, so the perpendicular directions need no extension.
    cell_width = pieces[0].bounds[1][0] - pieces[0].bounds[0][0]
    assert _lead_in_assembly_extension(records, "Left") == cell_width * 2
    assert _lead_in_assembly_extension(records, "Right") == cell_width * 2
    assert _lead_in_assembly_extension(records, "Up") == 0.0
    assert _lead_in_assembly_extension(records, "Down") == 0.0

    # Whole (unsplit) shapes never extend.
    assert _lead_in_assembly_extension([{"idx": 1}], "Left") == 0.0


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


def test_assign_unique_valves_keeps_first_and_reassigns_duplicates() -> None:
    from app import assign_unique_valves

    records = [
        _mm_member(1, 1, 10.0, 10.0, 10.0),
        _mm_member(2, 2, 10.0, 10.0, 10.0),
        _mm_member(3, 3, 10.0, 10.0, 10.0),
        _mm_member(4, 4, 10.0, 10.0, 10.0),
    ]
    records[0]["valve"] = 4
    records[1]["valve"] = 4  # duplicate: must move
    records[2]["valve"] = 5  # unique: must stay
    records[3]["valve"] = 0  # unset: must be assigned

    updated, rows = assign_unique_valves(records, None)
    valves = [record["valve"] for record in updated]
    assert valves[0] == 4  # first occurrence keeps its number
    assert valves[2] == 5
    assert len(set(valves)) == 4
    assert all(valve >= 4 for valve in valves)
    valve_pos = SHAPE_SETTINGS_HEADERS.index("Valve")
    assert [row[valve_pos] for row in rows] == valves


def test_undo_split_unwinds_multiple_splits() -> None:
    from shapely.geometry import MultiPolygon, box

    from app import _slice_params_snapshot, split_selected_shape_for_grid, undo_last_split
    from stl_slicer import LayerStack

    stack = LayerStack(
        layers=[MultiPolygon([box(0.0, 0.0, 20.0, 10.0)])],
        z_values=[0.4],
        bounds=((0.0, 0.0, 0.0), (20.0, 10.0, 0.8)),
        layer_height=0.8,
        name="solo",
    )
    record = _mm_member(1, 1, 20.0, 10.0, 0.8)
    record["name"] = "solo"
    record["layer_stack"] = stack
    record["slice_params"] = _slice_params_snapshot(record, 0.8, None, None)

    # Split once, then split one of the pieces: the undo stack grows.
    first = split_selected_shape_for_grid([record], None, None, 2, 1, False, 5, 9, 1.0)
    pieces = first[0]
    stack_one = first[-1]
    assert len(pieces) == 2
    assert isinstance(stack_one, list) and len(stack_one) == 1

    second = split_selected_shape_for_grid(
        pieces, None, None, 2, 1, False, 7, 11, 1.0, 0.8, None, stack_one
    )
    stack_two = second[-1]
    assert len(stack_two) == 2

    # Undo pops one split at a time, newest first.
    undo_one = undo_last_split(stack_two)
    assert [r["name"] for r in undo_one[0]] == [r["name"] for r in pieces]
    assert "1 earlier split" in undo_one[7]
    assert len(undo_one[-1]) == 1

    undo_two = undo_last_split(undo_one[-1])
    assert [r["name"] for r in undo_two[0]] == ["solo"]
    assert undo_two[0][0]["layer_stack"] is stack
    assert undo_two[-1] is None  # fully unwound

    # Nothing to undo: every data output is skipped.
    empty = undo_last_split(None)
    assert not isinstance(empty[0], list)


def test_generation_gives_pressure_ownership_to_one_shape_per_port() -> None:
    from pathlib import Path

    from shapely.geometry import MultiPolygon, box

    from app import generate_dynamic_gcode
    from stl_slicer import LayerStack

    def _record(idx: int, port: int, valve: int) -> dict:
        stack = LayerStack(
            layers=[MultiPolygon([box(0.0, 0.0, 5.0, 5.0)])],
            z_values=[0.4],
            bounds=((0.0, 0.0, 0.0), (5.0, 5.0, 0.8)),
            layer_height=0.8,
            name=f"p{idx}",
        )
        return {
            "idx": idx, "name": f"p{idx}", "stl_path": None,
            "target_x": 5.0, "target_y": 5.0, "target_z": 0.8,
            "pressure": 25.0, "valve": valve, "nozzle": idx, "port": port,
            "color": "#111111", "infill": 100.0, "layer_stack": stack,
        }

    # Shapes 1+2 share port 1 (one regulator); shape 3 is alone on port 2.
    records = [_record(1, 1, 4), _record(2, 1, 5), _record(3, 2, 6)]
    out = generate_dynamic_gcode(
        records, None, "X-direction raster", True, 5.0, 5.0, 3, "Left", 0.8, 0.8, None
    )
    texts = [Path(record["gcode_path"]).read_text() for record in out[0]]

    assert "setpress" in texts[0] and "togglepress" in texts[0]
    # The port-mate's file carries NO pressure commands at all.
    assert "serialPort" not in texts[1]
    # A shape on its own port owns its regulator.
    assert "setpress" in texts[2] and "serialPort2" in texts[2]
    # Valve control stays per shape in every file.
    assert "WAGO_ValveCommands(5, 1)" in texts[1]


def test_gcode_staleness_banner_flags_edits_after_generation() -> None:
    from app import (
        GCODE_STALE_MESSAGE,
        _gcode_settings_snapshot,
        check_gcode_staleness,
    )

    settings_args = ("X-direction raster", True, 5.0, 5.0, 3, "Left", 0.8, 0.8, None)
    record = _mm_member(1, 1, 20.0, 10.0, 5.0)
    record["gcode_path"] = "fake_gcode.txt"
    record["gcode_snapshot"] = _gcode_settings_snapshot(record, *settings_args)
    records = [record]
    rows = _shape_settings_rows(records)

    # Untouched settings: no banner.
    assert check_gcode_staleness(records, rows, *settings_args) == ""

    # Editing a table cell (pressure) makes the file stale.
    pressure_pos = SHAPE_SETTINGS_HEADERS.index("Pressure (psi)")
    edited = _shape_settings_rows(records)
    edited[0][pressure_pos] = 40.0
    assert check_gcode_staleness(records, edited, *settings_args) == GCODE_STALE_MESSAGE

    # Changing a generation option (raster pattern) is stale too.
    changed = ("Circle Spiral raster", *settings_args[1:])
    assert check_gcode_staleness(records, rows, *changed) == GCODE_STALE_MESSAGE

    # A shape added after generation (no G-code yet) is stale.
    with_new = [record, _mm_member(2, 2, 10.0, 10.0, 10.0)]
    assert (
        check_gcode_staleness(with_new, _shape_settings_rows(with_new), *settings_args)
        == GCODE_STALE_MESSAGE
    )

    # Nothing generated at all: nothing can be stale.
    fresh = [_mm_member(1, 1, 10.0, 10.0, 10.0)]
    assert check_gcode_staleness(fresh, _shape_settings_rows(fresh), *settings_args) == ""


def test_print_time_estimate_from_path_length_and_speed() -> None:
    from app import _parsed_path_length, _print_time_estimate

    parsed = {
        "print_segments": [[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 5.0, 0.0)]],
        "travel_segments": [[(10.0, 5.0, 0.0), (10.0, 5.0, 2.0)]],
    }
    assert _parsed_path_length(parsed) == 17.0

    assert _print_time_estimate(17.0, 1.0) == "17 s"
    assert _print_time_estimate(150.0, 1.0) == "2 min 30 s"
    assert _print_time_estimate(3600.0, 1.0) == "1 h 00 min"
    assert _print_time_estimate(29532.0, 10.0) == "49 min 13 s"
    # No estimate without a usable speed or path.
    assert _print_time_estimate(100.0, 0) is None
    assert _print_time_estimate(100.0, None) is None
    assert _print_time_estimate(0.0, 5.0) is None


def test_new_shapes_default_to_unique_valves() -> None:
    # Shapes sharing a valve dispense together, so defaults must not collide.
    first_load = _records_from_files(["a.stl", "b.stl", "c.stl"])
    assert [record["valve"] for record in first_load] == [4, 5, 6]

    # Re-syncing keeps the assigned valves.
    resync = _records_from_files(["a.stl", "b.stl", "c.stl"], first_load)
    assert [record["valve"] for record in resync] == [4, 5, 6]

    # User-set valves survive, and new shapes take the smallest unused number.
    first_load[0]["valve"] = 9
    more = _records_from_files(["a.stl", "d.stl"], first_load)
    assert more[0]["valve"] == 9
    assert more[1]["valve"] == 4


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
        origin_sink: dict = {}
        gcode_path = generate_vector_gcode(
            piece,
            shape_name=f"piece_{index}",
            pressure=25,
            valve=4 + index,
            port=1,
            fil_width=1.0,
            motion=reference,
            raster_pattern=raster_pattern,
            origin_sink=origin_sink,
            output_dir=tmp_path / f"piece_{index}",
        )
        records.append(
            {
                "idx": index + 1,
                "name": piece.name,
                "nozzle": index + 1,
                "color": "#000000",
                "gcode_path": str(gcode_path),
                "path_origin": origin_sink.get("path_origin"),
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
        origin_sink: dict = {}
        gcode_path = generate_vector_gcode(
            piece,
            shape_name=f"col_{index}",
            pressure=25,
            valve=4 + index,
            port=1,
            fil_width=fil,
            raster_pattern=RASTER_PATTERN_Y_DIRECTION,
            origin_sink=origin_sink,
            output_dir=tmp_path / f"col_{index}",
        )
        records.append(
            {
                "idx": index + 1,
                "name": piece.name,
                "nozzle": index + 1,
                "color": "#000000",
                "gcode_path": str(gcode_path),
                "path_origin": origin_sink.get("path_origin"),
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
    assert [record["name"] for record in first_outputs[1]] == ["first", "last"]

    second_outputs = delete_shape_from_settings(
        first_outputs[1],
        first_outputs[2],
        first_outputs[-1],
        Event(),
    )

    # Blocked by the cooldown: every output is skipped (nothing rewritten).
    assert not isinstance(second_outputs[1], list)


def test_non_delete_cell_selection_touches_nothing() -> None:
    # The select handler fires on EVERY cell click; unless the click is on
    # the Delete column it must skip all outputs — echoing the table here
    # raced (and clobbered) the Keep Proportions dimension recompute.
    class Event:
        index = (0, 3)  # a Target Y cell

    records = [
        {"idx": 1, "name": "first", "stl_path": "first.stl", "target_x": 10.0, "target_y": 11.0, "target_z": 12.0, "pressure": 25, "valve": 4, "port": 1, "color": "#111111"},
    ]

    outputs = delete_shape_from_settings(records, _shape_settings_rows(records), 0.0, Event())

    assert len(outputs) == 9
    assert all(not isinstance(value, list) for value in outputs)
    assert not isinstance(outputs[1], list)  # records State untouched


def test_group_split_splits_all_materials_on_one_shared_grid() -> None:
    from shapely.geometry import MultiPolygon, box

    from app import split_selected_shape_for_grid
    from stl_slicer import LayerStack

    def _material(polygon, name: str) -> LayerStack:
        layers = [MultiPolygon([polygon]), MultiPolygon([polygon])]
        x_min, y_min, x_max, y_max = polygon.bounds
        return LayerStack(
            layers=layers,
            z_values=[0.5, 1.5],
            bounds=((x_min, y_min, 0.0), (x_max, y_max, 2.0)),
            layer_height=1.0,
            name=name,
        )

    def _record(idx: int, name: str, stack: LayerStack) -> dict:
        return {
            "idx": idx,
            "name": name,
            "stl_path": f"{name}.stl",
            "target_x": 20.0,
            "target_y": 5.0,
            "target_z": 2.0,
            "pressure": 25.0,
            "valve": 4,
            "nozzle": 1,  # both materials on nozzle 1 -> one assembly
            "port": 1,
            "color": "#111111",
            "layer_stack": stack,
        }

    # Two materials tiling one 20x10 shape as horizontal strips.
    records = [
        _record(1, "bottom", _material(box(0.0, 0.0, 20.0, 5.0), "bottom")),
        _record(2, "top", _material(box(0.0, 5.0, 20.0, 10.0), "top")),
    ]
    # Mark the stacks as freshly sliced for the split's auto-slice step (the
    # fixture stl paths don't exist, so a re-slice attempt would wipe them).
    from app import _slice_params_snapshot

    for record in records:
        record["slice_params"] = _slice_params_snapshot(record, 0.8, None, None)

    outputs = split_selected_shape_for_grid(
        records,
        None,  # selected -> defaults to the first record
        None,  # settings table
        2,  # columns
        1,  # rows
        False,  # overlapping layers
        5,  # starting nozzle
        9,  # starting valve
        1.0,  # fil width
    )
    next_records = outputs[0]

    pieces = [record for record in next_records if record.get("split_group_id")]
    assert len(pieces) == 4  # 2 cells x 2 materials

    # Cell-major: cell 1 pieces share nozzle 5, cell 2 pieces share nozzle 6;
    # every piece gets its own valve.
    assert [piece["nozzle"] for piece in pieces] == [5, 5, 6, 6]
    assert [piece["valve"] for piece in pieces] == [9, 10, 11, 12]
    assert [piece["name"] for piece in pieces] == [
        "bottom - R1C1",
        "top - R1C1",
        "bottom - R1C2",
        "top - R1C2",
    ]

    # Cell-mates carry IDENTICAL nominal cell bounds (the shared grid) and
    # one shared scan frame covering the whole assembly.
    for cell_first, cell_second in ((0, 1), (2, 3)):
        assert pieces[cell_first]["layer_stack"].bounds == pieces[cell_second]["layer_stack"].bounds
    frames = {piece["layer_stack"].scan_frame for piece in pieces}
    assert frames == {(0.0, 0.0, 20.0, 10.0)}

    # Geometry: each piece is its material clipped to its cell.
    assert pieces[0]["layer_stack"].layers[0].bounds == (0.0, 0.0, 10.0, 5.0)
    assert pieces[1]["layer_stack"].layers[0].bounds == (0.0, 5.0, 10.0, 10.0)
    assert pieces[2]["layer_stack"].layers[0].bounds == (10.0, 0.0, 20.0, 5.0)

    # Contours exclude BOTH the material interface (y=5) and the cut seam
    # (x=10): piece R1C1 of `bottom` keeps only its west + south edges.
    for path in pieces[0]["layer_stack"].contour_paths[0]:
        for x, y in path:
            assert x <= 10.0 - 0.5 + 1e-9 or y <= 5.0 - 0.5 + 1e-9


def test_describe_split_source_warns_about_group_splits() -> None:
    from shapely.geometry import MultiPolygon, box

    from app import SPLIT_STATUS_DEFAULT, describe_split_source
    from stl_slicer import LayerStack

    def _stack(name: str) -> LayerStack:
        return LayerStack(
            layers=[MultiPolygon([box(0.0, 0.0, 1.0, 1.0)])],
            z_values=[0.5],
            bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            layer_height=1.0,
            name=name,
        )

    records = [
        {"idx": 1, "name": "black", "stl_path": "black.stl", "nozzle": 1, "layer_stack": _stack("black")},
        {"idx": 2, "name": "gold", "stl_path": "gold.stl", "nozzle": 1, "layer_stack": _stack("gold")},
        {"idx": 3, "name": "solo", "stl_path": "solo.stl", "nozzle": 2, "layer_stack": _stack("solo")},
    ]

    grouped_note = describe_split_source(records, "1: black")
    assert "whole group as one shape" in grouped_note
    assert "black" in grouped_note and "gold" in grouped_note
    assert "nozzle 1" in grouped_note

    assert describe_split_source(records, "3: solo") == SPLIT_STATUS_DEFAULT
    # No selection defaults to the first shape - which is grouped here.
    assert "whole group as one shape" in describe_split_source(records, None)
    assert describe_split_source([], None) == SPLIT_STATUS_DEFAULT


def test_keep_proportions_is_stale_echo_proof() -> None:
    # The table's .change event delivers stale/echoed tables out of order.
    # Anchoring must come from the ROW's own ratios (odd one out), never from
    # diffing against fresher records - that mis-anchored and reverted the
    # first edit. Self-consistent rows must skip the table write entirely.
    import gradio as gr

    def _record(**overrides):
        record = {
            "idx": 1, "name": "cube", "stl_path": "cube.stl",
            "original_x": 38.1, "original_y": 38.1, "original_z": 32.99557,
            "target_x": 38.1, "target_y": 38.1, "target_z": 32.99557,
            "pressure": 25.0, "valve": 4, "nozzle": 1, "port": 1,
            "color": "#111111", "last_scaled_axis": "target_x",
        }
        record.update(overrides)
        return record

    # 1) A user edit (odd ratio on Y) rescales everything and writes the table.
    records = [_record()]
    rows = _shape_settings_rows(records)
    rows[0][3] = 50.0
    updated, table_out = normalize_shape_dimensions_for_mode(records, rows, SCALE_MODE_UNIFORM_FACTOR)
    assert updated[0]["target_x"] == 50.0
    assert updated[0]["target_z"] == 43.3  # tenths grid (originals round too)
    assert isinstance(table_out, list)  # table written

    # 2) A stale PRE-EDIT echo arrives while records already hold the scaled
    #    dims: the row is self-consistent, so no re-anchor, no revert write.
    stale_rows = _shape_settings_rows([_record()])
    updated2, table_out2 = normalize_shape_dimensions_for_mode(updated, stale_rows, SCALE_MODE_UNIFORM_FACTOR)
    assert not isinstance(table_out2, list)  # table output skipped

    # 3) The echo of our own scaled write-back is also a no-op.
    scaled_rows = _shape_settings_rows(updated)
    updated3, table_out3 = normalize_shape_dimensions_for_mode(updated, scaled_rows, SCALE_MODE_UNIFORM_FACTOR)
    assert not isinstance(table_out3, list)
    assert updated3[0]["target_x"] == 50.0
    assert updated3[0]["target_z"] == 43.3


def _mm_member(idx: int, nozzle: int, ox: float, oy: float, oz: float) -> dict:
    return {
        "idx": idx, "name": f"part{idx}", "stl_path": f"part{idx}.stl",
        "original_x": ox, "original_y": oy, "original_z": oz,
        "target_x": ox, "target_y": oy, "target_z": oz,
        "pressure": 25.0, "valve": 4, "nozzle": nozzle, "port": 1,
        "color": "#111111", "last_scaled_axis": "target_x",
    }


def test_group_members_share_scale_factors_in_independent_mode() -> None:
    from app import SCALE_MODE_TARGET_DIMENSIONS

    # Two assembly parts of DIFFERENT sizes on nozzle 1, a solo on nozzle 2.
    records = [
        _mm_member(1, 1, 40.0, 20.0, 10.0),
        _mm_member(2, 1, 20.0, 20.0, 10.0),
        _mm_member(3, 2, 30.0, 30.0, 30.0),
    ]
    rows = _shape_settings_rows(records)
    rows[0][2] = 60.0  # X of part 1: factor 1.5

    updated, table_out = normalize_shape_dimensions_for_mode(
        records, rows, SCALE_MODE_TARGET_DIMENSIONS
    )

    # Part 2 gets the same FACTOR (x 20 -> 30), not the same absolute value.
    assert updated[0]["target_x"] == 60.0
    assert updated[1]["target_x"] == 30.0
    # Unedited axes keep factor 1.
    assert updated[0]["target_y"] == 20.0 and updated[1]["target_y"] == 20.0
    # Solo shape on its own nozzle is untouched.
    assert updated[2]["target_x"] == 30.0
    assert isinstance(table_out, list)  # propagation must be written back


def test_group_members_share_scale_factors_in_keep_proportions() -> None:
    records = [
        _mm_member(1, 1, 40.0, 20.0, 10.0),
        _mm_member(2, 1, 20.0, 20.0, 10.0),
    ]
    rows = _shape_settings_rows(records)
    rows[0][3] = 30.0  # Y of part 1: factor 1.5

    updated, table_out = normalize_shape_dimensions_for_mode(
        records, rows, SCALE_MODE_UNIFORM_FACTOR
    )

    # Part 1 rescales proportionally; part 2 follows with the same factor.
    assert [updated[0][k] for k in ("target_x", "target_y", "target_z")] == [60.0, 30.0, 15.0]
    assert [updated[1][k] for k in ("target_x", "target_y", "target_z")] == [30.0, 30.0, 15.0]
    assert isinstance(table_out, list)

    # The echo of that write-back is a converged no-op.
    echoed_rows = _shape_settings_rows(updated)
    updated2, table_out2 = normalize_shape_dimensions_for_mode(
        updated, echoed_rows, SCALE_MODE_UNIFORM_FACTOR
    )
    assert not isinstance(table_out2, list)
    assert updated2[1]["target_x"] == 30.0


def test_group_propagation_skips_when_the_source_is_ambiguous() -> None:
    from app import SCALE_MODE_TARGET_DIMENSIONS

    # A stale echo can make SEVERAL members look edited at once: the
    # propagation must not guess a source (guessing reverted edits).
    records = [
        _mm_member(1, 1, 40.0, 20.0, 10.0),
        _mm_member(2, 1, 20.0, 20.0, 10.0),
    ]
    records[0]["target_x"] = 60.0  # records already hold part 1 scaled...
    rows = _shape_settings_rows(records)
    rows[0][2] = 44.0  # ...while the table flags edits on BOTH members
    rows[1][2] = 24.0

    updated, _table_out = normalize_shape_dimensions_for_mode(
        records, rows, SCALE_MODE_TARGET_DIMENSIONS
    )

    # Both edits applied as-is; no propagation happened (factors differ).
    assert updated[0]["target_x"] == 44.0
    assert updated[1]["target_x"] == 24.0


def test_joining_a_nozzle_group_adopts_the_group_scale() -> None:
    from app import SCALE_MODE_TARGET_DIMENSIONS

    # Nozzle-1 group already scaled x1.5; a solo shape moves onto nozzle 1
    # via the Nozzle column and must adopt the group's factors.
    member_a = _mm_member(1, 1, 40.0, 20.0, 10.0)
    member_b = _mm_member(2, 1, 20.0, 20.0, 10.0)
    for member in (member_a, member_b):
        for axis in ("x", "y", "z"):
            member[f"target_{axis}"] = member[f"original_{axis}"] * 1.5
    solo = _mm_member(3, 2, 30.0, 10.0, 10.0)

    records = [member_a, member_b, solo]
    rows = _shape_settings_rows(records)
    nozzle_pos = SHAPE_SETTINGS_HEADERS.index("Nozzle")
    rows[2][nozzle_pos] = 1  # solo joins the assembly

    updated, table_out = normalize_shape_dimensions_for_mode(
        records, rows, SCALE_MODE_TARGET_DIMENSIONS
    )

    assert [updated[2][k] for k in ("target_x", "target_y", "target_z")] == [45.0, 15.0, 15.0]
    # Incumbents unchanged.
    assert updated[0]["target_x"] == 60.0
    assert isinstance(table_out, list)

    # The echo of that write-back converges (nozzle now matches the record).
    echoed = _shape_settings_rows(updated)
    updated2, table_out2 = normalize_shape_dimensions_for_mode(
        updated, echoed, SCALE_MODE_TARGET_DIMENSIONS
    )
    assert not isinstance(table_out2, list)
    assert updated2[2]["target_x"] == 45.0


def test_table_dimensions_display_and_store_to_the_tenths_place() -> None:
    from app import SCALE_MODE_TARGET_DIMENSIONS

    # Headers shrank: "Target X (mm)" -> "X (mm)".
    assert SHAPE_SETTINGS_HEADERS[2:5] == ["X (mm)", "Y (mm)", "Z (mm)"]

    record = _mm_member(1, 1, 38.1, 38.1, 33.0)
    record["target_z"] = 32.99557  # legacy noisy value from a mesh extent
    rows = _shape_settings_rows([record])
    assert rows[0][2:5] == [38.1, 38.1, 33.0]

    # The echo of the rounded display converges: the stored value snaps to
    # the displayed tenths and no further table rewrite happens.
    updated, table_out = normalize_shape_dimensions_for_mode(
        [record], rows, SCALE_MODE_TARGET_DIMENSIONS
    )
    assert updated[0]["target_z"] == 33.0
    assert not isinstance(table_out, list)

    # Keep Proportions recomputes land on the tenths grid too.
    updated[0]["last_scaled_axis"] = "target_x"
    rows2 = _shape_settings_rows(updated)
    rows2[0][2] = 50.0
    updated2, _out = normalize_shape_dimensions_for_mode(
        updated, rows2, SCALE_MODE_UNIFORM_FACTOR
    )
    for key in ("target_x", "target_y", "target_z"):
        assert updated2[0][key] == round(updated2[0][key], 1)


def test_pressure_edit_propagates_to_shapes_sharing_the_port() -> None:
    from app import SCALE_MODE_TARGET_DIMENSIONS

    # Shapes 1+2 share port 1 (one regulator); shape 3 is alone on port 2.
    records = [
        _mm_member(1, 1, 40.0, 20.0, 10.0),
        _mm_member(2, 2, 20.0, 20.0, 10.0),
        _mm_member(3, 3, 30.0, 30.0, 30.0),
    ]
    records[2]["port"] = 2
    records[2]["pressure"] = 40.0
    rows = _shape_settings_rows(records)
    pressure_pos = SHAPE_SETTINGS_HEADERS.index("Pressure (psi)")
    rows[0][pressure_pos] = 32.0

    updated, table_out = normalize_shape_dimensions_for_mode(
        records, rows, SCALE_MODE_TARGET_DIMENSIONS
    )

    assert updated[0]["pressure"] == 32.0
    assert updated[1]["pressure"] == 32.0  # port-mate follows
    assert updated[2]["pressure"] == 40.0  # other port untouched
    assert isinstance(table_out, list)  # propagation must be written back

    # The echo of that write-back is a converged no-op.
    echoed = _shape_settings_rows(updated)
    updated2, table_out2 = normalize_shape_dimensions_for_mode(
        updated, echoed, SCALE_MODE_TARGET_DIMENSIONS
    )
    assert not isinstance(table_out2, list)
    assert updated2[1]["pressure"] == 32.0


def test_moving_a_shape_onto_a_port_adopts_that_ports_pressure() -> None:
    from app import SCALE_MODE_TARGET_DIMENSIONS

    first = _mm_member(1, 1, 40.0, 20.0, 10.0)
    second = _mm_member(2, 2, 20.0, 20.0, 10.0)
    mover = _mm_member(3, 3, 30.0, 30.0, 30.0)
    first["pressure"] = 32.0
    second["pressure"] = 32.0
    mover["port"] = 2
    mover["pressure"] = 40.0

    records = [first, second, mover]
    rows = _shape_settings_rows(records)
    port_pos = SHAPE_SETTINGS_HEADERS.index("Port")
    rows[2][port_pos] = 1  # mover joins port 1

    updated, table_out = normalize_shape_dimensions_for_mode(
        records, rows, SCALE_MODE_TARGET_DIMENSIONS
    )

    assert updated[2]["port"] == 1
    assert updated[2]["pressure"] == 32.0  # newcomer adopts the port pressure
    assert updated[0]["pressure"] == 32.0  # incumbents unchanged
    assert isinstance(table_out, list)


def test_new_shapes_adopt_the_existing_port_pressure() -> None:
    from app import _records_from_files

    previous = _records_from_files(["first.stl"], None)
    previous[0]["pressure"] = 40.0

    records = _records_from_files(["first.stl", "second.stl"], previous)

    assert records[0]["pressure"] == 40.0
    assert records[1]["port"] == records[0]["port"]
    assert records[1]["pressure"] == 40.0  # same port -> same regulator


def test_split_pieces_are_never_rescaled_by_keep_proportions() -> None:
    # Regression: split pieces inherit the parent's original_* dims while
    # their targets are the CELL sizes; a table echo in Keep Proportions
    # used to "restore" the parent dimensions (shapes visibly reset after
    # e.g. a color change).
    piece = {
        "idx": 1, "name": "cube - R1C1", "stl_path": None,
        "original_x": 30.0, "original_y": 30.0, "original_z": 30.0,
        "target_x": 15.2, "target_y": 15.2, "target_z": 30.0,
        "pressure": 25.0, "valve": 4, "nozzle": 1, "port": 1,
        "color": "#111111", "split_group_id": "split-1", "split_columns": 2,
        "split_rows": 2, "last_scaled_axis": "target_x",
    }
    records = [piece, dict(piece, idx=2, name="cube - R1C2", nozzle=1, valve=5)]
    rows = _shape_settings_rows(records)

    updated, table_out = normalize_shape_dimensions_for_mode(
        records, rows, SCALE_MODE_UNIFORM_FACTOR
    )

    assert updated[0]["target_x"] == 15.2  # NOT reset to 30
    assert updated[0]["target_z"] == 30.0
    assert updated[1]["target_x"] == 15.2
    assert not isinstance(table_out, list)  # nothing changed, no write-back


def test_select_all_string_bools_trigger_a_canonical_rewrite() -> None:
    # Gradio's header "select all" writes STRING "true"/"false" into the
    # checkbox columns (rendered as text / stray stale checkboxes). The
    # normalizer must answer with real rows so the table re-renders with
    # proper booleans; a clean payload must stay a no-op.
    from app import SCALE_MODE_TARGET_DIMENSIONS

    records = [
        _mm_member(1, 1, 10.0, 10.0, 10.0),
        _mm_member(2, 2, 10.0, 10.0, 10.0),
    ]
    rows = _shape_settings_rows(records)
    lead_in_pos = SHAPE_SETTINGS_HEADERS.index("Lead In")
    for row in rows:
        row[lead_in_pos] = "true"  # select-all artifact

    updated, table_out = normalize_shape_dimensions_for_mode(
        records, rows, SCALE_MODE_TARGET_DIMENSIONS
    )

    assert all(record["lead_in"] is True for record in updated)
    assert isinstance(table_out, list)  # canonical rewrite issued
    assert all(row[lead_in_pos] is True for row in table_out)  # real booleans

    # The rewrite's echo is clean -> converges to a no-op.
    updated2, table_out2 = normalize_shape_dimensions_for_mode(
        updated, table_out, SCALE_MODE_TARGET_DIMENSIONS
    )
    assert not isinstance(table_out2, list)

    # Unchecking via select-all ("false" strings) round-trips too.
    rows_off = _shape_settings_rows(updated2)
    for row in rows_off:
        row[lead_in_pos] = "false"
    updated3, table_out3 = normalize_shape_dimensions_for_mode(
        updated2, rows_off, SCALE_MODE_TARGET_DIMENSIONS
    )
    assert all(record["lead_in"] is False for record in updated3)
    assert isinstance(table_out3, list)


def test_apply_bulk_bool_selection_sets_a_whole_column() -> None:
    from app import apply_bulk_bool_selection

    records = [
        _mm_member(1, 1, 10.0, 10.0, 10.0),
        _mm_member(2, 2, 10.0, 10.0, 10.0),
        _mm_member(3, 3, 10.0, 10.0, 10.0),
    ]
    lead_pos = SHAPE_SETTINGS_HEADERS.index("Lead In")
    contour_pos = SHAPE_SETTINGS_HEADERS.index("Contour Tracing")

    updated, rows = apply_bulk_bool_selection(records, None, f"{lead_pos}|1")
    assert all(record["lead_in"] is True for record in updated)
    assert all(record.get("contour_tracing") is not True for record in updated)  # no bleed
    assert all(row[lead_pos] is True for row in rows)
    assert all(row[contour_pos] is False for row in rows)

    # Unchecking clears the whole column; junk payloads change nothing.
    cleared, rows2 = apply_bulk_bool_selection(updated, rows, f"{lead_pos}|0")
    assert all(record["lead_in"] is False for record in cleared)
    same, _rows3 = apply_bulk_bool_selection(cleared, rows2, "garbage")
    assert all(record["lead_in"] is False for record in same)
    # Non-bool columns are refused.
    color_pos = SHAPE_SETTINGS_HEADERS.index("Color")
    refused, _rows4 = apply_bulk_bool_selection(cleared, rows2, f"{color_pos}|1")
    assert refused[0].get("color") == cleared[0].get("color")


def test_download_all_zip_bundles_gcode_files_by_shape_name(tmp_path) -> None:
    import zipfile

    from app import _gcode_zip_update

    first = tmp_path / "circle_gcode.txt"
    first.write_text("G91\n")
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    duplicate = other_dir / "circle_gcode.txt"
    duplicate.write_text("G91\n")
    square = tmp_path / "square_gcode.txt"
    square.write_text("G91\n")

    update = _gcode_zip_update(
        [
            {"idx": 1, "gcode_path": str(first)},
            {"idx": 2, "gcode_path": str(duplicate)},
            {"idx": 3, "gcode_path": str(square)},
            {"idx": 4, "gcode_path": None},
        ]
    )
    assert update["visible"] is True
    with zipfile.ZipFile(update["value"]) as bundle:
        assert sorted(bundle.namelist()) == [
            "circle_gcode.txt",
            "circle_gcode_2.txt",
            "square_gcode.txt",
        ]

    # Nothing generated yet: the button hides instead of offering an empty zip.
    hidden = _gcode_zip_update([{"idx": 1, "gcode_path": None}])
    assert hidden["visible"] is False


def test_layer_preview_draws_the_selection_and_its_nozzle_group() -> None:
    from shapely.geometry import MultiPolygon, box

    from app import update_layer_preview
    from stl_slicer import LayerStack

    def _preview_stack(name: str, poly) -> LayerStack:
        return LayerStack(
            layers=[MultiPolygon([poly]), MultiPolygon([poly])],
            z_values=[0.4, 1.2],
            bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 1.6)),
            layer_height=0.8,
            name=name,
        )

    records = [
        {"idx": 1, "name": "base", "stl_path": "base.stl", "nozzle": 1, "color": "#ff0000",
         "layer_stack": _preview_stack("base", box(0.0, 0.0, 10.0, 10.0))},
        {"idx": 2, "name": "stripe", "stl_path": "stripe.stl", "nozzle": 1, "color": "#0000ff",
         "layer_stack": _preview_stack("stripe", box(2.0, 2.0, 8.0, 8.0))},
        {"idx": 3, "name": "solo", "stl_path": "solo.stl", "nozzle": 2, "color": "#00ff00",
         "layer_stack": _preview_stack("solo", box(0.0, 0.0, 5.0, 5.0))},
    ]

    # Selected shape plus its same-nozzle assembly sibling are both drawn.
    slider, fig = update_layer_preview(records, "1: base", None, 2)
    assert slider["maximum"] == 2
    assert slider["value"] == 2
    assert len(fig.axes[0].patches) == 2

    # A shape alone on its nozzle draws only itself; out-of-range slider
    # values clamp to the layer count.
    slider_solo, fig_solo = update_layer_preview(records, "3: solo", None, 99)
    assert slider_solo["value"] == 2
    assert len(fig_solo.axes[0].patches) == 1

    # Unsliced shape: slider resets, figure carries the hint (no patches).
    unsliced = [{"idx": 1, "name": "raw", "stl_path": "raw.stl", "nozzle": 1, "color": "#ff0000", "layer_stack": None}]
    slider_reset, fig_hint = update_layer_preview(unsliced, "1: raw", None, 5)
    assert slider_reset["maximum"] == 1
    assert not fig_hint.axes[0].patches


def test_multi_material_demo_set_groups_parts_onto_shared_nozzles() -> None:
    from app import load_sample_shapes

    outputs = load_sample_shapes(None, [], None, "Multi-Material Demo")
    records = outputs[1]
    assert [record["name"] for record in records] == [
        "Checkerboard_Cube_1",
        "Checkerboard_Cube_2",
        "Wrapped_Egg_Inside",
        "Wrapped_Egg_Outside",
        "Space_Helmet_Glass",
        "Space_Helmet_Shell",
    ]
    # Parts of the same model share a nozzle (three assemblies)...
    assert [record["nozzle"] for record in records] == [1, 1, 2, 2, 3, 3]
    # ...while every part keeps its own valve.
    valves = [record["valve"] for record in records]
    assert len(set(valves)) == 6
    # The table rows carry the grouped nozzles too.
    nozzle_pos = SHAPE_SETTINGS_HEADERS.index("Nozzle")
    assert [row[nozzle_pos] for row in outputs[2]] == [1, 1, 2, 2, 3, 3]


def test_project_settings_export_import_round_trip(tmp_path) -> None:
    from app import export_project_settings, import_project_settings

    records = _records_from_files(["egg_inside.stl", "egg_outside.stl"], None)
    records[0].update(nozzle=2, valve=9, pressure=40.0, infill=50.0, contour_tracing=True, color="#d62728")
    records[1].update(nozzle=2, valve=10, pressure=40.0)

    settings_path, status = export_project_settings(
        records, None, 0.4, 0.4, None, "Circle Spiral raster", False, 1.2,
        6.0, 7.0, 4, "Down", "Horizontal", 15.0,
    )
    assert settings_path and "2 shape(s)" in status

    # A fresh session re-uploads the same files, then imports.
    fresh = _records_from_files(["egg_inside.stl", "egg_outside.stl", "extra.stl"], None)
    outputs = import_project_settings([settings_path], fresh, None)
    updated, rows, message = outputs[0], outputs[1], outputs[2]
    assert "2 shape(s)" in message
    assert updated[0]["nozzle"] == 2 and updated[0]["valve"] == 9
    assert updated[0]["pressure"] == 40.0 and updated[0]["infill"] == 50.0
    assert updated[0]["contour_tracing"] is True and updated[0]["color"] == "#d62728"
    assert updated[1]["nozzle"] == 2 and updated[1]["valve"] == 10
    assert updated[2]["name"] == "extra"  # untouched shape keeps defaults
    # Generation options come back as component updates.
    option_updates = outputs[3:]
    assert option_updates[0]["value"] == "Circle Spiral raster"
    assert option_updates[1]["value"] is False
    assert option_updates[2]["value"] == 1.2
    assert option_updates[8]["value"] == 0.4  # layer height
    assert option_updates[11]["value"] == 15.0  # nozzle speed

    # A file listed in the export but not loaded is reported.
    partial = import_project_settings([settings_path], _records_from_files(["egg_inside.stl"], None), None)
    assert "egg_outside.stl" in partial[2]

    # Garbage input fails gracefully.
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    failed = import_project_settings([str(bad)], fresh, None)
    assert "Import failed" in failed[2]


def test_load_sample_shapes_respects_the_selected_set() -> None:
    from app import DEFAULT_SAMPLE_STL_SET, SAMPLE_STL_SETS, load_sample_shapes

    simple = load_sample_shapes(None, [], None, "Simple Shapes")
    assert [record["name"] for record in simple[1]] == [
        "Simple_Circle", "Simple_Square", "Simple_Triangle",
    ]

    standard = load_sample_shapes(None, [], None, "Standard Shapes")
    assert [record["name"] for record in standard[1]] == [
        "Hollow_Pyramid", "Rounded_Cube_Through_Holes", "halfsphere",
    ]

    # Unknown/empty selection falls back to the default set.
    fallback = load_sample_shapes(None, [], None, None)
    assert [record["name"] for record in fallback[1]] == [
        "Hollow_Pyramid", "Rounded_Cube_Through_Holes", "halfsphere",
    ]
    assert DEFAULT_SAMPLE_STL_SET in SAMPLE_STL_SETS
