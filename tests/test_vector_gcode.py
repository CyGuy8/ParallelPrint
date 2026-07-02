from __future__ import annotations

import math

from shapely.geometry import MultiPolygon, Polygon, box

from stl_slicer import LayerStack
from vector_gcode import generate_vector_gcode
from vector_toolpath import (
    RASTER_PATTERN_CIRCLE_SPIRAL,
    RASTER_PATTERN_RECTANGULAR_SPIRAL,
    RASTER_PATTERN_WOODPILE,
    RASTER_PATTERN_Y_DIRECTION,
    ContourSource,
    _append_layer_contours,
    _circle_spiral_points,
    _layer_contour_loops,
    _rectangular_spiral_polyline,
    build_reference_stack,
    split_layer_stack_grid,
)


def _stack(
    *layers: Polygon | MultiPolygon | None,
    layer_height: float = 1.0,
    name: str = "shape",
) -> LayerStack:
    multipolygons: list[MultiPolygon] = []
    for layer in layers:
        if layer is None:
            multipolygons.append(MultiPolygon())
        elif isinstance(layer, MultiPolygon):
            multipolygons.append(layer)
        else:
            multipolygons.append(MultiPolygon([layer]))

    bounds_list = [layer.bounds for layer in multipolygons if not layer.is_empty]
    if bounds_list:
        x_min = min(b[0] for b in bounds_list)
        y_min = min(b[1] for b in bounds_list)
        x_max = max(b[2] for b in bounds_list)
        y_max = max(b[3] for b in bounds_list)
    else:
        x_min = y_min = x_max = y_max = 0.0

    return LayerStack(
        layers=multipolygons,
        z_values=[(index + 0.5) * layer_height for index in range(len(multipolygons))],
        bounds=((x_min, y_min, 0.0), (x_max, y_max, len(multipolygons) * layer_height)),
        layer_height=layer_height,
        name=name,
    )


def _move_signature(gcode_text: str) -> list[tuple[float | None, float | None, float | None]]:
    signature: list[tuple[float | None, float | None, float | None]] = []
    for line in gcode_text.splitlines():
        if not line.startswith(("G0", "G1")):
            continue
        axes: dict[str, float] = {}
        for token in line.split():
            if token[:1] in {"X", "Y", "Z"}:
                axes[token[0]] = float(token[1:])
        signature.append((axes.get("X"), axes.get("Y"), axes.get("Z")))
    return signature


def _move_endpoints_for_color(gcode_text: str, color: int) -> list[tuple[float, float]]:
    x = y = 0.0
    endpoints: list[tuple[float, float]] = []
    for line in gcode_text.splitlines():
        if not line.startswith(("G0", "G1")):
            continue
        start = (x, y)
        for token in line.split():
            if token.startswith("X"):
                x += float(token[1:])
            if token.startswith("Y"):
                y += float(token[1:])
        if f"; Color {color}" in line:
            endpoints.extend([start, (x, y)])
    return endpoints


def _moves_with_colors(gcode_text: str) -> list[dict]:
    x = y = z = 0.0
    moves: list[dict] = []
    for line in gcode_text.splitlines():
        if not line.startswith(("G0", "G1")):
            continue
        start = (x, y, z)
        for token in line.split():
            if token.startswith("X"):
                x += float(token[1:])
            if token.startswith("Y"):
                y += float(token[1:])
            if token.startswith("Z"):
                z += float(token[1:])
        color = None
        if "; Color " in line:
            color = int(line.rsplit("; Color ", 1)[1])
        moves.append({"start": start, "end": (x, y, z), "color": color})
    return moves


def _pressure_set_count(gcode_text: str) -> int:
    return gcode_text.count("\\x30\\x38\\x50\\x53") + gcode_text.count("setpress(")


def test_gcode_writes_fixed_point_coordinates_never_scientific(tmp_path) -> None:
    from vector_gcode import write_gcode_file

    gcode_path = tmp_path / "noise.txt"
    write_gcode_file(
        gcode_path,
        [
            {"X": -5.1e-08, "Y": 0.8, "Color": 0},
            {"X": 1.2e-05, "Y": 0.0, "Color": 255},
            {"X": 4.0, "Y": -0.0, "Color": 0},
        ],
        pressure=25,
        valve=7,
        port=3,
        increase_pressure_per_layer=0.1,
        pressure_ramp_enabled=True,
        all_g1=False,
    )

    move_lines = [
        line for line in gcode_path.read_text().splitlines() if line.startswith(("G0", "G1"))
    ]
    assert move_lines == [
        "G0 X0.0 Y0.8 ; Color 0",
        "G1 X0.000012 Y0.0 ; Color 255",
        "G0 X4.0 Y0.0 ; Color 0",
    ]


def test_slanted_shape_gcode_round_trips_through_the_viewer(tmp_path) -> None:
    from gcode_viewer import parse_gcode_path

    # Slanted edges produce float-noise sweep bounds that differ between rows
    # (the pyramid failure mode); the parsed positions must stay inside the
    # material footprint on every layer.
    layers = [
        Polygon(
            [
                (inset, inset),
                (20.0 - inset, inset),
                (20.0 - inset, 20.0 - inset),
                (inset, 20.0 - inset),
            ]
        )
        for inset in (0.0, 0.57735026918962, 1.15470053837925, 1.73205080756887)
    ]
    gcode_path = generate_vector_gcode(
        _stack(*layers),
        shape_name="slanted",
        pressure=25,
        valve=7,
        port=3,
        fil_width=0.8,
        layer_height=1.0,
        output_dir=tmp_path,
    )

    parsed = parse_gcode_path(gcode_path.read_text())
    for segment in parsed["print_segments"]:
        for x, y, _z in segment:
            # Origin sits one fil_width left of the layer-0 sweep start; all
            # print positions stay within the 20 mm footprint plus buffers.
            assert -1.0 <= x <= 21.0
            assert -1.0 <= y <= 21.0


def test_gcode_header_writes_presets_before_initial_aux_commands(tmp_path) -> None:
    gcode_path = generate_vector_gcode(
        _stack(box(0.0, 0.0, 1.0, 1.0)),
        shape_name="header_order",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        output_dir=tmp_path,
    )

    lines = [
        line.strip()
        for line in gcode_path.read_text().splitlines()
        if line.strip()
    ]

    assert lines[0] == "G91"
    assert lines[1] == "{aux_command}WAGO_ValveCommands(7, 0)"
    assert lines[2] == "serialPort3.write(eval(setpress(25)))"
    assert lines[3] == "serialPort3.write(eval(togglepress()))"
    assert lines[4].startswith("{aux_command}WAGO_ValveCommands(")
    assert lines[5].startswith("{aux_command}WAGO_ValveCommands(")


def test_gcode_lead_in_runs_once_before_first_layer(tmp_path) -> None:
    gcode_path = generate_vector_gcode(
        _stack(box(0.0, 0.0, 0.5, 0.5), box(0.0, 0.0, 0.5, 0.5)),
        shape_name="lead_in",
        pressure=25,
        valve=7,
        port=3,
        fil_width=0.5,
        layer_height=1.0,
        lead_in_enabled=True,
        lead_in_length=3.0,
        lead_in_clearance=4.0,
        lead_in_lines=3,
        output_dir=tmp_path,
    )

    moves = _moves_with_colors(gcode_path.read_text())

    assert moves[:7] == [
        {"start": (0.0, 0.0, 0.0), "end": (-7.0, 0.0, 0.0), "color": 0},
        {"start": (-7.0, 0.0, 0.0), "end": (-4.0, 0.0, 0.0), "color": 255},
        {"start": (-4.0, 0.0, 0.0), "end": (-4.0, 0.5, 0.0), "color": 0},
        {"start": (-4.0, 0.5, 0.0), "end": (-7.0, 0.5, 0.0), "color": 255},
        {"start": (-7.0, 0.5, 0.0), "end": (-7.0, 1.0, 0.0), "color": 0},
        {"start": (-7.0, 1.0, 0.0), "end": (-4.0, 1.0, 0.0), "color": 255},
        {"start": (-4.0, 1.0, 0.0), "end": (0.0, 0.0, 0.0), "color": 0},
    ]
    assert all(move["end"][2] == 0.0 for move in moves[:7])

    first_z_index = next(index for index, move in enumerate(moves) if move["end"][2] > 0.0)
    assert first_z_index > 7
    assert not any(
        move["start"][0] < -3.0 or move["end"][0] < -3.0
        for move in moves[first_z_index:]
    )


def test_gcode_pressure_ramp_can_be_disabled(tmp_path) -> None:
    stack = _stack(box(0.0, 0.0, 1.0, 1.0), box(0.0, 0.0, 1.0, 1.0))

    ramped_path = generate_vector_gcode(
        stack,
        shape_name="pressure_ramped",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        layer_height=1.0,
        pressure_ramp_enabled=True,
        output_dir=tmp_path / "ramped",
    )
    fixed_path = generate_vector_gcode(
        stack,
        shape_name="pressure_fixed",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        layer_height=1.0,
        pressure_ramp_enabled=False,
        output_dir=tmp_path / "fixed",
    )

    assert _pressure_set_count(ramped_path.read_text()) > 1
    assert _pressure_set_count(fixed_path.read_text()) == 1


def test_gcode_uses_g1_for_print_and_g0_for_travel(tmp_path) -> None:
    gcode_path = generate_vector_gcode(
        _stack(box(0.0, 0.0, 1.0, 1.0)),
        shape_name="move_types",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        output_dir=tmp_path,
    )

    move_lines = [
        line.strip()
        for line in gcode_path.read_text().splitlines()
        if line.startswith(("G0", "G1"))
    ]

    assert any(line.startswith("G1") and "; Color 255" in line for line in move_lines)
    assert all(not line.startswith("G0") for line in move_lines if "; Color 255" in line)
    assert all(not line.startswith("G1") for line in move_lines if "; Color 0" in line)


def test_woodpile_raster_switches_print_axis_between_layers(tmp_path) -> None:
    layer = box(0.0, 0.0, 3.0, 2.0)
    gcode_path = generate_vector_gcode(
        _stack(layer, layer, layer, layer),
        shape_name="woodpile",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        raster_pattern=RASTER_PATTERN_WOODPILE,
        output_dir=tmp_path,
    )

    gcode_text = gcode_path.read_text()
    move_lines = [
        line.strip()
        for line in gcode_text.splitlines()
        if line.startswith(("G0", "G1"))
    ]
    z_move_index = next(i for i, line in enumerate(move_lines) if " Z" in line)
    first_layer_prints = [
        line
        for line in move_lines[:z_move_index]
        if line.startswith("G1") and "; Color 255" in line
    ]
    second_layer_end = next(
        (i for i, line in enumerate(move_lines[z_move_index + 1 :], start=z_move_index + 1) if " Z" in line),
        len(move_lines),
    )
    second_layer_prints = [
        line
        for line in move_lines[z_move_index + 1 : second_layer_end]
        if line.startswith("G1") and "; Color 255" in line
    ]

    assert move_lines[0] == "G0 X1.0 Y0.0 ; Color 0"
    # Layer 0 prints sweep along X, layer 1 prints sweep along Y.
    assert first_layer_prints
    assert all("Y0.0" in line for line in first_layer_prints)
    assert second_layer_prints
    assert all("X0.0" in line for line in second_layer_prints)

    x = y = 0.0
    x_positions = [x]
    y_positions = [y]
    for line in move_lines:
        for token in line.split():
            if token.startswith("X"):
                x += float(token[1:])
            if token.startswith("Y"):
                y += float(token[1:])
        x_positions.append(x)
        y_positions.append(y)
    assert min(x_positions) == 0.0
    assert max(x_positions) == 5.0
    assert min(y_positions) == -1.5
    assert max(y_positions) == 2.5

    # Each layer restarts at the sweep-start candidate nearest the previous
    # layer's endpoint. Candidates are the four buffered sweep corners, here
    # in cumulative coordinates (origin = layer 0's start at world (-1, 0.5)).
    y_axis_candidates = [(1.5, -1.5), (1.5, 2.5), (3.5, -1.5), (3.5, 2.5)]
    x_axis_candidates = [(0.0, 0.0), (0.0, 1.0), (5.0, 0.0), (5.0, 1.0)]
    moves = _moves_with_colors(gcode_text)
    layer_changes = [move for move in moves if move["end"][2] > move["start"][2]]
    assert len(layer_changes) == 3
    for layer_number, layer_change in enumerate(layer_changes, start=1):
        candidates = y_axis_candidates if layer_number % 2 == 1 else x_axis_candidates
        start = layer_change["start"][:2]
        end = layer_change["end"][:2]
        assert end in candidates
        best = min(math.dist(start, candidate) for candidate in candidates)
        assert math.dist(start, end) <= best + 1e-9


def test_y_direction_raster_prints_each_layer_along_y_axis(tmp_path) -> None:
    layer = box(0.0, 0.0, 3.0, 2.0)
    gcode_path = generate_vector_gcode(
        _stack(layer, layer),
        shape_name="y_direction",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        raster_pattern=RASTER_PATTERN_Y_DIRECTION,
        output_dir=tmp_path,
    )

    gcode_text = gcode_path.read_text()
    move_lines = [
        line.strip()
        for line in gcode_text.splitlines()
        if line.startswith(("G0", "G1"))
    ]
    print_lines = [
        line
        for line in move_lines
        if line.startswith("G1") and "; Color 255" in line
    ]
    assert print_lines
    assert move_lines[0] == "G0 X0.0 Y1.0 ; Color 0"
    assert all("X0.0" in line and "Y0.0" not in line for line in print_lines)

    x = y = 0.0
    x_positions = [x]
    y_positions = [y]
    for line in move_lines:
        for token in line.split():
            if token.startswith("X"):
                x += float(token[1:])
            if token.startswith("Y"):
                y += float(token[1:])
        x_positions.append(x)
        y_positions.append(y)
    assert min(x_positions) == 0.0
    assert max(x_positions) == 2.0
    assert min(y_positions) == 0.0
    assert max(y_positions) == 4.0

    moves = _moves_with_colors(gcode_text)
    first_layer_change = next(
        move for move in moves if move["end"][2] > move["start"][2]
    )
    assert first_layer_change["start"][:2] == first_layer_change["end"][:2]


def test_raster_crosses_interior_holes_with_valve_off(tmp_path) -> None:
    hollow = Polygon(
        box(0.0, 0.0, 6.0, 6.0).exterior.coords,
        [list(box(2.0, 2.0, 4.0, 4.0).exterior.coords)],
    )
    gcode_path = generate_vector_gcode(
        _stack(hollow),
        shape_name="hollow",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        output_dir=tmp_path,
    )

    moves = _moves_with_colors(gcode_path.read_text())
    print_moves = [move for move in moves if move["color"] == 255]
    travel_moves = [move for move in moves if move["color"] == 0]

    # Middle sweeps must split into two print runs around the hole.
    assert len(print_moves) == 4 + 4  # 4 full-width rows + 2 rows split in two
    # Some interior travel (crossing the hole) exists besides the buffers.
    assert any(
        0.0 < move["start"][0] < 7.0 and 0.0 < move["end"][0] < 7.0
        for move in travel_moves
    )


def test_rectangular_spiral_polyline_reverses_center_to_edge() -> None:
    inward = _rectangular_spiral_polyline((0.0, 0.0, 3.0, 3.0), 1.0)
    outward = _rectangular_spiral_polyline((0.0, 0.0, 3.0, 3.0), 1.0, reverse=True)

    assert inward[0] != inward[-1]
    assert outward[0] == inward[-1]
    assert outward[-1] == inward[0]


def test_rectangular_spiral_raster_reverses_between_layers(tmp_path) -> None:
    layer = box(0.0, 0.0, 3.0, 3.0)
    gcode_path = generate_vector_gcode(
        _stack(layer, layer),
        shape_name="rectangular_spiral",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        layer_height=1.0,
        raster_pattern=RASTER_PATTERN_RECTANGULAR_SPIRAL,
        output_dir=tmp_path,
    )

    moves = _moves_with_colors(gcode_path.read_text())
    first_layer_change = next(
        move for move in moves if move["end"][2] > move["start"][2]
    )

    assert first_layer_change["start"][:2] == first_layer_change["end"][:2]
    end_x, end_y, end_z = moves[-1]["end"]
    assert abs(end_x) < 1e-9
    assert abs(end_y) < 1e-9
    assert end_z == 1.0


def test_circle_spiral_points_decrease_radius_to_center() -> None:
    points = _circle_spiral_points(2.0, 3.0, outer_radius=4.0, pitch=1.0)
    radii = [math.hypot(x - 2.0, y - 3.0) for x, y in points]

    assert radii[0] == 4.0
    assert radii[-1] == 0.0
    assert all(
        current <= previous + 1e-9
        for previous, current in zip(radii, radii[1:])
    )


def test_circle_spiral_raster_reverses_between_layers(tmp_path) -> None:
    layer = box(0.0, 0.0, 5.0, 5.0)
    gcode_path = generate_vector_gcode(
        _stack(layer, layer),
        shape_name="circle_spiral",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        layer_height=1.0,
        raster_pattern=RASTER_PATTERN_CIRCLE_SPIRAL,
        output_dir=tmp_path,
    )

    moves = _moves_with_colors(gcode_path.read_text())
    first_layer_change = next(
        move for move in moves if move["end"][2] > move["start"][2]
    )

    assert first_layer_change["start"][:2] == first_layer_change["end"][:2]
    end_x, end_y, end_z = moves[-1]["end"]
    assert abs(end_x) < 1e-9
    assert abs(end_y) < 1e-9
    assert end_z == 1.0


def test_layer_contour_loops_follow_polygon_rings() -> None:
    hollow = MultiPolygon(
        [
            Polygon(
                box(0.0, 0.0, 4.0, 4.0).exterior.coords,
                [list(box(1.0, 1.0, 2.0, 2.0).exterior.coords)],
            )
        ]
    )

    loops = _layer_contour_loops(hollow)

    assert len(loops) == 2
    # Largest loop (the exterior) sorts first.
    assert set(loops[0]) == {(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)}
    assert set(loops[1]) == {(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)}
    assert loops[0][0] == loops[0][-1]
    assert loops[1][0] == loops[1][-1]


def test_contour_tracing_travels_to_nearest_border_after_infill(tmp_path) -> None:
    layer = box(0.0, 0.0, 1.0, 1.0)
    stack = _stack(layer)
    gcode_path = generate_vector_gcode(
        stack,
        shape_name="nearest_border_contour",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        all_g1=True,
        contour_sources=[ContourSource(owner_idx=1, stack=stack)],
        active_contour_owner=1,
        output_dir=tmp_path,
    )

    moves = _moves_with_colors(gcode_path.read_text())

    # Move 0 is the valve-settle approach, move 1 the single infill sweep.
    assert moves[1]["color"] == 255
    infill_end = moves[1]["end"]
    # The contour starts printing from the point nearest the infill end,
    # with no travel in between (the trailing buffer is rewound).
    assert moves[2]["color"] == 255
    assert moves[2]["start"] == infill_end


def test_contour_tracing_closes_loop_and_restores_raster_endpoint(tmp_path) -> None:
    layer = box(0.0, 0.0, 2.0, 2.0)
    stack = _stack(layer, layer)
    gcode_path = generate_vector_gcode(
        stack,
        shape_name="contour_loop",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        layer_height=1.0,
        all_g1=True,
        contour_sources=[ContourSource(owner_idx=1, stack=stack)],
        active_contour_owner=1,
        output_dir=tmp_path,
    )

    all_moves = _moves_with_colors(gcode_path.read_text())
    for layer_z in (0.0, 1.0):
        layer_moves = [
            move
            for move in all_moves
            if move["start"][2] == layer_z and move["end"][2] == layer_z
        ]
        layer_prints = [move for move in layer_moves if move["color"] == 255]
        assert layer_prints

        # The contour is a closed loop: the last print returns to where the
        # contour started.
        contour_prints = layer_prints[2:]
        assert contour_prints
        assert contour_prints[-1]["end"] == contour_prints[0]["start"]

        # After the contour, a travel move restores the raster endpoint.
        last_print_index = max(
            idx for idx, move in enumerate(layer_moves) if move["color"] == 255
        )
        trailing = layer_moves[last_print_index + 1 :]
        assert trailing
        assert all(move["color"] == 0 for move in trailing)


def test_contour_tracing_keeps_hollow_rings_separate() -> None:
    output = [{"X": 0.0, "Y": 0.0, "Color": 255}]
    contour_layers = [
        [
            {
                "owner_idx": 1,
                "contours": [
                    [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)],
                    [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0), (1.0, 1.0)],
                ],
            }
        ]
    ]

    current_x, current_y = _append_layer_contours(
        output,
        0.0,
        0.0,
        contour_layers,
        layer_number=0,
        active_owner_idx=1,
    )

    contour_print_moves = [move for move in output[1:] if move["Color"] == 255]

    assert len(contour_print_moves) == 8
    assert (current_x, current_y) == (1.0, 1.0)


def test_contour_tracing_skips_inactive_nozzle_outline(tmp_path) -> None:
    blank_stack = _stack(None)
    contour_stack = _stack(box(0.0, 0.0, 1.0, 1.0))
    contour_sources = [ContourSource(owner_idx=1, stack=contour_stack)]

    active_path = generate_vector_gcode(
        blank_stack,
        shape_name="active_contour",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        all_g1=True,
        contour_sources=contour_sources,
        active_contour_owner=1,
        output_dir=tmp_path / "active",
    )
    inactive_path = generate_vector_gcode(
        blank_stack,
        shape_name="inactive_contour",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        all_g1=True,
        contour_sources=contour_sources,
        active_contour_owner=2,
        output_dir=tmp_path / "inactive",
    )

    active_text = active_path.read_text()
    inactive_text = inactive_path.read_text()

    assert _move_signature(active_text)
    assert _move_signature(inactive_text) == []
    assert any(
        line.startswith("G1") and "; Color 255" in line
        for line in active_text.splitlines()
    )
    assert not any("; Color 255" in line for line in inactive_text.splitlines())


def test_inactive_contour_tracing_preserves_original_raster_moves(tmp_path) -> None:
    layer = box(1.0, 1.0, 3.0, 2.0)
    stack = _stack(layer, layer)

    original_path = generate_vector_gcode(
        stack,
        shape_name="original_raster",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        all_g1=True,
        output_dir=tmp_path / "original",
    )
    inactive_path = generate_vector_gcode(
        stack,
        shape_name="inactive_contour_raster",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        all_g1=True,
        contour_sources=[ContourSource(owner_idx=2, stack=stack)],
        active_contour_owner=1,
        output_dir=tmp_path / "inactive",
    )

    assert _move_signature(inactive_path.read_text()) == _move_signature(
        original_path.read_text()
    )


def test_reference_motion_shares_path_and_gates_valve_per_shape(tmp_path) -> None:
    small = _stack(box(0.0, 0.0, 2.0, 2.0), name="small")
    big = _stack(box(0.0, 0.0, 4.0, 4.0), name="big")
    reference = build_reference_stack([small, big])
    assert reference is not None

    def _generate(stack: LayerStack, label: str):
        return generate_vector_gcode(
            stack,
            shape_name=label,
            pressure=25,
            valve=7,
            port=3,
            fil_width=1.0,
            motion=reference,
            output_dir=tmp_path / label,
        )

    small_moves = _moves_with_colors(_generate(small, "small").read_text())
    big_moves = _moves_with_colors(_generate(big, "big").read_text())

    # Both shapes follow the same shared motion path: identical final position
    # and identical total path length (the moves split at different valve
    # boundaries, but the traversed polyline is the same).
    assert small_moves[-1]["end"] == big_moves[-1]["end"]

    def _total_length(moves: list[dict]) -> float:
        return sum(math.dist(move["start"][:2], move["end"][:2]) for move in moves)

    assert abs(_total_length(small_moves) - _total_length(big_moves)) < 1e-6

    def _print_length(moves: list[dict]) -> float:
        return sum(
            math.dist(move["start"][:2], move["end"][:2])
            for move in moves
            if move["color"] == 255
        )

    # The big shape dispenses over more of the shared path than the small one.
    assert _print_length(small_moves) > 0
    assert _print_length(big_moves) > _print_length(small_moves)
    # The small shape's total print length matches its own area coverage:
    # 2mm-wide rows on the shared 4-row sweep -> only rows inside the small box.
    assert _print_length(small_moves) < _print_length(big_moves) / 2 + 4.0


def test_build_reference_stack_unions_center_aligned_layers() -> None:
    first = _stack(box(0.0, 0.0, 2.0, 2.0), name="first")
    second = _stack(box(10.0, 10.0, 14.0, 14.0), name="second")

    reference = build_reference_stack([first, second])

    assert reference is not None
    # The second stack is re-centred onto the first stack's bbox centre (1, 1).
    assert reference.bounds == ((-1.0, -1.0, 0.0), (3.0, 3.0, 1.0))
    assert reference.layers[0].area == 16.0
    assert len(reference.layers) == 1
    assert reference.z_values == [0.5]


def test_split_layer_stack_grid_produces_row_major_cells() -> None:
    layer = box(10.0, -2.0, 12.5, -1.0)
    stack = _stack(layer, name="strip")

    pieces = split_layer_stack_grid(stack, columns=2, rows=1)

    assert [piece.name for piece in pieces] == ["strip_r01_c01", "strip_r01_c02"]
    assert pieces[0].bounds[0][0] == 10.0
    assert pieces[1].bounds[0][0] == 11.25
    total_area = sum(piece.layers[0].area for piece in pieces)
    assert abs(total_area - layer.area) < 1e-9


def test_split_layer_stack_grid_orders_rows_top_down() -> None:
    layer = box(0.0, 0.0, 4.0, 4.0)
    stack = _stack(layer, name="grid")

    pieces = split_layer_stack_grid(stack, columns=2, rows=2)

    assert [piece.name for piece in pieces] == [
        "grid_r01_c01",
        "grid_r01_c02",
        "grid_r02_c01",
        "grid_r02_c02",
    ]
    # Row 1 is the top strip (max-Y side).
    assert pieces[0].bounds == ((0.0, 2.0, 0.0), (2.0, 4.0, 1.0))
    assert pieces[3].bounds == ((2.0, 0.0, 0.0), (4.0, 2.0, 1.0))
    assert all(piece.layers[0].area == 4.0 for piece in pieces)


def test_split_layer_stack_grid_overlap_alternates_between_layers() -> None:
    layer = box(0.0, 0.0, 4.0, 2.0)
    stack = _stack(layer, layer, name="interlock")

    pieces = split_layer_stack_grid(
        stack,
        columns=2,
        rows=1,
        overlapping_layers=True,
        overlap=0.5,
    )

    left, right = pieces
    # The cut line alternates by +/- overlap between layers, so each piece's
    # area differs between layer 0 and layer 1 while the totals stay constant.
    assert left.layers[0].area != left.layers[1].area
    assert abs(left.layers[0].area - left.layers[1].area) == 2.0  # 2*(0.5*2)
    for index in range(2):
        combined = left.layers[index].area + right.layers[index].area
        assert abs(combined - layer.area) < 1e-9
    # Nominal bounds stay the un-shifted cells.
    assert left.bounds == ((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    assert right.bounds == ((2.0, 0.0, 0.0), (4.0, 2.0, 2.0))
