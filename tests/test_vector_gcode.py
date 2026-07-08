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
    # World anchor of the relative toolpath (origin at sweep start (-1, 0.5)).
    assert lines[1] == "; PathOrigin X-1.0 Y0.5"
    assert lines[2] == "{aux_command}WAGO_ValveCommands(7, 0)"
    assert lines[3] == "serialPort3.write(eval(setpress(25)))"
    assert lines[4] == "serialPort3.write(eval(togglepress()))"
    assert lines[5].startswith("{aux_command}WAGO_ValveCommands(")
    assert lines[6].startswith("{aux_command}WAGO_ValveCommands(")


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

    assert moves[:9] == [
        {"start": (0.0, 0.0, 0.0), "end": (-7.0, 0.0, 0.0), "color": 0},
        {"start": (-7.0, 0.0, 0.0), "end": (-4.0, 0.0, 0.0), "color": 255},
        {"start": (-4.0, 0.0, 0.0), "end": (-4.0, 0.5, 0.0), "color": 0},
        {"start": (-4.0, 0.5, 0.0), "end": (-7.0, 0.5, 0.0), "color": 255},
        {"start": (-7.0, 0.5, 0.0), "end": (-7.0, 1.0, 0.0), "color": 0},
        {"start": (-7.0, 1.0, 0.0), "end": (-4.0, 1.0, 0.0), "color": 255},
        # Return route: exit the patch one spacing to the outside, travel
        # home through the clearance lane, then step onto the start point —
        # never dragging the primed nozzle back across the purge lines.
        {"start": (-4.0, 1.0, 0.0), "end": (-4.0, -0.5, 0.0), "color": 0},
        {"start": (-4.0, -0.5, 0.0), "end": (0.0, -0.5, 0.0), "color": 0},
        {"start": (0.0, -0.5, 0.0), "end": (0.0, 0.0, 0.0), "color": 0},
    ]
    assert all(move["end"][2] == 0.0 for move in moves[:9])

    first_z_index = next(index for index, move in enumerate(moves) if move["end"][2] > 0.0)
    assert first_z_index > 9
    assert not any(
        move["start"][0] < -3.0 or move["end"][0] < -3.0
        for move in moves[first_z_index:]
    )


def test_gcode_lead_in_direction_points_the_purge_patch(tmp_path) -> None:
    from vector_toolpath import LEAD_IN_DIRECTION_UP

    gcode_path = generate_vector_gcode(
        _stack(box(0.0, 0.0, 0.5, 0.5)),
        shape_name="lead_in_up",
        pressure=25,
        valve=7,
        port=3,
        fil_width=0.5,
        lead_in_enabled=True,
        lead_in_length=3.0,
        lead_in_clearance=4.0,
        lead_in_lines=2,
        lead_in_direction=LEAD_IN_DIRECTION_UP,
        output_dir=tmp_path,
    )

    moves = _moves_with_colors(gcode_path.read_text())
    # Patch is ABOVE the start: first travel goes +7 in Y, purge strokes are
    # vertical and sit in y in [4, 7]; only the half-fil-wide return lane
    # dips to the negative lateral side.
    assert moves[0]["end"] == (0.0, 7.0, 0.0)
    lead_prints = [m for m in moves[:6] if m["color"] == 255]
    assert lead_prints
    assert all(abs(m["end"][0] - m["start"][0]) < 1e-9 for m in lead_prints)
    assert all(
        3.9 <= min(m["start"][1], m["end"][1]) and max(m["start"][1], m["end"][1]) <= 7.1
        for m in lead_prints
    )
    assert all(m["end"][0] >= -0.5 - 1e-9 for m in moves[:9])


def test_lead_in_opt_out_travels_shared_patch_but_skips_it_solo(tmp_path) -> None:
    small = _stack(box(0.0, 0.0, 2.0, 2.0), name="small")
    big = _stack(box(0.0, 0.0, 4.0, 4.0), name="big")
    reference = build_reference_stack([small, big])

    def _generate(stack: LayerStack, dispense: bool, motion, label: str):
        path = generate_vector_gcode(
            stack,
            shape_name=label,
            pressure=25,
            valve=7,
            port=3,
            fil_width=1.0,
            motion=motion,
            lead_in_enabled=True,
            lead_in_length=3.0,
            lead_in_clearance=4.0,
            lead_in_lines=3,
            lead_in_dispense=dispense,
            output_dir=tmp_path / label,
        )
        return _moves_with_colors(path.read_text())

    # Shared motion: the opted-out head traverses the identical patch with
    # the valve shut; totals and endpoints match the dispensing head exactly.
    priming = _generate(big, True, reference, "priming")
    passive = _generate(small, False, reference, "passive")
    assert priming[-1]["end"] == passive[-1]["end"]
    assert abs(_total_length(priming) - _total_length(passive)) < 1e-6
    assert any(m["color"] == 255 for m in priming[:6])
    assert all(m["color"] == 0 for m in passive[:9])

    # Solo (no shared motion): the opted-out shape skips the lead-in
    # entirely — its first move is the raster approach, not the purge travel.
    solo = _generate(small, False, None, "solo")
    with_lead = _generate(small, True, None, "with_lead")
    assert len(solo) < len(with_lead)
    assert solo[0]["end"] != (-7.0, 0.0, 0.0)
    assert with_lead[0]["end"] == (-7.0, 0.0, 0.0)


def test_gcode_lead_in_return_never_crosses_the_purge_lines(tmp_path) -> None:
    for lines in (1, 2, 3, 4):
        gcode_path = generate_vector_gcode(
            _stack(box(0.0, 0.0, 0.5, 0.5)),
            shape_name=f"lead_return_{lines}",
            pressure=25,
            valve=7,
            port=3,
            fil_width=0.5,
            lead_in_enabled=True,
            lead_in_length=3.0,
            lead_in_clearance=4.0,
            lead_in_lines=lines,
            output_dir=tmp_path / str(lines),
        )
        moves = _moves_with_colors(gcode_path.read_text())
        lead_end = next(i for i, m in enumerate(moves) if m["end"] == (0.0, 0.0, 0.0))
        prints = [m for m in moves[: lead_end + 1] if m["color"] == 255]
        travels = [m for m in moves[: lead_end + 1] if m["color"] == 0]
        # No travel move's interior crosses a printed purge line: every
        # printed line sits on a lane y = k*0.5, and travels only run along
        # x = const (lane changes at line ends) or at y = -0.5 / y <= 0.
        for travel in travels[1:]:
            y0, y1 = travel["start"][1], travel["end"][1]
            x0, x1 = travel["start"][0], travel["end"][0]
            if abs(y1 - y0) < 1e-9 and abs(x1 - x0) > 1e-9:
                # Horizontal travel: must be outside the printed lanes.
                assert y0 < -1e-9 or not any(
                    abs(p["start"][1] - y0) < 1e-9 for p in prints
                ), (lines, travel)


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


def test_diagonal_woodpile_rotates_45_degrees_per_layer(tmp_path) -> None:
    from vector_toolpath import RASTER_PATTERN_DIAGONAL_WOODPILE

    layer = box(0.0, 0.0, 8.0, 8.0)
    gcode_path = generate_vector_gcode(
        _stack(layer, layer, layer, layer),
        shape_name="diagonal",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        layer_height=1.0,
        raster_pattern=RASTER_PATTERN_DIAGONAL_WOODPILE,
        output_dir=tmp_path,
    )

    moves = _moves_with_colors(gcode_path.read_text())
    directions_by_layer: dict[float, set[float]] = {}
    intercepts_45: set[float] = set()
    for move in moves:
        if move["color"] != 255:
            continue
        z = round(move["start"][2], 6)
        dx = move["end"][0] - move["start"][0]
        dy = move["end"][1] - move["start"][1]
        angle = round(math.degrees(math.atan2(dy, dx)) % 180.0, 1)
        directions_by_layer.setdefault(z, set()).add(angle)
        if z == 1.0:
            intercepts_45.add(
                round((move["start"][1] - move["start"][0]) / math.sqrt(2), 5)
            )

    # The raster angle cycles 0 -> 45 -> 90 -> 135 across layers.
    assert directions_by_layer == {
        0.0: {0.0},
        1.0: {45.0},
        2.0: {90.0},
        3.0: {135.0},
    }
    # Diagonal lines keep an exact one-fil perpendicular pitch.
    ordered = sorted(intercepts_45)
    assert len(ordered) > 3
    assert {round(b - a, 4) for a, b in zip(ordered, ordered[1:])} == {1.0}


def test_diagonal_woodpile_shares_reference_motion(tmp_path) -> None:
    from vector_toolpath import RASTER_PATTERN_DIAGONAL_WOODPILE

    big = _stack(*([box(0.0, 0.0, 8.0, 8.0)] * 4), name="big")
    small = _stack(*([box(2.0, 2.0, 6.0, 6.0)] * 4), name="small")
    reference = build_reference_stack([big, small], grid=1.0)

    totals = []
    for stack, label in ((big, "dbig"), (small, "dsmall")):
        gcode_path = generate_vector_gcode(
            stack,
            shape_name=label,
            pressure=25,
            valve=7,
            port=3,
            fil_width=1.0,
            motion=reference,
            raster_pattern=RASTER_PATTERN_DIAGONAL_WOODPILE,
            output_dir=tmp_path / label,
        )
        totals.append(_total_length(_moves_with_colors(gcode_path.read_text())))

    assert abs(totals[0] - totals[1]) < 1e-2


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


def test_circle_spiral_steps_radius_by_whole_pitches() -> None:
    # Each revolution is a true circle at a constant radius; the radius drops
    # by exactly one pitch in a single radial jump between revolutions.
    points = _circle_spiral_points(2.0, 3.0, outer_radius=4.0, pitch=0.8)
    radii = [math.hypot(x - 2.0, y - 3.0) for x, y in points]

    distinct = sorted({round(radius, 6) for radius in radii})
    assert distinct == [0.0, 0.8, 1.6, 2.4, 3.2, 4.0]

    ring_transitions = sum(
        1
        for previous, current in zip(radii, radii[1:])
        if abs(current - previous) > 1e-9
    )
    assert ring_transitions == 5


def test_circle_spiral_ring_steps_travel_with_valve_shut(tmp_path) -> None:
    from gcode_viewer import parse_gcode_path
    from vector_toolpath import RASTER_PATTERN_CIRCLE_SPIRAL

    layer = box(0.0, 0.0, 10.0, 10.0)
    gcode_path = generate_vector_gcode(
        _stack(layer, layer),
        shape_name="ring_steps",
        pressure=25,
        valve=7,
        port=3,
        fil_width=0.8,
        layer_height=1.0,
        raster_pattern=RASTER_PATTERN_CIRCLE_SPIRAL,
        output_dir=tmp_path,
    )

    parsed = parse_gcode_path(gcode_path.read_text())
    origin_x, origin_y = parsed["path_origin"]
    center_x = center_y = 5.0

    # Print moves stay on a constant-radius ring (within chord flattening);
    # the inward steps between rings — including pieces clipped by the
    # material boundary at the edges — are always valve-off travel.
    worst = 0.0
    for segment in parsed["print_segments"]:
        for a, b in zip(segment, segment[1:]):
            radius_a = math.hypot(a[0] + origin_x - center_x, a[1] + origin_y - center_y)
            radius_b = math.hypot(b[0] + origin_x - center_x, b[1] + origin_y - center_y)
            worst = max(worst, abs(radius_b - radius_a))
    assert worst < 0.11


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


def _total_length(moves: list[dict]) -> float:
    return sum(math.dist(move["start"][:2], move["end"][:2]) for move in moves)


def _print_length(moves: list[dict]) -> float:
    return sum(
        math.dist(move["start"][:2], move["end"][:2])
        for move in moves
        if move["color"] == 255
    )


def test_half_infill_skips_alternate_lines_but_keeps_the_same_path(tmp_path) -> None:
    layer = box(0.0, 0.0, 4.0, 4.0)
    stack = _stack(layer, layer)

    def _generate(infill: float, label: str):
        path = generate_vector_gcode(
            stack,
            shape_name=label,
            pressure=25,
            valve=7,
            port=3,
            fil_width=1.0,
            layer_height=1.0,
            infill=infill,
            output_dir=tmp_path / label,
        )
        return _moves_with_colors(path.read_text())

    full = _generate(1.0, "full")
    half = _generate(0.5, "half")

    # Identical motion: same final position and same total traversed length.
    assert full[-1]["end"] == half[-1]["end"]
    assert abs(_total_length(full) - _total_length(half)) < 1e-6

    # Half the lines dispense: 2 of the 4 sweeps per layer print.
    assert abs(_print_length(half) - _print_length(full) / 2) < 1e-6

    # The printing sweeps sit on alternating scanlines (one fil apart x2).
    half_print_rows = sorted({round(m["start"][1], 6) for m in half if m["color"] == 255 and m["start"][2] == 0.0})
    assert len(half_print_rows) == 2
    assert abs((half_print_rows[1] - half_print_rows[0]) - 2.0) < 1e-9


def test_infill_selection_is_shared_across_reference_motion(tmp_path) -> None:
    small = _stack(box(0.0, 0.0, 2.0, 2.0), name="small")
    big = _stack(box(0.0, 0.0, 4.0, 4.0), name="big")
    reference = build_reference_stack([small, big])

    def _generate(stack: LayerStack, infill: float, label: str):
        path = generate_vector_gcode(
            stack,
            shape_name=label,
            pressure=25,
            valve=7,
            port=3,
            fil_width=1.0,
            motion=reference,
            infill=infill,
            output_dir=tmp_path / label,
        )
        return _moves_with_colors(path.read_text())

    sparse = _generate(small, 0.5, "sparse")
    dense = _generate(big, 1.0, "dense")

    # Different infill per shape, one shared motion path.
    assert sparse[-1]["end"] == dense[-1]["end"]
    assert abs(_total_length(sparse) - _total_length(dense)) < 1e-6
    assert 0 < _print_length(sparse) < _print_length(dense)


def test_spiral_infill_skips_rings_and_keeps_the_path(tmp_path) -> None:
    layer = box(0.0, 0.0, 6.0, 6.0)
    stack = _stack(layer)

    def _generate(pattern: str, infill: float, label: str):
        path = generate_vector_gcode(
            stack,
            shape_name=label,
            pressure=25,
            valve=7,
            port=3,
            fil_width=1.0,
            raster_pattern=pattern,
            infill=infill,
            output_dir=tmp_path / label,
        )
        return _moves_with_colors(path.read_text())

    for pattern in (RASTER_PATTERN_RECTANGULAR_SPIRAL, RASTER_PATTERN_CIRCLE_SPIRAL):
        full = _generate(pattern, 1.0, f"{pattern}-full".replace(" ", "_"))
        half = _generate(pattern, 0.5, f"{pattern}-half".replace(" ", "_"))
        # Identical path within the writer's micron-level delta rounding
        # (segment split points differ, so the rounding accumulates
        # differently by up to ~1 um over tens of thousands of moves).
        assert math.dist(full[-1]["end"], half[-1]["end"]) < 1e-4, pattern
        assert abs(_total_length(full) - _total_length(half)) < 1e-3, pattern
        assert 0 < _print_length(half) < _print_length(full), pattern


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


def test_reference_motion_contours_share_path_and_gate_valve_per_shape(tmp_path) -> None:
    small = _stack(box(0.0, 0.0, 2.0, 2.0), name="small")
    big = _stack(box(0.0, 0.0, 4.0, 4.0), name="big")
    reference = build_reference_stack([small, big])
    assert reference is not None
    contour_sources = [
        ContourSource(owner_idx=1, stack=small),
        ContourSource(owner_idx=2, stack=big),
    ]

    def _generate(stack: LayerStack, owner: int, label: str):
        path = generate_vector_gcode(
            stack,
            shape_name=label,
            pressure=25,
            valve=7,
            port=3,
            fil_width=1.0,
            motion=reference,
            contour_sources=contour_sources,
            active_contour_owner=owner,
            output_dir=tmp_path / label,
        )
        return _moves_with_colors(path.read_text())

    small_moves = _generate(small, 1, "small")
    big_moves = _generate(big, 2, "big")

    # The motion including EVERY shape's contour tour is identical: same final
    # position and same total traversed length for both heads.
    assert small_moves[-1]["end"] == big_moves[-1]["end"]

    def _total_length(moves: list[dict]) -> float:
        return sum(math.dist(move["start"][:2], move["end"][:2]) for move in moves)

    assert abs(_total_length(small_moves) - _total_length(big_moves)) < 1e-6

    # In the shared frame (origin at the motion sweep start (-2, -0.5); big is
    # re-centred to (-1,-1)..(3,3), small stays (0,0)..(2,2)):
    small_corners = {(2.0, 0.5), (4.0, 0.5), (4.0, 2.5), (2.0, 2.5)}
    big_corners = {(1.0, -0.5), (5.0, -0.5), (5.0, 3.5), (1.0, 3.5)}

    def _endpoints(moves: list[dict], color: int) -> set[tuple[float, float]]:
        return {
            (round(move["end"][0], 6), round(move["end"][1], 6))
            for move in moves
            if move["color"] == color
        }

    # Each shape PRINTS its own outline and TRAVELS the other shape's outline.
    assert small_corners <= _endpoints(small_moves, 255)
    assert big_corners <= _endpoints(small_moves, 0)
    assert big_corners <= _endpoints(big_moves, 255)
    assert small_corners <= _endpoints(big_moves, 0)


def test_solo_contours_still_trace_only_own_shape(tmp_path) -> None:
    small = _stack(box(0.0, 0.0, 2.0, 2.0), name="small")
    big = _stack(box(0.0, 0.0, 4.0, 4.0), name="big")
    contour_sources = [
        ContourSource(owner_idx=1, stack=small),
        ContourSource(owner_idx=2, stack=big),
    ]

    gcode_path = generate_vector_gcode(
        small,
        shape_name="solo",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        contour_sources=contour_sources,
        active_contour_owner=1,
        output_dir=tmp_path,
    )

    moves = _moves_with_colors(gcode_path.read_text())
    # Without reference motion the other shape's contour must NOT be traced:
    # nothing ever moves outside the small shape's buffered footprint
    # (origin at world (-1, 0.5), so relative x spans [0, 4], y [-0.5, 1.5];
    # the big shape's contour would reach (5.0, 3.5)).
    for move in moves:
        assert -0.5 <= move["end"][0] <= 4.2
        assert -1.0 <= move["end"][1] <= 2.0


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


def test_grid_split_pads_equal_whole_fil_cells() -> None:
    # 20.0 / 4 = 5.0 mm cells, which is 6.25 fil widths — not representable.
    # With `grid`, every cell rounds UP to 7 fils (5.6 mm) and the 2.4 mm
    # leftover becomes blank margin split evenly outside the outer edges.
    layer = box(0.0, 0.0, 20.0, 4.0)
    stack = _stack(layer, name="wide")

    pieces = split_layer_stack_grid(stack, columns=4, rows=1, grid=0.8)

    widths = {round(piece.bounds[1][0] - piece.bounds[0][0], 6) for piece in pieces}
    assert widths == {5.6}
    # Padding is centred: 1.2 mm of blank space beyond each outer edge.
    assert round(pieces[0].bounds[0][0], 6) == -1.2
    assert round(pieces[-1].bounds[1][0], 6) == 21.2
    # No material is lost or duplicated by the padded cells.
    total_area = sum(piece.layers[0].area for piece in pieces)
    assert abs(total_area - layer.area) < 1e-9


def test_grid_split_reference_deltas_are_uniform() -> None:
    from vector_toolpath import _centering_delta

    layer = box(0.0, 0.0, 20.0, 4.0)
    stack = _stack(layer, name="wide")
    pieces = split_layer_stack_grid(stack, columns=4, rows=1, grid=0.8)
    reference = build_reference_stack(pieces, grid=0.8)
    assert reference is not None

    deltas = [_centering_delta(piece, reference)[0] for piece in pieces]
    diffs = {round(a - b, 6) for a, b in zip(deltas, deltas[1:])}
    # Uniform spacing between every consecutive pair (one cell = 7 fils),
    # so the physical nozzle offsets are the same for every connection.
    assert diffs == {5.6}


def test_split_overlap_seam_raster_distance_is_equal_on_both_sides() -> None:
    from vector_toolpath import _axis_raster_segments

    # 7.5 mm is deliberately not a multiple of the 1 mm fil width, so raster
    # quantization leaves slack. The slack must be split evenly: both pieces'
    # lines sit the same distance from the (shifted) cut on every layer.
    layer = MultiPolygon([box(0.0, 0.0, 7.5, 4.0)])
    stack = LayerStack(
        layers=[layer, layer],
        z_values=[0.5, 1.5],
        bounds=((0.0, 0.0, 0.0), (7.5, 4.0, 2.0)),
        layer_height=1.0,
        name="seam",
    )
    left, right = split_layer_stack_grid(
        stack, columns=2, rows=1, overlapping_layers=True, overlap=0.5
    )

    for layer_number in range(2):
        left_columns = sorted(
            {seg[0] for seg in _axis_raster_segments(
                left.layers[layer_number], left.layers[layer_number], 1.0, "Y"
            ) if seg[4] == 255}
        )
        right_columns = sorted(
            {seg[0] for seg in _axis_raster_segments(
                right.layers[layer_number], right.layers[layer_number], 1.0, "Y"
            ) if seg[4] == 255}
        )
        seam = left.layers[layer_number].bounds[2]
        left_distance = seam - left_columns[-1]
        right_distance = right_columns[0] - seam
        assert abs(left_distance - right_distance) < 1e-9


def test_split_contour_paths_exclude_the_cut_seams() -> None:
    layer = MultiPolygon([box(0.0, 0.0, 9.0, 4.0)])
    stack = _stack(layer, name="bar")

    left, middle, right = split_layer_stack_grid(stack, columns=3, rows=1, grid=1.0)

    # Middle piece: only the parent's top and bottom edges, as open arcs —
    # no vertical paths along the cuts at x=3 and x=6.
    assert middle.contour_paths[0] == [
        [(3.0, 0.0), (6.0, 0.0)],
        [(3.0, 4.0), (6.0, 4.0)],
    ]
    # Edge pieces get one open C-shaped path around their outer three sides.
    (left_path,) = left.contour_paths[0]
    assert left_path[0] != left_path[-1]
    assert all(abs(x - 3.0) > 1e-9 or y in (0.0, 4.0) for x, y in left_path)

    # A fully interior piece has no contour at all.
    grid = split_layer_stack_grid(
        _stack(MultiPolygon([box(0.0, 0.0, 9.0, 9.0)]), name="sq"),
        columns=3,
        rows=3,
        grid=1.0,
    )
    assert grid[4].contour_paths[0] == []

    # A hole entirely inside one piece stays a closed ring.
    hollow = MultiPolygon(
        [
            Polygon(
                box(0.0, 0.0, 9.0, 4.0).exterior.coords,
                [list(box(1.0, 1.0, 2.0, 2.0).exterior.coords)],
            )
        ]
    )
    hole_left, _hm, _hr = split_layer_stack_grid(
        _stack(hollow, name="hollow"), columns=3, rows=1, grid=1.0
    )
    closed_paths = [p for p in hole_left.contour_paths[0] if p[0] == p[-1]]
    assert len(closed_paths) == 1


def test_split_contour_gcode_never_traces_the_cuts(tmp_path) -> None:
    layer = MultiPolygon([box(0.0, 0.0, 9.0, 4.0)])
    stack = _stack(layer, layer, name="bar")
    pieces = split_layer_stack_grid(stack, columns=3, rows=1, grid=1.0)
    reference = build_reference_stack(pieces, grid=1.0)
    sources = [
        ContourSource(owner_idx=index + 1, stack=piece)
        for index, piece in enumerate(pieces)
    ]

    all_moves = []
    for index, piece in enumerate(pieces):
        gcode_path = generate_vector_gcode(
            piece,
            shape_name=f"seam{index}",
            pressure=25,
            valve=4 + index,
            port=3,
            fil_width=1.0,
            motion=reference,
            contour_sources=sources,
            active_contour_owner=index + 1,
            output_dir=tmp_path / f"seam{index}",
        )
        all_moves.append(_moves_with_colors(gcode_path.read_text()))

    # All heads still share one motion path, contours included.
    totals = {round(_total_length(moves), 4) for moves in all_moves}
    assert len(totals) == 1
    assert len({moves[-1]["end"] for moves in all_moves}) == 1

    # The middle piece's contour arcs are horizontal: with the horizontal
    # X-raster infill, it must emit NO vertical print move at all (a vertical
    # print could only be a traced cut seam).
    middle = all_moves[1]
    vertical_prints = [
        move
        for move in middle
        if move["color"] == 255
        and abs(move["end"][0] - move["start"][0]) < 1e-9
        and abs(move["end"][1] - move["start"][1]) > 1e-9
    ]
    assert vertical_prints == []


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
