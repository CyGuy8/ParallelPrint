"""G-code emission for vector layer stacks.

Writes the same machine dialect as the old TIFF pipeline: G91 relative moves,
G0 travel / G1 print (or all-G1), WAGO valve commands on every valve-state
change, and serial pressure preset/toggle commands with an optional per-layer
pressure ramp.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from stl_slicer import LayerStack
from vector_toolpath import (
    LEAD_IN_DIRECTION_CHOICES,
    LEAD_IN_DIRECTION_LEFT,
    RASTER_PATTERN_CHOICES,
    RASTER_PATTERN_CIRCLE_SPIRAL,
    RASTER_PATTERN_DIAGONAL_WOODPILE,
    RASTER_PATTERN_RECTANGULAR_SPIRAL,
    RASTER_PATTERN_SAME_DIRECTION,
    RASTER_PATTERN_WOODPILE,
    RASTER_PATTERN_Y_DIRECTION,
    ContourSource,
    _centering_delta,
    _lead_in_moves,
    _normalize_raster_pattern,
    align_stack_to,
    build_contour_layers,
    circle_wall_radius,
    plan_layer_moves,
)

# The pressure box protocol ("08PS" + pressure*10 over serial) drives a
# regulator with a 0-100 psi range; values are encoded to the tenths place.
MAX_PRESSURE_PSI = 100.0

__all__ = [
    "LEAD_IN_DIRECTION_CHOICES",
    "LEAD_IN_DIRECTION_LEFT",
    "MAX_PRESSURE_PSI",
    "RASTER_PATTERN_CHOICES",
    "RASTER_PATTERN_CIRCLE_SPIRAL",
    "RASTER_PATTERN_DIAGONAL_WOODPILE",
    "RASTER_PATTERN_RECTANGULAR_SPIRAL",
    "RASTER_PATTERN_SAME_DIRECTION",
    "RASTER_PATTERN_WOODPILE",
    "RASTER_PATTERN_Y_DIRECTION",
    "ContourSource",
    "generate_vector_gcode",
    "write_gcode_file",
]


def _pressure_variable_name(port: int) -> str:
    """The hand-editable pressure variable, one per serial port.

    The name carries the port number because the print host execs every
    file's commands in ONE shared namespace: with several port-owning files
    in a print, a single shared name would leave every ramp reading the
    LAST file's base pressure. Pressure is a port property, so `pressureN`
    is also the natural granularity.
    """
    return f"pressure{port}"


def _pressure_variable_cmd(port: int, pressure: float) -> str:
    """The one hand-editable knob: `pressureN = X` on the file's first line.

    The exact form is dictated by how the TCode print host consumes files:

    - Only lines containing `{aux_command}` or `serialPort` are routed as
      commands; anything else is parsed as a G-code move. The `{aux_command}`
      marker (stripped by the host) is what keeps this line off the motion
      timeline.
    - The host joins all `{preset}` commands into ONE list literal and
      exec()s it at the controller's START signal, so the line must be a
      pure EXPRESSION — a bare `pressure = X` assignment would be a syntax
      error there. `globals().update(pressureN = X)` is an expression that
      still reads like the assignment, and it lands the name in the host's
      module globals where the setpress preset (same exec, evaluated next)
      and the mid-file ramp commands (exec'd individually) can see it.

    Editing this single number retunes every pressure command in the file —
    no need to regenerate the G-code just to print at a different pressure.
    """
    return (
        f"{{preset}}{{aux_command}}globals().update("
        f"{_pressure_variable_name(port)} = {pressure:g})"
    )


def _setpress_cmd(port: str, expression: str, start: bool) -> str:
    """Pressure preset in the readable host dialect.

    Every pressure command uses the same `eval(setpress(...))` form (the
    print host defines `setpress`, which builds the serial byte string).
    `expression` is evaluated by the host — it references the `pressureN`
    variable declared at the top of the file rather than a baked-in number,
    so hand-editing that variable retunes the preset and the whole ramp.
    `start` adds the {preset} marker: the host executes presets at the
    controller's START signal, before the initial toggles; ramp commands
    mid-file run at their scheduled times instead.
    """
    if start:
        return f"\n\r{{preset}}{port}.write(eval(setpress({expression})))"
    return f"\n\r{port}.write(eval(setpress({expression})))"


def _ramp_expression(port: int, offset: float) -> str:
    """Ramp step as an expression of the top-of-file `pressureN` variable.

    The regulator tops out at MAX_PRESSURE_PSI and resolves tenths, so the
    clamp and rounding travel inside the expression — they must hold for
    whatever value the user hand-edits the variable to, which rules out
    pre-computing them here.
    """
    return (
        f"round(min({_pressure_variable_name(port)} + {offset:g}, "
        f"{MAX_PRESSURE_PSI:g}), 1)"
    )


def _toggle_cmd(port: str, start: bool) -> str:
    if start:
        return f"\n\r{{preset}}{port}.write(eval(togglepress()))"
    return f"\n\r{port}.write(eval(togglepress()))"


def _valve_cmd(valve: int, command: int) -> str:
    return f"\n{{aux_command}}WAGO_ValveCommands({valve}, {command})\n"


def _coord(value: float) -> str:
    """Format a coordinate in fixed-point notation, never scientific.

    Python's repr writes small floats as e.g. "-5.1e-08", which G-code axis
    parsers (including this project's viewer) misread as "-5.1".
    """
    text = f"{float(value):.6f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    if text in ("-0.0", "-0"):
        return "0.0"
    return text


def write_gcode_file(
    gcode_path: Path,
    gcode_list: list[dict],
    pressure: float,
    valve: int,
    port: int,
    increase_pressure_per_layer: float,
    pressure_ramp_enabled: bool,
    all_g1: bool,
    emit_pressure_commands: bool = True,
    pressure_variable: bool = True,
) -> None:
    """Write the move list as a G-code file.

    `emit_pressure_commands` gates EVERY pressure command (the top-of-file
    `pressure` variable, preset, toggle, per-layer ramp, closing toggle):
    the pressure regulator is a PORT
    device, so when several shapes share a serial port only ONE of their
    files may own it — the print host compiles all files onto one timeline,
    and duplicated toggles would flip the regulator on/off/on at start.

    `pressure_variable` picks the pressure command style: True writes the
    hand-editable `pressureN` variable on the first line with every
    setpress referencing it; False writes plain numeric setpress commands
    (the pre-variable format) with the ramp values baked in.
    """
    off_color = 0
    com_port = f"serialPort{port}"
    color_dict: dict[int, int] = {0: 100, 255: valve}

    # The regulator tops out at MAX_PRESSURE_PSI and resolves tenths: clamp
    # the default the file ships with; the ramp expressions carry their own
    # clamp so hand-edited values stay legal too.
    pressure = round(min(max(float(pressure), 0.0), MAX_PRESSURE_PSI), 1)
    preset_expression = (
        _pressure_variable_name(port) if pressure_variable else f"{pressure:g}"
    )
    setpress_lines = [_setpress_cmd(com_port, preset_expression, start=True)]
    pressure_on_lines = [_toggle_cmd(com_port, start=True)]
    pressure_off_lines = [_toggle_cmd(com_port, start=False)]

    ramp_layer = 0
    pressure_cur = float(pressure)

    with open(gcode_path, "w") as f:
        if emit_pressure_commands and pressure_variable:
            f.write(f"{_pressure_variable_cmd(port, pressure)}\n")
        f.write("G91\n")
        f.write(_valve_cmd(valve, 0))
        if emit_pressure_commands:
            for line in setpress_lines:
                f.write(f"{line}\n")
            for line in pressure_on_lines:
                f.write(f"{line}\n")

        pressure_next: str | None = None
        for i, move in enumerate(gcode_list):
            prev_color = gcode_list[i - 1]["Color"] if i > 0 else 0
            cur_color = move["Color"]
            if prev_color != cur_color:
                if cur_color == off_color:
                    f.write(_valve_cmd(color_dict[prev_color], 0))
                else:
                    if prev_color == off_color:
                        f.write(_valve_cmd(color_dict[cur_color], 1))
                    else:
                        f.write(_valve_cmd(color_dict[cur_color], 1))
                        f.write(_valve_cmd(color_dict[prev_color], 0))

            # When all_g1 is set, every move is emitted as G1 regardless of
            # valve state; the valve commands still mark print vs travel.
            move_type = "G1" if (all_g1 or cur_color != off_color) else "G0"
            if "Z" in move:
                line = (
                    f"{move_type} X{_coord(move['X'])} Y{_coord(move['Y'])} "
                    f"Z{_coord(move['Z'])} ; Color {move['Color']}"
                )
                if pressure_ramp_enabled and emit_pressure_commands:
                    if pressure_variable:
                        ramp_layer += 1
                        offset = round(increase_pressure_per_layer * ramp_layer, 6)
                        ramp_expression = _ramp_expression(port, offset)
                    else:
                        pressure_cur = round(
                            min(
                                pressure_cur + increase_pressure_per_layer,
                                MAX_PRESSURE_PSI,
                            ),
                            1,
                        )
                        ramp_expression = f"{pressure_cur:g}"
                    pressure_next = _setpress_cmd(
                        com_port, ramp_expression, start=False
                    )
                else:
                    pressure_next = None
            else:
                line = (
                    f"{move_type} X{_coord(move['X'])} Y{_coord(move['Y'])} "
                    f"; Color {move['Color']}"
                )
                pressure_next = None

            f.write(f"{line}\n")
            if pressure_next is not None:
                f.write(f"{pressure_next}\n")
                pressure_next = None

        f.write(_valve_cmd(valve, 0))
        if emit_pressure_commands:
            for line in pressure_off_lines:
                f.write(f"{line}\n")


def generate_vector_gcode(
    shape: LayerStack,
    *,
    shape_name: str,
    pressure: float,
    valve: int,
    port: int,
    fil_width: float,
    layer_height: float | None = None,
    raster_pattern: str | None = RASTER_PATTERN_SAME_DIRECTION,
    motion: LayerStack | None = None,
    contour_sources: list[ContourSource] | None = None,
    active_contour_owner: int | None = None,
    contour_order: str | None = None,
    infill: float = 1.0,
    motion_infill_fractions: list[float] | None = None,
    emit_pressure_commands: bool = True,
    pressure_variable: bool = True,
    sweep_buffer: float | None = None,
    increase_pressure_per_layer: float = 0.1,
    pressure_ramp_enabled: bool = True,
    all_g1: bool = False,
    lead_in_enabled: bool = False,
    lead_in_length: float = 5.0,
    lead_in_clearance: float = 5.0,
    lead_in_lines: int = 3,
    lead_in_direction: str = LEAD_IN_DIRECTION_LEFT,
    lead_in_orientation: str | None = None,
    lead_in_dispense: bool = True,
    wall_sources: list[LayerStack] | None = None,
    origin_sink: dict | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate G-code for one sliced shape.

    Without `motion`, the shape's own layers drive both the nozzle path and
    the valve. With `motion` (the combined reference stack), the nozzle
    follows the shared reference path while the valve opens only inside this
    shape's own geometry, aligned into the reference frame — so parallel
    heads share one motion but each dispenses only its own shape.

    `wall_sources` (all shapes in the job, whole shapes only) matters for
    the Circle Spiral under shared motion: every shape's own wall radius
    joins the ONE shared ring set, so each shape keeps a smooth complete
    outer circle. Pass the SAME list to every shape's generation call.

    `motion_infill_fractions` lists EVERY shape's infill fraction (again the
    same list for every call): raster lines/rings that no head dispenses on
    are dropped from the shared motion instead of swept valve-off. When
    omitted, this shape's own fraction bounds its motion.
    """
    if shape is None or not shape.layers:
        raise ValueError("The shape has no sliced layers to generate G-code from.")
    if fil_width <= 0:
        raise ValueError("Filament width must be greater than zero.")

    raster_pattern = _normalize_raster_pattern(raster_pattern)
    if layer_height is None:
        layer_height = shape.layer_height

    if motion is not None:
        if not motion.layers:
            raise ValueError("The reference stack has no layers for motion.")
        motion_layers = motion.layers
        valve_layers = align_stack_to(shape, motion, len(motion.layers))
        contour_reference = motion
    else:
        motion_layers = shape.layers
        valve_layers = shape.layers
        contour_reference = None

    contour_layers = build_contour_layers(
        contour_sources,
        len(motion_layers),
        reference=contour_reference,
    )

    # Anchor the raster scan grid (and the diagonal-raster pivot) to the
    # motion stack's frame (a split piece's frame is its parent shape's
    # bounds) so lines stack across layers and stay on one continuous grid
    # across split pieces.
    frame_stack = motion if motion is not None else shape
    if frame_stack.scan_frame is not None:
        scan_frame = frame_stack.scan_frame
    else:
        (frame_x_min, frame_y_min, _fz), (frame_x_max, frame_y_max, _fz2) = frame_stack.bounds
        scan_frame = (frame_x_min, frame_y_min, frame_x_max, frame_y_max)

    # Circle Spiral under shared motion (whole shapes): rings centre on the
    # reference ALIGN centre (each shape is concentric with it) and every
    # shape's outermost-fitting grid ring joins the shared ring set, so each
    # shape keeps a complete, uniformly spaced outer circle.
    extra_wall_radii = None
    ring_center = None
    if (
        raster_pattern == RASTER_PATTERN_CIRCLE_SPIRAL
        and motion is not None
        and motion.scan_frame is None
        and shape.scan_frame is None
    ):
        if motion.align_center is not None:
            ring_center = motion.align_center
        else:
            ring_center = (
                (scan_frame[0] + scan_frame[2]) / 2.0,
                (scan_frame[1] + scan_frame[3]) / 2.0,
            )
        n_layers = len(motion_layers)
        extra_wall_radii = [[] for _ in range(n_layers)]
        sources = [
            source
            for source in (wall_sources or [])
            if source is not None and source.layers and source.scan_frame is None
        ]
        if sources:
            for source in sources:
                aligned = align_stack_to(source, motion, n_layers)
                for index in range(n_layers):
                    wall = circle_wall_radius(
                        aligned[index], ring_center[0], ring_center[1], fil_width
                    )
                    if wall is not None:
                        extra_wall_radii[index].append(wall)
        else:
            # No source list: at least this shape's own outer ring.
            for index in range(n_layers):
                wall = circle_wall_radius(
                    valve_layers[index], ring_center[0], ring_center[1], fil_width
                )
                if wall is not None:
                    extra_wall_radii[index].append(wall)

    gcode_list, toolpath_origin = plan_layer_moves(
        motion_layers,
        valve_layers,
        fil_width,
        float(layer_height),
        raster_pattern,
        contour_layers,
        active_contour_owner,
        shared_motion=motion is not None,
        scan_frame=scan_frame,
        infill_fraction=max(0.0, min(1.0, float(infill))),
        extra_wall_radii=extra_wall_radii,
        ring_center=ring_center,
        motion_infill_fractions=(
            [max(0.0, min(1.0, float(fraction))) for fraction in motion_infill_fractions]
            if motion_infill_fractions is not None
            else None
        ),
        # Valve-settle travel before/after each raster sweep; one fil_width
        # when not given. Pass the SAME value for every shape sharing motion.
        sweep_buffer=sweep_buffer,
        # Same value for every shape sharing motion: contour placement is
        # part of the shared path.
        contour_order=contour_order,
    )

    # World anchor: the toolpath origin expressed in the shape's own frame.
    # With reference motion the geometry was translated by the centering
    # delta, so subtract it to get back to the shape's coordinates. It is
    # handed back through `origin_sink` (NOT written into the G-code — the
    # printed file stays free of metadata): the app stores it on the shape
    # record for Auto Align Split Parts and the visualizations.
    if motion is not None:
        delta_x, delta_y = _centering_delta(shape, motion)
    else:
        delta_x = delta_y = 0.0
    path_origin = (toolpath_origin[0] - delta_x, toolpath_origin[1] - delta_y)
    if origin_sink is not None:
        origin_sink["path_origin"] = path_origin

    # A shape that opts out of the lead-in still TRAVELS the purge patch when
    # motion is shared (all heads must move identically) but keeps its valve
    # shut; printing solo, it skips the lead-in moves entirely.
    lead_in = _lead_in_moves(
        lead_in_enabled and (lead_in_dispense or motion is not None),
        lead_in_length,
        lead_in_clearance,
        lead_in_lines,
        fil_width,
        255 if lead_in_dispense else 0,
        0,
        direction=lead_in_direction,
        orientation=lead_in_orientation,
    )
    if lead_in:
        gcode_list = [*lead_in, *gcode_list]

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="vector_gcode_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    gcode_path = output_dir / f"{shape_name}_gcode.txt"
    write_gcode_file(
        gcode_path,
        gcode_list,
        pressure=float(pressure),
        valve=int(valve),
        port=int(port),
        increase_pressure_per_layer=float(increase_pressure_per_layer),
        pressure_ramp_enabled=bool(pressure_ramp_enabled),
        all_g1=bool(all_g1),
        emit_pressure_commands=bool(emit_pressure_commands),
        pressure_variable=bool(pressure_variable),
    )
    return gcode_path
