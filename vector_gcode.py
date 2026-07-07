"""G-code emission for vector layer stacks.

Writes the same machine dialect as the old TIFF pipeline: G91 relative moves,
G0 travel / G1 print (or all-G1), WAGO valve commands on every valve-state
change, and serial pressure preset/toggle commands with an optional per-layer
pressure ramp.
"""

from __future__ import annotations

import tempfile
from codecs import encode
from pathlib import Path
from textwrap import wrap

from stl_slicer import LayerStack
from vector_toolpath import (
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
    plan_layer_moves,
)

__all__ = [
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


def _setpress(pressure: float) -> str:
    pressure_str = str(int(pressure * 10)).zfill(4)
    command_bytes = bytes("08PS  " + pressure_str, "utf-8")
    hex_command = encode(command_bytes, "hex").decode("utf-8")
    format_command = "\\x" + "\\x".join(
        hex_command[i : i + 2] for i in range(0, len(hex_command), 2)
    )

    hex_pairs = wrap(hex_command, 2)
    decimal_sum = sum(int(pair, 16) for pair in hex_pairs)
    checksum_bin = bin(decimal_sum % 256)[2:].zfill(8)
    inverted = int("".join("1" if c == "0" else "0" for c in checksum_bin), 2) + 1
    checksum_hex = hex(inverted)[2:].upper()
    format_checksum = "\\x" + "\\x".join(
        checksum_hex[i : i + 2] for i in range(0, len(checksum_hex), 2)
    )

    return "b'" + "\\x05\\x02" + format_command + format_checksum + "\\x03" + "'"


def _togglepress() -> str:
    return "b'\\x05\\x02\\x30\\x34\\x44\\x49\\x20\\x20\\x43\\x46\\x03'"


def _setpress_cmd(port: str, pressure: float, start: bool) -> str:
    if start:
        return f"\n\r{port}.write(eval(setpress({pressure:g})))"
    insert = ""
    return f"\n\r{insert}{port}.write({_setpress(pressure)})"


def _toggle_cmd(port: str, start: bool) -> str:
    if start:
        return f"\n\r{port}.write(eval(togglepress()))"
    insert = ""
    return f"\n\r{insert}{port}.write({_togglepress()})"


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
    path_origin: tuple[float, float] | None = None,
) -> None:
    off_color = 0
    com_port = f"serialPort{port}"
    color_dict: dict[int, int] = {0: 100, 255: valve}

    setpress_lines = [_setpress_cmd(com_port, pressure, start=True)]
    pressure_on_lines = [_toggle_cmd(com_port, start=True)]
    pressure_off_lines = [_toggle_cmd(com_port, start=False)]

    pressure_cur = float(pressure)

    with open(gcode_path, "w") as f:
        f.write("G91\n")
        if path_origin is not None:
            # World anchor of the relative toolpath: absolute position (in the
            # shape's own coordinate frame) of the point the moves start from.
            # Lets tools place parallel parts so split pieces reassemble.
            f.write(
                f"; PathOrigin X{_coord(path_origin[0])} Y{_coord(path_origin[1])}\n"
            )
        f.write(_valve_cmd(valve, 0))
        for line in setpress_lines:
            f.write(f"{line}\n")
        for line in pressure_on_lines:
            f.write(f"{line}\n")
        for color in color_dict:
            f.write(_valve_cmd(color_dict[color], 0))

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
                if pressure_ramp_enabled:
                    pressure_cur += increase_pressure_per_layer
                    pressure_next = _setpress_cmd(com_port, pressure_cur, start=False)
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

        for color in color_dict:
            f.write(_valve_cmd(color_dict[color], 0))
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
    infill: float = 1.0,
    increase_pressure_per_layer: float = 0.1,
    pressure_ramp_enabled: bool = True,
    all_g1: bool = False,
    lead_in_enabled: bool = False,
    lead_in_length: float = 5.0,
    lead_in_clearance: float = 5.0,
    lead_in_lines: int = 3,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate G-code for one sliced shape.

    Without `motion`, the shape's own layers drive both the nozzle path and
    the valve. With `motion` (the combined reference stack), the nozzle
    follows the shared reference path while the valve opens only inside this
    shape's own geometry, aligned into the reference frame — so parallel
    heads share one motion but each dispenses only its own shape.
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
    )

    # World anchor: the toolpath origin expressed in the shape's own frame.
    # With reference motion the geometry was translated by the centering
    # delta, so subtract it to get back to the shape's coordinates.
    if motion is not None:
        delta_x, delta_y = _centering_delta(shape, motion)
    else:
        delta_x = delta_y = 0.0
    path_origin = (toolpath_origin[0] - delta_x, toolpath_origin[1] - delta_y)

    lead_in = _lead_in_moves(
        lead_in_enabled,
        lead_in_length,
        lead_in_clearance,
        lead_in_lines,
        fil_width,
        255,
        0,
    )
    if lead_in:
        gcode_list = [*lead_in, *gcode_list]

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="vector_gcode_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    gcode_path = output_dir / f"{shape_name}_SnakePath_gcode.txt"
    write_gcode_file(
        gcode_path,
        gcode_list,
        pressure=float(pressure),
        valve=int(valve),
        port=int(port),
        increase_pressure_per_layer=float(increase_pressure_per_layer),
        pressure_ramp_enabled=bool(pressure_ramp_enabled),
        all_g1=bool(all_g1),
        path_origin=path_origin,
    )
    return gcode_path
