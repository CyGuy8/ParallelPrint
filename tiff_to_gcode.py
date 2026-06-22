from __future__ import annotations

import os
import tempfile
import zipfile
from codecs import encode
from pathlib import Path
from textwrap import wrap

import numpy as np
from PIL import Image


RASTER_PATTERN_SAME_DIRECTION = "X-direction raster"
RASTER_PATTERN_Y_DIRECTION = "Y-direction raster"
RASTER_PATTERN_WOODPILE = "Woodpile raster"
RASTER_PATTERN_CHOICES = (
    RASTER_PATTERN_SAME_DIRECTION,
    RASTER_PATTERN_Y_DIRECTION,
    RASTER_PATTERN_WOODPILE,
)


def _normalize_raster_pattern(pattern: str | None) -> str:
    if pattern == RASTER_PATTERN_WOODPILE:
        return RASTER_PATTERN_WOODPILE
    if pattern == RASTER_PATTERN_Y_DIRECTION:
        return RASTER_PATTERN_Y_DIRECTION
    return RASTER_PATTERN_SAME_DIRECTION


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
    insert = "{preset}" if start else ""
    return f"\n\r{insert}{port}.write({_setpress(pressure)})"


def _toggle_cmd(port: str, start: bool) -> str:
    insert = "{preset}" if start else ""
    return f"\n\r{insert}{port}.write({_togglepress()})"


def _valve_cmd(valve: int, command: int) -> str:
    return f"\n{{aux_command}}WAGO_ValveCommands({valve}, {command})\n"


def _gcode_layer(
    path_img: np.ndarray,
    color_img: np.ndarray,
    output_list: list[dict],
    pixel_size: float,
    direction: int,
    layer_number: int,
) -> int:
    mask = path_img > 0
    first_nonblack = np.where(mask.any(axis=1), mask.argmax(axis=1), -1)
    last_nonblack = np.where(
        mask.any(axis=1),
        mask.shape[1] - 1 - np.fliplr(mask).argmax(axis=1),
        -1,
    )

    stored_gcode: list[dict] = []
    nonblank_rows = np.where(first_nonblack != -1)[0]

    for idx, i in enumerate(nonblank_rows):
        f_idx, l_idx = int(first_nonblack[i]), int(last_nonblack[i])
        if f_idx == -1:
            continue

        if direction < 0:
            rng = range(f_idx, l_idx + 1)
        else:
            rng = range(l_idx, f_idx - 1, -1)
        direction *= -1

        prev_color = None
        color_len = 0
        buffer = direction
        stored_gcode.append({"X": buffer * pixel_size, "Y": 0, "Color": 0})

        for j in rng:
            this_color = int(color_img[i, j])
            if prev_color is None:
                prev_color = this_color
                color_len = 1
            elif this_color == prev_color:
                color_len += 1
            else:
                stored_gcode.append(
                    {
                        "X": direction * color_len * pixel_size,
                        "Y": 0,
                        "Color": prev_color,
                    }
                )
                color_len = 1
                prev_color = this_color

        if color_len > 0:
            stored_gcode.append(
                {
                    "X": direction * color_len * pixel_size,
                    "Y": 0,
                    "Color": prev_color,
                }
            )

        stored_gcode.append({"X": buffer * pixel_size, "Y": 0, "Color": 0})

        curr_x = l_idx if direction > 0 else f_idx
        curr_x += buffer

        if idx + 1 < len(nonblank_rows):
            next_i = int(nonblank_rows[idx + 1])
            y_travel_dist = next_i - int(i)
            nf, nl = int(first_nonblack[next_i]), int(last_nonblack[next_i])
            if nf == -1:
                continue
            next_start = nf if direction < 0 else nl
            travel_x = (next_start + buffer) - curr_x
            y_dir = -1 if layer_number % 2 == 1 else 1
            stored_gcode.append(
                {
                    "X": travel_x * pixel_size,
                    "Y": y_travel_dist * pixel_size * y_dir,
                    "Color": 0,
                }
            )

    output_list.extend(stored_gcode)
    return direction


def _sort_key(filename: str) -> int:
    digits = "".join(filter(str.isdigit, filename))
    return int(digits) if digits else 2**31


def _extract_zip_tiffs(zip_path: Path, dest: Path) -> list[Path]:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)

    tiffs: list[Path] = []
    for root, _, files in os.walk(dest):
        for name in files:
            if name.lower().endswith((".tif", ".tiff")):
                tiffs.append(Path(root) / name)
    tiffs.sort(key=lambda p: _sort_key(p.name))
    return tiffs


def _load_grayscale(path: Path, invert: bool) -> np.ndarray:
    with Image.open(path) as image:
        array = np.array(image.convert("L"), dtype=np.uint8)
    if invert:
        array = 255 - array
    return array


def _center_on_canvas(
    img: np.ndarray, canvas_h: int, canvas_w: int, fill: int = 0
) -> np.ndarray:
    """Place `img` centred on a (canvas_h, canvas_w) canvas filled with `fill`.

    Mirrors the centring used to build the reference stack, so a shape's slice
    lines up pixel-for-pixel with the reference (motion) slice of the same layer.
    """
    h, w = img.shape[:2]
    out = np.full((canvas_h, canvas_w), fill, dtype=img.dtype)
    y_off = max(0, (canvas_h - h) // 2)
    x_off = max(0, (canvas_w - w) // 2)
    out[y_off : y_off + h, x_off : x_off + w] = img[: canvas_h, : canvas_w]
    return out


def _append_relative_move(
    output_list: list[dict],
    current_x: float,
    current_y: float,
    target_x: float,
    target_y: float,
    color: int,
    z_step: float | None = None,
) -> tuple[float, float]:
    dx = target_x - current_x
    dy = target_y - current_y
    if dx == 0 and dy == 0 and z_step is None:
        return current_x, current_y
    move = {"X": dx, "Y": dy, "Color": color}
    if z_step is not None:
        move["Z"] = z_step
    output_list.append(move)
    return target_x, target_y


def _woodpile_layer_segments(
    path_img: np.ndarray,
    color_img: np.ndarray,
    pixel_size: float,
    raster_axis: str,
) -> list[tuple[float, float, float, float, int]]:
    mask = path_img > 0
    segments: list[tuple[float, float, float, float, int]] = []

    if raster_axis == "Y":
        first_nonblank = np.where(mask.any(axis=0), mask.argmax(axis=0), -1)
        last_nonblank = np.where(
            mask.any(axis=0),
            mask.shape[0] - 1 - np.flipud(mask).argmax(axis=0),
            -1,
        )
        for col_number, col in enumerate(np.where(first_nonblank != -1)[0]):
            f_idx, l_idx = int(first_nonblank[col]), int(last_nonblank[col])
            if f_idx == -1:
                continue
            forward = col_number % 2 == 0
            row_values = list(range(f_idx, l_idx + 1)) if forward else list(range(l_idx, f_idx - 1, -1))
            run_start = row_values[0]
            prev_row = row_values[0]
            prev_color = int(color_img[prev_row, col])
            x = (int(col) + 0.5) * pixel_size
            for row in row_values[1:]:
                this_color = int(color_img[row, col])
                if this_color == prev_color:
                    prev_row = row
                    continue
                if forward:
                    start_y = run_start * pixel_size
                    end_y = (prev_row + 1) * pixel_size
                else:
                    start_y = (run_start + 1) * pixel_size
                    end_y = prev_row * pixel_size
                segments.append((x, start_y, x, end_y, prev_color))
                run_start = prev_row = row
                prev_color = this_color
            if forward:
                start_y = run_start * pixel_size
                end_y = (prev_row + 1) * pixel_size
            else:
                start_y = (run_start + 1) * pixel_size
                end_y = prev_row * pixel_size
            segments.append((x, start_y, x, end_y, prev_color))
        return segments

    first_nonblank = np.where(mask.any(axis=1), mask.argmax(axis=1), -1)
    last_nonblank = np.where(
        mask.any(axis=1),
        mask.shape[1] - 1 - np.fliplr(mask).argmax(axis=1),
        -1,
    )
    for row_number, row in enumerate(np.where(first_nonblank != -1)[0]):
        f_idx, l_idx = int(first_nonblank[row]), int(last_nonblank[row])
        if f_idx == -1:
            continue
        forward = row_number % 2 == 0
        col_values = list(range(f_idx, l_idx + 1)) if forward else list(range(l_idx, f_idx - 1, -1))
        run_start = col_values[0]
        prev_col = col_values[0]
        prev_color = int(color_img[row, prev_col])
        y = (int(row) + 0.5) * pixel_size
        for col in col_values[1:]:
            this_color = int(color_img[row, col])
            if this_color == prev_color:
                prev_col = col
                continue
            if forward:
                start_x = run_start * pixel_size
                end_x = (prev_col + 1) * pixel_size
            else:
                start_x = (run_start + 1) * pixel_size
                end_x = prev_col * pixel_size
            segments.append((start_x, y, end_x, y, prev_color))
            run_start = prev_col = col
            prev_color = this_color
        if forward:
            start_x = run_start * pixel_size
            end_x = (prev_col + 1) * pixel_size
        else:
            start_x = (run_start + 1) * pixel_size
            end_x = prev_col * pixel_size
        segments.append((start_x, y, end_x, y, prev_color))
    return segments


def _raster_axis_for_pattern(pattern: str, layer_number: int) -> str:
    if pattern == RASTER_PATTERN_Y_DIRECTION:
        return "Y"
    if pattern == RASTER_PATTERN_WOODPILE and layer_number % 2 == 1:
        return "Y"
    return "X"


def _build_footprint_raster_gcode_list(
    path_ref_list: list[np.ndarray],
    color_ref_list: list[np.ndarray],
    pixel_size: float,
    layer_height: float,
    raster_pattern: str,
) -> list[dict]:
    gcode_list: list[dict] = []
    current_x = 0.0
    current_y = 0.0

    for layer_number, (path_img, color_img) in enumerate(zip(path_ref_list, color_ref_list)):
        raster_axis = _raster_axis_for_pattern(raster_pattern, layer_number)
        segments = _woodpile_layer_segments(path_img, color_img, pixel_size, raster_axis)
        if not segments:
            if layer_number > 0:
                gcode_list.append({"X": 0.0, "Y": 0.0, "Z": layer_height, "Color": 0})
            continue

        first_x, first_y = segments[0][0], segments[0][1]
        if layer_number > 0:
            current_x, current_y = _append_relative_move(
                gcode_list,
                current_x,
                current_y,
                first_x,
                first_y,
                0,
                z_step=layer_height,
            )
        else:
            current_x, current_y = _append_relative_move(
                gcode_list,
                current_x,
                current_y,
                first_x,
                first_y,
                0,
            )

        for start_x, start_y, end_x, end_y, color in segments:
            current_x, current_y = _append_relative_move(
                gcode_list,
                current_x,
                current_y,
                start_x,
                start_y,
                0,
            )
            current_x, current_y = _append_relative_move(
                gcode_list,
                current_x,
                current_y,
                end_x,
                end_y,
                color,
            )

    return gcode_list


def generate_snake_path_gcode(
    zip_path: str | Path,
    shape_name: str,
    pressure: float,
    valve: int,
    port: int,
    layer_height: float = 0.8,
    fil_width: float = 0.8,
    invert: bool = True,
    increase_pressure_per_layer: float = 0.1,
    all_g1: bool = False,
    motion_tiffs: list[str] | None = None,
    raster_pattern: str | None = RASTER_PATTERN_SAME_DIRECTION,
) -> Path:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    raster_pattern = _normalize_raster_pattern(raster_pattern)

    work_dir = Path(tempfile.mkdtemp(prefix="tiff_gcode_"))
    extract_dir = work_dir / "tiffs"
    extract_dir.mkdir(parents=True, exist_ok=True)
    tiff_files = _extract_zip_tiffs(zip_path, extract_dir)
    if not tiff_files:
        raise ValueError("No TIFF files found in the ZIP archive.")

    off_color = 0
    com_port = f"serialPort{port}"
    color_dict: dict[int, int] = {0: 100, 255: valve}

    # Two non-flipped source image lists. The "path" images drive the nozzle
    # motion (which rows are swept, the sweep extent, the inter-layer shifts);
    # the "color" images decide the valve state (material) at each swept pixel.
    # Normally both are this shape's own slices. When reference motion tiffs are
    # supplied, motion comes from the combined reference stack while the valve is
    # still driven by this shape's slices, centred onto the reference canvas — so
    # parallel heads share one motion path but each dispenses only its geometry.
    shape_imgs = [_load_grayscale(p, invert=invert) for p in tiff_files]

    if motion_tiffs:
        motion_paths = sorted(
            (Path(p) for p in motion_tiffs), key=lambda p: _sort_key(p.name)
        )
        path_ref_list = [_load_grayscale(p, invert=invert) for p in motion_paths]
        if not path_ref_list:
            raise ValueError("No reference TIFF files provided for motion.")
        color_ref_list: list[np.ndarray] = []
        for li, motion_img in enumerate(path_ref_list):
            h_c, w_c = motion_img.shape[:2]
            if li < len(shape_imgs):
                color_ref_list.append(
                    _center_on_canvas(shape_imgs[li], h_c, w_c, fill=off_color)
                )
            else:
                # Reference is taller than this shape: move but dispense nothing.
                color_ref_list.append(np.full((h_c, w_c), off_color, dtype=np.uint8))
    else:
        path_ref_list = [im.copy() for im in shape_imgs]
        color_ref_list = [im.copy() for im in shape_imgs]

    setpress_lines = [_setpress_cmd(com_port, pressure, start=True)]
    pressure_on_lines = [_toggle_cmd(com_port, start=True)]
    pressure_off_lines = [_toggle_cmd(com_port, start=False)]

    if raster_pattern in (RASTER_PATTERN_Y_DIRECTION, RASTER_PATTERN_WOODPILE):
        gcode_list = _build_footprint_raster_gcode_list(
            path_ref_list,
            color_ref_list,
            fil_width,
            layer_height,
            raster_pattern,
        )
    else:
        gcode_list: list[dict] = []
        dist_sign_long = 1
        current_offsets_x: list[int] = []
        use_flip_y = False
        direction = -1

        for layers in range(len(path_ref_list)):
            current_image_ref = path_ref_list[layers]
            last_image_ref = path_ref_list[layers - 1] if layers > 0 else None
            y_ref = current_image_ref.shape[0]

            def find_first_valid_y(row: np.ndarray | None, flip: bool = False) -> int | None:
                if row is None:
                    return None
                row_data = np.flip(row) if flip else row
                for j, pixel in enumerate(row_data):
                    if np.any(pixel) != off_color:
                        return y_ref - 1 - j if flip else j
                return None

            last_x = last_y = None
            if current_offsets_x:
                use_flip_x = layers % 2 == 1
                last_x = current_offsets_x[-1] if use_flip_x else current_offsets_x[0]
                last_row = (
                    last_image_ref[last_x] if last_image_ref is not None else None
                )
                last_y = find_first_valid_y(last_row, flip=use_flip_y)
                current_offsets_x.clear()

            current_offsets_x = [
                i for i, row in enumerate(current_image_ref) if np.any(row) != off_color
            ]

            first_x = first_y = None
            if current_offsets_x:
                use_flip_x = layers % 2 == 1
                first_x = current_offsets_x[-1] if use_flip_x else current_offsets_x[0]
                first_row = current_image_ref[first_x]
                first_y = find_first_valid_y(first_row, flip=use_flip_y)

            if None in (last_x, last_y, first_x, first_y):
                shift_x = shift_y = 0
            else:
                shift_x = (first_x - last_x) * fil_width
                shift_y = (first_y - last_y) * fil_width * dist_sign_long
                if use_flip_y:
                    shift_y = -shift_y

            if len(current_offsets_x) % 2 == 1:
                use_flip_y = not use_flip_y

            if layers > 0:
                gcode_list.append(
                    {"X": shift_y, "Y": shift_x, "Z": layer_height, "Color": 0}
                )

            for row in current_image_ref:
                if all(p == off_color for p in row):
                    dist_sign_long = -dist_sign_long
                dist_sign_long = -dist_sign_long

            # Flip path and color together on even layers so they stay aligned.
            even_layer = (layers + 1) % 2 == 0
            ref_for_path = (
                np.flipud(current_image_ref) if even_layer else current_image_ref.copy()
            )
            current_image = (
                np.flipud(color_ref_list[layers]) if even_layer else color_ref_list[layers]
            )

            if layers == 0:
                direction = -1
            direction = _gcode_layer(
                ref_for_path,
                current_image,
                gcode_list,
                fil_width,
                direction,
                layers,
            )

    gcode_path = work_dir / f"{shape_name}_SnakePath_gcode.txt"
    pressure_cur = float(pressure)

    with open(gcode_path, "w") as f:
        f.write("G91\n")
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
                    f"{move_type} X{move['X']} Y{move['Y']} Z{move['Z']} "
                    f"; Color {move['Color']}"
                )
                pressure_cur += increase_pressure_per_layer
                pressure_next = _setpress_cmd(com_port, pressure_cur, start=False)
            else:
                line = (
                    f"{move_type} X{move['X']} Y{move['Y']} ; Color {move['Color']}"
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

    return gcode_path
