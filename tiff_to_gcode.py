from __future__ import annotations

import math
import os
import tempfile
import zipfile
from collections import defaultdict
from codecs import encode
from pathlib import Path
from textwrap import wrap

import numpy as np
from PIL import Image


RASTER_PATTERN_SAME_DIRECTION = "X-direction raster"
RASTER_PATTERN_Y_DIRECTION = "Y-direction raster"
RASTER_PATTERN_WOODPILE = "Woodpile raster"
RASTER_PATTERN_RECTANGULAR_SPIRAL = "Rectangular Spiral raster"
RASTER_PATTERN_CIRCLE_SPIRAL = "Circle Spiral raster"
RASTER_PATTERN_CHOICES = (
    RASTER_PATTERN_SAME_DIRECTION,
    RASTER_PATTERN_Y_DIRECTION,
    RASTER_PATTERN_WOODPILE,
    RASTER_PATTERN_RECTANGULAR_SPIRAL,
    RASTER_PATTERN_CIRCLE_SPIRAL,
)
CONTOUR_MODE_EXACT = "Exact pixel border"
CONTOUR_MODE_ROW_ENVELOPE = "Shape-optimized row envelope"
CONTOUR_MODE_CHOICES = (
    CONTOUR_MODE_EXACT,
    CONTOUR_MODE_ROW_ENVELOPE,
)

def _normalize_raster_pattern(pattern: str | None) -> str:
    if pattern == RASTER_PATTERN_CIRCLE_SPIRAL:
        return RASTER_PATTERN_CIRCLE_SPIRAL
    if pattern == RASTER_PATTERN_RECTANGULAR_SPIRAL:
        return RASTER_PATTERN_RECTANGULAR_SPIRAL
    if pattern == RASTER_PATTERN_WOODPILE:
        return RASTER_PATTERN_WOODPILE
    if pattern == RASTER_PATTERN_Y_DIRECTION:
        return RASTER_PATTERN_Y_DIRECTION
    return RASTER_PATTERN_SAME_DIRECTION


def _normalize_contour_mode(mode: str | None) -> str:
    if mode == CONTOUR_MODE_ROW_ENVELOPE:
        return CONTOUR_MODE_ROW_ENVELOPE
    return CONTOUR_MODE_EXACT


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


def _lead_in_moves(
    enabled: bool,
    length: float,
    clearance: float,
    line_count: int,
    line_spacing: float,
    print_color: int,
    off_color: int,
) -> list[dict]:
    if not enabled:
        return []
    lead_length = max(0.0, float(length))
    if lead_length <= 0.0:
        return []
    lead_clearance = max(0.0, float(clearance))
    pass_count = max(1, int(line_count))
    spacing = max(0.0, float(line_spacing))

    moves: list[dict] = []
    current_x = 0.0
    current_y = 0.0

    def append_move(dx: float, dy: float, color: int) -> None:
        nonlocal current_x, current_y
        if dx == 0.0 and dy == 0.0:
            return
        moves.append({"X": dx, "Y": dy, "Color": color})
        current_x += dx
        current_y += dy

    append_move(-(lead_clearance + lead_length), 0.0, off_color)
    direction = 1.0
    for pass_index in range(pass_count):
        append_move(direction * lead_length, 0.0, print_color)
        direction *= -1.0
        if pass_index < pass_count - 1:
            append_move(0.0, spacing, off_color)
    append_move(-current_x, -current_y, off_color)
    return moves


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


def _simplify_closed_contour(
    points: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if len(points) < 4:
        return points
    if points[0] != points[-1]:
        points = [*points, points[0]]

    ring = points[:-1]
    simplified: list[tuple[int, int]] = []
    for idx, point in enumerate(ring):
        prev_point = ring[idx - 1]
        next_point = ring[(idx + 1) % len(ring)]
        dx1 = point[0] - prev_point[0]
        dy1 = point[1] - prev_point[1]
        dx2 = next_point[0] - point[0]
        dy2 = next_point[1] - point[1]
        if dx1 * dy2 == dy1 * dx2:
            continue
        simplified.append(point)

    if len(simplified) < 3:
        simplified = ring
    simplified.append(simplified[0])
    return simplified


def _contour_area2(points: list[tuple[float, float]]) -> float:
    return sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:])
    )


def _contour_sort_key(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (-abs(_contour_area2(points)), min(ys), min(xs))


def _trace_mask_contours(
    mask: np.ndarray,
    pixel_size: float,
    x_offset_px: float = 0.0,
    y_offset_px: float = 0.0,
) -> list[list[tuple[float, float]]]:
    if not np.any(mask):
        return []

    mask = mask.astype(bool)
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    height, width = mask.shape

    def is_on(row: int, col: int) -> bool:
        return 0 <= row < height and 0 <= col < width and bool(mask[row, col])

    for row in range(height):
        for col in range(width):
            if not mask[row, col]:
                continue
            if not is_on(row - 1, col):
                segments.append(((col, row), (col + 1, row)))
            if not is_on(row, col + 1):
                segments.append(((col + 1, row), (col + 1, row + 1)))
            if not is_on(row + 1, col):
                segments.append(((col + 1, row + 1), (col, row + 1)))
            if not is_on(row, col - 1):
                segments.append(((col, row + 1), (col, row)))

    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for start, end in segments:
        outgoing[start].append(end)

    contours: list[list[tuple[float, float]]] = []
    remaining = set(segments)
    while remaining:
        start, end = min(remaining)
        remaining.remove((start, end))
        contour = [start, end]
        current = end

        while current != start:
            next_point = next(
                (
                    candidate
                    for candidate in outgoing.get(current, [])
                    if (current, candidate) in remaining
                ),
                None,
            )
            if next_point is None:
                break
            remaining.remove((current, next_point))
            current = next_point
            contour.append(current)

        if len(contour) > 3 and contour[-1] == contour[0]:
            simplified = _simplify_closed_contour(contour)
            contours.append(
                [
                    ((x + x_offset_px) * pixel_size, (y + y_offset_px) * pixel_size)
                    for x, y in simplified
                ]
            )

    contours.sort(key=_contour_sort_key)
    return contours


def _trace_row_envelope_contours(
    mask: np.ndarray,
    pixel_size: float,
    x_offset_px: float = 0.0,
    y_offset_px: float = 0.0,
) -> list[list[tuple[float, float]]]:
    """Trace one outside contour using each active row's left/right extent."""
    if not np.any(mask):
        return []

    mask = mask.astype(bool)
    first = np.where(mask.any(axis=1), mask.argmax(axis=1), -1)
    last = np.where(
        mask.any(axis=1),
        mask.shape[1] - 1 - np.fliplr(mask).argmax(axis=1),
        -1,
    )
    rows = np.where(first != -1)[0]
    if len(rows) == 0:
        return []

    points: list[tuple[float, float]] = []

    for row in rows:
        points.append((float(first[row] - 1), float(row)))

    bottom = int(rows[-1])
    for col in range(int(first[bottom]) - 1, int(last[bottom]) + 1):
        points.append((float(col), float(bottom) + 0.5))

    for row in rows[::-1]:
        points.append((float(last[row]), float(row)))

    top = int(rows[0])
    for col in range(int(last[top]), int(first[top]) - 2, -1):
        points.append((float(col), float(top) - 0.5))

    return [
        [
            ((x + x_offset_px) * pixel_size, (y + y_offset_px) * pixel_size)
            for x, y in points
        ]
    ]


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


def _point_distance_sq(
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    return (ax - bx) ** 2 + (ay - by) ** 2


def _closest_point_on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float, float]:
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return ax, ay, 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return ax + t * dx, ay + t * dy, t


def _rotate_closed_contour_to_nearest_border(
    contour: list[tuple[float, float]],
    current_x: float,
    current_y: float,
    approach_dx: float = 0.0,
    approach_dy: float = 0.0,
) -> list[tuple[float, float]]:
    if len(contour) < 3:
        return contour

    ring = contour[:-1] if contour[0] == contour[-1] else contour
    if len(ring) < 2:
        return contour

    best_idx = 0
    best_t = 0.0
    best_point = ring[0]
    best_dist = float("inf")
    for idx, (ax, ay) in enumerate(ring):
        bx, by = ring[(idx + 1) % len(ring)]
        point_x, point_y, t = _closest_point_on_segment(
            current_x,
            current_y,
            ax,
            ay,
            bx,
            by,
        )
        distance = _point_distance_sq(current_x, current_y, point_x, point_y)
        if distance < best_dist:
            best_dist = distance
            best_idx = idx
            best_t = t
            best_point = (point_x, point_y)

    eps = 1e-9
    def choose_direction(
        forward: list[tuple[float, float]],
        reverse: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if (
            len(forward) < 2
            or len(reverse) < 2
            or (approach_dx == 0 and approach_dy == 0)
        ):
            return forward

        def score(candidate: list[tuple[float, float]]) -> float:
            tx = candidate[1][0] - candidate[0][0]
            ty = candidate[1][1] - candidate[0][1]
            # Positive when the shape interior, opposite the approach vector,
            # is to the left of the contour's first move.
            return (ty * approach_dx) - (tx * approach_dy)

        return reverse if score(reverse) > score(forward) else forward

    if best_t <= eps:
        forward = ring[best_idx:] + ring[:best_idx] + [ring[best_idx]]
        reverse_ring = list(reversed(ring))
        reverse_idx = reverse_ring.index(ring[best_idx])
        reverse = (
            reverse_ring[reverse_idx:]
            + reverse_ring[:reverse_idx]
            + [reverse_ring[reverse_idx]]
        )
        return choose_direction(forward, reverse)

    next_idx = (best_idx + 1) % len(ring)
    if best_t >= 1.0 - eps:
        forward = ring[next_idx:] + ring[:next_idx] + [ring[next_idx]]
        reverse_ring = list(reversed(ring))
        reverse_idx = reverse_ring.index(ring[next_idx])
        reverse = (
            reverse_ring[reverse_idx:]
            + reverse_ring[:reverse_idx]
            + [reverse_ring[reverse_idx]]
        )
        return choose_direction(forward, reverse)

    forward = [best_point]
    for step in range(1, len(ring) + 1):
        forward.append(ring[(best_idx + step) % len(ring)])
    forward.append(best_point)

    reverse = [best_point]
    for step in range(0, len(ring)):
        reverse.append(ring[(best_idx - step) % len(ring)])
    reverse.append(best_point)
    return choose_direction(forward, reverse)


def _last_print_reference(output_list: list[dict]) -> tuple[float, float, float, float]:
    x = y = 0.0
    last_x = last_y = 0.0
    last_dx = last_dy = 0.0
    for move in output_list:
        dx = float(move.get("X", 0.0))
        dy = float(move.get("Y", 0.0))
        x += dx
        y += dy
        if move.get("Color") == 255 and "Z" not in move:
            last_x = x
            last_y = y
            last_dx = dx
            last_dy = dy
    return last_x, last_y, last_dx, last_dy


def _rewind_trailing_travel(
    output_list: list[dict],
    current_x: float,
    current_y: float,
) -> tuple[float, float]:
    if not output_list:
        return current_x, current_y

    last_move = output_list[-1]
    if last_move.get("Color") != 0 or "Z" in last_move:
        return current_x, current_y

    has_layer_print = any(
        move.get("Color") == 255 and "Z" not in move
        for move in reversed(output_list[:-1])
    )
    if not has_layer_print:
        return current_x, current_y

    output_list.pop()
    return (
        current_x - float(last_move.get("X", 0.0)),
        current_y - float(last_move.get("Y", 0.0)),
    )


def _contour_source_paths(source: dict, extract_root: Path, source_pos: int) -> list[Path]:
    paths = source.get("tiff_paths") or source.get("paths") or []
    if paths:
        return sorted((Path(path) for path in paths), key=lambda path: _sort_key(path.name))

    zip_path = source.get("zip_path")
    if not zip_path:
        return []
    extract_dir = extract_root / f"source_{source_pos:03d}"
    extract_dir.mkdir(parents=True, exist_ok=True)
    return _extract_zip_tiffs(Path(zip_path), extract_dir)


def _active_pixel_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _first_raster_row_start(mask: np.ndarray) -> tuple[int, int] | None:
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return None
    row = int(rows[0])
    cols = np.where(mask[row])[0]
    if len(cols) == 0:
        return None
    return row, int(cols[0])


def _contour_pixel_offsets(
    raster_pattern: str,
    motion_mask: np.ndarray,
    contour_mask: np.ndarray,
    contour_mode: str = CONTOUR_MODE_EXACT,
) -> tuple[float, float]:
    if raster_pattern == RASTER_PATTERN_SAME_DIRECTION:
        first_row_start = _first_raster_row_start(motion_mask)
        if first_row_start is None:
            first_row_start = _first_raster_row_start(contour_mask)
        if first_row_start is None:
            return 0.0, 0.0
        first_row, first_col = first_row_start
        # The legacy X-raster infill path anchors every row to the first
        # rastered row's starting pixel. Wider lower rows can extend left of
        # that point, so using the layer-wide min column shifts pyramid-like
        # contours away from the infill.
        x_offset, y_offset = 1.0 - first_col, -0.5 - first_row
    else:
        x_offset, y_offset = 0.0, 0.0

    if contour_mode == CONTOUR_MODE_ROW_ENVELOPE:
        x_offset += 1.0
        y_offset += 0.5
    return x_offset, y_offset


def _build_contour_layers(
    contour_tiff_sets: list[dict] | None,
    path_ref_list: list[np.ndarray],
    pixel_size: float,
    invert: bool,
    off_color: int,
    work_dir: Path,
    raster_pattern: str,
    layer_start_directions: list[int] | None = None,
) -> list[list[dict]]:
    contour_layers: list[list[dict]] = [[] for _ in path_ref_list]
    if not contour_tiff_sets:
        return contour_layers

    extract_root = work_dir / "contour_sources"
    for source_pos, source in enumerate(contour_tiff_sets):
        try:
            owner_idx = int(source.get("owner_idx", source.get("idx", source_pos + 1)))
        except (TypeError, ValueError):
            owner_idx = source_pos + 1

        contour_mode = _normalize_contour_mode(
            source.get("contour_mode") or source.get("mode")
        )
        source_paths = _contour_source_paths(source, extract_root, source_pos)
        for layer_number, path_img in enumerate(path_ref_list):
            if layer_number >= len(source_paths):
                continue
            canvas_h, canvas_w = path_img.shape[:2]
            contour_img = _load_grayscale(source_paths[layer_number], invert=invert)
            if contour_img.shape[:2] != (canvas_h, canvas_w):
                contour_img = _center_on_canvas(
                    contour_img,
                    canvas_h,
                    canvas_w,
                    fill=off_color,
                )
            contour_path_img = path_img
            if raster_pattern == RASTER_PATTERN_SAME_DIRECTION and layer_number % 2 == 1:
                contour_path_img = np.flipud(contour_path_img)
                contour_img = np.flipud(contour_img)
            if (
                raster_pattern == RASTER_PATTERN_SAME_DIRECTION
                and layer_start_directions is not None
                and layer_number < len(layer_start_directions)
                and layer_start_directions[layer_number] > 0
            ):
                contour_path_img = np.fliplr(contour_path_img)
                contour_img = np.fliplr(contour_img)

            contour_mask = contour_img > off_color
            x_offset_px, y_offset_px = _contour_pixel_offsets(
                raster_pattern,
                contour_path_img > off_color,
                contour_mask,
                contour_mode,
            )
            if contour_mode == CONTOUR_MODE_ROW_ENVELOPE:
                contours = _trace_row_envelope_contours(
                    contour_mask,
                    pixel_size,
                    x_offset_px=x_offset_px,
                    y_offset_px=y_offset_px,
                )
            else:
                contours = _trace_mask_contours(
                    contour_mask,
                    pixel_size,
                    x_offset_px=x_offset_px,
                    y_offset_px=y_offset_px,
                )
            if contours:
                contour_layers[layer_number].append(
                    {
                        "owner_idx": owner_idx,
                        "contour_mode": contour_mode,
                        "contours": contours,
                    }
                )

    return contour_layers


def _same_direction_layer_start_directions(
    path_ref_list: list[np.ndarray],
    off_color: int,
) -> list[int]:
    directions: list[int] = []
    direction = -1
    for path_img in path_ref_list:
        directions.append(direction)
        nonblank_rows = int(np.count_nonzero((path_img > off_color).any(axis=1)))
        if nonblank_rows % 2 == 1:
            direction *= -1
    return directions


def _append_layer_contours(
    output_list: list[dict],
    current_x: float,
    current_y: float,
    contour_layers: list[list[dict]],
    layer_number: int,
    active_owner_idx: int | None,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
) -> tuple[float, float]:
    if layer_number >= len(contour_layers):
        return current_x, current_y

    active_sources = [
        source
        for source in contour_layers[layer_number]
        if source.get("owner_idx") == active_owner_idx
        and any(len(contour) >= 2 for contour in source.get("contours", []))
    ]
    if not active_sources:
        return current_x, current_y

    current_x, current_y = _rewind_trailing_travel(
        output_list,
        current_x,
        current_y,
    )

    use_infill_reference = True

    for source in active_sources:
        color = 255
        for contour in source.get("contours", []):
            if len(contour) < 2:
                continue
            contour = [
                (origin_x + (x_scale * x), origin_y + (y_scale * y))
                for x, y in contour
            ]
            if use_infill_reference:
                nearest_x, nearest_y, approach_dx, approach_dy = _last_print_reference(
                    output_list
                )
            else:
                nearest_x, nearest_y = current_x, current_y
                approach_dx, approach_dy = 0.0, 0.0
            contour = _rotate_closed_contour_to_nearest_border(
                contour,
                nearest_x,
                nearest_y,
                approach_dx,
                approach_dy,
            )
            if contour[0] != contour[-1]:
                contour = [*contour, contour[0]]
            start_x, start_y = contour[0]
            use_infill_reference = False
            current_x, current_y = _append_relative_move(
                output_list,
                current_x,
                current_y,
                start_x,
                start_y,
                0,
            )
            for target_x, target_y in contour[1:]:
                current_x, current_y = _append_relative_move(
                    output_list,
                    current_x,
                    current_y,
                    target_x,
                    target_y,
                    color,
                )

    return current_x, current_y


def _woodpile_layer_segments(
    path_img: np.ndarray,
    color_img: np.ndarray,
    pixel_size: float,
    raster_axis: str,
    reverse_order: bool = False,
    start_forward: bool = True,
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
        columns = list(np.where(first_nonblank != -1)[0])
        if reverse_order:
            columns.reverse()
        for col_number, col in enumerate(columns):
            f_idx, l_idx = int(first_nonblank[col]), int(last_nonblank[col])
            if f_idx == -1:
                continue
            forward = (col_number % 2 == 0) == start_forward
            row_values = list(range(f_idx, l_idx + 1)) if forward else list(range(l_idx, f_idx - 1, -1))
            run_start = row_values[0]
            prev_row = row_values[0]
            prev_color = int(color_img[prev_row, col])
            x = (int(col) + 0.5) * pixel_size
            if forward:
                sweep_start_y = f_idx * pixel_size
                segments.append(
                    (x, sweep_start_y - pixel_size, x, sweep_start_y, 0)
                )
            else:
                sweep_start_y = (l_idx + 1) * pixel_size
                segments.append(
                    (x, sweep_start_y + pixel_size, x, sweep_start_y, 0)
                )
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
            if forward:
                segments.append((x, end_y, x, end_y + pixel_size, 0))
            else:
                segments.append((x, end_y, x, end_y - pixel_size, 0))
        return segments

    first_nonblank = np.where(mask.any(axis=1), mask.argmax(axis=1), -1)
    last_nonblank = np.where(
        mask.any(axis=1),
        mask.shape[1] - 1 - np.fliplr(mask).argmax(axis=1),
        -1,
    )
    rows = list(np.where(first_nonblank != -1)[0])
    if reverse_order:
        rows.reverse()
    for row_number, row in enumerate(rows):
        f_idx, l_idx = int(first_nonblank[row]), int(last_nonblank[row])
        if f_idx == -1:
            continue
        forward = (row_number % 2 == 0) == start_forward
        col_values = list(range(f_idx, l_idx + 1)) if forward else list(range(l_idx, f_idx - 1, -1))
        run_start = col_values[0]
        prev_col = col_values[0]
        prev_color = int(color_img[row, prev_col])
        y = (int(row) + 0.5) * pixel_size
        if forward:
            sweep_start_x = f_idx * pixel_size
            segments.append((sweep_start_x - pixel_size, y, sweep_start_x, y, 0))
        else:
            sweep_start_x = (l_idx + 1) * pixel_size
            segments.append((sweep_start_x + pixel_size, y, sweep_start_x, y, 0))
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
        if forward:
            segments.append((end_x, y, end_x + pixel_size, y, 0))
        else:
            segments.append((end_x, y, end_x - pixel_size, y, 0))
    return segments


def _raster_axis_for_pattern(pattern: str, layer_number: int) -> str:
    if pattern == RASTER_PATTERN_Y_DIRECTION:
        return "Y"
    if pattern == RASTER_PATTERN_WOODPILE and layer_number % 2 == 1:
        return "Y"
    return "X"


def _oriented_woodpile_layer_segments(
    path_img: np.ndarray,
    color_img: np.ndarray,
    pixel_size: float,
    raster_axis: str,
    current_x: float,
    current_y: float,
    prefer_default: bool = False,
) -> list[tuple[float, float, float, float, int]]:
    default_segments = _woodpile_layer_segments(
        path_img,
        color_img,
        pixel_size,
        raster_axis,
    )
    if prefer_default or not default_segments:
        return default_segments

    candidates: list[list[tuple[float, float, float, float, int]]] = [default_segments]
    for reverse_order in (False, True):
        for start_forward in (False, True):
            segments = _woodpile_layer_segments(
                path_img,
                color_img,
                pixel_size,
                raster_axis,
                reverse_order=reverse_order,
                start_forward=start_forward,
            )
            if segments and segments not in candidates:
                candidates.append(segments)

    return min(
        candidates,
        key=lambda segments: _point_distance_sq(
            current_x,
            current_y,
            segments[0][0],
            segments[0][1],
        ),
    )


def _rectangular_spiral_positions(
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            positions.append((top, col))
        top += 1

        for row in range(top, bottom + 1):
            positions.append((row, right))
        right -= 1

        if top <= bottom:
            for col in range(right, left - 1, -1):
                positions.append((bottom, col))
            bottom -= 1

        if left <= right:
            for row in range(bottom, top - 1, -1):
                positions.append((row, left))
            left += 1

    return positions


def _append_colored_segment(
    segments: list[tuple[float, float, float, float, int]],
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    color: int,
) -> None:
    if start_x == end_x and start_y == end_y:
        return

    if segments:
        prev_start_x, prev_start_y, prev_end_x, prev_end_y, prev_color = segments[-1]
        if (
            prev_color == color
            and prev_end_x == start_x
            and prev_end_y == start_y
        ):
            prev_dx = prev_end_x - prev_start_x
            prev_dy = prev_end_y - prev_start_y
            next_dx = end_x - start_x
            next_dy = end_y - start_y
            if abs((prev_dx * next_dy) - (prev_dy * next_dx)) < 1e-9:
                segments[-1] = (
                    prev_start_x,
                    prev_start_y,
                    end_x,
                    end_y,
                    color,
                )
                return

    segments.append((start_x, start_y, end_x, end_y, color))


def _rectangular_spiral_layer_segments(
    path_img: np.ndarray,
    color_img: np.ndarray,
    pixel_size: float,
    reverse: bool = False,
) -> list[tuple[float, float, float, float, int]]:
    bounds = _active_pixel_bounds(path_img > 0)
    if bounds is None:
        return []

    top, bottom, left, right = bounds
    positions = _rectangular_spiral_positions(top, bottom, left, right)
    if reverse:
        positions.reverse()
    if not positions:
        return []

    def center(row: int, col: int) -> tuple[float, float]:
        return (float(col) + 0.5) * pixel_size, (float(row) + 0.5) * pixel_size

    def pixel_color(row: int, col: int) -> int:
        return int(color_img[row, col])

    segments: list[tuple[float, float, float, float, int]] = []
    if len(positions) == 1:
        row, col = positions[0]
        y = (float(row) + 0.5) * pixel_size
        _append_colored_segment(
            segments,
            float(col) * pixel_size,
            y,
            (float(col) + 1.0) * pixel_size,
            y,
            pixel_color(row, col),
        )
        return segments

    centers = [center(row, col) for row, col in positions]
    first_dx = centers[1][0] - centers[0][0]
    first_dy = centers[1][1] - centers[0][1]
    last_dx = centers[-1][0] - centers[-2][0]
    last_dy = centers[-1][1] - centers[-2][1]

    start_x = centers[0][0] - (first_dx * 0.5)
    start_y = centers[0][1] - (first_dy * 0.5)
    first_row, first_col = positions[0]
    _append_colored_segment(
        segments,
        start_x,
        start_y,
        centers[0][0],
        centers[0][1],
        pixel_color(first_row, first_col),
    )

    for idx, ((row, col), (next_row, next_col)) in enumerate(
        zip(positions, positions[1:])
    ):
        start_center_x, start_center_y = centers[idx]
        end_center_x, end_center_y = centers[idx + 1]
        mid_x = (start_center_x + end_center_x) / 2.0
        mid_y = (start_center_y + end_center_y) / 2.0
        _append_colored_segment(
            segments,
            start_center_x,
            start_center_y,
            mid_x,
            mid_y,
            pixel_color(row, col),
        )
        _append_colored_segment(
            segments,
            mid_x,
            mid_y,
            end_center_x,
            end_center_y,
            pixel_color(next_row, next_col),
        )

    last_row, last_col = positions[-1]
    end_x = centers[-1][0] + (last_dx * 0.5)
    end_y = centers[-1][1] + (last_dy * 0.5)
    _append_colored_segment(
        segments,
        centers[-1][0],
        centers[-1][1],
        end_x,
        end_y,
        pixel_color(last_row, last_col),
    )
    return segments


def _circle_spiral_points(
    center_x: float,
    center_y: float,
    outer_radius: float,
    pitch: float,
) -> list[tuple[float, float]]:
    if outer_radius <= 0.0:
        return [(center_x, center_y)]

    pitch = max(float(pitch), 1e-9)
    sample_spacing = pitch
    theta_max = (outer_radius / pitch) * 2.0 * math.pi
    theta = 0.0
    points = [(center_x + outer_radius, center_y)]

    while theta < theta_max:
        radius = max(outer_radius - (pitch * theta / (2.0 * math.pi)), 0.0)
        d_theta = min(math.pi / 10.0, sample_spacing / max(radius, pitch))
        theta = min(theta + d_theta, theta_max)
        radius = max(outer_radius - (pitch * theta / (2.0 * math.pi)), 0.0)
        points.append(
            (
                center_x + (radius * math.cos(theta)),
                center_y + (radius * math.sin(theta)),
            )
        )

    if points[-1] != (center_x, center_y):
        points.append((center_x, center_y))
    return points


def _circle_spiral_layer_segments(
    path_img: np.ndarray,
    color_img: np.ndarray,
    pixel_size: float,
    reverse: bool = False,
) -> list[tuple[float, float, float, float, int]]:
    bounds = _active_pixel_bounds(path_img > 0)
    if bounds is None:
        return []

    top, bottom, left, right = bounds
    left_x = float(left) * pixel_size
    right_x = float(right + 1) * pixel_size
    top_y = float(top) * pixel_size
    bottom_y = float(bottom + 1) * pixel_size
    center_x = (left_x + right_x) / 2.0
    center_y = (top_y + bottom_y) / 2.0
    outer_radius = max(
        math.hypot(corner_x - center_x, corner_y - center_y)
        for corner_x, corner_y in (
            (left_x, top_y),
            (right_x, top_y),
            (right_x, bottom_y),
            (left_x, bottom_y),
        )
    )

    points = _circle_spiral_points(center_x, center_y, outer_radius, pixel_size)
    if reverse:
        points.reverse()

    height, width = color_img.shape[:2]

    def point_color(x: float, y: float) -> int:
        col = int(math.floor(x / pixel_size))
        row = int(math.floor(y / pixel_size))
        if row < 0 or row >= height or col < 0 or col >= width:
            return 0
        return int(color_img[row, col])

    segments: list[tuple[float, float, float, float, int]] = []
    for (start_x, start_y), (end_x, end_y) in zip(points, points[1:]):
        mid_x = (start_x + end_x) / 2.0
        mid_y = (start_y + end_y) / 2.0
        _append_colored_segment(
            segments,
            start_x,
            start_y,
            end_x,
            end_y,
            point_color(mid_x, mid_y),
        )
    return segments


def _build_footprint_raster_gcode_list(
    path_ref_list: list[np.ndarray],
    color_ref_list: list[np.ndarray],
    pixel_size: float,
    layer_height: float,
    raster_pattern: str,
    contour_layers: list[list[dict]] | None = None,
    active_contour_owner: int | None = None,
) -> list[dict]:
    gcode_list: list[dict] = []
    current_x = 0.0
    current_y = 0.0
    raster_origin_initialized = False
    contour_layers = contour_layers or []

    for layer_number, (path_img, color_img) in enumerate(zip(path_ref_list, color_ref_list)):
        if raster_pattern == RASTER_PATTERN_CIRCLE_SPIRAL:
            segments = _circle_spiral_layer_segments(
                path_img,
                color_img,
                pixel_size,
                reverse=layer_number % 2 == 1,
            )
        elif raster_pattern == RASTER_PATTERN_RECTANGULAR_SPIRAL:
            segments = _rectangular_spiral_layer_segments(
                path_img,
                color_img,
                pixel_size,
                reverse=layer_number % 2 == 1,
            )
        else:
            raster_axis = _raster_axis_for_pattern(raster_pattern, layer_number)
            segments = _oriented_woodpile_layer_segments(
                path_img,
                color_img,
                pixel_size,
                raster_axis,
                current_x,
                current_y,
                prefer_default=not raster_origin_initialized,
            )
        if not segments:
            if layer_number > 0:
                gcode_list.append({"X": 0.0, "Y": 0.0, "Z": layer_height, "Color": 0})
            layer_end_x, layer_end_y = current_x, current_y
            current_x, current_y = _append_layer_contours(
                gcode_list,
                current_x,
                current_y,
                contour_layers,
                layer_number,
                active_contour_owner,
            )
            current_x, current_y = _append_relative_move(
                gcode_list,
                current_x,
                current_y,
                layer_end_x,
                layer_end_y,
                0,
            )
            continue

        first_x, first_y = segments[0][0], segments[0][1]
        if not raster_origin_initialized:
            if layer_number > 0:
                current_x, current_y = _append_relative_move(
                    gcode_list,
                    current_x,
                    current_y,
                    current_x,
                    current_y,
                    0,
                    z_step=layer_height,
                )
            current_x, current_y = first_x, first_y
            raster_origin_initialized = True
        elif layer_number > 0:
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

        layer_end_x, layer_end_y = current_x, current_y
        current_x, current_y = _append_layer_contours(
            gcode_list,
            current_x,
            current_y,
            contour_layers,
            layer_number,
            active_contour_owner,
        )
        current_x, current_y = _append_relative_move(
            gcode_list,
            current_x,
            current_y,
            layer_end_x,
            layer_end_y,
            0,
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
    pressure_ramp_enabled: bool = True,
    all_g1: bool = False,
    motion_tiffs: list[str] | None = None,
    raster_pattern: str | None = RASTER_PATTERN_SAME_DIRECTION,
    contour_tiff_sets: list[dict] | None = None,
    active_contour_owner: int | None = None,
    lead_in_enabled: bool = False,
    lead_in_length: float = 5.0,
    lead_in_clearance: float = 5.0,
    lead_in_lines: int = 3,
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

    layer_start_directions = None
    if raster_pattern == RASTER_PATTERN_SAME_DIRECTION:
        layer_start_directions = _same_direction_layer_start_directions(
            path_ref_list,
            off_color,
        )

    contour_layers = _build_contour_layers(
        contour_tiff_sets,
        path_ref_list,
        fil_width,
        invert,
        off_color,
        work_dir,
        raster_pattern,
        layer_start_directions,
    )

    setpress_lines = [_setpress_cmd(com_port, pressure, start=True)]
    pressure_on_lines = [_toggle_cmd(com_port, start=True)]
    pressure_off_lines = [_toggle_cmd(com_port, start=False)]

    if raster_pattern in (
        RASTER_PATTERN_Y_DIRECTION,
        RASTER_PATTERN_WOODPILE,
        RASTER_PATTERN_RECTANGULAR_SPIRAL,
        RASTER_PATTERN_CIRCLE_SPIRAL,
    ):
        gcode_list = _build_footprint_raster_gcode_list(
            path_ref_list,
            color_ref_list,
            fil_width,
            layer_height,
            raster_pattern,
            contour_layers,
            active_contour_owner,
        )
    else:
        gcode_list: list[dict] = []
        current_x = 0.0
        current_y = 0.0
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
                current_x += shift_y
                current_y += shift_x

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
            layer_start = len(gcode_list)
            layer_origin_x, layer_origin_y = current_x, current_y
            layer_x_scale = 1.0 if direction < 0 else -1.0
            layer_y_scale = -1.0 if layers % 2 == 1 else 1.0
            direction = _gcode_layer(
                ref_for_path,
                current_image,
                gcode_list,
                fil_width,
                direction,
                layers,
            )
            for move in gcode_list[layer_start:]:
                current_x += float(move.get("X", 0.0))
                current_y += float(move.get("Y", 0.0))

            layer_end_x, layer_end_y = current_x, current_y
            current_x, current_y = _append_layer_contours(
                gcode_list,
                current_x,
                current_y,
                contour_layers,
                layers,
                active_contour_owner,
                layer_origin_x,
                layer_origin_y,
                layer_x_scale,
                layer_y_scale,
            )
            current_x, current_y = _append_relative_move(
                gcode_list,
                current_x,
                current_y,
                layer_end_x,
                layer_end_y,
                off_color,
            )

    lead_in = _lead_in_moves(
        lead_in_enabled,
        lead_in_length,
        lead_in_clearance,
        lead_in_lines,
        fil_width,
        255,
        off_color,
    )
    if lead_in:
        gcode_list = [*lead_in, *gcode_list]

    gcode_path = work_dir / f"{shape_name}_SnakePath_gcode.txt"
    pressure_cur = float(pressure)

    with open(gcode_path, "w") as f:
        f.write("G91\n")
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
                    f"{move_type} X{move['X']} Y{move['Y']} Z{move['Z']} "
                    f"; Color {move['Color']}"
                )
                if pressure_ramp_enabled:
                    pressure_cur += increase_pressure_per_layer
                    pressure_next = _setpress_cmd(com_port, pressure_cur, start=False)
                else:
                    pressure_next = None
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
