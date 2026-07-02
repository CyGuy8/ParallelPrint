"""Vector toolpath generation: shapely layer polygons -> printable move lists.

Replaces the pixel-raster core of the old TIFF pipeline. All geometry lives in
world-XY millimetres; every motion segment is a 5-tuple
``(x0, y0, x1, y1, color)`` where color 255 means the dispensing valve is open
and 0 means travel. `fil_width` is both the raster line spacing and the
filament width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely import prepare
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPolygon, Point, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from stl_slicer import LayerStack, _as_multipolygon


EPS = 1e-6

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

Seg = tuple[float, float, float, float, int]


@dataclass(slots=True)
class ContourSource:
    owner_idx: int
    stack: LayerStack


def _normalize_raster_pattern(pattern: str | None) -> str:
    if pattern in RASTER_PATTERN_CHOICES:
        return pattern
    return RASTER_PATTERN_SAME_DIRECTION


def _raster_axis_for_pattern(pattern: str, layer_number: int) -> str:
    if pattern == RASTER_PATTERN_Y_DIRECTION:
        return "Y"
    if pattern == RASTER_PATTERN_WOODPILE and layer_number % 2 == 1:
        return "Y"
    return "X"


def _iter_linestrings(geometry: object):
    if geometry is None or getattr(geometry, "is_empty", True):
        return
    geom_type = geometry.geom_type
    if geom_type == "LineString":
        yield geometry
    elif geom_type in ("MultiLineString", "GeometryCollection"):
        for part in geometry.geoms:
            yield from _iter_linestrings(part)


def _iter_coords(geometry: object):
    if geometry is None or getattr(geometry, "is_empty", True):
        return
    geom_type = geometry.geom_type
    if geom_type == "Point":
        yield geometry.x, geometry.y
    elif geom_type == "LineString":
        for coord in geometry.coords:
            yield coord[0], coord[1]
    elif geom_type in ("MultiPoint", "MultiLineString", "GeometryCollection"):
        for part in geometry.geoms:
            yield from _iter_coords(part)


def _point_distance_sq(ax: float, ay: float, bx: float, by: float) -> float:
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


def _append_colored_segment(
    segments: list[Seg],
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


def _append_relative_move(
    output_list: list[dict],
    current_x: float,
    current_y: float,
    target_x: float,
    target_y: float,
    color: int,
    z_step: float | None = None,
) -> tuple[float, float]:
    # Round away GEOS float noise (sub-micron residues, -0.0) so the emitted
    # G-code stays clean; position tracking keeps the exact targets. Six
    # decimals (1 nm) is far below any printable resolution, and rounding here
    # also guarantees the writer can format every delta without scientific
    # notation (which G-code parsers misread: "X-5.1e-08" parses as X-5.1).
    dx = round(target_x - current_x, 6) + 0.0
    dy = round(target_y - current_y, 6) + 0.0
    if dx == 0 and dy == 0 and z_step is None:
        return current_x, current_y
    move = {"X": dx, "Y": dy, "Color": color}
    if z_step is not None:
        move["Z"] = z_step
    output_list.append(move)
    return target_x, target_y


def _chord_runs(
    region: MultiPolygon,
    axis: str,
    coord: float,
    eps: float = EPS,
) -> list[tuple[float, float]]:
    """Sorted, merged (lo, hi) intervals where an axis-aligned scanline is inside `region`.

    Axis "X": horizontal scanline at y=coord, runs measured along X.
    Axis "Y": vertical scanline at x=coord, runs measured along Y.
    """
    if region is None or region.is_empty:
        return []

    min_x, min_y, max_x, max_y = region.bounds
    if axis == "Y":
        if coord < min_x - eps or coord > max_x + eps:
            return []
        line = LineString([(coord, min_y - 1.0), (coord, max_y + 1.0)])
        index = 1
    else:
        if coord < min_y - eps or coord > max_y + eps:
            return []
        line = LineString([(min_x - 1.0, coord), (max_x + 1.0, coord)])
        index = 0

    runs: list[tuple[float, float]] = []
    for piece in _iter_linestrings(region.intersection(line)):
        if piece.length <= eps:
            continue
        values = [coords[index] for coords in piece.coords]
        runs.append((min(values), max(values)))

    runs.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in runs:
        if merged and lo <= merged[-1][1] + eps:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _point_along(
    t: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[float, float]:
    if t <= 0.0:
        return x0, y0
    if t >= 1.0:
        return x1, y1
    return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t


def _classify_polyline(
    points: list[tuple[float, float]],
    valve: MultiPolygon,
    eps: float = EPS,
) -> list[Seg]:
    """Split a motion polyline at valve-boundary crossings and color the pieces.

    Preserves path order (a whole-polyline shapely intersection would not).
    Pieces on the boundary count as printing (`covers`), matching the raster
    convention that boundary-grazing sweeps dispense.
    """
    segments: list[Seg] = []
    has_valve = valve is not None and not valve.is_empty
    if has_valve:
        prepare(valve)
        boundary = valve.boundary

    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len <= eps:
            continue

        if not has_valve:
            _append_colored_segment(segments, x0, y0, x1, y1, 0)
            continue

        ts = {0.0, 1.0}
        crossings = LineString([(x0, y0), (x1, y1)]).intersection(boundary)
        for cx, cy in _iter_coords(crossings):
            t = ((cx - x0) * dx + (cy - y0) * dy) / (seg_len * seg_len)
            ts.add(max(0.0, min(1.0, t)))

        t_eps = eps / seg_len
        ordered = sorted(ts)
        deduped = [ordered[0]]
        for t in ordered[1:]:
            if t - deduped[-1] > t_eps:
                deduped.append(t)
            else:
                deduped[-1] = max(deduped[-1], t)

        for t0, t1 in zip(deduped, deduped[1:]):
            if (t1 - t0) * seg_len <= eps:
                continue
            mid_x, mid_y = _point_along((t0 + t1) / 2.0, x0, y0, x1, y1)
            color = 255 if valve.covers(Point(mid_x, mid_y)) else 0
            start = _point_along(t0, x0, y0, x1, y1)
            end = _point_along(t1, x0, y0, x1, y1)
            _append_colored_segment(segments, *start, *end, color)

    return segments


def _scan_coords(lo: float, hi: float, fil_width: float) -> list[float]:
    """Scanline positions at half-fil_width offsets; always at least one line."""
    if hi - lo < fil_width:
        return [(lo + hi) / 2.0]

    coords: list[float] = []
    value = lo + fil_width / 2.0
    limit = hi - fil_width / 2.0 + 1e-9
    while value <= limit:
        coords.append(value)
        value += fil_width
    return coords


def _axis_raster_segments(
    motion: MultiPolygon,
    valve: MultiPolygon,
    fil_width: float,
    axis: str,
    reverse_order: bool = False,
    start_forward: bool = True,
) -> list[Seg]:
    """Snake raster of `motion`, dispensing only inside `valve`.

    Axis "X" sweeps along X with rows stacked in Y; axis "Y" sweeps along Y
    with columns stacked in X. Each sweep spans from the first to the last
    motion chord (crossing interior gaps valve-off) and gets a fil_width
    valve-settle travel buffer before and after.
    """
    if motion is None or motion.is_empty:
        return []

    min_x, min_y, max_x, max_y = motion.bounds
    if axis == "Y":
        scan_lo, scan_hi = min_x, max_x
    else:
        scan_lo, scan_hi = min_y, max_y

    coords = _scan_coords(scan_lo, scan_hi, fil_width)
    if reverse_order:
        coords = coords[::-1]

    segments: list[Seg] = []
    sweep_number = 0
    for coord in coords:
        motion_runs = _chord_runs(motion, axis, coord)
        if not motion_runs:
            continue

        sweep_lo = motion_runs[0][0]
        sweep_hi = motion_runs[-1][1]
        valve_runs = []
        for lo, hi in _chord_runs(valve, axis, coord):
            lo = max(lo, sweep_lo)
            hi = min(hi, sweep_hi)
            if hi - lo > EPS:
                valve_runs.append((lo, hi))

        def emit(a: float, b: float, color: int) -> None:
            if axis == "Y":
                _append_colored_segment(segments, coord, a, coord, b, color)
            else:
                _append_colored_segment(segments, a, coord, b, coord, color)

        forward = (sweep_number % 2 == 0) == start_forward
        sweep_number += 1

        if forward:
            emit(sweep_lo - fil_width, sweep_lo, 0)
            current = sweep_lo
            for lo, hi in valve_runs:
                if lo - current > EPS:
                    emit(current, lo, 0)
                    current = lo
                start = max(lo, current)
                if hi - start > EPS:
                    emit(start, hi, 255)
                    current = hi
            if sweep_hi - current > EPS:
                emit(current, sweep_hi, 0)
                current = sweep_hi
            emit(current, current + fil_width, 0)
        else:
            emit(sweep_hi + fil_width, sweep_hi, 0)
            current = sweep_hi
            for lo, hi in reversed(valve_runs):
                if current - hi > EPS:
                    emit(current, hi, 0)
                    current = hi
                start = min(hi, current)
                if start - lo > EPS:
                    emit(start, lo, 255)
                    current = lo
            if current - sweep_lo > EPS:
                emit(current, sweep_lo, 0)
                current = sweep_lo
            emit(current, current - fil_width, 0)

    return segments


def _oriented_axis_raster_segments(
    motion: MultiPolygon,
    valve: MultiPolygon,
    fil_width: float,
    axis: str,
    current_x: float,
    current_y: float,
    prefer_default: bool = False,
) -> list[Seg]:
    """Pick the raster orientation whose start is nearest the current position."""
    default_segments = _axis_raster_segments(motion, valve, fil_width, axis)
    if prefer_default or not default_segments:
        return default_segments

    candidates: list[list[Seg]] = [default_segments]
    for reverse_order in (False, True):
        for start_forward in (False, True):
            segments = _axis_raster_segments(
                motion,
                valve,
                fil_width,
                axis,
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


def _extend_polyline_ends(
    points: list[tuple[float, float]],
    length: float,
) -> list[tuple[float, float]]:
    """Extend both polyline ends by `length` along their local directions."""
    if len(points) < 2:
        return points

    (x0, y0), (x1, y1) = points[0], points[1]
    distance = math.hypot(x1 - x0, y1 - y0)
    if distance > 0:
        points = [
            (x0 - (x1 - x0) / distance * length, y0 - (y1 - y0) / distance * length),
            *points,
        ]

    (xa, ya), (xb, yb) = points[-2], points[-1]
    distance = math.hypot(xb - xa, yb - ya)
    if distance > 0:
        points = [
            *points,
            (xb + (xb - xa) / distance * length, yb + (yb - ya) / distance * length),
        ]
    return points


def _rectangular_spiral_polyline(
    bounds: tuple[float, float, float, float],
    fil_width: float,
    reverse: bool = False,
) -> list[tuple[float, float]]:
    """Corner polyline spiralling from the bounds inward (or outward if reversed)."""
    min_x, min_y, max_x, max_y = bounds
    half = fil_width / 2.0
    left = min_x + half
    right = max_x - half
    bottom = min_y + half
    top = max_y - half
    if right < left:
        left = right = (min_x + max_x) / 2.0
    if top < bottom:
        bottom = top = (min_y + max_y) / 2.0

    eps = 1e-9
    points: list[tuple[float, float]] = []

    def add(x: float, y: float) -> None:
        if not points or _point_distance_sq(points[-1][0], points[-1][1], x, y) > eps:
            points.append((x, y))

    while left <= right + eps and bottom <= top + eps:
        add(left, top)
        add(right, top)
        top -= fil_width

        if bottom <= top + eps:
            add(right, bottom)
        right -= fil_width

        if bottom <= top + eps:
            add(left, bottom)
            bottom += fil_width

        if left <= right + eps:
            if bottom <= top + eps:
                add(left, top)
            left += fil_width

    if len(points) == 1:
        center_x, center_y = points[0]
        points = [(center_x - half, center_y), (center_x + half, center_y)]
        if reverse:
            points.reverse()
        return points

    if reverse:
        points.reverse()
    return _extend_polyline_ends(points, half)


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


def _circle_spiral_polyline(
    bounds: tuple[float, float, float, float],
    fil_width: float,
    reverse: bool = False,
) -> list[tuple[float, float]]:
    min_x, min_y, max_x, max_y = bounds
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    outer_radius = max(
        math.hypot(corner_x - center_x, corner_y - center_y)
        for corner_x, corner_y in (
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        )
    )

    points = _circle_spiral_points(center_x, center_y, outer_radius, fil_width)
    if reverse:
        points.reverse()
    return points


def _contour_area2(points: list[tuple[float, float]]) -> float:
    return sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:])
    )


def _contour_sort_key(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (-abs(_contour_area2(points)), min(ys), min(xs))


def _layer_contour_loops(layer: MultiPolygon) -> list[list[tuple[float, float]]]:
    """Closed contour loops of a layer: exterior rings first, holes after."""
    loops: list[list[tuple[float, float]]] = []
    for polygon in layer.geoms:
        oriented = orient(polygon)
        for ring in (oriented.exterior, *oriented.interiors):
            simplified = ring.simplify(0)
            coords = [(float(x), float(y)) for x, y in simplified.coords]
            if len(coords) >= 4:
                loops.append(coords)

    loops.sort(key=_contour_sort_key)
    return loops


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


def build_contour_layers(
    sources: list[ContourSource] | None,
    n_layers: int,
    reference: LayerStack | None = None,
) -> list[list[dict]]:
    """Per-layer contour loops per owning shape.

    When `reference` is given (reference-motion mode) each source stack is
    translated by its centering delta so contours land in the shared frame.
    """
    contour_layers: list[list[dict]] = [[] for _ in range(n_layers)]
    for source in sources or []:
        stack = source.stack
        if stack is None or not stack.layers:
            continue

        if reference is not None:
            layers = align_stack_to(stack, reference, n_layers)
        else:
            layers = stack.layers

        for layer_number in range(min(n_layers, len(layers))):
            layer = layers[layer_number]
            if layer is None or layer.is_empty:
                continue
            contours = _layer_contour_loops(layer)
            if contours:
                contour_layers[layer_number].append(
                    {
                        "owner_idx": source.owner_idx,
                        "contours": contours,
                    }
                )

    return contour_layers


def _append_layer_contours(
    output_list: list[dict],
    current_x: float,
    current_y: float,
    contour_layers: list[list[dict]],
    layer_number: int,
    active_owner_idx: int | None,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> tuple[float, float]:
    # `origin_x/origin_y` is the world position of the move list's start:
    # _last_print_reference sums relative moves from zero, so its result must
    # be shifted back into the world frame the contours live in.
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
            if use_infill_reference:
                nearest_x, nearest_y, approach_dx, approach_dy = _last_print_reference(
                    output_list
                )
                nearest_x += origin_x
                nearest_y += origin_y
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


def plan_layer_moves(
    motion_layers: list[MultiPolygon],
    valve_layers: list[MultiPolygon],
    fil_width: float,
    layer_height: float,
    raster_pattern: str,
    contour_layers: list[list[dict]] | None = None,
    active_contour_owner: int | None = None,
) -> list[dict]:
    """Assemble per-layer segments into a relative move list for all patterns."""
    raster_pattern = _normalize_raster_pattern(raster_pattern)
    gcode_list: list[dict] = []
    current_x = 0.0
    current_y = 0.0
    origin_x = 0.0
    origin_y = 0.0
    raster_origin_initialized = False
    contour_layers = contour_layers or []

    for layer_number, (motion, valve) in enumerate(zip(motion_layers, valve_layers)):
        if motion is None or motion.is_empty:
            segments: list[Seg] = []
        elif raster_pattern == RASTER_PATTERN_CIRCLE_SPIRAL:
            points = _circle_spiral_polyline(
                motion.bounds,
                fil_width,
                reverse=layer_number % 2 == 1,
            )
            segments = _classify_polyline(points, valve)
        elif raster_pattern == RASTER_PATTERN_RECTANGULAR_SPIRAL:
            points = _rectangular_spiral_polyline(
                motion.bounds,
                fil_width,
                reverse=layer_number % 2 == 1,
            )
            segments = _classify_polyline(points, valve)
        else:
            raster_axis = _raster_axis_for_pattern(raster_pattern, layer_number)
            segments = _oriented_axis_raster_segments(
                motion,
                valve,
                fil_width,
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
                origin_x,
                origin_y,
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
            origin_x, origin_y = first_x, first_y
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
            origin_x,
            origin_y,
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


def _stack_center(stack: LayerStack) -> tuple[float, float]:
    (x_min, y_min, _z_min), (x_max, y_max, _z_max) = stack.bounds
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def _centering_delta(stack: LayerStack, reference: LayerStack) -> tuple[float, float]:
    reference_x, reference_y = _stack_center(reference)
    center_x, center_y = _stack_center(stack)
    return reference_x - center_x, reference_y - center_y


def align_stack_to(
    stack: LayerStack,
    reference: LayerStack,
    n_layers: int,
) -> list[MultiPolygon]:
    """Translate a stack's layers into the reference frame, padded with empties."""
    delta_x, delta_y = _centering_delta(stack, reference)
    layers: list[MultiPolygon] = []
    for index in range(n_layers):
        if index < len(stack.layers) and not stack.layers[index].is_empty:
            layers.append(
                _as_multipolygon(
                    translate(stack.layers[index], xoff=delta_x, yoff=delta_y)
                )
            )
        else:
            layers.append(MultiPolygon())
    return layers


def build_reference_stack(stacks: list[LayerStack | None]) -> LayerStack | None:
    """Union all shapes into one shared motion stack, bbox-centres aligned.

    Vector analog of the old centered "black wins" TIFF merge: every stack is
    translated so its XY bbox centre lands on the first stack's centre, then
    each layer is the union of the translated layers.
    """
    valid = [stack for stack in stacks if stack is not None and stack.layers]
    if not valid:
        return None

    n_layers = max(len(stack.layers) for stack in valid)
    reference_x, reference_y = _stack_center(valid[0])

    layer_parts: list[list[MultiPolygon]] = [[] for _ in range(n_layers)]
    x_min = y_min = z_min = math.inf
    x_max = y_max = z_max = -math.inf

    for stack in valid:
        center_x, center_y = _stack_center(stack)
        delta_x = reference_x - center_x
        delta_y = reference_y - center_y

        (sx_min, sy_min, sz_min), (sx_max, sy_max, sz_max) = stack.bounds
        x_min = min(x_min, sx_min + delta_x)
        x_max = max(x_max, sx_max + delta_x)
        y_min = min(y_min, sy_min + delta_y)
        y_max = max(y_max, sy_max + delta_y)
        z_min = min(z_min, sz_min)
        z_max = max(z_max, sz_max)

        for index, layer in enumerate(stack.layers):
            if layer is None or layer.is_empty:
                continue
            layer_parts[index].append(translate(layer, xoff=delta_x, yoff=delta_y))

    layers = [
        _as_multipolygon(unary_union(parts)) if parts else MultiPolygon()
        for parts in layer_parts
    ]

    z_values: list[float] = []
    for index in range(n_layers):
        z_value = 0.0
        for stack in valid:
            if index < len(stack.z_values):
                z_value = stack.z_values[index]
                break
        z_values.append(z_value)

    return LayerStack(
        layers=layers,
        z_values=z_values,
        bounds=((x_min, y_min, z_min), (x_max, y_max, z_max)),
        layer_height=valid[0].layer_height,
        name="reference",
    )


def _split_boundaries(
    lo: float,
    hi: float,
    count: int,
    layer_index: int,
    overlap: float,
) -> list[float]:
    """Cell boundaries for one axis; interior ones alternate by ±overlap per layer."""
    edges = [lo + index * (hi - lo) / count for index in range(count + 1)]
    if overlap <= 0.0 or count <= 1:
        return edges

    adjusted = list(edges)
    for boundary_index in range(1, count):
        direction = 1 if (layer_index + boundary_index) % 2 == 1 else -1
        lower = adjusted[boundary_index - 1] + overlap
        upper = edges[boundary_index + 1] - overlap
        shifted = edges[boundary_index] + direction * overlap
        if lower <= upper:
            shifted = max(lower, min(upper, shifted))
        else:
            shifted = edges[boundary_index]
        adjusted[boundary_index] = shifted
    return adjusted


def split_layer_stack_grid(
    stack: LayerStack,
    columns: int,
    rows: int,
    overlapping_layers: bool = False,
    overlap: float = 0.0,
) -> list[LayerStack]:
    """Split a sliced shape into a rows x columns grid of piece stacks.

    Pieces are returned row-major with row 1 the top strip (max-Y side),
    matching the legacy image-grid ordering. With `overlapping_layers`, the
    interior cut lines alternate by ±overlap between layers so neighbouring
    pieces interlock. Piece `bounds` are the nominal (un-shifted) cell boxes.
    """
    columns = max(1, int(columns))
    rows = max(1, int(rows))
    (x_min, y_min, z_min), (x_max, y_max, z_max) = stack.bounds

    overlap_x = overlap if (overlapping_layers and columns > 1) else 0.0
    overlap_y = overlap if (overlapping_layers and rows > 1) else 0.0

    base_x_edges = _split_boundaries(x_min, x_max, columns, 0, 0.0)
    base_y_edges = _split_boundaries(y_min, y_max, rows, 0, 0.0)
    layer_x_edges = [
        _split_boundaries(x_min, x_max, columns, index, overlap_x)
        for index in range(len(stack.layers))
    ]
    layer_y_edges = [
        _split_boundaries(y_min, y_max, rows, index, overlap_y)
        for index in range(len(stack.layers))
    ]

    base_name = stack.name or "shape"
    pieces: list[LayerStack] = []
    for row_index in range(1, rows + 1):
        # Row 1 is the top strip: count y-cells down from the max edge.
        y_cell = rows - row_index
        for col_index in range(1, columns + 1):
            x_cell = col_index - 1

            layers: list[MultiPolygon] = []
            for layer_number, layer in enumerate(stack.layers):
                if layer is None or layer.is_empty:
                    layers.append(MultiPolygon())
                    continue
                x_edges = layer_x_edges[layer_number]
                y_edges = layer_y_edges[layer_number]
                cell = box(
                    x_edges[x_cell],
                    y_edges[y_cell],
                    x_edges[x_cell + 1],
                    y_edges[y_cell + 1],
                )
                layers.append(_as_multipolygon(layer.intersection(cell)))

            pieces.append(
                LayerStack(
                    layers=layers,
                    z_values=list(stack.z_values),
                    bounds=(
                        (base_x_edges[x_cell], base_y_edges[y_cell], z_min),
                        (base_x_edges[x_cell + 1], base_y_edges[y_cell + 1], z_max),
                    ),
                    layer_height=stack.layer_height,
                    name=f"{base_name}_r{row_index:02d}_c{col_index:02d}",
                )
            )

    return pieces
