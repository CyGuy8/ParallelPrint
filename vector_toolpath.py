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
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, box
from shapely.geometry.polygon import orient
from shapely.ops import linemerge, unary_union

from stl_slicer import LayerStack, _as_multipolygon


EPS = 1e-6

RASTER_PATTERN_SAME_DIRECTION = "X-direction raster"
RASTER_PATTERN_Y_DIRECTION = "Y-direction raster"
RASTER_PATTERN_WOODPILE = "90° Woodpile raster"
RASTER_PATTERN_DIAGONAL_WOODPILE = "45° Woodpile raster"
RASTER_PATTERN_RECTANGULAR_SPIRAL = "Rectangular Spiral raster"
RASTER_PATTERN_CIRCLE_SPIRAL = "Circle Spiral raster"
RASTER_PATTERN_CHOICES = (
    RASTER_PATTERN_SAME_DIRECTION,
    RASTER_PATTERN_Y_DIRECTION,
    RASTER_PATTERN_WOODPILE,
    RASTER_PATTERN_DIAGONAL_WOODPILE,
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
    if pattern == RASTER_PATTERN_DIAGONAL_WOODPILE and layer_number % 4 == 2:
        return "Y"
    return "X"


def _diagonal_layer_angle(layer_number: int) -> float | None:
    """Diagonal-woodpile raster angle for a layer; None for the axis layers.

    The pattern cycles 0, 45, 90, 135 degrees: axis layers (0/90) go through
    the regular X/Y raster; only the odd layers need the rotated raster.
    """
    phase = layer_number % 4
    if phase == 1:
        return 45.0
    if phase == 3:
        return 135.0
    return None


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
    keep_point=None,
    keep_segment=None,
) -> list[Seg]:
    """Split a motion polyline at valve-boundary crossings and color the pieces.

    Preserves path order (a whole-polyline shapely intersection would not).
    Pieces on the boundary count as printing (`covers`), matching the raster
    convention that boundary-grazing sweeps dispense. `keep_point(x, y)` can
    additionally veto dispensing for a piece (evaluated at its midpoint) —
    used for partial infill. `keep_segment(x0, y0, x1, y1)` vetoes a whole
    source segment BEFORE it is split, so a vetoed transition move stays
    valve-off even where the material boundary cuts through it. The motion
    is unaffected by either gate.
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

        if not has_valve or (
            keep_segment is not None and not keep_segment(x0, y0, x1, y1)
        ):
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
            dispensing = valve.covers(Point(mid_x, mid_y))
            if dispensing and keep_point is not None:
                dispensing = keep_point(mid_x, mid_y)
            color = 255 if dispensing else 0
            start = _point_along(t0, x0, y0, x1, y1)
            end = _point_along(t1, x0, y0, x1, y1)
            _append_colored_segment(segments, *start, *end, color)

    return segments


def _infill_line_keep(fraction: float):
    """Line-selection gate for partial infill; None means print every line.

    Uses a Bresenham-style distribution so kept lines spread evenly: line k
    dispenses iff floor((k+1)*f) > floor(k*f). At f=0.5 every other line
    prints; at f=1 all do; at f=0 none. `k` is a global grid index, so the
    same lines are selected across layers, across split pieces sharing a
    parent frame, and across shapes sharing reference motion — the motion
    path itself is never affected.
    """
    if fraction >= 1.0:
        return None
    fraction = max(0.0, float(fraction))

    def keep(line_index: int) -> bool:
        return math.floor((line_index + 1) * fraction) - math.floor(line_index * fraction) >= 1

    return keep


def _combined_infill_keep(fractions):
    """Union infill gate over every shape sharing the motion; None = keep all.

    A grid line is traversed iff ANY shape's infill pattern dispenses on it —
    lines nobody prints are dropped from the MOTION entirely instead of being
    swept valve-off (e.g. every shape at 50% infill halves the path). Every
    shape must be given the same fraction list so the shared motion stays
    identical across heads.
    """
    keeps = []
    for fraction in fractions or []:
        keep = _infill_line_keep(fraction)
        if keep is None:
            return None  # someone prints every line: no motion line can drop
        keeps.append(keep)
    if not keeps:
        return None

    def keep_any(line_index: int) -> bool:
        return any(keep(line_index) for keep in keeps)

    return keep_any


def _scan_anchor(lo: float, hi: float, fil_width: float) -> float:
    """First scanline position of the centred grid spanning [lo, hi].

    The grid is centred: when the extent is not an exact multiple of
    fil_width, the leftover slack is split evenly between both edges.
    """
    extent = hi - lo
    if extent < fil_width:
        return (lo + hi) / 2.0
    count = int(math.floor(extent / fil_width + 1e-9))
    slack = max(0.0, extent - count * fil_width)
    return lo + slack / 2.0 + fil_width / 2.0


def _scan_coords(
    lo: float,
    hi: float,
    fil_width: float,
    anchor: float | None = None,
) -> list[float]:
    """Scanline positions in [lo, hi], fil_width apart; at least one line.

    Without `anchor` the set is centred in [lo, hi] (always at least one
    line). With `anchor` the lines lie on the global grid
    {anchor + k*fil_width}: grid-split pieces anchored to their parent's
    frame then raster on one continuous line grid, so the assembled seams
    keep an exact one-fil-width pitch instead of drifting by each piece's own
    quantization slack. The interval is half-open at `hi` so a line exactly
    on a cut belongs to exactly one piece, and a sliver too thin to contain a
    grid line gets none — the neighbouring grid line (at most one fil away)
    covers it with filament width.
    """
    if anchor is None:
        anchor = _scan_anchor(lo, hi, fil_width)
        if hi - lo < fil_width:
            return [anchor]
        count = int(math.floor((hi - lo) / fil_width + 1e-9))
        return [anchor + index * fil_width for index in range(count)]

    # The epsilon is in grid-cell units and must swamp float noise from the
    # anchor/edge arithmetic: a piece whose material starts EXACTLY on a grid
    # line (a split cut on the line) can compute (lo-anchor)/fil as
    # -1.0000000000000009, and a 1e-9 epsilon then ceils the boundary line
    # away — nobody prints it and every assembled seam gets a one-fil gap.
    k_lo = math.ceil((lo - anchor) / fil_width - 1e-6)
    k_hi = math.floor((hi - anchor) / fil_width - 1e-6)
    return [anchor + k * fil_width for k in range(int(k_lo), int(k_hi) + 1)]


def _axis_raster_segments(
    motion: MultiPolygon,
    valve: MultiPolygon,
    fil_width: float,
    axis: str,
    reverse_order: bool = False,
    start_forward: bool = True,
    scan_anchor: float | None = None,
    infill_keep=None,
    motion_keep=None,
) -> list[Seg]:
    """Snake raster of `motion`, dispensing only inside `valve`.

    Axis "X" sweeps along X with rows stacked in Y; axis "Y" sweeps along Y
    with columns stacked in X. Each sweep spans from the first to the last
    motion chord (crossing interior gaps valve-off) and gets a fil_width
    valve-settle travel buffer before and after. `scan_anchor` pins the
    scanlines to a global grid (see `_scan_coords`). `infill_keep(k)` gates
    dispensing per grid line for partial infill: skipped lines are still
    swept with the valve closed. `motion_keep(k)` drops a grid line from the
    MOTION entirely — used when NO shape sharing the motion dispenses there
    (the union of every head's infill pattern), so nobody sweeps dead lines.
    """
    if motion is None or motion.is_empty:
        return []

    min_x, min_y, max_x, max_y = motion.bounds
    if axis == "Y":
        scan_lo, scan_hi = min_x, max_x
    else:
        scan_lo, scan_hi = min_y, max_y

    coords = _scan_coords(scan_lo, scan_hi, fil_width, anchor=scan_anchor)
    if reverse_order:
        coords = coords[::-1]

    # Grid index base: orientation-independent, and shared across pieces when
    # anchored to a common frame, so infill line selection lines up at seams.
    index_base = scan_anchor if scan_anchor is not None else _scan_anchor(
        scan_lo, scan_hi, fil_width
    )

    # A grid line can graze the material boundary from OUTSIDE by float ulps
    # (split cuts sit exactly on grid lines, and the piece's material edge IS
    # the cut): probe the chords a hair inside the bounds so the boundary
    # sweep is still found, while emitting at the true grid coordinate.
    # A dropped boundary sweep prints a one-fil gap at every assembled seam.
    probe_eps = fil_width * 1e-6
    probe_lo = min(scan_lo + probe_eps, (scan_lo + scan_hi) / 2.0)
    probe_hi = max(scan_hi - probe_eps, (scan_lo + scan_hi) / 2.0)

    segments: list[Seg] = []
    sweep_number = 0
    for coord in coords:
        line_index = int(round((coord - index_base) / fil_width))
        if motion_keep is not None and not motion_keep(line_index):
            continue  # no head prints this line: drop it from the motion
        probe = min(max(coord, probe_lo), probe_hi)
        motion_runs = _chord_runs(motion, axis, probe)
        if not motion_runs:
            continue

        sweep_lo = motion_runs[0][0]
        sweep_hi = motion_runs[-1][1]
        if infill_keep is not None and not infill_keep(line_index):
            valve_runs: list[tuple[float, float]] = []
        else:
            valve_runs = []
            for lo, hi in _chord_runs(valve, axis, probe):
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
    scan_anchor: float | None = None,
    infill_keep=None,
    motion_keep=None,
) -> list[Seg]:
    """Pick the raster orientation whose start is nearest the current position."""
    default_segments = _axis_raster_segments(
        motion,
        valve,
        fil_width,
        axis,
        scan_anchor=scan_anchor,
        infill_keep=infill_keep,
        motion_keep=motion_keep,
    )
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
                scan_anchor=scan_anchor,
                infill_keep=infill_keep,
                motion_keep=motion_keep,
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


def _rotated_raster_segments(
    motion: MultiPolygon,
    valve: MultiPolygon,
    fil_width: float,
    angle_degrees: float,
    current_x: float,
    current_y: float,
    prefer_default: bool,
    frame: tuple[float, float, float, float],
    infill_keep=None,
    motion_keep=None,
) -> list[Seg]:
    """Snake raster at an arbitrary angle, reusing the axis raster machinery.

    Rotates motion/valve by -angle about the scan frame's centre, rasters
    with the standard X-axis sweep (snake, buffers, valve gating, scan grid,
    infill selection all identical), then rotates the segments back. The
    pivot and the rotated scan anchor both derive from the shared frame, so
    split pieces and reference-motion shapes keep one continuous diagonal
    line grid across seams.
    """
    if motion is None or motion.is_empty:
        return []

    pivot_x = (frame[0] + frame[2]) / 2.0
    pivot_y = (frame[1] + frame[3]) / 2.0
    pivot = (pivot_x, pivot_y)

    rotated_motion = _as_multipolygon(rotate(motion, -angle_degrees, origin=pivot))
    if valve is motion:
        rotated_valve = rotated_motion
    elif valve is None or valve.is_empty:
        rotated_valve = MultiPolygon()
    else:
        rotated_valve = _as_multipolygon(rotate(valve, -angle_degrees, origin=pivot))

    rotated_frame_bounds = rotate(box(*frame), -angle_degrees, origin=pivot).bounds
    anchor = _scan_anchor(rotated_frame_bounds[1], rotated_frame_bounds[3], fil_width)

    theta = math.radians(-angle_degrees)
    cos_f, sin_f = math.cos(theta), math.sin(theta)
    rotated_current_x = pivot_x + (current_x - pivot_x) * cos_f - (current_y - pivot_y) * sin_f
    rotated_current_y = pivot_y + (current_x - pivot_x) * sin_f + (current_y - pivot_y) * cos_f

    segments = _oriented_axis_raster_segments(
        rotated_motion,
        rotated_valve,
        fil_width,
        "X",
        rotated_current_x,
        rotated_current_y,
        prefer_default=prefer_default,
        scan_anchor=anchor,
        infill_keep=infill_keep,
        motion_keep=motion_keep,
    )

    theta_back = math.radians(angle_degrees)
    cos_b, sin_b = math.cos(theta_back), math.sin(theta_back)

    def back(x: float, y: float) -> tuple[float, float]:
        return (
            pivot_x + (x - pivot_x) * cos_b - (y - pivot_y) * sin_b,
            pivot_y + (x - pivot_x) * sin_b + (y - pivot_y) * cos_b,
        )

    return [
        (*back(x0, y0), *back(x1, y1), color)
        for x0, y0, x1, y1, color in segments
    ]


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


def _drop_short_print_runs(segments: list[Seg], min_length: float) -> list[Seg]:
    """Close the valve over isolated dispensing runs shorter than min_length.

    Motion is untouched — only the color flips to travel, so shared-motion
    parallel heads stay in sync. Used against boundary-grazing flicker,
    which no valve can deposit cleanly anyway.
    """
    result = list(segments)
    index = 0
    while index < len(result):
        if result[index][4] != 255:
            index += 1
            continue
        end = index
        run_length = 0.0
        while end < len(result) and result[end][4] == 255:
            x0, y0, x1, y1, _color = result[end]
            run_length += math.hypot(x1 - x0, y1 - y0)
            end += 1
        if run_length < min_length:
            for position in range(index, end):
                x0, y0, x1, y1, _color = result[position]
                result[position] = (x0, y0, x1, y1, 0)
        index = end
    return result


def circle_wall_radius(
    layer: MultiPolygon | None,
    center_x: float,
    center_y: float,
    fil_width: float,
) -> float | None:
    """A layer's outermost COMPLETE ring radius about the ring centre.

    Uses the minimum OUTER-boundary distance (holes ignored; the material
    may sit slightly off the ring centre). When a global grid ring already
    lies close under the boundary it IS the outer ring (no extra wall — a
    custom wall a fraction of a bead away would just double-deposit);
    otherwise the wall sits half a bead inside the closest boundary point.
    Only meaningful when the material surrounds the centre.
    """
    if layer is None or layer.is_empty:
        return None
    center = Point(center_x, center_y)
    for polygon in layer.geoms:
        if not polygon.covers(center):
            continue
        d_min = float(polygon.exterior.distance(center))
        grid_j = int(math.floor(d_min / fil_width - 0.5 + 1e-9))
        if grid_j >= 0:
            # ALWAYS the outermost grid ring that fits: an off-grid
            # edge-hugging wall would land within half a bead of the grid
            # ring below it (visibly "too close"), and suppressing that
            # ring instead leaves a visibly skipped line. Staying on the
            # grid keeps ring spacing uniform; the rim sits at most half a
            # bead further in at unlucky dimensions.
            return (grid_j + 0.5) * fil_width
        wall = d_min - fil_width / 2.0
        return wall if wall > fil_width * 0.25 else None
    return None


def _frame_spiral_bounds(
    frame: tuple[float, float, float, float],
    material_bounds: tuple[float, float, float, float],
    fil_width: float,
) -> tuple[float, float, float, float]:
    """Outer bounds for a rectangular spiral, on the FRAME's loop family.

    The spiral's loops live on the family "frame inset by half + k*fil per
    side" so every layer (and every split sibling) walks the same
    rectangles and walls stack. Loops that enclose this layer's material
    with more than half a bead of margin on EVERY side are pure travel:
    skip them by starting k0 loops in. Returns the frame shrunk by k0*fil
    per side (still on the family).
    """
    frame_left, frame_bottom, frame_right, frame_top = frame
    mat_left, mat_bottom, mat_right, mat_top = material_bounds
    half = fil_width / 2.0
    # Margin between the base loop (frame inset by half) and the material.
    min_margin = min(
        mat_left - (frame_left + half),
        mat_bottom - (frame_bottom + half),
        (frame_right - half) - mat_right,
        (frame_top - half) - mat_top,
    )
    skip = max(0, int(math.ceil((min_margin - half) / fil_width - 1e-9)))
    # Never shrink past the material's own footprint.
    max_skip_x = (frame_right - frame_left - (mat_right - mat_left)) / (2.0 * fil_width)
    max_skip_y = (frame_top - frame_bottom - (mat_top - mat_bottom)) / (2.0 * fil_width)
    skip = min(skip, max(0, int(min(max_skip_x, max_skip_y))))
    inset = skip * fil_width
    return (
        frame_left + inset,
        frame_bottom + inset,
        frame_right - inset,
        frame_top - inset,
    )


def _circle_ring_radii(
    motion: MultiPolygon,
    center_x: float,
    center_y: float,
    fil_width: float,
) -> tuple[list[float], tuple[float, ...]]:
    """Ring radii for one layer, outermost first: perimeter walls + grid fill.

    The outermost revolution is a PERIMETER WALL hugging the layer's material
    edge (its farthest boundary distance minus half a bead), so the printed
    silhouette follows the shape smoothly instead of staircasing by whole
    grid steps. If the layer has a central hole, a matching inner wall hugs
    the hole edge. The fill between the walls comes from a global grid —
    ring j at (j + 1/2) * fil_width from the frame centre, the same grid on
    every layer (and every split sibling) so interior rings stack vertically.
    Grid rings that would overlap a wall bead, or whose circle cannot cross
    the layer's material at all, are skipped instead of traveled.

    Returns (radii outermost-first, wall radii). Walls always dispense even
    under partial infill, like contour tracing.
    """
    if motion is None or motion.is_empty:
        return [], ()

    max_dist = 0.0
    for polygon in motion.geoms:
        for ring in (polygon.exterior, *polygon.interiors):
            for x, y in ring.coords:
                max_dist = max(max_dist, math.hypot(x - center_x, y - center_y))
    if max_dist <= 0.0:
        return [], ()
    min_dist = float(motion.distance(Point(center_x, center_y)))

    pitch = max(float(fil_width), 1e-9)
    half = pitch / 2.0

    outer_wall = max_dist - half
    if outer_wall <= EPS:
        # Material thinner than one bead (e.g. the dome cap): one tiny ring
        # through the middle of it so the layer still gets motion.
        radius = max(max_dist / 2.0, EPS)
        return [radius], (radius,)

    walls = [outer_wall]
    inner_wall: float | None = None
    if min_dist > pitch / 4.0:
        candidate = min_dist + half
        if candidate <= outer_wall - half:
            inner_wall = candidate
            walls.append(candidate)

    grid_hi = outer_wall - half
    grid_lo = (inner_wall + half) if inner_wall is not None else max(min_dist - half, 0.0)
    j_hi = int(math.floor(grid_hi / pitch - 0.5 + 1e-9))
    j_lo = max(0, int(math.ceil(grid_lo / pitch - 0.5 - 1e-9)))

    radii = [outer_wall]
    if j_hi >= j_lo:
        radii.extend((j + 0.5) * pitch for j in range(j_hi, j_lo - 1, -1))
    if inner_wall is not None:
        radii.append(inner_wall)
    return radii, tuple(walls)


def _circle_rings_polyline(
    center_x: float,
    center_y: float,
    radii: list[float],
    pitch: float,
) -> list[tuple[float, float]]:
    """Concentric-ring "spiral": one full circle per radius, in list order.

    Each revolution stays at a CONSTANT radius (so the printed walls are true
    smooth circles); consecutive rings are joined by a radial jump at theta 0,
    which the caller classifies as valve-off travel.
    """
    pitch = max(float(pitch), 1e-9)
    points: list[tuple[float, float]] = []
    for radius in radii:
        if radius <= 0.0:
            continue
        # Sample roughly one pitch of arc length per step, at least 20/ring.
        d_theta = min(math.pi / 10.0, pitch / max(radius, pitch))
        steps = max(8, int(math.ceil((2.0 * math.pi) / d_theta)))
        for index in range(steps + 1):
            theta = (2.0 * math.pi) * index / steps
            points.append(
                (
                    center_x + (radius * math.cos(theta)),
                    center_y + (radius * math.sin(theta)),
                )
            )
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


def _last_motion_reference(output_list: list[dict]) -> tuple[float, float, float, float]:
    """Endpoint and direction of the last XY move, regardless of valve state.

    Used instead of `_last_print_reference` when shapes share a reference
    motion path: anchoring contours off the valve state would give every
    shape a different contour tour, breaking the shared path.
    """
    x = y = 0.0
    last_x = last_y = 0.0
    last_dx = last_dy = 0.0
    for move in output_list:
        dx = float(move.get("X", 0.0))
        dy = float(move.get("Y", 0.0))
        x += dx
        y += dy
        if "Z" not in move and (dx != 0.0 or dy != 0.0):
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
            delta_x, delta_y = _centering_delta(stack, reference)
        else:
            delta_x = delta_y = 0.0

        if stack.contour_paths is not None:
            # Split pieces carry seam-free contour paths computed from the
            # parent shape's outline; use them directly.
            for layer_number in range(min(n_layers, len(stack.contour_paths))):
                contours = [
                    [(x + delta_x, y + delta_y) for x, y in path]
                    for path in stack.contour_paths[layer_number]
                    if len(path) >= 2
                ]
                if contours:
                    contour_layers[layer_number].append(
                        {
                            "owner_idx": source.owner_idx,
                            "contours": contours,
                        }
                    )
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
    shared_motion: bool = False,
) -> tuple[float, float]:
    # `origin_x/origin_y` is the world position of the move list's start:
    # _last_print_reference sums relative moves from zero, so its result must
    # be shifted back into the world frame the contours live in.
    #
    # With `shared_motion` (reference-stack printing) every shape must trace
    # EVERY source's contours in the same order so all parallel heads follow
    # one identical path; the valve opens only on the active owner's contours.
    # Anchoring then uses the last motion move instead of the last print, and
    # the trailing-travel rewind is skipped — both depend on how the raster
    # moves were split at valve boundaries, which differs per shape and would
    # desynchronise the shared path.
    if layer_number >= len(contour_layers):
        return current_x, current_y

    active_sources = [
        source
        for source in contour_layers[layer_number]
        if (shared_motion or source.get("owner_idx") == active_owner_idx)
        and any(len(contour) >= 2 for contour in source.get("contours", []))
    ]
    if not active_sources:
        return current_x, current_y

    if not shared_motion:
        current_x, current_y = _rewind_trailing_travel(
            output_list,
            current_x,
            current_y,
        )

    use_infill_reference = True

    for source in active_sources:
        if shared_motion and source.get("owner_idx") != active_owner_idx:
            color = 0
        else:
            color = 255
        for contour in source.get("contours", []):
            if len(contour) < 2:
                continue
            if use_infill_reference:
                reference = (
                    _last_motion_reference(output_list)
                    if shared_motion
                    else _last_print_reference(output_list)
                )
                nearest_x, nearest_y, approach_dx, approach_dy = reference
                nearest_x += origin_x
                nearest_y += origin_y
            else:
                nearest_x, nearest_y = current_x, current_y
                approach_dx, approach_dy = 0.0, 0.0
            is_closed = len(contour) >= 4 and contour[0] == contour[-1]
            if is_closed:
                contour = _rotate_closed_contour_to_nearest_border(
                    contour,
                    nearest_x,
                    nearest_y,
                    approach_dx,
                    approach_dy,
                )
                if contour[0] != contour[-1]:
                    contour = [*contour, contour[0]]
            else:
                # Open arc (a split piece's seam-free outline): approach the
                # nearer end and print to the other. Never close the loop —
                # the closing chord would run along the cut seam this path
                # deliberately excludes.
                if _point_distance_sq(
                    nearest_x, nearest_y, contour[-1][0], contour[-1][1]
                ) < _point_distance_sq(
                    nearest_x, nearest_y, contour[0][0], contour[0][1]
                ):
                    contour = list(reversed(contour))
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


LEAD_IN_DIRECTION_LEFT = "Left"
LEAD_IN_DIRECTION_RIGHT = "Right"
LEAD_IN_DIRECTION_UP = "Up"
LEAD_IN_DIRECTION_DOWN = "Down"
LEAD_IN_DIRECTION_CHOICES = (
    LEAD_IN_DIRECTION_LEFT,
    LEAD_IN_DIRECTION_RIGHT,
    LEAD_IN_DIRECTION_UP,
    LEAD_IN_DIRECTION_DOWN,
)
# Away-from-the-part axis and the lateral line-stepping axis per direction.
_LEAD_IN_AXES = {
    LEAD_IN_DIRECTION_LEFT: ((-1.0, 0.0), (0.0, 1.0)),
    LEAD_IN_DIRECTION_RIGHT: ((1.0, 0.0), (0.0, 1.0)),
    LEAD_IN_DIRECTION_UP: ((0.0, 1.0), (1.0, 0.0)),
    LEAD_IN_DIRECTION_DOWN: ((0.0, -1.0), (1.0, 0.0)),
}


def _normalize_lead_in_direction(direction: str | None) -> str:
    if direction in _LEAD_IN_AXES:
        return direction
    return LEAD_IN_DIRECTION_LEFT


def _lead_in_moves(
    enabled: bool,
    length: float,
    clearance: float,
    line_count: int,
    line_spacing: float,
    print_color: int,
    off_color: int,
    direction: str | None = LEAD_IN_DIRECTION_LEFT,
) -> list[dict]:
    """Purge patch printed before layer 1, in the chosen direction.

    The patch sits `clearance` away from the toolpath start along the purge
    direction and snakes `line_count` strokes of `length`, stepping one
    `line_spacing` laterally between strokes. The return route exits the
    patch one lateral step to the OUTSIDE and comes home through virgin
    ground, so the freshly primed nozzle never drags back across the wet
    purge lines.
    """
    if not enabled:
        return []
    lead_length = max(0.0, float(length))
    if lead_length <= 0.0:
        return []
    lead_clearance = max(0.0, float(clearance))
    pass_count = max(1, int(line_count))
    spacing = max(0.0, float(line_spacing))
    away, lateral = _LEAD_IN_AXES[_normalize_lead_in_direction(direction)]

    moves: list[dict] = []
    # Patch-local frame: `a` runs along the away axis, `v` along the lateral.
    current_a = 0.0
    current_v = 0.0

    def append_move(delta_a: float, delta_v: float, color: int) -> None:
        nonlocal current_a, current_v
        if delta_a == 0.0 and delta_v == 0.0:
            return
        moves.append(
            {
                "X": delta_a * away[0] + delta_v * lateral[0],
                "Y": delta_a * away[1] + delta_v * lateral[1],
                "Color": color,
            }
        )
        current_a += delta_a
        current_v += delta_v

    append_move(lead_clearance + lead_length, 0.0, off_color)
    stroke = -1.0  # first stroke prints back toward the part
    for pass_index in range(pass_count):
        append_move(stroke * lead_length, 0.0, print_color)
        stroke *= -1.0
        if pass_index < pass_count - 1:
            append_move(0.0, spacing, off_color)

    # Return: step one spacing outside the patch laterally, travel home
    # through the clearance lane, then step back onto the start point.
    if spacing > 0.0:
        append_move(0.0, -(current_v + spacing), off_color)
        append_move(-current_a, 0.0, off_color)
        append_move(0.0, -current_v, off_color)
    else:
        append_move(-current_a, -current_v, off_color)
    return moves


def plan_layer_moves(
    motion_layers: list[MultiPolygon],
    valve_layers: list[MultiPolygon],
    fil_width: float,
    layer_height: float,
    raster_pattern: str,
    contour_layers: list[list[dict]] | None = None,
    active_contour_owner: int | None = None,
    shared_motion: bool = False,
    scan_frame: tuple[float, float, float, float] | None = None,
    infill_fraction: float = 1.0,
    extra_wall_radii: list[list[float]] | None = None,
    ring_center: tuple[float, float] | None = None,
    motion_infill_fractions: list[float] | None = None,
) -> tuple[list[dict], tuple[float, float]]:
    """Assemble per-layer segments into a relative move list for all patterns.

    `scan_frame` (an XY box) pins the rasters' scanlines to a global grid so
    lines stack across layers and across split pieces, and provides the
    rotation pivot for diagonal layers.

    `infill_fraction` < 1 skips dispensing on evenly-distributed lines (grid
    lines for axis rasters, rings for the circle spiral). Lines that NO head
    prints are dropped from the motion entirely: `motion_infill_fractions`
    lists every shape sharing the motion (pass the SAME list to every shape,
    or the shared paths diverge). When omitted, a SOLO shape's own fraction
    bounds its motion, while shared motion is never restricted (the other
    heads' infill is unknown). The rectangular spiral keeps its full
    continuous walk (a skipped loop would break the spiral into
    disconnected rectangles).

    Returns the move list and the toolpath origin — the world position the
    relative moves start from (the first segment start of the first non-empty
    layer). Callers need it to map relative G-code back into world space.
    """
    raster_pattern = _normalize_raster_pattern(raster_pattern)
    infill_keep = _infill_line_keep(infill_fraction)
    if motion_infill_fractions is not None:
        motion_keep = _combined_infill_keep(motion_infill_fractions)
    elif shared_motion:
        motion_keep = None
    else:
        motion_keep = infill_keep
    if scan_frame is not None:
        anchor_x = _scan_anchor(scan_frame[0], scan_frame[2], fil_width)
        anchor_y = _scan_anchor(scan_frame[1], scan_frame[3], fil_width)
    else:
        anchor_x = anchor_y = None
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
            # Rings live on one global radii grid anchored at the frame
            # centre (ring j at (j + 1/2) * fil): every layer draws from the
            # same radii, so walls stack across layers instead of aliasing,
            # and rings that never touch this layer's material are skipped
            # instead of swept as full travel circles.
            frame = scan_frame if scan_frame is not None else motion.bounds
            if ring_center is not None:
                # Shared-motion whole shapes: rings centred where every
                # shape's centre was ALIGNED to (the reference align
                # centre), so each shape is concentric with the ring set —
                # a bbox-union centre can sit off it when shapes differ in
                # size, making rings graze boundaries.
                center_x, center_y = ring_center
            else:
                center_x = (frame[0] + frame[2]) / 2.0
                center_y = (frame[1] + frame[3]) / 2.0
            radii, wall_radii = _circle_ring_radii(motion, center_x, center_y, fil_width)

            # Shared-motion parallel printing: every shape's own wall radius
            # joins the ONE ring set (all heads travel all walls; each
            # dispenses only on its own), so every shape keeps a smooth,
            # complete outer circle instead of whatever grid ring happens to
            # graze its boundary.
            extra_walls = (
                extra_wall_radii[layer_number]
                if extra_wall_radii is not None and layer_number < len(extra_wall_radii)
                else []
            )
            if extra_walls:
                merged = sorted(set(radii) | set(extra_walls), reverse=True)
                radii = []
                for radius in merged:
                    if radius <= EPS:
                        continue
                    if radii and radii[-1] - radius < fil_width * 0.05:
                        continue  # near-duplicate wall/ring
                    radii.append(radius)
                wall_radii = tuple(sorted(set(wall_radii) | set(extra_walls), reverse=True))

            if motion_keep is not None:
                # Rings NO head dispenses on drop out of the motion (walls
                # always print, so they always stay).
                radii = [
                    radius
                    for radius in radii
                    if any(abs(radius - wall) <= fil_width * 0.25 for wall in wall_radii)
                    or motion_keep(max(0, int(round(radius / fil_width - 0.5))))
                ]

            points = _circle_rings_polyline(center_x, center_y, radii, fil_width)
            if layer_number % 2 == 1:
                points.reverse()

            # Dispense only ON a ring: the radial jump between revolutions
            # travels with the valve shut — otherwise every jump would
            # extrude a radial seam, worst on the outer wall. Vetoing whole
            # source segments by their radial change also kills the step
            # pieces the material boundary would otherwise split off.
            def keep_segment(x0: float, y0: float, x1: float, y1: float) -> bool:
                radius_0 = math.hypot(x0 - center_x, y0 - center_y)
                radius_1 = math.hypot(x1 - center_x, y1 - center_y)
                if abs(radius_1 - radius_0) > fil_width * 0.25:
                    return False  # radial jump between rings: travel only
                radius_mid = (radius_0 + radius_1) / 2.0
                if infill_keep is None:
                    return True
                if any(abs(radius_mid - wall) <= fil_width * 0.25 for wall in wall_radii):
                    return True  # perimeter walls always print, like contours
                ring = round(radius_mid / fil_width - 0.5)
                return infill_keep(max(0, int(ring)))

            segments = _classify_polyline(points, valve, keep_segment=keep_segment)
            # A ring grazing a faceted boundary flickers in/out of material,
            # printing sub-bead dashes (the spotty outer ring); real fill
            # arcs — a square's corner fill, a triangle's lobes — are far
            # longer and must stay. Close the valve over runs too short to
            # deposit cleanly; the motion is untouched.
            segments = _drop_short_print_runs(segments, fil_width * 1.5)
        elif raster_pattern == RASTER_PATTERN_RECTANGULAR_SPIRAL:
            # Anchor the loop family to the FRAME (constant across layers,
            # shapes, and split pieces), not each layer's own material
            # bounds: layers with smaller footprints would otherwise spiral
            # at their own offsets and the walls would not stack. Outer
            # loops that enclose this layer's material with more than half a
            # bead to spare are skipped instead of traveled.
            frame = scan_frame if scan_frame is not None else motion.bounds
            spiral_bounds = _frame_spiral_bounds(frame, motion.bounds, fil_width)
            points = _rectangular_spiral_polyline(
                spiral_bounds,
                fil_width,
                reverse=layer_number % 2 == 1,
            )
            keep_point = None
            if infill_keep is not None:
                # A spiral "line" is one ring; index rings from the FRAME's
                # base loop so the selection lines up across layers.
                min_x, min_y, max_x, max_y = frame
                half = fil_width / 2.0
                left, right = min_x + half, max_x - half
                bottom, top = min_y + half, max_y - half

                def keep_point(x: float, y: float) -> bool:
                    inset = min(x - left, right - x, y - bottom, top - y)
                    return infill_keep(max(0, int(round(inset / fil_width))))

            segments = _classify_polyline(points, valve, keep_point=keep_point)
        elif (
            raster_pattern == RASTER_PATTERN_DIAGONAL_WOODPILE
            and _diagonal_layer_angle(layer_number) is not None
        ):
            angle = _diagonal_layer_angle(layer_number)
            frame = scan_frame if scan_frame is not None else motion.bounds
            segments = _rotated_raster_segments(
                motion,
                valve,
                fil_width,
                angle,
                current_x,
                current_y,
                prefer_default=not raster_origin_initialized,
                frame=frame,
                infill_keep=infill_keep,
                motion_keep=motion_keep,
            )
        else:
            raster_axis = _raster_axis_for_pattern(raster_pattern, layer_number)
            # Axis "X" sweeps along X with rows stacked in Y, so its
            # scanlines use the Y anchor (and vice versa).
            axis_anchor = anchor_x if raster_axis == "Y" else anchor_y
            segments = _oriented_axis_raster_segments(
                motion,
                valve,
                fil_width,
                raster_axis,
                current_x,
                current_y,
                prefer_default=not raster_origin_initialized,
                scan_anchor=axis_anchor,
                infill_keep=infill_keep,
                motion_keep=motion_keep,
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
                shared_motion,
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
            shared_motion,
        )
        current_x, current_y = _append_relative_move(
            gcode_list,
            current_x,
            current_y,
            layer_end_x,
            layer_end_y,
            0,
        )

    return gcode_list, (origin_x, origin_y)


def _stack_center(stack: LayerStack) -> tuple[float, float]:
    (x_min, y_min, _z_min), (x_max, y_max, _z_max) = stack.bounds
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def _alignment_center(stack: LayerStack) -> tuple[float, float]:
    """The point a stack is centred by: its multi-material group frame's
    centre when it belongs to a group, else its own bbox centre. Group
    members share one frame, so they all get the same delta and keep their
    modelled positions relative to each other."""
    if stack.align_frame is not None:
        x_min, y_min, x_max, y_max = stack.align_frame
        return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
    return _stack_center(stack)


def _snap_to_grid(value: float, grid: float | None) -> float:
    if not grid or grid <= 0.0:
        return value
    # Half-up, not Python's banker's rounding: values that differ by exact
    # grid multiples must snap to results that differ by the same multiples
    # (banker's rounds exact halves toward even, breaking that translation
    # invariance and with it uniform split-piece spacing).
    return math.floor(value / grid + 0.5 + 1e-9) * grid


def _centering_delta(stack: LayerStack, reference: LayerStack) -> tuple[float, float]:
    """Translation that aligns `stack` into `reference`'s shared frame.

    Uses the alignment target and snap grid stamped on the reference by
    `build_reference_stack`, so re-deriving the delta later reproduces the
    exact translation the reference was built with. Snapping the delta to the
    fil grid keeps every shape's world scan-grid phase intact, so split
    pieces printed with shared reference motion still tile at an exact
    one-fil-width line pitch.

    Split siblings (stacks sharing the reference's scan frame) are aligned by
    their cell corner within that frame instead of their centre: with cells
    sized in whole grid multiples the deltas — and with them the required
    nozzle spacing — come out uniform across all pieces, whereas snapping the
    centres would wobble by up to one fil where the last cell's width (and so
    its centre phase) differs.

    Multi-material group members (stacks carrying a shared `align_frame`)
    are aligned by the group frame's centre instead of their own bbox
    centre: every member gets the same delta, so the group moves as one
    rigid unit and parts keep their modelled relative positions.
    """
    grid = reference.align_grid
    if (
        grid
        and stack.scan_frame is not None
        and stack.scan_frame == reference.scan_frame
    ):
        (stack_min_x, stack_min_y, _sz), _stack_max = (
            stack.bounds[0],
            stack.bounds[1],
        )
        frame_min_x, frame_min_y = stack.scan_frame[0], stack.scan_frame[1]
        return (
            _snap_to_grid(frame_min_x - stack_min_x, grid),
            _snap_to_grid(frame_min_y - stack_min_y, grid),
        )

    if reference.align_center is not None:
        reference_x, reference_y = reference.align_center
    else:
        reference_x, reference_y = _stack_center(reference)
    center_x, center_y = _alignment_center(stack)
    # No grid snap here: whole shapes centre EXACTLY on the align centre
    # (circle-spiral rings are centred there, and a snapped residue of up
    # to half a fil would make rings graze the shape's boundary). The snap
    # only ever mattered for split pieces, which take the scan-frame corner
    # rule above.
    return (reference_x - center_x, reference_y - center_y)


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


def build_reference_stack(
    stacks: list[LayerStack | None],
    grid: float | None = None,
) -> LayerStack | None:
    """Union all shapes into one shared motion stack, alignment-centres aligned.

    Vector analog of the old centered "black wins" TIFF merge: every stack is
    translated so its alignment centre (its own XY bbox centre, or its
    multi-material group frame's centre — see `_alignment_center`) lands on
    the first stack's alignment centre, then each layer is the union of the
    translated layers. Multi-material group members share one frame, so the
    group translates as a rigid unit and its parts keep their modelled
    relative positions; the group holding the first stack does not move at
    all. Group members should be sliced on one common Z grid so layer indices
    line up; a part that starts higher simply contributes nothing to the
    lower layers.

    `grid` (the fil width) snaps each translation to grid multiples so every
    shape's world scan-grid phase survives the alignment — required for split
    pieces to keep an exact one-fil line pitch under shared reference motion.
    The alignment target and grid are stamped on the result so later
    `_centering_delta` calls reproduce the exact same translations.
    """
    valid = [stack for stack in stacks if stack is not None and stack.layers]
    if not valid:
        return None

    n_layers = max(len(stack.layers) for stack in valid)
    reference_x, reference_y = _alignment_center(valid[0])
    target = LayerStack(
        layers=[],
        z_values=[],
        bounds=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        layer_height=valid[0].layer_height,
        scan_frame=valid[0].scan_frame,
        align_center=(reference_x, reference_y),
        align_grid=grid,
    )

    layer_parts: list[list[MultiPolygon]] = [[] for _ in range(n_layers)]
    x_min = y_min = z_min = math.inf
    x_max = y_max = z_max = -math.inf

    for stack in valid:
        delta_x, delta_y = _centering_delta(stack, target)

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

    # The first stack's delta is zero (snap(0) == 0), so its scan frame is
    # already in the shared frame and anchors the reference's scan grid.
    return LayerStack(
        layers=layers,
        z_values=z_values,
        bounds=((x_min, y_min, z_min), (x_max, y_max, z_max)),
        layer_height=valid[0].layer_height,
        name="reference",
        scan_frame=valid[0].scan_frame,
        align_center=(reference_x, reference_y),
        align_grid=grid,
    )


def _base_split_edges(
    lo: float,
    hi: float,
    count: int,
    grid: float | None = None,
) -> list[float]:
    """Cell edges for one axis: equal cells, padded to whole grid multiples.

    With `grid` (the fil width), every cell gets the SAME width — the extent
    divided by `count`, rounded UP to a whole number of grid units — and the
    leftover becomes blank margin split evenly outside the shape's outer
    edges. This is the vector analog of the old pixel splitter's padded
    canvas: identical piece sizes, so the required nozzle spacing is one cell
    everywhere, including under shared reference motion.
    """
    extent = hi - lo
    if grid and grid > 0.0 and count > 1 and extent > 0.0:
        units = max(1, math.ceil(extent / (count * grid) - 1e-9))
        cell = units * grid
        pad = cell * count - extent
        start = lo - pad / 2.0
        return [start + index * cell for index in range(count + 1)]
    return [lo + index * extent / count for index in range(count + 1)]


def _shifted_split_edges(
    edges: list[float],
    layer_index: int,
    overlap: float,
) -> list[float]:
    """Per-layer cell edges; interior ones alternate by ±overlap per layer."""
    count = len(edges) - 1
    if overlap <= 0.0 or count <= 1:
        return list(edges)

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


def _linework_to_paths(geometry: object) -> list[list[tuple[float, float]]]:
    """Merge linework into maximal polylines, ordered/oriented
    deterministically so shapes sharing reference motion trace them
    identically. Open paths allowed; closed rings keep first == last."""
    pieces = list(_iter_linestrings(geometry))
    if not pieces:
        return []
    merged = linemerge(MultiLineString(pieces)) if len(pieces) > 1 else pieces[0]

    paths: list[list[tuple[float, float]]] = []
    for line in _iter_linestrings(merged):
        coords = [(float(x), float(y)) for x, y in line.coords]
        if len(coords) < 2 or LineString(coords).length <= EPS:
            continue
        # Normalize open-path orientation (closed rings keep first == last).
        if coords[0] != coords[-1] and coords[-1] < coords[0]:
            coords.reverse()
        paths.append(coords)

    paths.sort(
        key=lambda path: (
            min(point[1] for point in path),
            min(point[0] for point in path),
            -len(path),
        )
    )
    return paths


def _clip_contour_paths(
    source_lines: object,
    cell: object,
) -> list[list[tuple[float, float]]]:
    """Contour polylines of a split piece: the parent outline inside the cell.

    Clipping the PARENT's boundary linework (instead of taking the piece
    polygon's own boundary) excludes the cut seams between sibling pieces —
    only the true outer surface remains.
    """
    return _linework_to_paths(source_lines.intersection(cell))


def group_contour_paths(
    member: LayerStack,
    siblings: list[LayerStack],
    tolerance: float,
) -> list[list[list[tuple[float, float]]]]:
    """Per-layer seam-free contour polylines for one multi-material member.

    Parts sharing a nozzle assemble into ONE shape, so where a member's
    boundary meets a sibling material — or comes within `tolerance` of it
    (fit gaps between materials) — that edge is an internal interface, not a
    printable surface. Only the member boundary on the assembly's true
    outside is kept; a member fully embedded in the assembly gets no
    contours at all. Members should share one Z grid so layer indices align.
    """
    tolerance = max(float(tolerance), EPS)
    contour_paths: list[list[list[tuple[float, float]]]] = []
    for layer_number, layer in enumerate(member.layers):
        if layer is None or layer.is_empty:
            contour_paths.append([])
            continue
        sibling_parts = [
            sibling.layers[layer_number]
            for sibling in siblings
            if layer_number < len(sibling.layers)
            and sibling.layers[layer_number] is not None
            and not sibling.layers[layer_number].is_empty
        ]
        boundary = layer.boundary
        if sibling_parts:
            boundary = boundary.difference(unary_union(sibling_parts).buffer(tolerance))
        contour_paths.append(_linework_to_paths(boundary))
    return contour_paths


def split_layer_stack_grid(
    stack: LayerStack,
    columns: int,
    rows: int,
    overlapping_layers: bool = False,
    overlap: float = 0.0,
    grid: float | None = None,
    frame: tuple[float, float, float, float] | None = None,
) -> list[LayerStack]:
    """Split a sliced shape into a rows x columns grid of piece stacks.

    Pieces are returned row-major with row 1 the top strip (max-Y side),
    matching the legacy image-grid ordering. With `overlapping_layers`, the
    interior cut lines alternate by ±overlap between layers so neighbouring
    pieces interlock. `grid` (the fil width) sizes the cells in whole grid
    multiples (see `_base_split_edges`). Piece `bounds` are the nominal
    (un-shifted) cell boxes.

    `frame` overrides the XY box the cell grid is computed over. Splitting
    every member of a multi-material group with the group's combined bounds
    as the frame clips all materials by the SAME cells (and one shared scan
    frame), so cell-mates assemble exactly.
    """
    columns = max(1, int(columns))
    rows = max(1, int(rows))
    (x_min, y_min, z_min), (x_max, y_max, z_max) = stack.bounds
    if frame is not None:
        x_min, y_min, x_max, y_max = (float(value) for value in frame)

    overlap_x = overlap if (overlapping_layers and columns > 1) else 0.0
    overlap_y = overlap if (overlapping_layers and rows > 1) else 0.0

    base_x_edges = _base_split_edges(x_min, x_max, columns, grid)
    base_y_edges = _base_split_edges(y_min, y_max, rows, grid)
    layer_x_edges = [
        _shifted_split_edges(base_x_edges, index, overlap_x)
        for index in range(len(stack.layers))
    ]
    layer_y_edges = [
        _shifted_split_edges(base_y_edges, index, overlap_y)
        for index in range(len(stack.layers))
    ]

    # Pieces inherit the parent's scan frame so they all raster on ONE
    # continuous line grid: the assembled seams then keep an exact
    # one-fil-width pitch instead of each piece re-centring its own lines.
    scan_frame = stack.scan_frame or (x_min, y_min, x_max, y_max)

    base_name = stack.name or "shape"
    pieces: list[LayerStack] = []
    for row_index in range(1, rows + 1):
        # Row 1 is the top strip: count y-cells down from the max edge.
        y_cell = rows - row_index
        for col_index in range(1, columns + 1):
            x_cell = col_index - 1

            layers: list[MultiPolygon] = []
            contour_paths: list[list[list[tuple[float, float]]]] = []
            for layer_number, layer in enumerate(stack.layers):
                if layer is None or layer.is_empty:
                    layers.append(MultiPolygon())
                    contour_paths.append([])
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

                # Contours come from the parent's outline (or, when
                # re-splitting a piece, its already-seam-free paths) so the
                # cut lines between siblings are never traced.
                if stack.contour_paths is not None:
                    parent_paths = (
                        stack.contour_paths[layer_number]
                        if layer_number < len(stack.contour_paths)
                        else []
                    )
                    source_lines = MultiLineString(
                        [path for path in parent_paths if len(path) >= 2]
                    )
                else:
                    source_lines = layer.boundary
                contour_paths.append(_clip_contour_paths(source_lines, cell))

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
                    scan_frame=scan_frame,
                    contour_paths=contour_paths,
                )
            )

    return pieces
