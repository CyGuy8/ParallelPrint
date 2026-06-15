from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


# A move is any G0/G1 (or G00/G01) line. Coordinates may list any subset of
# X/Y/Z in any order, mixed with other tokens (F feed rate, E extrusion); only
# the axes named on a line change. This matches standard slicer/firmware G-code
# as well as this app's own always-paired "X Y" output.
_CMD_RE = re.compile(r"^G0*([01])(?![0-9])", re.IGNORECASE)
_AXIS_RE = re.compile(r"([XYZ])\s*([-+]?(?:\d*\.\d+|\d+\.?))", re.IGNORECASE)
# Pneumatic valve toggle: WAGO_ValveCommands(<valve>, <0=close|1=open>). Some
# generators emit every move as G1 and convey extrusion only through the valve.
_VALVE_RE = re.compile(r"WAGO_ValveCommands\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE)


def parse_gcode_path(gcode_text: str) -> dict:
    relative = True
    x = y = z = 0.0

    # Decide how to tell print from travel. The valve physically controls
    # material flow, so when valve commands are present they are the ground
    # truth: valve open = printing, valve closed = travel. This is correct for
    # the app's own output (where valve state and G1/G0 agree) and for external
    # generators whose G0/G1 labels are unreliable — some omit G0 entirely
    # (every move G1), others invert G0/G1 relative to the valve. Only fall back
    # to the G1 = print / G0 = travel convention when there is no valve to read.
    use_valve = bool(_VALVE_RE.search(gcode_text))
    open_valves: set[str] = set()

    print_segments: list[list[tuple[float, float, float]]] = []
    travel_segments: list[list[tuple[float, float, float]]] = []
    moves: list[dict] = []
    current_kind: str | None = None
    current_segment: list[tuple[float, float, float]] = []

    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = []

    def flush_segment() -> None:
        nonlocal current_segment, current_kind
        if current_segment and current_kind is not None:
            target = print_segments if current_kind == "print" else travel_segments
            target.append(current_segment)
        current_segment = []
        current_kind = None

    for raw_line in gcode_text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_segment()
            continue

        # Drop inline comments so axis letters in comment text are never read.
        code = line.split(";", 1)[0].strip()
        upper = code.upper()
        if upper.startswith("G90"):
            relative = False
            continue
        if upper.startswith("G91"):
            relative = True
            continue

        cmd_match = _CMD_RE.match(code)
        if not cmd_match:
            # Track valve open/close so all-G1 files can be split into
            # print (valve open) and travel (valve closed) runs.
            valve_match = _VALVE_RE.search(code)
            if valve_match:
                valve, state = valve_match.group(1), valve_match.group(2)
                if state == "0":
                    open_valves.discard(valve)
                else:
                    open_valves.add(valve)
            flush_segment()
            continue

        axes = {a.upper(): float(v) for a, v in _AXIS_RE.findall(code)}
        if not axes:
            # A G0/G1 with no coordinates (e.g. "G1 F1800") is not a move.
            flush_segment()
            continue

        prev_pos = (x, y, z)

        if relative:
            x += axes.get("X", 0.0)
            y += axes.get("Y", 0.0)
            z += axes.get("Z", 0.0)
        else:
            if "X" in axes:
                x = axes["X"]
            if "Y" in axes:
                y = axes["Y"]
            if "Z" in axes:
                z = axes["Z"]

        if use_valve:
            kind = "print" if open_valves else "travel"
        else:
            kind = "print" if cmd_match.group(1) == "1" else "travel"
        moves.append({"kind": kind, "start": prev_pos, "end": (x, y, z)})

        if kind != current_kind:
            flush_segment()
            current_kind = kind
            current_segment = [prev_pos]

        current_segment.append((x, y, z))
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)

    flush_segment()

    if all_x:
        bounds = (
            (min(all_x), min(all_y), min(all_z)),
            (max(all_x), max(all_y), max(all_z)),
        )
    else:
        bounds = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    # Assign a layer index to every move. Layers are the distinct Z heights at
    # which printing (G1) happens; travel moves (including the Z lift between
    # layers) are attributed to the layer of the next print move so a layer's
    # timeline starts with the approach travel and ends with its last print.
    print_z = sorted({round(m["end"][2], 6) for m in moves if m["kind"] == "print"})
    z_to_layer = {z: i for i, z in enumerate(print_z)}
    next_print_layer = len(print_z) - 1 if print_z else 0
    for move in reversed(moves):
        if move["kind"] == "print":
            next_print_layer = z_to_layer[round(move["end"][2], 6)]
        move["layer"] = next_print_layer

    return {
        "print_segments": print_segments,
        "travel_segments": travel_segments,
        "moves": moves,
        "layer_count": len(print_z),
        "bounds": bounds,
        "point_count": len(all_x),
    }


def _move_length(move: dict) -> float:
    (x0, y0, z0), (x1, y1, z1) = move["start"], move["end"]
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)


def _chronological_trace_arrays(moves: list[dict]) -> dict:
    """Build per-kind polyline arrays with a shared time axis for animation.

    Each point gets a timestamp equal to the cumulative path length (print +
    travel) at which the nozzle reaches it, so the browser can reveal both
    traces in lockstep by slicing at a time cutoff. Gap markers (None) close a
    trace's polyline whenever the move kind switches and carry the timestamp
    of the segment they terminate.
    """
    arrays: dict[str, dict[str, list]] = {
        "print": {"x": [], "y": [], "z": [], "t": []},
        "travel": {"x": [], "y": [], "z": [], "t": []},
    }
    cum = 0.0
    prev_kind: str | None = None
    layer_end: dict[int, float] = {}

    for move in moves:
        trace = arrays[move["kind"]]
        if move["kind"] != prev_kind:
            if prev_kind is not None:
                prev_trace = arrays[prev_kind]
                prev_trace["x"].append(None)
                prev_trace["y"].append(None)
                prev_trace["z"].append(None)
                prev_trace["t"].append(cum)
            sx, sy, sz = move["start"]
            trace["x"].append(sx)
            trace["y"].append(sy)
            trace["z"].append(sz)
            trace["t"].append(cum)
            prev_kind = move["kind"]
        cum += _move_length(move)
        ex, ey, ez = move["end"]
        trace["x"].append(ex)
        trace["y"].append(ey)
        trace["z"].append(ez)
        trace["t"].append(cum)
        layer_end[move["layer"]] = cum

    layer_count = (max(layer_end) + 1) if layer_end else 0
    layer_t_end: list[float] = []
    last = 0.0
    for i in range(layer_count):
        last = max(last, layer_end.get(i, last))
        layer_t_end.append(last)

    return {
        "print": arrays["print"],
        "travel": arrays["travel"],
        "total_length": cum,
        "layer_t_end": layer_t_end,
    }


def _path_arrays(moves: list[dict]) -> dict:
    """Chronological nozzle positions with cumulative-length timestamps."""
    xs = [moves[0]["start"][0]]
    ys = [moves[0]["start"][1]]
    zs = [moves[0]["start"][2]]
    ts = [0.0]
    cum = 0.0
    for move in moves:
        cum += _move_length(move)
        ex, ey, ez = move["end"]
        xs.append(ex)
        ys.append(ey)
        zs.append(ez)
        ts.append(cum)
    return {"x": xs, "y": ys, "z": zs, "t": ts}


def _build_path_tube(
    moves: list[dict],
    radius: float,
    kind: str = "print",
    sides: int = 12,
    max_rings: int = 5000,
) -> dict:
    """Extrude a tube of physical radius along the moves of the given kind.

    Returns Mesh3d-ready vertex and face arrays. Rings are laid down in
    chronological order and long moves are subdivided, so faces can be
    revealed progressively by slicing the (sorted) per-face timestamps.
    """
    # Group consecutive moves of this kind into continuous runs (global times).
    runs: list[tuple[list, list]] = []
    cur_pts: list | None = None
    cur_ts: list | None = None
    cum = 0.0
    for move in moves:
        length = _move_length(move)
        if move["kind"] == kind:
            if cur_pts is None:
                cur_pts = [move["start"]]
                cur_ts = [cum]
            cur_pts.append(move["end"])
            cur_ts.append(cum + length)
        elif cur_pts is not None:
            runs.append((cur_pts, cur_ts))
            cur_pts = cur_ts = None
        cum += length
    if cur_pts is not None:
        runs.append((cur_pts, cur_ts))

    total_print = sum(ts[-1] - ts[0] for _pts, ts in runs)
    step = max(radius * 2.0, total_print / max_rings) if total_print > 0 else radius

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    fi: list[int] = []
    fj: list[int] = []
    fk: list[int] = []
    face_t: list[float] = []
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    for pts, ts in runs:
        # Subdivide long moves so the tube grows smoothly during playback.
        sub_p = [np.asarray(pts[0], dtype=float)]
        sub_t = [ts[0]]
        for a in range(len(pts) - 1):
            p0 = np.asarray(pts[a], dtype=float)
            p1 = np.asarray(pts[a + 1], dtype=float)
            seg = float(np.linalg.norm(p1 - p0))
            pieces = max(1, math.ceil(seg / step))
            for s in range(1, pieces + 1):
                f = s / pieces
                sub_p.append(p0 + (p1 - p0) * f)
                sub_t.append(ts[a] + (ts[a + 1] - ts[a]) * f)

        points = np.vstack(sub_p)
        n_rings = len(points)
        if n_rings < 2:
            continue

        # Per-ring tangents (averaged at interior points) and a perpendicular
        # frame; vertical tangents fall back to the X axis for the side vector.
        tangents = np.zeros_like(points)
        tangents[1:-1] = points[2:] - points[:-2]
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tangents /= norms

        side_vec = np.cross(tangents, np.array([0.0, 0.0, 1.0]))
        side_norm = np.linalg.norm(side_vec, axis=1)
        vertical = side_norm < 1e-6
        if vertical.any():
            side_vec[vertical] = np.cross(tangents[vertical], np.array([1.0, 0.0, 0.0]))
        side_vec /= np.maximum(np.linalg.norm(side_vec, axis=1, keepdims=True), 1e-12)
        up_vec = np.cross(side_vec, tangents)

        base = len(xs)
        rings = (
            points[:, None, :]
            + radius * (cos_a[None, :, None] * side_vec[:, None, :]
                        + sin_a[None, :, None] * up_vec[:, None, :])
        )
        flat = np.round(rings.reshape(-1, 3), 4)
        xs.extend(flat[:, 0].tolist())
        ys.extend(flat[:, 1].tolist())
        zs.extend(flat[:, 2].tolist())

        # Center vertices for the end caps that close the tube.
        cap_start = len(xs)
        xs.append(round(float(points[0][0]), 4))
        ys.append(round(float(points[0][1]), 4))
        zs.append(round(float(points[0][2]), 4))
        cap_end = len(xs)
        xs.append(round(float(points[-1][0]), 4))
        ys.append(round(float(points[-1][1]), 4))
        zs.append(round(float(points[-1][2]), 4))

        t_start = round(sub_t[0], 4)
        t_end = round(sub_t[-1], 4)

        # Start cap (fan around the first ring).
        for k in range(sides):
            k_next = (k + 1) % sides
            fi.append(cap_start)
            fj.append(base + k_next)
            fk.append(base + k)
            face_t.append(t_start)

        for r in range(n_rings - 1):
            r0 = base + r * sides
            r1 = r0 + sides
            t_face = round(sub_t[r + 1], 4)
            for k in range(sides):
                k_next = (k + 1) % sides
                fi.extend((r0 + k, r0 + k))
                fj.extend((r1 + k, r1 + k_next))
                fk.extend((r1 + k_next, r0 + k_next))
                face_t.extend((t_face, t_face))

        # End cap (fan around the last ring).
        last_ring = base + (n_rings - 1) * sides
        for k in range(sides):
            k_next = (k + 1) % sides
            fi.append(cap_end)
            fj.append(last_ring + k)
            fk.append(last_ring + k_next)
            face_t.append(t_end)

    return {"x": xs, "y": ys, "z": zs, "i": fi, "j": fj, "k": fk, "face_t": face_t}


def _segments_to_xyz(
    segments: list[list[tuple[float, float, float]]],
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for segment in segments:
        for px, py, pz in segment:
            xs.append(px)
            ys.append(py)
            zs.append(pz)
        xs.append(None)
        ys.append(None)
        zs.append(None)
    return xs, ys, zs


def build_toolpath_figure(
    parsed: dict,
    travel_opacity: float = 0.2,
    print_opacity: float = 1.0,
    travel_color: str = "#969696",
    print_color: str = "#1f77b4",
    print_width: float = 0.8,
    travel_width: float = 0.2,
    tube: bool = True,
) -> go.Figure:
    moves = parsed.get("moves") or []

    fig = go.Figure()
    meta = None

    def add_tube_trace(tube: dict, name: str, color: str, opacity: float) -> None:
        fig.add_trace(
            go.Mesh3d(
                x=tube["x"],
                y=tube["y"],
                z=tube["z"],
                i=tube["i"],
                j=tube["j"],
                k=tube["k"],
                color=color,
                opacity=opacity,
                name=name,
                showlegend=True,
                hoverinfo="skip",
                lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15, roughness=0.6),
            )
        )

    if moves and tube:
        chrono = _chronological_trace_arrays(moves)

        # Physical-width tubes along both paths: filament-like rendering whose
        # thickness scales with zoom (widths are diameters in mm).
        travel_tube = _build_path_tube(
            moves, radius=max(travel_width, 0.05) / 2.0, kind="travel"
        )
        if travel_tube["i"]:
            add_tube_trace(travel_tube, "Travel (G0)", travel_color, travel_opacity)

        print_tube = _build_path_tube(
            moves, radius=max(print_width, 0.05) / 2.0, kind="print"
        )
        if print_tube["i"]:
            add_tube_trace(print_tube, "Print (G1)", print_color, print_opacity)

        # Nozzle position marker, driven client-side during playback.
        end_x, end_y, end_z = moves[-1]["end"]
        fig.add_trace(
            go.Scatter3d(
                x=[end_x],
                y=[end_y],
                z=[end_z],
                mode="markers",
                name="Nozzle",
                marker=dict(size=5, color="#d62728"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        path = _path_arrays(moves)
        meta = {
            "animation": {
                "travel_face_t": travel_tube["face_t"],
                "print_face_t": print_tube["face_t"],
                "path_x": path["x"],
                "path_y": path["y"],
                "path_z": path["z"],
                "path_t": path["t"],
                "layer_t_end": chrono["layer_t_end"],
                "total_length": chrono["total_length"],
            }
        }
    else:
        travel_xs, travel_ys, travel_zs = _segments_to_xyz(parsed["travel_segments"])
        if travel_xs:
            fig.add_trace(
                go.Scatter3d(
                    x=travel_xs,
                    y=travel_ys,
                    z=travel_zs,
                    mode="lines",
                    name="Travel (G0)",
                    opacity=travel_opacity,
                    line=dict(color=travel_color, width=2),
                    hoverinfo="skip",
                )
            )
        print_xs, print_ys, print_zs = _segments_to_xyz(parsed["print_segments"])
        if print_xs:
            fig.add_trace(
                go.Scatter3d(
                    x=print_xs,
                    y=print_ys,
                    z=print_zs,
                    mode="lines",
                    name="Print (G1)",
                    opacity=print_opacity,
                    line=dict(color=print_color, width=4),
                    hovertemplate="X=%{x:.2f}<br>Y=%{y:.2f}<br>Z=%{z:.2f}<extra></extra>",
                )
            )

    (x_min, y_min, z_min), (x_max, y_max, z_max) = parsed["bounds"]
    fig.update_layout(
        meta=meta,
        height=700,
        uirevision="toolpath",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
        title=(
            f"Tool path — {len(parsed['print_segments'])} print / "
            f"{len(parsed['travel_segments'])} travel segments   "
            f"X[{x_min:.1f},{x_max:.1f}]  Y[{y_min:.1f},{y_max:.1f}]  "
            f"Z[{z_min:.1f},{z_max:.1f}]"
        ),
    )
    return fig


def build_parallel_figure(
    parts: list[dict],
    gap: float = 5.0,
    filament_width: float = 0.8,
    travel_width: float = 0.2,
    travel_opacity: float = 0.2,
    print_opacity: float = 1.0,
    tube: bool = True,
) -> go.Figure:
    """Render several parsed shapes side by side, offset along X so they don't
    overlap. `tube` True draws filament tubes with a shared-time animation
    timeline; False draws fast thin scatter lines (no animation).

    `parts` is a list of {"idx": int, "color": str, "parsed": dict}. Each part's
    print and travel traces (and, in tube mode, a nozzle marker) are named by idx
    so the client-side animation/recolor can address them.
    """
    fig = go.Figure()
    anim_parts: list[dict] = []
    total_length = 0.0
    rendered = False
    n_parts = 0
    bx0 = by0 = bz0 = float("inf")
    bx1 = by1 = bz1 = float("-inf")

    running_x = 0.0
    for part in parts:
        idx = part["idx"]
        color = part["color"]
        parsed = part["parsed"]
        moves = parsed.get("moves") or []
        if not moves:
            continue

        (pxmin, pymin, pzmin), (pxmax, pymax, pzmax) = parsed["bounds"]
        width = pxmax - pxmin
        x_off = running_x - pxmin
        running_x += width + gap

        if tube:
            print_tube = _build_path_tube(moves, radius=max(filament_width, 0.05) / 2.0, kind="print")
            travel_tube = _build_path_tube(moves, radius=max(travel_width, 0.05) / 2.0, kind="travel")
            path = _path_arrays(moves)

            px = [v + x_off for v in print_tube["x"]]
            tx = [v + x_off for v in travel_tube["x"]]
            path_x = [v + x_off for v in path["x"]]

            if travel_tube["i"]:
                fig.add_trace(
                    go.Mesh3d(
                        x=tx, y=travel_tube["y"], z=travel_tube["z"],
                        i=travel_tube["i"], j=travel_tube["j"], k=travel_tube["k"],
                        color=color, opacity=travel_opacity, name=f"Travel {idx}",
                        showlegend=False, hoverinfo="skip",
                        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.1, roughness=0.6),
                    )
                )
            if print_tube["i"]:
                fig.add_trace(
                    go.Mesh3d(
                        x=px, y=print_tube["y"], z=print_tube["z"],
                        i=print_tube["i"], j=print_tube["j"], k=print_tube["k"],
                        color=color, opacity=print_opacity, name=f"Shape {idx}",
                        showlegend=True, hoverinfo="skip",
                        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15, roughness=0.6),
                    )
                )
            fig.add_trace(
                go.Scatter3d(
                    x=[path_x[-1]], y=[path["y"][-1]], z=[path["z"][-1]],
                    mode="markers", name=f"Nozzle {idx}",
                    marker=dict(size=4, color=color), showlegend=False, hoverinfo="skip",
                )
            )

            part_total = path["t"][-1] if path["t"] else 0.0
            total_length = max(total_length, part_total)
            anim_parts.append({
                "printName": f"Shape {idx}",
                "travelName": f"Travel {idx}",
                "nozzleName": f"Nozzle {idx}",
                "print_face_t": print_tube["face_t"],
                "travel_face_t": travel_tube["face_t"],
                "path_x": path_x, "path_y": path["y"], "path_z": path["z"], "path_t": path["t"],
            })
        else:
            t_xs, t_ys, t_zs = _segments_to_xyz(parsed["travel_segments"])
            p_xs, p_ys, p_zs = _segments_to_xyz(parsed["print_segments"])
            t_xs = [v + x_off if v is not None else None for v in t_xs]
            p_xs = [v + x_off if v is not None else None for v in p_xs]
            if t_xs:
                fig.add_trace(
                    go.Scatter3d(
                        x=t_xs, y=t_ys, z=t_zs, mode="lines", name=f"Travel {idx}",
                        opacity=travel_opacity, line=dict(color=color, width=2),
                        showlegend=False, hoverinfo="skip",
                    )
                )
            if p_xs:
                fig.add_trace(
                    go.Scatter3d(
                        x=p_xs, y=p_ys, z=p_zs, mode="lines", name=f"Shape {idx}",
                        opacity=print_opacity, line=dict(color=color, width=4),
                        showlegend=True, hoverinfo="skip",
                    )
                )

        rendered = True
        n_parts += 1
        bx0 = min(bx0, pxmin + x_off); bx1 = max(bx1, pxmax + x_off)
        by0 = min(by0, pymin); by1 = max(by1, pymax)
        bz0 = min(bz0, pzmin); bz1 = max(bz1, pzmax)

    if not rendered:
        fig.update_layout(height=700)
        return fig

    meta = {"animation": {"total_length": total_length, "parts": anim_parts}} if anim_parts else None
    pad = max(bx1 - bx0, by1 - by0, bz1 - bz0, 1.0) * 0.05
    fig.update_layout(
        meta=meta,
        height=700,
        uirevision="parallel",
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            xaxis_range=[bx0 - pad, bx1 + pad],
            yaxis_range=[by0 - pad, by1 + pad],
            zaxis_range=[bz0 - pad, bz1 + pad],
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
        title=f"Parallel print — {n_parts} part(s)",
    )
    return fig


def build_parallel_gif(
    parts: list[dict],
    out_path: str | Path,
    gap: float = 5.0,
    duration: float = 6.0,
    fps: int = 10,
    travel_opacity: float = 0.15,
    travel_color: str = "#9a9a9a",
    print_width: float = 4.0,
    travel_width: float = 1.5,
    elev: float = 22.0,
    azim: float = -60.0,
    progress_cb=None,
) -> Path | None:
    """Render the parallel print as an animated GIF using Matplotlib (CPU Agg
    backend — no WebGL/headless browser, works on Hugging Face).

    Each part's toolpath is drawn as growing colored lines (print solid, travel
    faint), three parts in parallel on a shared cumulative-length time axis.
    `parts` is a list of {"idx": int, "color": str, "parsed": dict}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    pdata: list[dict] = []
    running_x = 0.0
    total_length = 0.0
    bx0 = by0 = bz0 = float("inf")
    bx1 = by1 = bz1 = float("-inf")

    for part in parts:
        parsed = part["parsed"]
        moves = parsed.get("moves") or []
        if not moves:
            continue
        (pxmin, pymin, pzmin), (pxmax, pymax, pzmax) = parsed["bounds"]
        x_off = running_x - pxmin
        running_x += (pxmax - pxmin) + gap

        cum = 0.0
        mlist: list[tuple] = []
        for m in moves:
            s = (m["start"][0] + x_off, m["start"][1], m["start"][2])
            e = (m["end"][0] + x_off, m["end"][1], m["end"][2])
            seg_len = math.dist(s, e)
            mlist.append((m["kind"], s, e, cum, cum + seg_len))
            cum += seg_len
        total_length = max(total_length, cum)
        pdata.append({
            "color": part["color"], "moves": mlist,
            "last": mlist[-1][2], "first": mlist[0][1],
        })

        bx0 = min(bx0, pxmin + x_off); bx1 = max(bx1, pxmax + x_off)
        by0 = min(by0, pymin); by1 = max(by1, pymax)
        bz0 = min(bz0, pzmin); bz1 = max(bz1, pzmax)

    if not pdata or total_length <= 0:
        return None

    n_frames = max(2, int(round(duration * fps)))

    def segs_at(mlist: list[tuple], kind: str, cutoff: float) -> list:
        out = []
        for k, s, e, t0, t1 in mlist:
            if k != kind:
                continue
            if t1 <= cutoff:
                out.append([s, e])
            elif t0 < cutoff:
                f = (cutoff - t0) / (t1 - t0) if t1 > t0 else 1.0
                ei = (s[0] + (e[0] - s[0]) * f, s[1] + (e[1] - s[1]) * f, s[2] + (e[2] - s[2]) * f)
                out.append([s, ei])
        return out

    def nozzle_at(mlist: list[tuple], cutoff: float) -> tuple:
        last = mlist[0][1]
        for _k, s, e, t0, t1 in mlist:
            if cutoff >= t1:
                last = e
            elif t0 <= cutoff <= t1:
                f = (cutoff - t0) / (t1 - t0) if t1 > t0 else 1.0
                return (s[0] + (e[0] - s[0]) * f, s[1] + (e[1] - s[1]) * f, s[2] + (e[2] - s[2]) * f)
            else:
                return last
        return last

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    # Honour explicit zorder instead of depth-sorting, so the nozzle markers
    # always draw on top of the toolpath lines.
    try:
        ax.computed_zorder = False
    except Exception:
        pass
    pad = max(bx1 - bx0, by1 - by0, bz1 - bz0, 1.0) * 0.05
    ax.set_xlim(bx0 - pad, bx1 + pad)
    ax.set_ylim(by0 - pad, by1 + pad)
    ax.set_zlim(bz0 - pad, bz1 + pad)
    try:
        ax.set_box_aspect((bx1 - bx0 + 1e-6, by1 - by0 + 1e-6, bz1 - bz0 + 1e-6))
    except Exception:
        pass
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.view_init(elev=elev, azim=azim)

    artists = []
    for pd in pdata:
        # Seed with a degenerate segment: matplotlib 3.11's add_collection3d
        # errors on an empty collection. Axis limits are fixed above, so this
        # placeholder doesn't affect scaling; update() replaces it each frame.
        seed = [[pd["first"], pd["first"]]]
        # Travel drawn in neutral grey (distinct from the part's print color)
        # and faint, so travel and print are easy to tell apart.
        travel_col = Line3DCollection(seed, colors=travel_color, linewidths=travel_width, alpha=travel_opacity, zorder=1)
        print_col = Line3DCollection(seed, colors=pd["color"], linewidths=print_width, zorder=2)
        ax.add_collection3d(travel_col)
        ax.add_collection3d(print_col)
        # Nozzle marker: white fill with a black outline, drawn on top (high
        # zorder + computed_zorder disabled) so it stays visible against any
        # part color and the light background.
        noz = ax.scatter(
            [pd["last"][0]], [pd["last"][1]], [pd["last"][2]],
            color="white", edgecolors="black", linewidths=1.4, s=90,
            depthshade=False, zorder=10,
        )
        artists.append((print_col, travel_col, noz))

    def update(frame: int):
        cutoff = (frame / (n_frames - 1)) * total_length
        if progress_cb is not None:
            progress_cb(frame, n_frames)
        drawn = []
        for (print_col, travel_col, noz), pd in zip(artists, pdata):
            print_col.set_segments(segs_at(pd["moves"], "print", cutoff))
            travel_col.set_segments(segs_at(pd["moves"], "travel", cutoff))
            nx, ny, nz = nozzle_at(pd["moves"], cutoff)
            noz._offsets3d = ([nx], [ny], [nz])
            drawn += [print_col, travel_col, noz]
        return drawn

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    out_path = Path(out_path)
    anim.save(str(out_path), writer=PillowWriter(fps=int(fps)))
    plt.close(fig)
    return out_path


def render_gcode_file(path: str | Path) -> tuple[go.Figure, dict]:
    text = Path(path).read_text()
    parsed = parse_gcode_path(text)
    return build_toolpath_figure(parsed), parsed
