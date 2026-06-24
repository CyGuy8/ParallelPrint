from __future__ import annotations

import zipfile

import numpy as np
from PIL import Image

from tiff_to_gcode import (
    RASTER_PATTERN_SAME_DIRECTION,
    RASTER_PATTERN_WOODPILE,
    RASTER_PATTERN_Y_DIRECTION,
    _build_contour_layers,
    _trace_mask_contours,
    generate_snake_path_gcode,
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


def test_trace_mask_contours_uses_tiff_pixel_border_edges() -> None:
    contours = _trace_mask_contours(
        np.array(
            [
                [True, True],
                [True, True],
            ],
            dtype=bool,
        ),
        pixel_size=1.0,
    )

    assert contours == [[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]]


def test_contour_tracing_aligns_default_raster_border_pixel_frame(tmp_path) -> None:
    raster_tiff = tmp_path / "raster_slice_0000.tif"
    raster_image = Image.new("L", (7, 6), 255)
    raster_image.putpixel((4, 3), 0)
    raster_image.save(raster_tiff)
    raster_zip = tmp_path / "raster_slices.zip"
    with zipfile.ZipFile(raster_zip, mode="w") as archive:
        archive.write(raster_tiff, arcname=raster_tiff.name)

    contour_tiff = tmp_path / "contour_slice_0000.tif"
    contour_image = Image.new("L", (7, 6), 255)
    contour_image.putpixel((4, 3), 0)
    contour_image.save(contour_tiff)

    gcode_path = generate_snake_path_gcode(
        raster_zip,
        shape_name="aligned_contour",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        all_g1=True,
        contour_tiff_sets=[{"owner_idx": 1, "tiff_paths": [str(contour_tiff)]}],
        active_contour_owner=1,
    )

    points = _move_endpoints_for_color(gcode_path.read_text(), 255)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    assert (min(xs), max(xs)) == (1.0, 2.0)
    assert (min(ys), max(ys)) == (-0.5, 0.5)


def test_contour_tracing_travels_to_nearest_border_after_infill(tmp_path) -> None:
    tiff_path = tmp_path / "slice_0000.tif"
    image = Image.new("L", (7, 6), 255)
    image.putpixel((4, 3), 0)
    image.save(tiff_path)
    zip_path = tmp_path / "slices.zip"
    with zipfile.ZipFile(zip_path, mode="w") as archive:
        archive.write(tiff_path, arcname=tiff_path.name)

    gcode_path = generate_snake_path_gcode(
        zip_path,
        shape_name="nearest_border_contour",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        all_g1=True,
        contour_tiff_sets=[{"owner_idx": 1, "tiff_paths": [str(tiff_path)]}],
        active_contour_owner=1,
    )

    moves = _moves_with_colors(gcode_path.read_text())
    assert moves[1] == {
        "start": (1.0, 0.0, 0.0),
        "end": (2.0, 0.0, 0.0),
        "color": 255,
    }
    assert moves[2]["color"] == 255
    assert moves[2]["start"] == (2.0, 0.0, 0.0)


def test_contour_tracing_anchors_to_expanding_raster_frame(tmp_path) -> None:
    tiff_path = tmp_path / "slice_0000.tif"
    image = Image.new("L", (7, 5), 255)
    for col in [3]:
        image.putpixel((col, 0), 0)
    for col in range(2, 5):
        image.putpixel((col, 1), 0)
    for col in range(1, 6):
        image.putpixel((col, 2), 0)
    image.save(tiff_path)

    zip_path = tmp_path / "slices.zip"
    with zipfile.ZipFile(zip_path, mode="w") as archive:
        archive.write(tiff_path, arcname=tiff_path.name)

    gcode_path = generate_snake_path_gcode(
        zip_path,
        shape_name="expanding_contour",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        all_g1=True,
        contour_tiff_sets=[{"owner_idx": 1, "tiff_paths": [str(tiff_path)]}],
        active_contour_owner=1,
    )

    print_moves = [
        move for move in _moves_with_colors(gcode_path.read_text()) if move["color"] == 255
    ]
    contour_points = [
        point
        for move in print_moves[3:]
        for point in (move["start"], move["end"])
    ]
    xs = [point[0] for point in contour_points]
    ys = [point[1] for point in contour_points]

    assert print_moves[:3] == [
        {"start": (1.0, 0.0, 0.0), "end": (2.0, 0.0, 0.0), "color": 255},
        {"start": (3.0, 1.0, 0.0), "end": (0.0, 1.0, 0.0), "color": 255},
        {"start": (-1.0, 2.0, 0.0), "end": (4.0, 2.0, 0.0), "color": 255},
    ]
    assert print_moves[3]["start"] == (4.0, 2.0, 0.0)
    assert (min(xs), max(xs)) == (-1.0, 4.0)
    assert (min(ys), max(ys)) == (-0.5, 2.5)


def test_contour_tracing_uses_shifted_layer_raster_frame(tmp_path) -> None:
    tiff_paths = []
    for index, pixel in enumerate([(4, 3), (5, 4)]):
        tiff_path = tmp_path / f"slice_{index:04d}.tif"
        image = Image.new("L", (8, 7), 255)
        image.putpixel(pixel, 0)
        image.save(tiff_path)
        tiff_paths.append(tiff_path)

    zip_path = tmp_path / "slices.zip"
    with zipfile.ZipFile(zip_path, mode="w") as archive:
        for tiff_path in tiff_paths:
            archive.write(tiff_path, arcname=tiff_path.name)

    gcode_path = generate_snake_path_gcode(
        zip_path,
        shape_name="shifted_layer_contour",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        layer_height=1.0,
        all_g1=True,
        contour_tiff_sets=[{"owner_idx": 1, "tiff_paths": [str(p) for p in tiff_paths]}],
        active_contour_owner=1,
    )

    layer_one_prints = [
        move
        for move in _moves_with_colors(gcode_path.read_text())
        if move["color"] == 255
        and move["start"][2] == 1.0
        and move["end"][2] == 1.0
    ]

    assert layer_one_prints[0] == {
        "start": (3.0, 1.0, 1.0),
        "end": (2.0, 1.0, 1.0),
        "color": 255,
    }
    contour_points = [
        point
        for move in layer_one_prints[1:]
        for point in (move["start"], move["end"])
    ]
    xs = [point[0] for point in contour_points]
    ys = [point[1] for point in contour_points]

    assert layer_one_prints[1]["start"] == (2.0, 1.0, 1.0)
    assert layer_one_prints[1]["end"] == (2.0, 0.5, 1.0)
    assert (min(xs), max(xs)) == (2.0, 3.0)
    assert (min(ys), max(ys)) == (0.5, 1.5)


def test_contour_tracing_mirrors_odd_layer_y_frame(tmp_path) -> None:
    tiff_paths = []
    for index in range(2):
        tiff_path = tmp_path / f"slice_{index:04d}.tif"
        image = Image.new("L", (8, 7), 255)
        image.putpixel((4, 3), 0)
        image.putpixel((4, 4), 0)
        image.putpixel((5, 4), 0)
        image.save(tiff_path)
        tiff_paths.append(tiff_path)

    zip_path = tmp_path / "slices.zip"
    with zipfile.ZipFile(zip_path, mode="w") as archive:
        for tiff_path in tiff_paths:
            archive.write(tiff_path, arcname=tiff_path.name)

    gcode_path = generate_snake_path_gcode(
        zip_path,
        shape_name="odd_layer_y_contour",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        layer_height=1.0,
        all_g1=True,
        contour_tiff_sets=[{"owner_idx": 1, "tiff_paths": [str(p) for p in tiff_paths]}],
        active_contour_owner=1,
    )

    layer_one_prints = [
        move
        for move in _moves_with_colors(gcode_path.read_text())
        if move["color"] == 255
        and move["start"][2] == 1.0
        and move["end"][2] == 1.0
    ]
    contour_points = [
        point
        for move in layer_one_prints[2:]
        for point in (move["start"], move["end"])
    ]
    xs = [point[0] for point in contour_points]
    ys = [point[1] for point in contour_points]

    assert layer_one_prints[:2] == [
        {"start": (1.0, 1.0, 1.0), "end": (3.0, 1.0, 1.0), "color": 255},
        {"start": (2.0, 0.0, 1.0), "end": (1.0, 0.0, 1.0), "color": 255},
    ]
    assert layer_one_prints[2]["start"] == (1.0, 0.0, 1.0)
    assert layer_one_prints[2]["end"] == (1.0, -0.5, 1.0)
    assert (min(xs), max(xs)) == (1.0, 3.0)
    assert (min(ys), max(ys)) == (-0.5, 1.5)


def test_contour_tracing_closes_loop_and_restores_raster_endpoint(tmp_path) -> None:
    tiff_paths = []
    for index in range(4):
        tiff_path = tmp_path / f"slice_{index:04d}.tif"
        image = Image.new("L", (8, 8), 255)
        image.putpixel((4, 2), 0)
        image.putpixel((4, 3), 0)
        image.putpixel((5, 4), 0)
        image.save(tiff_path)
        tiff_paths.append(tiff_path)

    zip_path = tmp_path / "slices.zip"
    with zipfile.ZipFile(zip_path, mode="w") as archive:
        for tiff_path in tiff_paths:
            archive.write(tiff_path, arcname=tiff_path.name)

    gcode_path = generate_snake_path_gcode(
        zip_path,
        shape_name="odd_layer_last_infill_anchor",
        pressure=25,
        valve=7,
        port=3,
        fil_width=1.0,
        layer_height=1.0,
        all_g1=True,
        contour_tiff_sets=[{"owner_idx": 1, "tiff_paths": [str(p) for p in tiff_paths]}],
        active_contour_owner=1,
    )

    all_moves = _moves_with_colors(gcode_path.read_text())
    for layer_z in (1.0, 3.0):
        layer_moves = [
            move
            for move in all_moves
            if move["start"][2] == layer_z
            and move["end"][2] == layer_z
        ]
        layer_prints = [
            move
            for move in layer_moves
            if move["color"] == 255
        ]

        assert layer_prints[:3] == [
            {"start": (3.0, 2.0, layer_z), "end": (2.0, 2.0, layer_z), "color": 255},
            {"start": (1.0, 1.0, layer_z), "end": (2.0, 1.0, layer_z), "color": 255},
            {"start": (2.0, 0.0, layer_z), "end": (1.0, 0.0, layer_z), "color": 255},
        ]
        contour_start = layer_prints[2]["end"]
        assert layer_prints[3]["start"] == contour_start
        assert layer_prints[-1]["end"] == contour_start

        last_print_index = max(
            idx for idx, move in enumerate(layer_moves) if move["color"] == 255
        )
        assert layer_moves[last_print_index + 1] == {
            "start": contour_start,
            "end": (0.0, 0.0, layer_z),
            "color": 0,
        }


def test_contour_tracing_follows_default_raster_layer_flip(tmp_path) -> None:
    tiff_paths = []
    motion_img = np.zeros((7, 8), dtype=np.uint8)
    motion_img[3, 4] = 255
    motion_img[4, 4] = 255
    motion_img[4, 5] = 255
    for index in range(2):
        tiff_path = tmp_path / f"l_shape_{index:04d}.tif"
        image = Image.new("L", (8, 7), 255)
        image.putpixel((4, 3), 0)
        image.putpixel((4, 4), 0)
        image.putpixel((5, 4), 0)
        image.save(tiff_path)
        tiff_paths.append(str(tiff_path))

    contour_layers = _build_contour_layers(
        [{"owner_idx": 1, "tiff_paths": tiff_paths}],
        [motion_img, motion_img],
        pixel_size=1.0,
        invert=True,
        off_color=0,
        work_dir=tmp_path,
        raster_pattern=RASTER_PATTERN_SAME_DIRECTION,
    )

    assert contour_layers[0][0]["contours"][0] == [
        (1.0, -0.5),
        (2.0, -0.5),
        (2.0, 0.5),
        (3.0, 0.5),
        (3.0, 1.5),
        (1.0, 1.5),
        (1.0, -0.5),
    ]
    assert contour_layers[1][0]["contours"][0] == [
        (1.0, -0.5),
        (3.0, -0.5),
        (3.0, 0.5),
        (2.0, 0.5),
        (2.0, 1.5),
        (1.0, 1.5),
        (1.0, -0.5),
    ]


def test_gcode_header_writes_presets_before_initial_aux_commands(tmp_path) -> None:
    tiff_path = tmp_path / "slice_0000.tif"
    Image.new("L", (1, 1), 0).save(tiff_path)

    zip_path = tmp_path / "slices.zip"
    with zipfile.ZipFile(zip_path, mode="w") as archive:
        archive.write(tiff_path, arcname=tiff_path.name)

    gcode_path = generate_snake_path_gcode(
        zip_path,
        shape_name="header_order",
        pressure=25,
        valve=7,
        port=3,
    )

    lines = [
        line.strip()
        for line in gcode_path.read_text().splitlines()
        if line.strip()
    ]

    assert lines[0] == "G91"
    assert lines[1].startswith("{preset}serialPort3.write(")
    assert lines[2].startswith("{preset}serialPort3.write(")
    assert lines[3].startswith("{aux_command}WAGO_ValveCommands(")
    assert lines[4].startswith("{aux_command}WAGO_ValveCommands(")


def test_gcode_uses_g1_for_print_and_g0_for_travel(tmp_path) -> None:
    tiff_path = tmp_path / "slice_0000.tif"
    Image.new("L", (1, 1), 0).save(tiff_path)

    zip_path = tmp_path / "slices.zip"
    with zipfile.ZipFile(zip_path, mode="w") as archive:
        archive.write(tiff_path, arcname=tiff_path.name)

    gcode_path = generate_snake_path_gcode(
        zip_path,
        shape_name="move_types",
        pressure=25,
        valve=7,
        port=3,
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
    tiff_paths = []
    for index in range(4):
        tiff_path = tmp_path / f"slice_{index:04d}.tif"
        Image.new("L", (3, 2), 0).save(tiff_path)
        tiff_paths.append(tiff_path)

    zip_path = tmp_path / "slices.zip"
    with zipfile.ZipFile(zip_path, mode="w") as archive:
        for tiff_path in tiff_paths:
            archive.write(tiff_path, arcname=tiff_path.name)

    gcode_path = generate_snake_path_gcode(
        zip_path,
        shape_name="woodpile",
        pressure=25,
        valve=7,
        port=3,
        raster_pattern=RASTER_PATTERN_WOODPILE,
    )

    move_lines = [
        line.strip()
        for line in gcode_path.read_text().splitlines()
        if line.startswith(("G0", "G1"))
    ]
    z_move_index = next(i for i, line in enumerate(move_lines) if " Z" in line)
    first_layer_prints = [line for line in move_lines[:z_move_index] if line.startswith("G1") and "; Color 255" in line]
    second_layer_prints = [line for line in move_lines[z_move_index + 1 :] if line.startswith("G1") and "; Color 255" in line]

    assert any("X" in line and "Y0" in line for line in first_layer_prints)
    assert any("X0" in line and "Y" in line for line in second_layer_prints)

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
    assert min(x_positions) >= 0.0
    assert max(x_positions) <= 3.0
    assert min(y_positions) >= 0.0
    assert max(y_positions) <= 2.0


def test_y_direction_raster_prints_each_layer_along_y_axis(tmp_path) -> None:
    tiff_paths = []
    for index in range(2):
        tiff_path = tmp_path / f"slice_{index:04d}.tif"
        Image.new("L", (3, 2), 0).save(tiff_path)
        tiff_paths.append(tiff_path)

    zip_path = tmp_path / "slices.zip"
    with zipfile.ZipFile(zip_path, mode="w") as archive:
        for tiff_path in tiff_paths:
            archive.write(tiff_path, arcname=tiff_path.name)

    gcode_path = generate_snake_path_gcode(
        zip_path,
        shape_name="y_direction",
        pressure=25,
        valve=7,
        port=3,
        raster_pattern=RASTER_PATTERN_Y_DIRECTION,
    )

    move_lines = [
        line.strip()
        for line in gcode_path.read_text().splitlines()
        if line.startswith(("G0", "G1"))
    ]
    print_lines = [
        line
        for line in move_lines
        if line.startswith("G1") and "; Color 255" in line
    ]
    assert print_lines
    assert all("X0" in line and "Y0" not in line for line in print_lines)

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
    assert min(x_positions) >= 0.0
    assert max(x_positions) <= 3.0
    assert min(y_positions) >= 0.0
    assert max(y_positions) <= 2.0


def test_contour_tracing_skips_inactive_nozzle_outline(tmp_path) -> None:
    blank_tiff = tmp_path / "blank_slice_0000.tif"
    Image.new("L", (1, 1), 255).save(blank_tiff)
    blank_zip = tmp_path / "blank_slices.zip"
    with zipfile.ZipFile(blank_zip, mode="w") as archive:
        archive.write(blank_tiff, arcname=blank_tiff.name)

    contour_tiff = tmp_path / "contour_slice_0000.tif"
    Image.new("L", (1, 1), 0).save(contour_tiff)
    contour_sources = [{"owner_idx": 1, "tiff_paths": [str(contour_tiff)]}]

    active_path = generate_snake_path_gcode(
        blank_zip,
        shape_name="active_contour",
        pressure=25,
        valve=7,
        port=3,
        all_g1=True,
        contour_tiff_sets=contour_sources,
        active_contour_owner=1,
    )
    inactive_path = generate_snake_path_gcode(
        blank_zip,
        shape_name="inactive_contour",
        pressure=25,
        valve=7,
        port=3,
        all_g1=True,
        contour_tiff_sets=contour_sources,
        active_contour_owner=2,
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
    tiff_paths = []
    for index in range(2):
        tiff_path = tmp_path / f"slice_{index:04d}.tif"
        image = Image.new("L", (4, 3), 255)
        image.putpixel((1, 1), 0)
        image.putpixel((2, 1), 0)
        image.save(tiff_path)
        tiff_paths.append(tiff_path)

    zip_path = tmp_path / "slices.zip"
    with zipfile.ZipFile(zip_path, mode="w") as archive:
        for tiff_path in tiff_paths:
            archive.write(tiff_path, arcname=tiff_path.name)

    original_path = generate_snake_path_gcode(
        zip_path,
        shape_name="original_raster",
        pressure=25,
        valve=7,
        port=3,
        all_g1=True,
    )
    inactive_path = generate_snake_path_gcode(
        zip_path,
        shape_name="inactive_contour_raster",
        pressure=25,
        valve=7,
        port=3,
        all_g1=True,
        contour_tiff_sets=[{"owner_idx": 2, "tiff_paths": [str(p) for p in tiff_paths]}],
        active_contour_owner=1,
    )

    assert _move_signature(inactive_path.read_text()) == _move_signature(
        original_path.read_text()
    )
