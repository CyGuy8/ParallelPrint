from __future__ import annotations

import zipfile

from PIL import Image

from tiff_to_gcode import generate_snake_path_gcode


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
