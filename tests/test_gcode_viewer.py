from __future__ import annotations

from gcode_viewer import parse_gcode_path


def test_parse_gcode_classifies_g0_as_travel_and_g1_as_print() -> None:
    parsed = parse_gcode_path(
        "\n".join(
            [
                "G91",
                "G0 X1 Y0 ; travel",
                "G1 X0 Y1 ; print",
            ]
        )
    )

    assert len(parsed["travel_segments"]) == 1
    assert len(parsed["print_segments"]) == 1
