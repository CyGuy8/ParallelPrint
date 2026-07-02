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


def test_parse_gcode_reads_scientific_notation_instead_of_mantissa() -> None:
    parsed = parse_gcode_path(
        "\n".join(
            [
                "G91",
                "G0 X-5.1e-08 Y0.8 ; tiny float-noise travel",
                "G1 X4.0 Y0.0 ; print",
            ]
        )
    )

    (segment,) = parsed["print_segments"]
    start, end = segment[0], segment[-1]
    # "X-5.1e-08" must move the head by ~0, not by -5.1.
    assert abs(start[0]) < 1e-6
    assert abs(end[0] - 4.0) < 1e-6


def test_parse_gcode_keeps_uppercase_extrusion_token_out_of_x_axis() -> None:
    parsed = parse_gcode_path(
        "\n".join(
            [
                "G91",
                "G1 X1.2E3 Y0.5",  # compact G-code: X=1.2, E=3 (not X=1200)
            ]
        )
    )

    (segment,) = parsed["print_segments"]
    assert abs(segment[-1][0] - 1.2) < 1e-9
