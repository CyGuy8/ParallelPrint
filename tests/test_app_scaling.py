from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image

from app import (
    SCALE_MODE_TARGET_DIMENSIONS,
    SCALE_MODE_UNIFORM_FACTOR,
    generate_dynamic_stacks,
    split_tiff_stack_grid,
    split_tiff_stack_left_right,
    _resolve_mesh_scale_factors,
    _uniform_target_extents_from_anchor,
)


def test_resolve_mesh_scale_factors_uses_x_target_for_uniform_scaling() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 8.0))

    scale_factors = _resolve_mesh_scale_factors(
        mesh,
        scale_to_target=True,
        scale_mode=SCALE_MODE_UNIFORM_FACTOR,
        target_x=10.0,
        target_y=20.0,
        target_z=30.0,
    )

    assert scale_factors == (5.0, 5.0, 5.0)


def test_resolve_mesh_scale_factors_fits_each_axis_in_target_mode() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 8.0))

    scale_factors = _resolve_mesh_scale_factors(
        mesh,
        scale_to_target=True,
        scale_mode=SCALE_MODE_TARGET_DIMENSIONS,
        target_x=10.0,
        target_y=20.0,
        target_z=4.0,
    )

    np.testing.assert_allclose(scale_factors, (5.0, 5.0, 0.5))


def test_uniform_target_extents_update_from_changed_side() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 8.0))

    target_extents = _uniform_target_extents_from_anchor(
        mesh,
        anchor_axis="Y",
        target_x=10.0,
        target_y=12.0,
        target_z=30.0,
    )

    np.testing.assert_allclose(target_extents, (6.0, 12.0, 24.0))


def test_generate_dynamic_stacks_empty_input_resets_reference_stack() -> None:
    outputs = generate_dynamic_stacks([], [], 0.8, 0.8, SCALE_MODE_TARGET_DIMENSIONS)

    assert len(outputs) == 11
    assert outputs[0] == []
    assert outputs[2] == "Upload at least one STL first."
    assert outputs[7]["tiff_paths"] == []
    assert outputs[9] == "No reference stack generated yet."


def test_split_tiff_stack_left_right_preserves_pixels_and_metadata(tmp_path) -> None:
    pixels = np.array(
        [
            [0, 255, 0, 255, 0],
            [255, 0, 255, 0, 255],
        ],
        dtype=np.uint8,
    )
    tiff_path = tmp_path / "slice_0000.tif"
    Image.fromarray(pixels, mode="L").save(tiff_path)
    state = {
        "tiff_paths": [str(tiff_path)],
        "z_values": [1.25],
        "pixel_size": 0.5,
        "x_min": 10.0,
        "y_min": -2.0,
        "image_width": 5,
        "image_height": 2,
    }

    left_state, right_state, left_zip, right_zip = split_tiff_stack_left_right(state, "wide-part")

    assert left_zip.exists()
    assert right_zip.exists()
    assert left_state["image_width"] == 3
    assert right_state["image_width"] == 2
    assert left_state["x_min"] == 10.0
    assert right_state["x_min"] == 11.5
    assert left_state["z_values"] == [1.25]
    assert right_state["z_values"] == [1.25]

    with Image.open(left_state["tiff_paths"][0]) as left_image:
        np.testing.assert_array_equal(np.asarray(left_image), pixels[:, :3])
    with Image.open(right_state["tiff_paths"][0]) as right_image:
        np.testing.assert_array_equal(np.asarray(right_image), pixels[:, 3:])


def test_split_tiff_stack_grid_preserves_pixels_and_offsets(tmp_path) -> None:
    pixels = np.arange(20, dtype=np.uint8).reshape((4, 5))
    tiff_path = tmp_path / "slice_0000.tif"
    Image.fromarray(pixels, mode="L").save(tiff_path)
    state = {
        "tiff_paths": [str(tiff_path)],
        "z_values": [2.0],
        "pixel_size": 0.25,
        "x_min": 4.0,
        "y_min": 10.0,
        "image_width": 5,
        "image_height": 4,
    }

    pieces = split_tiff_stack_grid(state, "grid-part", columns=2, rows=2)

    assert [(piece["row"], piece["col"]) for piece in pieces] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert [piece["state"]["image_width"] for piece in pieces] == [3, 2, 3, 2]
    assert [piece["state"]["image_height"] for piece in pieces] == [2, 2, 2, 2]
    assert [piece["state"]["x_min"] for piece in pieces] == [4.0, 4.75, 4.0, 4.75]
    assert [piece["state"]["y_min"] for piece in pieces] == [10.5, 10.5, 10.0, 10.0]
    assert all(piece["zip_path"].exists() for piece in pieces)

    expected = [
        pixels[:2, :3],
        pixels[:2, 3:],
        pixels[2:, :3],
        pixels[2:, 3:],
    ]
    for piece, expected_pixels in zip(pieces, expected):
        with Image.open(piece["state"]["tiff_paths"][0]) as image:
            np.testing.assert_array_equal(np.asarray(image), expected_pixels)


def test_split_tiff_stack_grid_overlapping_layers_keep_small_alignment_margin(tmp_path) -> None:
    layer0 = np.zeros((2, 6), dtype=np.uint8)
    layer1 = np.zeros((2, 6), dtype=np.uint8)
    layer0_path = tmp_path / "slice_0000.tif"
    layer1_path = tmp_path / "slice_0001.tif"
    Image.fromarray(layer0, mode="L").save(layer0_path)
    Image.fromarray(layer1, mode="L").save(layer1_path)
    state = {
        "tiff_paths": [str(layer0_path), str(layer1_path)],
        "z_values": [0.0, 1.0],
        "pixel_size": 0.5,
        "x_min": 10.0,
        "y_min": -2.0,
        "image_width": 6,
        "image_height": 2,
    }

    left, right = split_tiff_stack_grid(state, "overlap-part", columns=2, rows=1, overlapping_layers=True)

    assert left["state"]["image_width"] == 4
    assert right["state"]["image_width"] == 4
    assert left["state"]["x_min"] == 10.0
    assert right["state"]["x_min"] == 11.0
    with Image.open(left["state"]["tiff_paths"][0]) as left_layer0:
        expected = np.zeros((2, 4), dtype=np.uint8)
        np.testing.assert_array_equal(np.asarray(left_layer0), expected)
    with Image.open(right["state"]["tiff_paths"][0]) as right_layer0:
        expected = np.full((2, 4), 255, dtype=np.uint8)
        expected[:, 2:] = 0
        np.testing.assert_array_equal(np.asarray(right_layer0), expected)
    with Image.open(left["state"]["tiff_paths"][1]) as left_layer1:
        expected = np.full((2, 4), 255, dtype=np.uint8)
        expected[:, :2] = 0
        np.testing.assert_array_equal(np.asarray(left_layer1), expected)
    with Image.open(right["state"]["tiff_paths"][1]) as right_layer1:
        expected = np.zeros((2, 4), dtype=np.uint8)
        np.testing.assert_array_equal(np.asarray(right_layer1), expected)
