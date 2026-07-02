from __future__ import annotations

import numpy as np
import trimesh

from app import (
    SCALE_MODE_TARGET_DIMENSIONS,
    SCALE_MODE_UNIFORM_FACTOR,
    generate_dynamic_layer_stacks,
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


def test_generate_dynamic_layer_stacks_empty_input_resets_reference() -> None:
    records, status, ref_layers = generate_dynamic_layer_stacks(
        [],
        [],
        0.8,
        SCALE_MODE_TARGET_DIMENSIONS,
    )

    assert records == []
    assert status == "Upload at least one STL first."
    assert ref_layers is None


def test_generate_dynamic_layer_stacks_slices_shapes_and_builds_reference(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    stl_path = tmp_path / "cube.stl"
    mesh.export(stl_path)
    records = [
        {
            "idx": 1,
            "name": "cube",
            "stl_path": str(stl_path),
            "target_x": 2.0,
            "target_y": 2.0,
            "target_z": 2.0,
        }
    ]

    next_records, status, ref_layers = generate_dynamic_layer_stacks(
        records,
        None,
        0.5,
        SCALE_MODE_TARGET_DIMENSIONS,
    )

    stack = next_records[0]["layer_stack"]
    assert stack is not None
    assert len(stack.layers) == 4
    assert next_records[0]["slice_params"]["layer_height"] == 0.5
    assert "sliced 4 layers" in status
    assert ref_layers is not None
    assert len(ref_layers.layers) == 4
    assert ref_layers.layers[0].area == stack.layers[0].area
