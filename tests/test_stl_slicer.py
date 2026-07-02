from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
import trimesh

import pytest

from stl_slicer import (
    _compose_even_odd_polygons,
    calculate_z_levels,
    scale_factors_for_target_extents,
    scale_mesh,
    slice_stl_to_layers,
)


def test_calculate_z_levels_creates_single_layer_for_thin_mesh() -> None:
    z_values = calculate_z_levels(0.0, 0.01, 0.1)

    assert len(z_values) == 1
    assert 0.0 <= z_values[0] < 0.01


def test_slice_stl_to_layers_creates_layer_polygons(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    stl_path = tmp_path / "cube.stl"
    mesh.export(stl_path)

    stack = slice_stl_to_layers(stl_path, layer_height=0.5)

    assert len(stack.layers) == 4
    assert len(stack.z_values) == 4
    assert all(
        later > earlier
        for earlier, later in zip(stack.z_values, stack.z_values[1:])
    )
    for layer in stack.layers:
        assert layer.area == pytest.approx(4.0)

    (x_min, y_min, z_min), (x_max, y_max, z_max) = stack.bounds
    assert (x_max - x_min, y_max - y_min, z_max - z_min) == pytest.approx((2.0, 2.0, 2.0))
    assert stack.name == "cube"
    assert stack.layer_height == 0.5


def test_slice_stl_to_layers_applies_scale_factors(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    stl_path = tmp_path / "cube.stl"
    mesh.export(stl_path)

    stack = slice_stl_to_layers(
        stl_path,
        layer_height=0.5,
        scale_factors=(2.0, 1.0, 0.5),
    )

    bounds = np.array(stack.bounds)
    np.testing.assert_allclose(bounds[1] - bounds[0], (4.0, 2.0, 1.0))
    assert len(stack.layers) == 2
    assert stack.layers[0].area == pytest.approx(8.0)


def test_scale_mesh_matches_target_extents_and_preserves_min_corner() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 5.0))
    mesh.apply_translation((5.0, 6.0, 7.0))

    target_extents = (10.0, 8.0, 2.5)
    scale_factors = scale_factors_for_target_extents(mesh, target_extents)
    scaled = scale_mesh(mesh, scale_factors)

    np.testing.assert_allclose(scaled.extents, target_extents)
    np.testing.assert_allclose(scaled.bounds[0], mesh.bounds[0])


def test_compose_even_odd_polygons_preserves_holes() -> None:
    outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    inner = Polygon([(3, 3), (7, 3), (7, 7), (3, 7)])

    composed = _compose_even_odd_polygons([outer, inner])

    assert len(composed) == 1
    assert composed[0].area == outer.area - inner.area
    assert len(composed[0].interiors) == 1
