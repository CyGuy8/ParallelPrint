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


def test_slice_stl_unions_interpenetrating_bodies(tmp_path) -> None:
    # One STL packing several separate solids that overlap (e.g. the stripe
    # prisms of a flag part) must slice to their union — the whole-mesh
    # even-odd rule would XOR the overlap into a false hole.
    a = trimesh.creation.box(extents=(4.0, 2.0, 2.0))
    b = trimesh.creation.box(extents=(2.0, 4.0, 2.0))
    b.apply_translation((1.0, 0.0, 0.0))  # overlaps half of `a`
    stl_path = tmp_path / "cross.stl"
    trimesh.util.concatenate([a, b]).export(stl_path)

    stack = slice_stl_to_layers(stl_path, layer_height=1.0)

    # Union area: 8 + 8 - 2x2 overlap = 12 (XOR would give 8).
    for layer in stack.layers:
        assert layer.area == pytest.approx(12.0)
        assert all(not polygon.interiors for polygon in layer.geoms)


def test_slice_stl_subtracts_inverted_cavity_bodies(tmp_path) -> None:
    # A watertight body wound inside-out (negative volume) is a modeller's
    # cavity: it must stay a hole, not be unioned as a solid.
    outer = trimesh.creation.box(extents=(6.0, 6.0, 2.0))
    cavity = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    cavity.invert()
    stl_path = tmp_path / "hollow.stl"
    trimesh.util.concatenate([outer, cavity]).export(stl_path)

    stack = slice_stl_to_layers(stl_path, layer_height=1.0)

    for layer in stack.layers:
        assert layer.area == pytest.approx(36.0 - 4.0)
        assert sum(len(polygon.interiors) for polygon in layer.geoms) == 1


def test_slice_stl_handles_abutting_cells_and_stray_open_quads(tmp_path) -> None:
    # Checkerboard-style STL: watertight cells that touch at edges/corners,
    # plus stray open quad fragments (internal walls). The cells must slice
    # per body and union into the exact checker pattern; the open quads
    # produce no closed rings and drop out.
    cells = []
    for cx, cy in ((0, 0), (1, 1), (2, 0), (0, 2), (2, 2)):
        cell = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
        cell.apply_translation((cx * 10.0 + 5.0, cy * 10.0 + 5.0, 5.0))
        cells.append(cell)
    quad = trimesh.Trimesh(
        vertices=[(10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (10.0, 10.0, 10.0), (10.0, 0.0, 10.0)],
        faces=[(0, 1, 2), (0, 2, 3)],
    )
    stl_path = tmp_path / "checker.stl"
    trimesh.util.concatenate(cells + [quad]).export(stl_path)

    stack = slice_stl_to_layers(stl_path, layer_height=1.0)

    for layer in stack.layers:
        assert layer.area == pytest.approx(500.0)
        assert layer.bounds == pytest.approx((0.0, 0.0, 30.0, 30.0))
