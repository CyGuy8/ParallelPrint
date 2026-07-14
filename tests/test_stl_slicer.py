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


def test_scale_mesh_about_an_explicit_anchor() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh.apply_translation((6.0, 6.0, 6.0))  # spans 5..7 on every axis

    scaled = scale_mesh(mesh, (2.0, 2.0, 2.0), anchor=(0.0, 0.0, 0.0))

    # Scaling about the shared origin: 5..7 becomes 10..14 (own-corner
    # scaling would give 5..9 and shift the part within an assembly).
    np.testing.assert_allclose(scaled.bounds[0], (10.0, 10.0, 10.0))
    np.testing.assert_allclose(scaled.bounds[1], (14.0, 14.0, 14.0))


def test_slice_stl_scale_anchor_keeps_assembly_parts_together(tmp_path) -> None:
    # Two assembly parts side by side; both scaled x2 about the ASSEMBLY
    # corner must stay adjacent (B's min corner moves from 4 to 8).
    a = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    a.apply_translation((2.0, 2.0, 2.0))  # spans 0..4
    b = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    b.apply_translation((6.0, 2.0, 2.0))  # spans 4..8 in x
    path_a = tmp_path / "a.stl"
    path_b = tmp_path / "b.stl"
    a.export(path_a)
    b.export(path_b)

    anchor = (0.0, 0.0, 0.0)
    stack_a = slice_stl_to_layers(path_a, 1.0, scale_factors=(2.0, 2.0, 2.0), scale_anchor=anchor)
    stack_b = slice_stl_to_layers(path_b, 1.0, scale_factors=(2.0, 2.0, 2.0), scale_anchor=anchor)

    assert stack_a.bounds[0][0] == pytest.approx(0.0)
    assert stack_a.bounds[1][0] == pytest.approx(8.0)
    assert stack_b.bounds[0][0] == pytest.approx(8.0)  # still flush against A
    assert stack_b.bounds[1][0] == pytest.approx(16.0)


def test_flip_z_mirrors_the_shape_top_to_bottom(tmp_path) -> None:
    # Wide slab with a narrow tower on top; flipped, the tower prints first.
    slab = trimesh.creation.box(extents=(10.0, 10.0, 1.0))
    slab.apply_translation((5.0, 5.0, 0.5))       # z 0..1
    tower = trimesh.creation.box(extents=(2.0, 2.0, 1.0))
    tower.apply_translation((5.0, 5.0, 1.5))      # z 1..2
    stl_path = tmp_path / "tower.stl"
    trimesh.util.concatenate([slab, tower]).export(stl_path)

    normal = slice_stl_to_layers(stl_path, layer_height=1.0)
    flipped = slice_stl_to_layers(stl_path, layer_height=1.0, flip_z=True)

    assert [round(layer.area) for layer in normal.layers] == [100, 4]
    assert [round(layer.area) for layer in flipped.layers] == [4, 100]
    # Flip about the own midplane preserves the Z range.
    assert flipped.bounds[0][2] == pytest.approx(normal.bounds[0][2])
    assert flipped.bounds[1][2] == pytest.approx(normal.bounds[1][2])


def test_flip_z_about_a_group_midplane_flips_the_assembly_as_one(tmp_path) -> None:
    # Two assembly parts at different heights flip about the SHARED midplane:
    # the part that was on top lands on the bottom of the shared Z range.
    low = trimesh.creation.box(extents=(4.0, 4.0, 1.0))
    low.apply_translation((2.0, 2.0, 0.5))   # z 0..1
    high = trimesh.creation.box(extents=(4.0, 4.0, 1.0))
    high.apply_translation((6.0, 2.0, 2.5))  # z 2..3
    path_low = tmp_path / "low.stl"
    path_high = tmp_path / "high.stl"
    low.export(path_low)
    high.export(path_high)

    group_mid = 1.5  # shared z range 0..3
    z_levels = [0.5, 1.5, 2.5]
    stack_low = slice_stl_to_layers(path_low, 1.0, z_levels=z_levels, flip_z=True, z_flip_mid=group_mid)
    stack_high = slice_stl_to_layers(path_high, 1.0, z_levels=z_levels, flip_z=True, z_flip_mid=group_mid)

    # `low` (was z 0..1) now occupies z 2..3; `high` now z 0..1.
    assert [layer.is_empty for layer in stack_low.layers] == [True, True, False]
    assert [layer.is_empty for layer in stack_high.layers] == [False, True, True]
    assert stack_high.layers[0].area == pytest.approx(16.0)
