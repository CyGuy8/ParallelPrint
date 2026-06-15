from __future__ import annotations

import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import trimesh

from stl_slicer import (
    _compose_even_odd_polygons,
    calculate_z_levels,
    scale_factors_for_target_extents,
    scale_mesh,
    slice_stl_to_tiffs,
)


def test_calculate_z_levels_creates_single_layer_for_thin_mesh() -> None:
    z_values = calculate_z_levels(0.0, 0.01, 0.1)

    assert len(z_values) == 1
    assert 0.0 <= z_values[0] < 0.01


def test_slice_stl_to_tiffs_creates_non_empty_tiffs(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    stl_path = tmp_path / "cube.stl"
    mesh.export(stl_path)

    stack = slice_stl_to_tiffs(
        stl_path,
        layer_height=0.5,
        pixel_size=0.25,
        output_root=tmp_path / "generated",
    )

    assert len(stack.tiff_paths) == 4
    assert stack.zip_path.exists()
    assert all(path.exists() for path in stack.tiff_paths)

    with Image.open(stack.tiff_paths[0]) as first_image:
        pixels = np.array(first_image)

    assert np.any(pixels == 0)


def test_scale_mesh_matches_target_extents_and_preserves_min_corner() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 5.0))
    mesh.apply_translation((5.0, 6.0, 7.0))

    target_extents = (10.0, 8.0, 2.5)
    scale_factors = scale_factors_for_target_extents(mesh, target_extents)
    scaled = scale_mesh(mesh, scale_factors)

    np.testing.assert_allclose(scaled.extents, target_extents)
    np.testing.assert_allclose(scaled.bounds[0], mesh.bounds[0])


def test_slice_stl_to_tiffs_applies_scale_factors(tmp_path) -> None:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    stl_path = tmp_path / "cube.stl"
    mesh.export(stl_path)

    stack = slice_stl_to_tiffs(
        stl_path,
        layer_height=0.5,
        pixel_size=0.25,
        output_root=tmp_path / "scaled",
        scale_factors=(2.0, 1.0, 0.5),
    )

    bounds = np.array(stack.bounds)
    np.testing.assert_allclose(bounds[1] - bounds[0], (4.0, 2.0, 1.0))
    assert stack.image_size == (17, 9)
    assert len(stack.tiff_paths) == 2


def test_compose_even_odd_polygons_preserves_holes() -> None:
    outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    inner = Polygon([(3, 3), (7, 3), (7, 7), (3, 7)])

    composed = _compose_even_odd_polygons([outer, inner])

    assert len(composed) == 1
    assert composed[0].area == outer.area - inner.area
    assert len(composed[0].interiors) == 1
