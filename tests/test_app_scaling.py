from __future__ import annotations

import numpy as np
import trimesh

from app import (
    SCALE_MODE_TARGET_DIMENSIONS,
    SCALE_MODE_UNIFORM_FACTOR,
    _resolve_mesh_scale_factors,
)


def test_resolve_mesh_scale_factors_uses_uniform_factor_for_all_axes() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 8.0))

    scale_factors = _resolve_mesh_scale_factors(
        mesh,
        scale_to_target=True,
        scale_mode=SCALE_MODE_UNIFORM_FACTOR,
        target_x=10.0,
        target_y=20.0,
        target_z=30.0,
        uniform_scale=1.5,
    )

    assert scale_factors == (1.5, 1.5, 1.5)


def test_resolve_mesh_scale_factors_fits_each_axis_in_target_mode() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 8.0))

    scale_factors = _resolve_mesh_scale_factors(
        mesh,
        scale_to_target=True,
        scale_mode=SCALE_MODE_TARGET_DIMENSIONS,
        target_x=10.0,
        target_y=20.0,
        target_z=4.0,
        uniform_scale=1.5,
    )

    np.testing.assert_allclose(scale_factors, (5.0, 5.0, 0.5))
