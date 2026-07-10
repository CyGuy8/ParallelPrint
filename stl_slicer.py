from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import trimesh


ProgressCallback = Callable[[int, int], None] | None
ScaleFactors = tuple[float, float, float]
Bounds3D = tuple[tuple[float, float, float], tuple[float, float, float]]

MIN_POLYGON_AREA = 1e-9


@dataclass(slots=True)
class LayerStack:
    """A sliced shape as per-layer vector outlines in world-XY millimetres.

    `bounds` is the scaled mesh's axis-aligned bounding box. For grid-split
    pieces it is the piece's nominal cell box, which keeps reference-stack
    centring stable even when the clipped geometry shrinks.

    `scan_frame` is the XY box the raster scan grid is anchored to. Grid-split
    pieces inherit the parent shape's frame so all pieces raster on one
    continuous line grid and seams keep an exact one-fil-width pitch.

    `align_center`/`align_grid` are stamped by `build_reference_stack` on the
    combined reference stack: the common centre shapes were aligned to, and
    the grid the alignment deltas were snapped to — so later alignment of a
    shape against the reference reproduces exactly the same translation.
    """

    layers: list[MultiPolygon]
    z_values: list[float]
    bounds: Bounds3D
    layer_height: float
    name: str = ""
    scan_frame: tuple[float, float, float, float] | None = None
    align_center: tuple[float, float] | None = None
    align_grid: float | None = None
    # Multi-material group frame: the combined XY bbox of every part sharing
    # this shape's nozzle. Group members all carry the SAME frame and are
    # aligned by its centre instead of their own bbox centre, so they move as
    # one rigid unit and keep their modelled positions relative to each
    # other. None = align by the shape's own bbox centre (normal shapes).
    align_frame: tuple[float, float, float, float] | None = None
    # Grid-split pieces: per-layer contour polylines from the PARENT shape's
    # boundary clipped to this piece's cell — cut seams between sibling
    # pieces are excluded, so contour tracing only outlines the true outer
    # surface. None means "derive contours from the layer polygons" (whole
    # shapes). Paths may be open arcs; closed rings keep first == last.
    contour_paths: list[list[list[tuple[float, float]]]] | None = None


def load_mesh(stl_path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(stl_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("The STL file does not contain any mesh geometry.")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError("Unable to load a valid mesh from the STL file.")

    return mesh


def _normalize_scale_factors(scale_factors: Sequence[float] | None) -> ScaleFactors:
    if scale_factors is None:
        return (1.0, 1.0, 1.0)

    values = tuple(float(value) for value in scale_factors)
    if len(values) != 3:
        raise ValueError("Scale factors must contain X, Y, and Z values.")

    if any(value <= 0 for value in values):
        raise ValueError("Scale factors must be greater than zero.")

    return (values[0], values[1], values[2])


def scale_mesh(mesh: trimesh.Trimesh, scale_factors: Sequence[float] | None = None) -> trimesh.Trimesh:
    """Return a copy of `mesh` scaled around its minimum XYZ corner."""
    sx, sy, sz = _normalize_scale_factors(scale_factors)
    scaled = mesh.copy()

    if math.isclose(sx, 1.0) and math.isclose(sy, 1.0) and math.isclose(sz, 1.0):
        return scaled

    anchor = np.asarray(mesh.bounds[0], dtype=float)
    transform = np.eye(4)
    transform[0, 0] = sx
    transform[1, 1] = sy
    transform[2, 2] = sz
    transform[:3, 3] = anchor * (1.0 - np.array([sx, sy, sz], dtype=float))
    scaled.apply_transform(transform)
    return scaled


def scale_factors_for_target_extents(
    mesh: trimesh.Trimesh,
    target_extents: Sequence[float],
) -> ScaleFactors:
    target = _normalize_scale_factors(target_extents)
    extents = np.asarray(mesh.extents, dtype=float)
    if np.any(extents <= 0):
        raise ValueError("Cannot scale a mesh with a zero-sized X, Y, or Z extent.")

    return (
        target[0] / float(extents[0]),
        target[1] / float(extents[1]),
        target[2] / float(extents[2]),
    )


def calculate_z_levels(z_min: float, z_max: float, layer_height: float) -> list[float]:
    if layer_height <= 0:
        raise ValueError("Layer height must be greater than zero.")

    thickness = z_max - z_min
    if thickness <= 0:
        return [z_min]

    layer_count = max(1, math.ceil(thickness / layer_height))
    top_guard = math.nextafter(z_max, z_min)

    return [
        min(z_min + ((index + 0.5) * layer_height), top_guard)
        for index in range(layer_count)
    ]


def _ring_to_world_xy(ring_coords: object, to_3d: np.ndarray) -> np.ndarray:
    planar = np.asarray(ring_coords, dtype=float)
    if planar.ndim != 2 or planar.shape[1] < 2:
        raise ValueError("Encountered an invalid polygon ring while slicing.")

    planar_3d = np.column_stack([planar[:, 0], planar[:, 1], np.zeros(len(planar))])
    world = trimesh.transform_points(planar_3d, to_3d)
    return world[:, :2]


def _compose_even_odd_polygons(polygons: list[Polygon]) -> list[Polygon]:
    geometry: Polygon | MultiPolygon | GeometryCollection | None = None
    for polygon in polygons:
        geometry = polygon if geometry is None else geometry.symmetric_difference(polygon)

    if geometry is None or geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    if isinstance(geometry, GeometryCollection):
        return [geom for geom in geometry.geoms if isinstance(geom, Polygon) and not geom.is_empty]

    return []


def _extract_world_polygons(section: trimesh.path.Path3D) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    if hasattr(section, "to_2D"):
        planar, to_3d = section.to_2D()
    else:
        planar, to_3d = section.to_planar()

    # polygons_full composes rings by containment depth (ring in a ring is a
    # hole, ring in a hole is an island), so rings from separate solids that
    # merely OVERLAP each other stay separate solids — the caller unions
    # them. A flat even-odd XOR across all rings would punch false holes
    # wherever interpenetrating solids overlap (e.g. flag stripe prisms
    # packed into one STL).
    try:
        composed_polygons = [
            polygon
            for polygon in planar.polygons_full
            if polygon is not None and not polygon.is_empty
        ]
    except BaseException:
        composed_polygons = _compose_even_odd_polygons(list(planar.polygons_closed))
    polygons: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for polygon in composed_polygons:
        exterior = _ring_to_world_xy(polygon.exterior.coords, to_3d)
        holes = [_ring_to_world_xy(interior.coords, to_3d) for interior in polygon.interiors]
        polygons.append((exterior, holes))

    return polygons


def _as_multipolygon(geometry: object, min_area: float = MIN_POLYGON_AREA) -> MultiPolygon:
    """Flatten any shapely result into a MultiPolygon, dropping slivers.

    Single choke point for shapely's habit of returning mixed geometry types
    from overlay operations (Polygon / MultiPolygon / GeometryCollection /
    lines / points / empties).
    """
    polygons: list[Polygon] = []

    def collect(geom: object) -> None:
        if geom is None or getattr(geom, "is_empty", True):
            return
        if isinstance(geom, Polygon):
            if geom.area > min_area:
                polygons.append(geom)
        elif isinstance(geom, (MultiPolygon, GeometryCollection)):
            for part in geom.geoms:
                collect(part)

    collect(geometry)
    return MultiPolygon(polygons)


def _section_to_multipolygon(section: trimesh.path.Path3D | None) -> MultiPolygon:
    if section is None:
        return MultiPolygon()

    polygons = [
        make_valid(Polygon(exterior, holes))
        for exterior, holes in _extract_world_polygons(section)
    ]
    if not polygons:
        return MultiPolygon()

    # Union, not just collect: polygons from interpenetrating solids overlap.
    return _as_multipolygon(make_valid(unary_union(polygons)))


def _split_solids_and_cavities(
    mesh: trimesh.Trimesh,
) -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    """Connected bodies of a mesh, split into solids and explicit cavities.

    A single STL often packs several separate solids that touch or
    interpenetrate (checkerboard cells, overlapping stripe prisms).
    Sectioning such a mesh whole corrupts the ring reconstruction where
    rings cross or meet, so each WATERTIGHT body is sliced separately and
    the results unioned; a watertight body wound inside-out (negative
    volume) is a modeller's cavity, subtracted instead of unioned.

    Everything that is not watertight — stray internal quads, meshes whose
    face connectivity shreds into open fragments (T-vertices, unstitched
    fans) — is kept together as ONE remainder mesh and sectioned whole,
    where ring reconstruction can stitch across fragment boundaries; open
    slivers there simply produce no closed rings and drop out.
    """
    try:
        bodies = [body for body in mesh.split(only_watertight=False) if len(body.faces)]
    except Exception:
        return [mesh], []
    if len(bodies) <= 1:
        return [mesh], []

    solids: list[trimesh.Trimesh] = []
    cavities: list[trimesh.Trimesh] = []
    remainder: list[trimesh.Trimesh] = []
    for body in bodies:
        if not body.is_watertight:
            remainder.append(body)
        elif body.is_winding_consistent and float(body.volume) < 0.0:
            cavities.append(body)
        else:
            solids.append(body)
    if remainder:
        solids.append(
            trimesh.util.concatenate(remainder) if len(remainder) > 1 else remainder[0]
        )
    return solids, cavities


def slice_stl_to_layers(
    stl_path: str | Path,
    layer_height: float,
    progress_callback: ProgressCallback = None,
    scale_factors: Sequence[float] | None = None,
    name: str | None = None,
    z_levels: Sequence[float] | None = None,
) -> LayerStack:
    """Slice an STL into per-layer vector outlines (world-XY millimetres).

    `z_levels` overrides the per-mesh Z planes with an explicit (world) grid —
    used by multi-material assemblies so every part slices on ONE shared grid
    and parts that start higher simply get empty lower layers.
    """
    stl_path = Path(stl_path)
    mesh = scale_mesh(load_mesh(stl_path), scale_factors)
    (x_min, y_min, z_min), (x_max, y_max, z_max) = mesh.bounds

    if z_levels is not None:
        z_values = [float(z) for z in z_levels]
    else:
        z_values = calculate_z_levels(float(z_min), float(z_max), layer_height)

    solids, cavities = _split_solids_and_cavities(mesh)

    def section_at(body: trimesh.Trimesh, z_value: float) -> MultiPolygon:
        return _section_to_multipolygon(
            body.section(
                plane_origin=np.array([0.0, 0.0, z_value], dtype=float),
                plane_normal=np.array([0.0, 0.0, 1.0], dtype=float),
            )
        )

    layers: list[MultiPolygon] = []
    for index, z_value in enumerate(z_values):
        layer = unary_union([section_at(body, z_value) for body in solids])
        if cavities:
            layer = layer.difference(
                unary_union([section_at(body, z_value) for body in cavities])
            )
        layers.append(_as_multipolygon(make_valid(layer)))

        if progress_callback is not None:
            progress_callback(index + 1, len(z_values))

    return LayerStack(
        layers=layers,
        z_values=z_values,
        bounds=(
            (float(x_min), float(y_min), float(z_min)),
            (float(x_max), float(y_max), float(z_max)),
        ),
        layer_height=layer_height,
        name=name if name is not None else (stl_path.stem or "mesh"),
    )
