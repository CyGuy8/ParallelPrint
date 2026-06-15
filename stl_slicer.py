from __future__ import annotations

import math
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
import trimesh


ProgressCallback = Callable[[int, int], None] | None
ScaleFactors = tuple[float, float, float]


@dataclass(slots=True)
class SliceStack:
    output_dir: Path
    zip_path: Path
    tiff_paths: list[Path]
    z_values: list[float]
    image_size: tuple[int, int]
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    layer_height: float
    pixel_size: float


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


def _to_pixel_ring(
    coords: np.ndarray,
    x_min: float,
    y_min: float,
    pixel_size: float,
    image_height: int,
) -> list[tuple[int, int]]:
    pixels: list[tuple[int, int]] = []
    for x_value, y_value in coords:
        x_pixel = int(round((float(x_value) - x_min) / pixel_size))
        y_pixel = int(round((float(y_value) - y_min) / pixel_size))
        pixels.append((x_pixel, image_height - 1 - y_pixel))
    return pixels


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

    composed_polygons = _compose_even_odd_polygons(list(planar.polygons_closed))
    polygons: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for polygon in composed_polygons:
        exterior = _ring_to_world_xy(polygon.exterior.coords, to_3d)
        holes = [_ring_to_world_xy(interior.coords, to_3d) for interior in polygon.interiors]
        polygons.append((exterior, holes))

    return polygons


def _render_slice(
    section: trimesh.path.Path3D | None,
    x_min: float,
    y_min: float,
    image_size: tuple[int, int],
    pixel_size: float,
) -> Image.Image:
    image = Image.new("L", image_size, 255)

    if section is None:
        return image

    polygons = _extract_world_polygons(section)
    if not polygons:
        return image

    draw = ImageDraw.Draw(image)
    for exterior, holes in polygons:
        draw.polygon(
            _to_pixel_ring(exterior, x_min, y_min, pixel_size, image.height),
            fill=0,
        )
        for hole in holes:
            draw.polygon(
                _to_pixel_ring(hole, x_min, y_min, pixel_size, image.height),
                fill=255,
            )

    return image


def _make_output_paths(stl_path: Path, output_root: str | Path | None) -> tuple[Path, Path]:
    root = Path(output_root) if output_root else Path(tempfile.mkdtemp(prefix="stl_slices_"))
    stem = stl_path.stem or "mesh"
    job_dir = root / f"{stem}_{uuid.uuid4().hex[:8]}"
    slices_dir = job_dir / "tiff_slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    zip_path = job_dir / f"{stem}_tiff_slices.zip"
    return slices_dir, zip_path


def _zip_tiffs(tiff_paths: list[Path], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for tiff_path in tiff_paths:
            archive.write(tiff_path, arcname=tiff_path.name)


def slice_stl_to_tiffs(
    stl_path: str | Path,
    layer_height: float,
    pixel_size: float,
    output_root: str | Path | None = None,
    progress_callback: ProgressCallback = None,
    scale_factors: Sequence[float] | None = None,
) -> SliceStack:
    if pixel_size <= 0:
        raise ValueError("Pixel size must be greater than zero.")

    stl_path = Path(stl_path)
    mesh = scale_mesh(load_mesh(stl_path), scale_factors)
    bounds = mesh.bounds
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds

    z_values = calculate_z_levels(float(z_min), float(z_max), layer_height)
    width = max(1, math.ceil((float(x_max) - float(x_min)) / pixel_size) + 1)
    height = max(1, math.ceil((float(y_max) - float(y_min)) / pixel_size) + 1)
    image_size = (width, height)

    output_dir, zip_path = _make_output_paths(stl_path, output_root)
    tiff_paths: list[Path] = []

    for index, z_value in enumerate(z_values):
        section = mesh.section(
            plane_origin=np.array([0.0, 0.0, z_value], dtype=float),
            plane_normal=np.array([0.0, 0.0, 1.0], dtype=float),
        )
        image = _render_slice(section, float(x_min), float(y_min), image_size, pixel_size)
        tiff_path = output_dir / f"slice_{index:04d}.tif"
        image.save(tiff_path, compression="tiff_deflate")
        tiff_paths.append(tiff_path)

        if progress_callback is not None:
            progress_callback(index + 1, len(z_values))

    _zip_tiffs(tiff_paths, zip_path)

    return SliceStack(
        output_dir=output_dir,
        zip_path=zip_path,
        tiff_paths=tiff_paths,
        z_values=z_values,
        image_size=image_size,
        bounds=(
            (float(x_min), float(y_min), float(z_min)),
            (float(x_max), float(y_max), float(z_max)),
        ),
        layer_height=layer_height,
        pixel_size=pixel_size,
    )
