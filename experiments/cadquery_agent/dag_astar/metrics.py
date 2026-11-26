"""Deterministic solid-volume and surface metrics in one shared reference frame."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.stats import qmc
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GeometryMetricProtocol:
    """One immutable protocol shared by IoU, Surface F1 and Chamfer."""

    surface_samples: int = 8192
    surface_tau: float = 0.02
    voxel_resolution: int = 96
    normalize_by_reference_extent: bool = True
    seed: int = 20261115
    strict_iou_threshold: float = 0.50
    strict_f1_threshold: float = 0.50
    max_voxels_per_shape: int = 4_000_000
    fallback_surface_multiplier: int = 4
    max_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.surface_samples < 256:
            raise ValueError("surface_samples_must_be_at_least_256")
        if self.surface_tau <= 0:
            raise ValueError("surface_tau_must_be_positive")
        if self.voxel_resolution < 16:
            raise ValueError("voxel_resolution_must_be_at_least_16")
        if self.fallback_surface_multiplier < 2:
            raise ValueError("fallback_surface_multiplier_must_be_at_least_2")
        if self.max_seconds <= 0:
            raise ValueError("metric_max_seconds_must_be_positive")


DEFAULT_PROTOCOL = GeometryMetricProtocol()


@dataclass(frozen=True)
class MeshMetrics:
    chamfer_distance: float
    hausdorff_distance: float
    hausdorff_95_distance: float
    f1_score: float
    volumetric_iou: float
    precision: float
    recall: float
    bbox_center_error: float
    bbox_extent_relative_error: float
    volume_relative_error: float
    surface_backend: str
    protocol: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def is_strict_success(self) -> bool:
        return (
            self.volumetric_iou >= float(self.protocol["strict_iou_threshold"])
            and self.f1_score >= float(self.protocol["strict_f1_threshold"])
        )


@dataclass(frozen=True)
class CADDesignerCompatibleMetrics:
    """Supplementary metrics matching the CADDesigner paper protocol.

    These values are intentionally separate from :class:`MeshMetrics` because
    independent unit-cube normalization and ICP remove absolute scale, position,
    and orientation errors. They must not be used for strict requirement
    acceptance in this project.
    """

    volumetric_iou: float
    chamfer_distance: float
    hausdorff_distance: float
    icp_cost: float
    point_samples: int
    voxel_size: float
    protocol_name: str = "caddesigner_unit_cube_icp_2048_v1"

    def to_dict(self) -> dict:
        return asdict(self)


def _as_mesh(path: str) -> trimesh.Trimesh:
    if Path(path).suffix.lower() == ".stl":
        # Loading through ``trimesh.load(..., force='mesh')`` first builds a
        # Scene and then applies its transform graph.  Besides unnecessary
        # overhead for a single STL, that route can trigger a native DLL clash
        # on Windows CadQuery environments.  The format loader returns the
        # exact same triangle arrays without Scene transforms.
        with Path(path).open("rb") as handle:
            loaded = trimesh.Trimesh(
                **trimesh.exchange.stl.load_stl(handle),
                process=True,
            )
    else:
        loaded = trimesh.load(path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.faces.size == 0:
        raise ValueError(f"mesh_has_no_faces:{path}")
    mesh = loaded.copy()
    mesh.process(validate=True)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        # A valid OCC Compound can contain individually closed solids that
        # touch along an edge or vertex.  Its combined STL is non-manifold at
        # that contact even though every participating solid is watertight.
        # Accept only this proven multi-solid case; an actually open component
        # remains an error.  Voxel filling below is performed per component.
        components = list(mesh.split(only_watertight=False, repair=False))
        if not components or any(not component.is_watertight for component in components):
            raise ValueError(f"mesh_is_not_watertight:{path}")
    if not math.isfinite(float(mesh.volume)) or abs(float(mesh.volume)) <= 1e-12:
        raise ValueError(f"mesh_has_zero_volume:{path}")
    return mesh


def _shared_reference_frame(
    generated: trimesh.Trimesh, reference: trimesh.Trimesh, normalize: bool,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    generated = generated.copy()
    reference = reference.copy()
    if not normalize:
        return generated, reference
    reference_center = np.asarray(reference.bounds, dtype=np.float64).mean(axis=0)
    scale = float(np.asarray(reference.extents, dtype=np.float64).max())
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("reference_has_invalid_extent")
    # Apply one reference-derived transform to both meshes. Relative position
    # and size errors remain visible to all metrics.
    generated.vertices = (generated.vertices - reference_center) / scale
    reference.vertices = (reference.vertices - reference_center) / scale
    return generated, reference


def _independent_unit_cube(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center and scale one mesh independently into [-0.5, 0.5]^3."""
    normalized = mesh.copy()
    center = np.asarray(normalized.bounds, dtype=np.float64).mean(axis=0)
    scale = float(np.asarray(normalized.extents, dtype=np.float64).max())
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("mesh_has_invalid_extent")
    normalized.vertices = (normalized.vertices - center) / scale
    return normalized


def _sobol_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total_area = float(areas.sum())
    if not math.isfinite(total_area) or total_area <= 1e-15:
        raise ValueError("mesh_has_zero_surface_area")
    cumulative = np.cumsum(areas) / total_area
    power = int(math.ceil(math.log2(count)))
    samples = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(power)[:count]
    face_index = np.searchsorted(cumulative, samples[:, 0], side="right")
    face_index = np.minimum(face_index, len(triangles) - 1)
    chosen = triangles[face_index]
    root = np.sqrt(samples[:, 1])
    w0 = 1.0 - root
    w1 = root * (1.0 - samples[:, 2])
    w2 = root * samples[:, 2]
    return chosen[:, 0] * w0[:, None] + chosen[:, 1] * w1[:, None] + chosen[:, 2] * w2[:, None]


def _point_to_surface_distances(
    mesh: trimesh.Trimesh, points: np.ndarray, *, fallback_count: int, seed: int,
) -> tuple[np.ndarray, str]:
    try:
        return np.asarray(trimesh.proximity.closest_point(mesh, points)[1], dtype=np.float64), "rtree_exact"
    except (ModuleNotFoundError, ImportError):
        dense = _sobol_surface(mesh, fallback_count, seed)
        distances, _ = cKDTree(dense, compact_nodes=True, balanced_tree=True).query(
            points, k=1, workers=-1,
        )
        return np.asarray(distances, dtype=np.float64), "sampled_kdtree"


def _surface_metrics(
    left_mesh: trimesh.Trimesh, right_mesh: trimesh.Trimesh,
    left_points: np.ndarray, right_points: np.ndarray, tau: float,
    fallback_count: int, seed: int,
) -> tuple[float, float, float, float, float, float, str]:
    left_distance, left_backend = _point_to_surface_distances(
        right_mesh, left_points, fallback_count=fallback_count, seed=seed,
    )
    right_distance, right_backend = _point_to_surface_distances(
        left_mesh, right_points, fallback_count=fallback_count, seed=seed,
    )
    chamfer = float(np.mean(left_distance**2) + np.mean(right_distance**2))
    hausdorff = float(max(np.max(left_distance), np.max(right_distance)))
    hausdorff_95 = float(max(
        np.quantile(left_distance, 0.95), np.quantile(right_distance, 0.95),
    ))
    precision = float(np.mean(left_distance <= tau))
    recall = float(np.mean(right_distance <= tau))
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    backend = left_backend if left_backend == right_backend else f"{left_backend}+{right_backend}"
    return chamfer, hausdorff, hausdorff_95, f1, precision, recall, backend


def _solid_voxels(mesh: trimesh.Trimesh, pitch: float, maximum: int) -> np.ndarray:
    estimated_shape = np.ceil(np.asarray(mesh.extents, dtype=np.float64) / pitch).astype(np.int64) + 3
    estimated_cells = int(np.prod(estimated_shape))
    if estimated_cells > maximum:
        raise ValueError(f"voxel_budget_exceeded_before_allocation:{estimated_cells}>{maximum}")
    components = [mesh] if mesh.is_watertight else list(
        mesh.split(only_watertight=False, repair=False)
    )
    occupied: list[np.ndarray] = []
    allocated = 0
    for component in components:
        if not component.is_watertight:
            raise ValueError("mesh_component_is_not_watertight")
        grid = component.voxelized(pitch).fill()
        allocated += int(np.prod(grid.shape))
        if allocated > maximum:
            raise ValueError(f"voxel_budget_exceeded:{allocated}>{maximum}")
        points = np.asarray(grid.points, dtype=np.float64)
        occupied.append(np.rint(points / pitch).astype(np.int64))
    if not occupied:
        raise ValueError("mesh_has_no_solid_voxels")
    return np.unique(np.concatenate(occupied, axis=0), axis=0)


def _row_keys(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.int64)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()


def compare_stl(
    generated_stl: str, reference_stl: str, *,
    protocol: GeometryMetricProtocol | None = None,
) -> MeshMetrics:
    protocol = protocol or DEFAULT_PROTOCOL
    started = time.perf_counter()
    def check_time(stage: str) -> None:
        elapsed = time.perf_counter() - started
        if elapsed > protocol.max_seconds:
            raise TimeoutError(f"geometry_metric_timeout:{stage}:{elapsed:.3f}>{protocol.max_seconds:.3f}")
    generated_raw, reference_raw = _as_mesh(generated_stl), _as_mesh(reference_stl)
    check_time("load")
    reference_scale = max(float(np.asarray(reference_raw.extents).max()), 1e-12)
    center_error = float(np.linalg.norm(
        np.asarray(generated_raw.bounds).mean(axis=0) - np.asarray(reference_raw.bounds).mean(axis=0)
    ) / reference_scale)
    extent_error = float(np.mean(
        np.abs(np.asarray(generated_raw.extents) - np.asarray(reference_raw.extents))
        / np.maximum(np.asarray(reference_raw.extents), 1e-12)
    ))
    volume_error = float(
        abs(abs(float(generated_raw.volume)) - abs(float(reference_raw.volume)))
        / max(abs(float(reference_raw.volume)), 1e-12)
    )
    generated, reference = _shared_reference_frame(
        generated_raw, reference_raw, protocol.normalize_by_reference_extent,
    )
    # Keep the statistical protocol stable when callers request a smaller sample
    # count. 8192 is the calibrated reporting count; larger explicit requests are
    # still honoured for sensitivity studies.
    effective_surface_samples = max(8192, protocol.surface_samples)
    generated_points = _sobol_surface(generated, effective_surface_samples, protocol.seed)
    reference_points = _sobol_surface(reference, effective_surface_samples, protocol.seed)
    fallback_count = effective_surface_samples * protocol.fallback_surface_multiplier
    (
        chamfer, hausdorff, hausdorff_95, f1, precision, recall, surface_backend,
    ) = _surface_metrics(
        generated, reference, generated_points, reference_points, protocol.surface_tau,
        fallback_count, protocol.seed,
    )
    check_time("surface")
    pitch = 1.0 / protocol.voxel_resolution if protocol.normalize_by_reference_extent else max(
        float(np.asarray(reference.extents).max()) / protocol.voxel_resolution, 1e-9,
    )
    generated_voxels = _solid_voxels(generated, pitch, protocol.max_voxels_per_shape)
    check_time("generated_voxels")
    reference_voxels = _solid_voxels(reference, pitch, protocol.max_voxels_per_shape)
    check_time("reference_voxels")
    generated_keys, reference_keys = _row_keys(generated_voxels), _row_keys(reference_voxels)
    intersection = int(np.intersect1d(generated_keys, reference_keys, assume_unique=True).size)
    union = int(generated_keys.size + reference_keys.size - intersection)
    iou = float(intersection / union) if union else 0.0
    return MeshMetrics(
        chamfer_distance=chamfer,
        hausdorff_distance=hausdorff,
        hausdorff_95_distance=hausdorff_95,
        f1_score=f1,
        volumetric_iou=iou,
        precision=precision,
        recall=recall,
        bbox_center_error=center_error,
        bbox_extent_relative_error=extent_error,
        volume_relative_error=volume_error,
        surface_backend=surface_backend,
        protocol={
            **asdict(protocol),
            "effective_surface_samples": effective_surface_samples,
            "distance_units": "reference_max_extent",
            "hausdorff_definition": "symmetric_sampled_max",
            "hausdorff_95_definition": "max_of_directional_p95",
        },
    )


def compare_stl_caddesigner_protocol(
    generated_stl: str,
    reference_stl: str,
    *,
    point_samples: int = 2048,
    voxel_size: float = 0.02,
    seed: int = 20261115,
    icp_iterations: int = 50,
    max_voxels_per_shape: int = 4_000_000,
) -> CADDesignerCompatibleMetrics:
    """Compute the supplementary protocol reported by CADDesigner.

    Both shapes are independently centered and scaled to a unit cube, then the
    generated point cloud is rigidly aligned to the reference with ICP. CD and
    HD are evaluated on the aligned 2048-point clouds and IoU uses solid voxels
    with a fixed 0.02 voxel size. The protocol deliberately masks absolute
    position, orientation, and scale errors, so it is never used by
    :meth:`MeshMetrics.is_strict_success`.
    """
    if point_samples < 256:
        raise ValueError("point_samples_must_be_at_least_256")
    if voxel_size <= 0:
        raise ValueError("voxel_size_must_be_positive")
    generated = _independent_unit_cube(_as_mesh(generated_stl))
    reference = _independent_unit_cube(_as_mesh(reference_stl))
    generated_points = _sobol_surface(generated, point_samples, seed)
    reference_points = _sobol_surface(reference, point_samples, seed)
    transform, aligned_points, icp_cost = trimesh.registration.icp(
        generated_points,
        reference_points,
        max_iterations=icp_iterations,
        reflection=False,
        scale=False,
    )
    generated.apply_transform(transform)
    aligned_points = np.asarray(aligned_points, dtype=np.float64)
    generated_to_reference = cKDTree(
        reference_points, compact_nodes=True, balanced_tree=True,
    ).query(aligned_points, k=1, workers=-1)[0]
    reference_to_generated = cKDTree(
        aligned_points, compact_nodes=True, balanced_tree=True,
    ).query(reference_points, k=1, workers=-1)[0]
    chamfer = float(
        np.mean(generated_to_reference**2) + np.mean(reference_to_generated**2)
    )
    hausdorff = float(max(
        np.max(generated_to_reference), np.max(reference_to_generated),
    ))
    generated_voxels = _solid_voxels(
        generated, voxel_size, max_voxels_per_shape,
    )
    reference_voxels = _solid_voxels(
        reference, voxel_size, max_voxels_per_shape,
    )
    generated_keys = _row_keys(generated_voxels)
    reference_keys = _row_keys(reference_voxels)
    intersection = int(np.intersect1d(
        generated_keys, reference_keys, assume_unique=True,
    ).size)
    union = int(generated_keys.size + reference_keys.size - intersection)
    return CADDesignerCompatibleMetrics(
        volumetric_iou=float(intersection / union) if union else 0.0,
        chamfer_distance=chamfer,
        hausdorff_distance=hausdorff,
        icp_cost=float(icp_cost),
        point_samples=point_samples,
        voxel_size=voxel_size,
    )
