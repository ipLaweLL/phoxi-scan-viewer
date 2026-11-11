"""Frame data shared by the scanner and the 3D viewer."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScanFrame:
    points: np.ndarray
    colors: np.ndarray | None
    source: str
    frame_id: int | None = None
    elapsed_ms: float | None = None

    @classmethod
    def from_arrays(cls, xyz, texture=None, *, source, frame_id=None, elapsed_ms=None):
        points = np.asarray(xyz, dtype=np.float32)
        if points.ndim < 2 or points.shape[-1] != 3:
            raise ValueError("Expected XYZ point cloud with three coordinates per point.")

        sample_shape = points.shape[:-1]
        points = points.reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
        if not valid.any():
            raise ValueError("The frame contains no valid 3D points.")

        colors = None
        if texture is not None:
            values = np.asarray(texture)
            if values.shape == sample_shape:
                values = values.reshape(-1, 1)[valid]
            elif (values.ndim == len(sample_shape) + 1 and values.shape[:-1] == sample_shape
                  and values.shape[-1] in (1, 3)):
                values = values.reshape(-1, values.shape[-1])[valid]
            else:
                values = None
            if values is not None:
                colors = _display_colors(values)

        return cls(points[valid], colors, source, frame_id, elapsed_ms)


def _display_colors(values):
    # Float PhoXi textures may use sensor intensity units. Preserve RGB ratios
    # with a shared white point and discard invalid values.
    if np.issubdtype(values.dtype, np.integer):
        white = np.iinfo(values.dtype).max
    else:
        finite = values[np.isfinite(values)]
        white = 1.0
        if len(finite) and finite.max() > 1:
            white = max(1.0, float(np.percentile(finite, 99.5)))
    scaled = np.nan_to_num(values.astype(np.float32) / white, nan=0, posinf=1, neginf=0)
    colors = np.clip(scaled, 0, 1)
    return np.repeat(colors, 3, axis=1) if colors.shape[1] == 1 else colors


def load_point_cloud(path: Path) -> ScanFrame:
    import open3d as o3d

    if not path.is_file() or path.suffix.lower() not in {".ply", ".pcd"}:
        raise ValueError("Choose an existing PLY or PCD point cloud.")
    cloud = o3d.io.read_point_cloud(str(path))
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    return ScanFrame.from_arrays(np.asarray(cloud.points), colors, source=path.name)
