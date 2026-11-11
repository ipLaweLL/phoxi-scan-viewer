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
    """Convert scanner intensity or RGB to display colours, never changing XYZ.

    PhoXi intensity is often stored as uint16 although its useful dynamic
    range occupies only a small fraction of [0, 65535]. Using the dtype
    ceiling would render a perfectly valid scan almost black.
    """
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros((len(values), 3), dtype=np.float32)

    if values.shape[1] == 1:
        data = values.astype(np.float32)
        if np.issubdtype(values.dtype, np.integer) and values.dtype.itemsize == 1:
            # Byte intensities are already on a standard display scale.
            scaled = data / 255.0
        else:
            # Ignore extreme intensities; do not let sparse bright points
            # flatten the contrast of the whole cloud.
            low, high = np.percentile(finite, [1, 99.5])
            if high <= low:
                scaled = np.clip(data / max(float(high), 1.0), 0, 1)
            else:
                scaled = (data - low) / (high - low)
        scaled = np.clip(np.nan_to_num(scaled, nan=0, posinf=1, neginf=0), 0, 1)
        return np.repeat(scaled, 3, axis=1)

    # RGB: maintain ratios, avoiding per-channel percentile stretching.
    if np.issubdtype(values.dtype, np.integer) and values.dtype.itemsize == 1:
        white = 255.0
    else:
        white = max(float(np.percentile(finite, 99.5)), 1.0)
    scaled = np.nan_to_num(values.astype(np.float32) / white, nan=0, posinf=1, neginf=0)
    return np.clip(scaled, 0, 1)


def load_point_cloud(path: Path) -> ScanFrame:
    import open3d as o3d

    if not path.is_file() or path.suffix.lower() not in {".ply", ".pcd"}:
        raise ValueError("Choose an existing PLY or PCD point cloud.")
    cloud = o3d.io.read_point_cloud(str(path))
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None
    return ScanFrame.from_arrays(np.asarray(cloud.points), colors, source=path.name)
