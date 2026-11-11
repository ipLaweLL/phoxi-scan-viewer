"""Non-destructive RANSAC plane segmentation for PhoXi point clouds (mm).

The largest plane may be the object's face, especially in tight ROIs.
No segmentation is trusted automatically: the user must inspect the preview.
"""
from dataclasses import dataclass
import numpy as np
import open3d as o3d
from .frames import ScanFrame


@dataclass(frozen=True)
class Segmentation:
    source: ScanFrame
    plane_mask: np.ndarray
    equation: np.ndarray
    threshold_mm: float
    plane_extent_mm: tuple

    @property
    def plane_count(self):
        return int(np.count_nonzero(self.plane_mask))

    @property
    def foreground_count(self):
        return len(self.source.points) - self.plane_count

    @property
    def plane_fraction(self):
        return self.plane_count / len(self.source.points)

    def frame(self, which):
        if which not in ("Plane", "Foreground"):
            raise ValueError("Unknown segmentation view.")
        keep = self.plane_mask if which == "Plane" else ~self.plane_mask
        colors = self.source.colors[keep].copy() if self.source.colors is not None else None
        return ScanFrame(
            points=self.source.points[keep].copy(),
            colors=colors,
            source=self.source.source + " / " + which,
            frame_id=self.source.frame_id,
            elapsed_ms=self.source.elapsed_ms,
        )


def segment_plane(frame, threshold_mm=0.35, iterations=1000):
    threshold_mm = float(threshold_mm)
    if not np.isfinite(threshold_mm) or not (0.05 <= threshold_mm <= 3.0):
        raise ValueError("Plane threshold must be from 0.05 to 3.0 mm.")
    xyz = np.asarray(frame.points, dtype=np.float64)
    if len(xyz) < 100 or not np.isfinite(xyz).all():
        raise ValueError("Need at least 100 finite points.")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    equation, indices = cloud.segment_plane(
        distance_threshold=threshold_mm, ransac_n=3,
        num_iterations=int(iterations))
    mask = np.zeros(len(xyz), dtype=bool)
    mask[np.asarray(indices, dtype=np.int64)] = True
    if mask.sum() < 30 or (~mask).sum() < 50:
        raise ValueError("Segmentation does not preserve enough plane and foreground points.")
    # Planar footprint is diagnostic only, not an object-vs-table classifier.
    extent = np.ptp(xyz[mask], axis=0)
    return Segmentation(
        source=frame, plane_mask=mask, equation=np.asarray(equation, dtype=float),
        threshold_mm=threshold_mm, plane_extent_mm=tuple(map(float, extent)))


def preview_colors(segment):
    """Show red/orange plane and cyan foreground, preserving raw XYZ."""
    colors = np.empty((len(segment.source.points), 3), dtype=np.float32)
    colors[segment.plane_mask] = (1.0, 0.34, 0.10)
    colors[~segment.plane_mask] = (0.05, 0.85, 0.95)
    return ScanFrame(
        points=segment.source.points.copy(), colors=colors,
        source=segment.source.source + " / Plane vs foreground",
        frame_id=segment.source.frame_id,
        elapsed_ms=segment.source.elapsed_ms)
