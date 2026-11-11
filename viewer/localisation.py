"""Bridge between the viewer's in-memory frame and single-part localisation CLI.

Runs entirely on the application's worker thread. Live scans are saved into a
temporary PLY in millimetres because the existing algorithm expects a path.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import open3d as o3d



def select_pose(result):
    """Use assisted evidence if available; never display an ambiguous pose."""
    assisted = result.get("assisted")
    selected = assisted if assisted is not None else result["mesh_only"]
    if selected["status"] != "match" or selected.get("pose_cad_to_scan") is None:
        return None, selected["status"], selected.get("best_candidate")
    pose = np.asarray(selected["pose_cad_to_scan"], dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Localisation returned an invalid 4x4 pose.")
    return pose, "match", selected["best_candidate"]


def locate_frame(frame, cad_path, *, voxel=1.0, viewpoint_is_origin=True):
    from localisation.localise import load_mesh, localise

    cad_path = Path(cad_path)
    if not cad_path.is_file() or cad_path.suffix.lower() not in (".step", ".stp", ".stl"):
        raise ValueError("Choose an existing STEP, STP or STL CAD file.")
    points = np.asarray(frame.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 50:
        raise ValueError("Scan must have at least 50 XYZ points.")
    with TemporaryDirectory(prefix="phoxi_localise_") as directory:
        scan = Path(directory) / "frame.ply"
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        if not o3d.io.write_point_cloud(str(scan), cloud, write_ascii=False):
            raise OSError("Could not save temporary scan for localisation.")
        options = SimpleNamespace(
            cad=cad_path, scan=scan, voxel=float(voxel),
            stl_scale=1.0, scan_scale=1.0, roi=None,
            viewpoint=[0.0, 0.0, 0.0] if viewpoint_is_origin else None,
            mesh_only=False, candidates=False,
            min_coverage=0.25, min_scan_inliers=0.65,
            ambiguity_margin=0.018,
        )
        result = localise(options)
    pose, status, candidate = select_pose(result)
    # Mesh stays untransformed until a valid match has been verified.
    mesh = load_mesh(cad_path, float(voxel) * 0.15, 1.0) if pose is not None else None
    return result, pose, candidate, mesh
