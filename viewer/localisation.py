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


def diagnostic_rows(result):
    """Summarise every refined hypothesis without changing solver thresholds."""
    candidates = result.get("candidates", [])
    rows = []
    for index, candidate in enumerate(candidates):
        coverage = float(candidate["model_coverage"])
        inliers = float(candidate["scan_inlier_fraction"])
        rmse = candidate.get("inlier_rmse_mm")
        reasons = []
        if coverage < 0.25:
            reasons.append("low CAD coverage")
        if inliers < 0.65:
            reasons.append("low global scan-inlier fraction")
        for check in candidate.get("feature_checks", []):
            if check.get("state") == "contradicted":
                reasons.append("CAD feature contradicted")
                break
        # Prefer the actual engine result; thresholds above are explanatory
        # defaults matching the single-part CLI configuration.
        passed = bool(candidate["passes_overlap"])
        if not passed and not reasons:
            reasons.append("overlap threshold not passed")
        rows.append({
            "index": index,
            "passes_overlap": passed,
            "coverage": coverage,
            "inliers": inliers,
            "rmse": rmse,
            "mesh_score": float(candidate["mesh_score"]),
            "reasons": reasons,
            "pose": candidate["pose_cad_to_scan"],
        })
    return rows


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
            mesh_only=False, candidates=True,
            min_coverage=0.25, min_scan_inliers=0.65,
            ambiguity_margin=0.018,
        )
        result = localise(options)
    pose, status, candidate = select_pose(result)
    # Mesh stays untransformed until a valid match has been verified.
    mesh = load_mesh(cad_path, float(voxel) * 0.15, 1.0) if result.get("candidates") else None
    return result, pose, candidate, mesh
