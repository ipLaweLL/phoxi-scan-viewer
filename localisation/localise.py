"""Offline localisation of one known CAD part in a bounded 3D scan."""
import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

# Limit overhead from OpenMP on small point clouds. Users can override it.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REG = o3d.pipelines.registration


def cloud(points):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def load_mesh(path, tolerance, stl_scale):
    if path.suffix.lower() in {".step", ".stp"}:
        import cadquery as cq

        part = cq.importers.importStep(str(path))
        if not part.vals():
            raise ValueError("STEP contains no importable shapes")
        shape = cq.Compound.makeCompound(part.vals())
        vertices, triangles = shape.tessellate(tolerance, 0.1)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector([v.toTuple() for v in vertices]),
            o3d.utility.Vector3iVector(triangles),
        )
    elif path.suffix.lower() == ".stl":
        mesh = o3d.io.read_triangle_mesh(str(path))
        mesh.scale(stl_scale, center=(0, 0, 0))
    else:
        raise ValueError("CAD model must be STEP, STP or STL")
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    if len(mesh.triangles) == 0 or mesh.get_surface_area() <= 0:
        raise ValueError("CAD model has no usable surface")
    return mesh


def ray_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def load_scan(path, scale, roi):
    if path.suffix.lower() not in {".ply", ".pcd"}:
        raise ValueError("Scan must be PLY or PCD")
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points) * scale
    points = points[np.isfinite(points).all(axis=1)]
    if roi is not None:
        limits = np.array(roi).reshape(3, 2)
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("ROI minimums must be smaller than maximums")
        points = points[((points >= limits[:, 0]) & (points <= limits[:, 1])).all(axis=1)]
    if len(points) < 50:
        raise ValueError("Scan ROI contains fewer than 50 valid points")
    return points


def preprocess(points, voxel):
    reduced = points.voxel_down_sample(voxel)
    if len(reduced.points) < 30:
        raise ValueError("Too few points after downsampling. Use a smaller voxel size.")
    reduced.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(voxel * 2.5, 35))
    return reduced


def pca_seeds(source, target):
    def frame(points):
        center = points.mean(axis=0)
        _, axes = np.linalg.eigh(np.cov(points.T))
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        return center, axes

    source_center, source_axes = frame(np.asarray(source.points))
    target_center, target_axes = frame(np.asarray(target.points))
    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        pose = np.eye(4)
        pose[:3, :3] = (target_axes * signs) @ source_axes.T
        pose[:3, 3] = target_center - pose[:3, :3] @ source_center
        yield pose


def initial_poses(source, target, voxel):
    search = o3d.geometry.KDTreeSearchParamHybrid(voxel * 5, 100)
    source_fpfh = REG.compute_fpfh_feature(source, search)
    target_fpfh = REG.compute_fpfh_feature(target, search)
    poses = list(pca_seeds(source, target))
    for seed in (7, 19, 37):
        o3d.utility.random.seed(seed)
        result = REG.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True, voxel * 1.75,
            REG.TransformationEstimationPointToPoint(False), 3,
            [REG.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             REG.CorrespondenceCheckerBasedOnDistance(voxel * 1.75)],
            REG.RANSACConvergenceCriteria(25000, 0.999),
        )
        if result.fitness > 0:
            poses.append(result.transformation)
    return poses


def extract_holes(path):
    import quiddity

    records = json.loads(quiddity.recognise(str(path)))["document"]["records"]
    holes = []
    for record in records:
        if record["family"] != "holes" or record["evidence_status"] != "verified":
            continue
        params = record["parameters"]
        if params.get("bottom") != "through" or any(
            params.get(key) for key in ("cbore", "csink", "countersink")
        ):
            continue
        axis = params.get("axis")
        if isinstance(axis, str) and axis in "xyz" and len(axis) == 1:
            axis = np.eye(3)["xyz".index(axis)]
        location = params.get("location")
        radius = params.get("diameter", 0) / 2
        depth = params.get("depth", 0)
        if axis is None or location is None or radius <= 0 or depth <= 0:
            continue
        axis = np.asarray(axis, dtype=float)
        location = np.asarray(location, dtype=float)
        if axis.shape != (3,) or location.shape != (3,) or np.linalg.norm(axis) == 0:
            continue
        axis /= np.linalg.norm(axis)
        holes.append((record["id"], location, axis, radius, depth))
    return holes


def check_holes(holes, pose, scan, scan_tree, spacing, voxel, viewpoint):
    checks = []
    for feature_id, entry, axis, radius, depth in holes:
        entry = transform(entry, pose)
        axis = pose[:3, :3] @ axis
        states = []
        for center, outward in ((entry, -axis), (entry + axis * depth, axis)):
            if viewpoint is not None:
                view = viewpoint - center
                if view @ outward <= 0.25 * np.linalg.norm(view):
                    continue
            if radius < 3 * spacing:
                continue
            slab = max(2 * spacing, 0.35 * voxel)
            local = scan[scan_tree.query_ball_point(center, np.hypot(1.55 * radius, slab))] - center
            height = local @ axis
            radial = local - height[:, None] * axis
            in_plane = np.abs(height) <= slab
            radial = radial[in_plane]
            distance = np.linalg.norm(radial, axis=1)
            basis = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
            basis /= np.linalg.norm(basis)
            angles = np.arctan2(radial @ np.cross(axis, basis), radial @ basis)
            sectors = np.floor((angles + np.pi) * (12 / (2 * np.pi))).astype(int) % 12
            ring = (distance >= 1.05 * radius) & (distance <= 1.55 * radius)
            core = distance < 0.62 * radius
            ring_sectors = len(np.unique(sectors[ring]))
            core_sectors = len(np.unique(sectors[core]))
            if ring_sectors >= 8:
                states.append("supported" if core_sectors <= 1 else
                              "contradicted" if core_sectors >= 6 else "unobserved")
        state = "contradicted" if "contradicted" in states else (
            "supported" if "supported" in states else "unobserved"
        )
        checks.append({"id": feature_id, "state": state})
    return checks


def candidate_metrics(pose, sample, scan, scan_tree, geometry, distance, viewpoint):
    inverse = np.linalg.inv(pose)
    scan_in_model = transform(scan, inverse)
    query = o3d.core.Tensor(scan_in_model.astype(np.float32))
    scan_distance = geometry.compute_distance(query).numpy()
    scan_inliers = scan_distance <= distance

    visible = sample
    if viewpoint is not None:
        origin = transform(viewpoint, inverse)
        rays = np.column_stack((np.broadcast_to(origin, sample.shape), sample - origin))
        hit = geometry.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
        hit_distance = np.abs(hit - 1) * np.linalg.norm(sample - origin, axis=1)
        visible = sample[hit_distance <= min(distance * 0.25, 0.3)]
    model_distance = scan_tree.query(transform(visible, pose))[0]
    coverage = float(np.mean(model_distance <= distance)) if len(visible) else 0.0
    precision = float(np.mean(scan_inliers))
    rmse = float(np.sqrt(np.mean(scan_distance[scan_inliers] ** 2))) if np.any(scan_inliers) else None
    score = (precision + coverage) / 2 - (0.15 * rmse / distance if rmse is not None else 0.15)
    return coverage, precision, rmse, score


def select(candidates, assisted, ambiguity_margin, using_features):
    eligible = [c for c in candidates if c["passes_overlap"]]
    score_key = "assisted_score" if assisted else "mesh_score"
    if assisted:
        eligible = [c for c in eligible if not any(
            check["state"] == "contradicted" for check in c["feature_checks"]
        )]
    if not eligible:
        return {"status": "no_match", "pose_cad_to_scan": None, "best_candidate": None}
    ranked = sorted(eligible, key=lambda c: c[score_key], reverse=True)
    best = ranked[0]
    tied = len(ranked) > 1 and best[score_key] - ranked[1][score_key] < ambiguity_margin
    return {
        "status": "ambiguous" if tied else "match",
        "pose_cad_to_scan": None if tied else best["pose_cad_to_scan"],
        "best_candidate": best,
        **({"feature_evidence_available": using_features and any(
            check["state"] == "supported" for check in best["feature_checks"]
        )} if assisted else {}),
    }


def localise(args):
    started = perf_counter()
    voxel = args.voxel
    viewpoint = None if args.viewpoint is None else np.asarray(args.viewpoint)
    mesh = load_mesh(args.cad, voxel * 0.15, args.stl_scale)
    raw = load_scan(args.scan, args.scan_scale, args.roi)
    o3d.utility.random.seed(42)
    # Target roughly two samples per voxel square, with bounded memory.
    count = int(np.clip(2 * mesh.get_surface_area() / voxel**2, 5000, 80000))
    cad_cloud = mesh.sample_points_uniformly(count)
    source = preprocess(cad_cloud, voxel)
    target = preprocess(cloud(raw), voxel)
    target.orient_normals_towards_camera_location(np.zeros(3) if viewpoint is None else viewpoint)
    cad_points = np.asarray(source.points)
    scan_points = np.asarray(target.points)
    scan_tree = cKDTree(scan_points)
    raw_tree = cKDTree(raw)
    sample_indices = np.linspace(0, len(raw) - 1, min(2000, len(raw)), dtype=int)
    spacing = max(0.05 * voxel, float(np.median(raw_tree.query(raw[sample_indices], k=2)[0][:, 1])))
    geometry = ray_scene(mesh)
    feature_status = "disabled"
    holes = []
    if not args.mesh_only and args.cad.suffix.lower() in {".step", ".stp"}:
        try:
            holes = extract_holes(args.cad)
            feature_status = "ready" if holes else "no_supported_features"
        except (ImportError, ValueError, KeyError, TypeError) as exc:
            feature_status = f"unavailable: {exc}"
    elif not args.mesh_only:
        feature_status = "step_required"
    refine = REG.TransformationEstimationPointToPlane(REG.HuberLoss(voxel))
    criteria = REG.ICPConvergenceCriteria(max_iteration=45)
    candidates = []
    reference = cad_points.mean(axis=0)
    for seed in initial_poses(source, target, voxel):
        pose = REG.registration_icp(source, target, voxel * 3, seed, refine, criteria).transformation
        pose = REG.registration_icp(source, target, voxel * 1.5, pose, refine, criteria).transformation
        if not np.isfinite(pose).all():
            continue
        coverage, precision, rmse, score = candidate_metrics(
            pose, cad_points, scan_points, scan_tree, geometry, voxel * 1.5, viewpoint
        )
        checks = check_holes(holes, pose, raw, raw_tree, spacing, voxel, viewpoint)
        supported = sum(check["state"] == "supported" for check in checks)
        candidates.append({
            "pose_cad_to_scan": pose.tolist(),
            "model_coverage": coverage,
            "scan_inlier_fraction": precision,
            "inlier_rmse_mm": rmse,
            "mesh_score": score,
            "assisted_score": score + 0.04 * min(supported, 3),
            "feature_checks": checks,
            "passes_overlap": coverage >= args.min_coverage and precision >= args.min_scan_inliers,
        })
    unique = []
    for candidate in sorted(candidates, key=lambda c: c["mesh_score"], reverse=True):
        pose = np.asarray(candidate["pose_cad_to_scan"])
        center = transform(reference, pose)
        if any(
            np.linalg.norm(center - transform(reference, previous)) < 1.5 * voxel
            and Rotation.from_matrix(pose[:3, :3] @ previous[:3, :3].T).magnitude() < np.deg2rad(5)
            for previous in (np.asarray(other["pose_cad_to_scan"]) for other in unique)
        ):
            continue
        unique.append(candidate)
    result = {
        "cad": str(args.cad),
        "scan": str(args.scan),
        "units": "mm",
        "model_coverage_basis": "self-visible CAD surface" if viewpoint is not None else "whole CAD surface",
        "feature_status": feature_status,
        "checked_cad_holes": len(holes),
        "voxel_mm": voxel,
        "candidate_count": len(unique),
        "mesh_only": select(unique, False, args.ambiguity_margin, False),
        "assisted": select(unique, True, args.ambiguity_margin, bool(holes)) if holes else None,
        "runtime_seconds": perf_counter() - started,
    }
    if args.candidates:
        result["candidates"] = unique
    return result


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cad", type=Path)
    cli.add_argument("scan", type=Path)
    cli.add_argument("--voxel", type=float, default=1.0, help="Registration voxel size in mm")
    cli.add_argument("--scan-scale", type=float, default=1.0, help="PLY/PCD units to mm")
    cli.add_argument("--stl-scale", type=float, default=1.0, help="STL units to mm")
    cli.add_argument("--roi", type=float, nargs=6, metavar="MM", help="xmin xmax ymin ymax zmin zmax")
    cli.add_argument("--viewpoint", type=float, nargs=3, metavar="MM", help="Scanner optical centre in scan coordinates")
    cli.add_argument("--min-coverage", type=float, default=0.25)
    cli.add_argument("--min-scan-inliers", type=float, default=0.65)
    cli.add_argument("--ambiguity-margin", type=float, default=0.018)
    cli.add_argument("--mesh-only", action="store_true")
    cli.add_argument("--candidates", action="store_true", help="Include all distinct refined candidates")
    cli.add_argument("--output", type=Path, help="Save result as JSON")
    return cli


def main():
    cli = parser()
    args = cli.parse_args()
    numeric = [args.voxel, args.scan_scale, args.stl_scale, args.min_coverage,
               args.min_scan_inliers, args.ambiguity_margin, *(args.roi or []),
               *(args.viewpoint or [])]
    if not np.isfinite(numeric).all() or min(args.voxel, args.scan_scale, args.stl_scale) <= 0:
        cli.error("numeric options must be finite and voxel/scale must be positive")
    if not all(0 < x <= 1 for x in (args.min_coverage, args.min_scan_inliers, args.ambiguity_margin)):
        cli.error("coverage, inlier fraction, and ambiguity margin must be in (0, 1]")
    if not args.cad.is_file() or not args.scan.is_file():
        cli.error("CAD and scan files must both exist")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    try:
        result = json.dumps(localise(args), indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(result, encoding="utf-8")
        else:
            sys.stdout.write(result)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        cli.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
