"""Generate reproducible synthetic scans and compare both localisation modes."""
import argparse
import json
from pathlib import Path

from localise import cloud, load_mesh, localise, o3d, ray_scene, transform
from localise import parser as localisation_parser

import cadquery as cq
import numpy as np
from scipy.spatial.transform import Rotation


def model(kind):
    if kind in ("plate_a", "plate_b"):
        holes = [(17, 8), (-6, -10)] if kind == "plate_a" else [(17, -8), (-10, 10)]
        return (cq.Workplane("XY").box(76, 48, 12)
                .faces(">Z").workplane().pushPoints(holes).hole(6))
    if kind == "bracket":
        base = cq.Workplane("XY").box(64, 39, 10)
        upright = cq.Workplane("XY").box(14, 39, 23).translate((-25, 0, 16))
        rib = cq.Workplane("XY").box(28, 14, 7).translate((7, 10, 8.5))
        return base.union(upright).union(rib)
    if kind == "rounded":
        return (cq.Workplane("XY").box(68, 44, 20).edges("|Z").fillet(9)
                .union(cq.Workplane("XY").cylinder(12, 11).translate((14, 5, 13))))
    return cq.Workplane("XY").box(76, 48, 12)


def scan_part(mesh, pose, rng, noise, occlude=False):
    scene = ray_scene(mesh.transform(pose))
    rays = scene.create_rays_pinhole(36, pose[:3, 3], [0., 0., 0.], [0., 1., 0.], 360, 280)
    distance = scene.cast_rays(rays)["t_hit"].numpy().reshape(-1)
    rays = rays.numpy().reshape(-1, 6)
    valid = np.isfinite(distance)
    points = rays[valid, :3] + rays[valid, 3:] * distance[valid, None]
    if occlude:
        points = points[points[:, 0] < np.quantile(points[:, 0], 0.78)]
    points = points[rng.random(len(points)) > 0.05]
    points += rng.normal(0, noise, points.shape)
    stray = rng.uniform(points.min(axis=0) - 4, points.max(axis=0) + 4, (len(points) // 60, 3))
    return np.vstack((points, stray))


def compare(result, expected):
    if result is None:
        return None
    pose = result["pose_cad_to_scan"]
    metrics = {"status": result["status"]}
    if pose is not None and expected is not None:
        pose = np.asarray(pose)
        translation = float(np.linalg.norm(transform([0, 0, 0], pose) - expected[:3, 3]))
        angle = float(np.rad2deg(Rotation.from_matrix(pose[:3, :3] @ expected[:3, :3].T).magnitude()))
        metrics.update(translation_error_mm=round(translation, 3), rotation_error_deg=round(angle, 3))
    candidate = result["best_candidate"]
    if candidate:
        metrics["feature_checks"] = candidate["feature_checks"]
    return metrics


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, help="Output folder for generated CAD, scans, and results")
    args = cli.parse_args()
    output = args.directory
    output.mkdir(parents=True, exist_ok=True)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    cad_files = {}
    for kind in ("plate_a", "plate_b", "bracket", "rounded", "plain_plate"):
        path = output / f"{kind}.step"
        cq.exporters.export(model(kind), str(path))
        cad_files[kind] = path
    bracket_stl = output / "bracket.stl"
    cq.exporters.export(model("bracket"), str(bracket_stl), tolerance=0.1, angularTolerance=0.1)

    scenes = (
        ("plate_a", "plate_a", (166, 16, -24), False),
        ("plate_b", "plate_b", (158, -18, 35), False),
        ("bracket", "bracket", (146, 28, 47), False),
        ("rounded", "rounded", (162, -19, -42), False),
        ("partial_plate", "plate_a", (166, 16, -24), True),
        ("symmetric_plate", "plain_plate", (166, 16, -24), False),
    )
    comparison = []
    for index, (label, shape, angles, occlude) in enumerate(scenes):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        pose[:3, 3] = [8, -14, 224]
        scan = output / f"{label}.ply"
        points = scan_part(load_mesh(cad_files[shape], 0.07, 1), pose,
                           np.random.default_rng(13 + index), 0.12, occlude)
        if not o3d.io.write_point_cloud(str(scan), cloud(points)):
            raise RuntimeError(f"Could not write {scan}")
        candidates = [(cad_files[shape], True)]
        if label in ("plate_a", "plate_b"):
            other = "plate_b" if shape == "plate_a" else "plate_a"
            candidates.append((cad_files[other], False))
        elif label == "bracket":
            candidates.extend(((bracket_stl, True), (cad_files["plate_a"], False)))
        for cad, present in candidates:
            settings = localisation_parser().parse_args(
                [str(cad), str(scan), "--viewpoint", "0", "0", "0", "--candidates"]
            )
            result = localise(settings)
            result["ground_truth_cad_to_scan"] = pose.tolist() if present else None
            (output / f"{label}__{cad.stem}_{cad.suffix[1:]}.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            comparison.append({
                "scan": label, "cad": cad.name, "cad_present": present,
                "mesh_only": compare(result["mesh_only"], pose if present else None),
                "assisted": compare(result["assisted"], pose if present else None),
                "runtime_seconds": round(result["runtime_seconds"], 3),
            })
    summary = json.dumps(comparison, indent=2, allow_nan=False) + "\n"
    (output / "benchmark_results.json").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
