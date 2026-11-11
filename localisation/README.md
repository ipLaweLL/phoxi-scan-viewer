# CAD-to-point-cloud localisation prototype

Find one known rigid part in an offline 3D scan. The program reports a 4×4 CAD-to-scan transform in millimetres, an ambiguous pose, or no match. It also reports overlap and fit error so candidate results can be compared.

| Input | Role |
| --- | --- |
| STEP or STP | Preferred CAD input. CadQuery tessellates the part for registration. Quiddity supplies feature records for the optional hole checks. |
| STL | CAD input for mesh registration. STL has no standard unit field. Use `--stl-scale 1000` if the mesh coordinates are in metres. |
| PLY or PCD | Recorded point cloud containing XYZ. Use `--scan-scale 1000` if the coordinates are in metres. |

## Install

Use CPython 3.12. The prototype was tested on Linux. On Windows, create a virtual environment and run its Python executable.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux, use `python3.12 -m venv .venv` and `.venv/bin/python -m pip install -r requirements.txt`. The following commands assume that `python` points to the environment where the requirements are installed.

## Run

```text
python localise.py part.step scan.ply --output pose.json
python localise.py part.stl scan.pcd --mesh-only --output pose.json
python localise.py examples/plate_a.step examples/plate_a.ply --viewpoint 0 0 0 --candidates
python benchmark.py benchmark_output
```

- Use `--roi xmin xmax ymin ymax zmin zmax` to crop out the fixture and other objects. Bounds are in scan coordinates, in mm after any scale conversion.
- Supply `--viewpoint x y z` when the optical centre is known in the same coordinate system as the scan. Use `0 0 0` for a scan whose origin is the optical centre. This limits the model coverage calculation to CAD surfaces visible from that point. Without it, coverage includes the full CAD surface and is typically lower for single-view scans.
- Set `--voxel` to a size suited to the point spacing and part dimensions. The default is 1 mm. Use `--help` for scoring thresholds and other options.
- Inspect `mesh_only` and `assisted` in the same JSON to compare the same generated poses. The top-level `feature_status` says whether the STEP feature checks were available. An ambiguous or rejected result has a null pose. Any `best_candidate` under an ambiguous result is diagnostic.
- The transform maps a column vector from the CAD coordinate system to the scan coordinate system, with `p_scan = R @ p_cad + t`. The fit RMSE describes inlier distances to the model surface. Use a known reference pose to measure absolute pose accuracy.

## How it works

Coarse alignment uses Open3D FPFH with RANSAC, plus principal-axis pose seeds. Point-to-plane ICP refines each candidate. Overlap and distances to the CAD surface score the refined poses. With a scanner viewpoint, ray casting excludes CAD surfaces hidden by the model itself.

For STEP, Quiddity extracts verified simple through-holes. The feature check looks for points around the expected opening and checks whether the opening is filled at the plane of the rim. A clearly filled opening vetoes that pose. Sparse or obscured rims remain unobserved. Feature checks evaluate the generated poses, so a missed coarse pose cannot be recovered by the hole check.

## Scope and verification

This version expects a bounded search area with one isolated instance and known CAD. To compare known variants, run the same scan against each model. Pose ambiguity is estimated from the distinct candidates found by the search. Treat the result as an offline feasibility test until thresholds and errors are measured on representative scans from your scanner. The included benchmark generates synthetic single-view scans with known poses, noise and missing points. It exercises both CAD formats and saves detailed JSON for each run.

References

- [Quiddity feature recognition](https://github.com/pzfreo/quiddity)
- [CadQuery STEP import](https://cadquery.readthedocs.io/en/latest/importexport.html)
- [Open3D global registration](https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html)
