# PhoXi Scan Viewer

Desktop viewer for triggering one Photoneo scan through PhoXi Control and inspecting the resulting point cloud. The viewer also opens saved PLY and PCD clouds.

## Start on Windows

Install 64-bit Python 3.12 and [PhoXi Control](https://www.photoneo.com/downloads/phoxi-control/). Use the PhoXi Python API support in PhoXi Control 1.17 or newer. Start PhoXi Control and check that it can see the scanner. If the Python API cannot locate the native library after installation, reopen the terminal and check `$env:PHOXI_CONTROL_PATH` against the PhoXi Control installation folder.

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m viewer
```

Click **Discover**, select the scanner or enter its device ID, then click **Connect** and **Trigger Scan**. After connecting, the **Scanner Profile** dropdown loads the device's actual saved profiles using the PhoXi Python API. Choose one and click **Apply Profile** to activate it (selection alone changes nothing). **Refresh Profiles** reloads the list and currently active profile. Applying a profile changes live scanner settings; it does not modify the startup profile. The viewer switches the scanner to software trigger mode and requests XYZ and texture. Disconnecting restores the previous trigger mode, acquisition state and the output settings the viewer enabled. A PhoXi Control file camera can also be selected. Its recorded frame must contain point cloud data.

Drag to orbit, Shift-drag to pan, and scroll to zoom. Use **Reset View** to refit, adjust the point size, or switch between texture/intensity, white, colour depth and grayscale depth colouring. The frame panel reports valid point count and the elapsed acquisition and transfer time. Points at the all-zero invalid marker and non-finite coordinates are filtered before display. The viewer displays coordinates from PhoXi Control in millimetres. Float and high-bit-depth intensity textures are stretched across robust per-frame percentiles for display, avoiding almost-black uint16 scans; 8-bit intensities preserve their standard 0–255 scale. These display changes do not modify XYZ points.

To test with a saved cloud without running PhoXi Control:

```powershell
.\.venv\Scripts\python.exe -m viewer localisation\examples\plate_a.ply
```

## Modules

| Module | Responsibility |
| --- | --- |
| `viewer/__main__.py` | Starts the application |
| `viewer/app.py` | Controls, background jobs and display status |
| `viewer/phoxi.py` | Device discovery, software triggering, and scanner state restoration |
| `viewer/frames.py` | Frame data, XYZ and texture conversion, PLY/PCD loading |
| `viewer/viewport.py` | Point rendering and camera controls |
| `localisation/` | Original offline CAD localisation CLI, benchmark and examples |

The viewer's scan action displays the acquired frame. Run the [original CLI](localisation/README.md) independently when you need to test CAD localisation.

## Visualization controls

The right-hand panel scrolls and groups controls into collapsible Scanner, Scanner Profiles, Visualization, Depth Contrast, and Frame Information sections.

- Texture displays scan intensity/RGB with gamma adjustment. Plain White displays neutral geometry.
- Depth - Grayscale and Depth - Hue colorize scanner Z coordinates (millimetres). For a flat scene with little depth variation, Smart stretches the central 1st–99th percentile Z range, Full uses the entire observed range, and Apply Manual Range accepts explicit minimum and maximum Z bounds. These are **display mappings only** and do not change the XYZ data.
- Gamma changes the brightness distribution for Texture and depth modes (0.2–3.0).
- Screenshot exports the current 3D viewport as PNG (not the entire application window). Coordinate axes can be shown or hidden.
- Frame Information reports valid point count, acquisition time, and observed Z range.
- Color depth represents scanner-coordinate Z, **not** physical distance from the current orbit camera or plane-relative height. Printed labels may only be visible in Texture mode.

## Single-part localisation in the viewer

Install the extra solver dependencies **from the repository root**:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r localisation\\requirements.txt
```

Launch the GUI with the bundled test point cloud:

```powershell
.\\.venv\\Scripts\\python.exe -m viewer localisation\\examples\\plate_a.ply
```

In **SINGLE-PART LOCALISATION**, click **Choose CAD** and select
`localisation\\examples\\plate_a.step`. Keep the voxel size at 1.0 mm and
**Scan origin is camera optical centre** checked for this example, then select
**Locate Part**. An accepted matching pose draws a semi-transparent green CAD
mesh over the scan. The panel reports candidate count, model coverage, scan
inliers, inlier RMSE, runtime and translation. **Clear Overlay** removes it.

The checkbox must only be enabled for point clouds whose (0,0,0) is the
scanner optical centre. For scans already transformed into other coordinate
frames, uncheck it; incorrect viewpoint coordinates can invalidate scoring.
STEP dimensions are assumed to be millimetres; STL is also assumed mm in
this GUI. Real-world validation and adjustable units/ROI are upcoming work.

The solver uses a temporary PLY for the current displayed XYZ frame,
including scans acquired live from PhoXi. It runs in the existing background
worker so the GUI remains responsive. Ambiguous and no-match results **do
not** render a CAD pose, even if a diagnostic best candidate is available.
Running a new scan clears any previous overlay. The localisation module is
loaded only when requested; scanner viewing still works without its optional
dependencies.

## Preprocessing / plane segmentation experiment

Load a live PhoXi frame or saved PLY/PCD, expand **PREPROCESSING / PLANE**,
set a plane distance threshold (0.35 mm initially), and click **Detect Plane**.
The background worker fits the **largest planar surface** with Open3D RANSAC.
This is **not guaranteed to be the supporting table**. In a tight ROI containing
mostly a 2.5 mm coin, the detected plane may be the *coin's top face*.
Do **not** turn on foreground localisation until you verify the classification.

Choose a preview:
- **Raw**: the original texture/depth scan.
- **Combined**: RANSAC plane **orange/red**, non-plane foreground **cyan**.
- **Plane**: only the detected plane.
- **Foreground**: only the remaining points.

The panel reports plane/foreground counts, plane fraction, equation and
axis-aligned plane-point extents. These are diagnostics, not automated checks
that a supporting table was found. Confirm that the coin remains intact in
**Foreground** and the board has been selected in **Plane**, then explicitly
check **Use inspected foreground for Locate Part**. The next Locate Part run
uses only the foreground points. Unchecking reverts to the raw scan; triggering
a new frame discards the segmentation. This process does not alter the original
scan or registration acceptance thresholds.

If the plane finder selects the coin face, *leave the foreground option off*.
Use a somewhat larger ROI containing sufficient supporting table, or wait for
the later object-aware segmentation phase. Very little table in the ROI is a
failure case for naïve largest-plane RANSAC.

## Candidate diagnostics for live-scene failures

Each Locate Part run now retains **all refined candidates**, even when the
final decision is `no_match` or `ambiguous`. Open **CANDIDATE DIAGNOSTICS**
in the sidebar and choose hypothesis #1, #2, and so on. The panel lists
CAD coverage, **global** scan-inlier fraction, inlier RMSE, mesh score
(in the selector), reasons for failing overlap, CAD-origin translation and
the 3x3 rotation matrix. **Export Candidate JSON** saves the entire result,
including 4x4 transforms, for reproducible debugging.

Check **Preview selected candidate (unverified)** to overlay a hypothesis
in **orange**. This is a diagnostic visualization and **not** a verified 6D
pose or robot command. A genuine accepted pose remains **green**.
Changing candidate updates the orange preview. Unchecking preview restores
a valid green pose if one exists; new scans and CAD changes clear results.

A low global scan-inlier fraction can be caused by residual table points in
the ROI. Passed overlap alone is not sufficient to resolve rotational
symmetry: a round, featureless coin may still be `ambiguous`. We have
**not** relaxed the registration thresholds or changed the solver. This
feature is intended to expose why the solver rejected live scans.

## Tests

Run the acquisition and frame-conversion tests with `python -m unittest discover -s tests`. These tests use a simulated PhoXi device. Live triggering requires a running PhoXi Control instance and a reachable scanner or file camera.

[Photoneo Python API reference](https://docs.photoneo.com/docs/3d_sensors/1.18/wrappers/phoxi_python_api/1.1.1/phoxi_api.html)
