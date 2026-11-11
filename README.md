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

Run the acquisition and frame-conversion tests with `python -m unittest discover -s tests`. These tests use a simulated PhoXi device. Live triggering requires a running PhoXi Control instance and a reachable scanner or file camera.

[Photoneo Python API reference](https://docs.photoneo.com/docs/3d_sensors/1.18/wrappers/phoxi_python_api/1.1.1/phoxi_api.html)
