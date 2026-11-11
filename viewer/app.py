"""Desktop UI and serialized acquisition commands."""
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

from .frames import load_point_cloud
from .phoxi import PhoXiScanner
from .localisation import locate_frame, diagnostic_rows
from .segmentation import segment_plane, preview_colors
from .viewport import PointCloudViewport, gui

LOG = logging.getLogger(__name__)


class ScanViewer:
    def __init__(self, app):
        self.window = app.create_window("PhoXi Scan Viewer", 1280, 800)
        self._scanner = PhoXiScanner()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phoxi")
        self._pending = None
        self._closing = False
        self._ready_to_close = False
        self._device_id = None
        self._has_frame = False
        self._profile_names = []
        self._active_profile = None
        self._viewport = PointCloudViewport(self.window.renderer)
        self._cad_path = None
        self._current_frame = None
        self._diagnostic_result = None
        self._diagnostic_mesh = None
        self._diagnostic_rows = []
        self._segmentation = None
        self._seg_view = 'Raw'

        em = self.window.theme.font_size
        spacing = int(0.55 * em)
        self._sidebar_width = int(21 * em)
        self._sidebar = gui.ScrollableVert(spacing, gui.Margins(em, em, em, em))
        self._sidebar.background_color = gui.Color(0.095, 0.12, 0.16)

        # A sectioned, scrollable sidebar keeps future processing and calibration
        # controls from pushing scanner controls outside the window.
        self._scanner_section = gui.CollapsableVert("SCANNER", spacing)
        self._sidebar.add_child(self._scanner_section)
        self._devices = gui.Combobox()
        self._devices.set_on_selection_changed(self._choose_device)
        self._scanner_section.add_child(self._devices)
        self._refresh = self._button("Discover", self.discover, self._scanner_section)
        self._scanner_id = gui.TextEdit()
        self._scanner_id.placeholder_text = "Device ID (for example SED-134)"
        self._scanner_id.set_on_text_changed(lambda _value: self._update_controls())
        self._scanner_section.add_child(self._scanner_id)
        self._connect = self._button("Connect", self._toggle_connection, self._scanner_section)
        self._trigger = self._button("Trigger Scan", self._trigger_scan, self._scanner_section)
        self._trigger.background_color = gui.Color(0.16, 0.37, 0.66)

        self._profile_section = gui.CollapsableVert("SCANNER PROFILES", spacing)
        self._sidebar.add_child(self._profile_section)
        self._profiles = gui.Combobox()
        self._profile_section.add_child(self._profiles)
        profile_actions = gui.Horiz(int(0.4 * em))
        self._apply_profile = self._button("Apply Profile", self._change_profile, profile_actions)
        self._reload_profiles = self._button("Refresh", self._refresh_profiles, profile_actions)
        self._profile_section.add_child(profile_actions)
        self._open = self._button("Open PLY / PCD", self._open_dialog, self._profile_section)

        self._view_section = gui.CollapsableVert("VISUALIZATION", spacing)
        self._sidebar.add_child(self._view_section)
        self._mode = gui.Combobox()
        for mode in ("Texture", "Plain White", "Depth - Grayscale", "Depth - Hue"):
            self._mode.add_item(mode)
        self._mode.set_on_selection_changed(
            lambda mode, _index: self._viewport.set_mode(mode))
        self._view_section.add_child(self._mode)
        self._view_section.add_child(gui.Label("Point size"))
        size = gui.Slider(gui.Slider.INT)
        size.set_limits(1, 8)
        size.int_value = 2
        size.set_on_value_changed(self._viewport.set_point_size)
        self._view_section.add_child(size)
        view_actions = gui.Horiz(int(0.4 * em))
        self._reset = self._button("Reset View", self._viewport.reset_view, view_actions)
        self._screenshot = self._button("Screenshot", self._save_screenshot, view_actions)
        self._view_section.add_child(view_actions)
        self._axes = gui.Checkbox("Show coordinate axes")
        self._axes.set_on_checked(self._viewport.set_show_axes)
        self._view_section.add_child(self._axes)

        self._contrast_section = gui.CollapsableVert("DEPTH CONTRAST", spacing)
        self._sidebar.add_child(self._contrast_section)
        contrast_actions = gui.Horiz(int(0.4 * em))
        self._smart = self._button("Smart", lambda: self._set_contrast("Smart"), contrast_actions)
        self._full = self._button("Full", lambda: self._set_contrast("Full"), contrast_actions)
        self._contrast_section.add_child(contrast_actions)
        self._contrast_section.add_child(gui.Label("Manual depth range (Z, mm)"))
        range_row = gui.Horiz(int(0.4 * em))
        self._min_depth = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._max_depth = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._min_depth.set_value(0.0)
        self._max_depth.set_value(1000.0)
        range_row.add_child(gui.Label("Min"))
        range_row.add_child(self._min_depth)
        range_row.add_child(gui.Label("Max"))
        range_row.add_child(self._max_depth)
        self._contrast_section.add_child(range_row)
        self._manual_button = self._button(
            "Apply Manual Range", self._apply_manual_range, self._contrast_section)
        self._contrast_section.add_child(gui.Label("Gamma"))
        gamma_row = gui.Horiz(int(0.4 * em))
        self._gamma_slider = gui.Slider(gui.Slider.DOUBLE)
        self._gamma_slider.set_limits(0.2, 3.0)
        self._gamma_slider.double_value = 1.0
        self._gamma_slider.set_on_value_changed(self._viewport.set_gamma)
        gamma_row.add_child(self._gamma_slider)
        self._reset_gamma = self._button(
            "Reset", self._reset_gamma_value, gamma_row)
        self._contrast_section.add_child(gamma_row)
        self._contrast_section.add_child(gui.Label(
            "Depth colours use scanner Z; texture preserves captured intensity."))

        self._frame_section = gui.CollapsableVert("FRAME INFORMATION", spacing)
        self._sidebar.add_child(self._frame_section)
        self._source = gui.Label("No frame")
        self._points = gui.Label("Valid points: —")
        self._time = gui.Label("Capture: —")
        self._zrange = gui.Label("Z range: —")
        for label in (self._source, self._points, self._time, self._zrange):
            self._frame_section.add_child(label)

        self._seg_section = gui.CollapsableVert("PREPROCESSING / PLANE", spacing)
        self._sidebar.add_child(self._seg_section)
        self._seg_section.add_child(gui.Label("RANSAC plane threshold (mm)"))
        self._plane_threshold = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._plane_threshold.set_value(0.35)
        self._seg_section.add_child(self._plane_threshold)
        self._run_segmentation = self._button(
            "Detect Plane", self._detect_plane, self._seg_section)
        self._seg_view_selector = gui.Combobox()
        for item in ("Raw", "Combined", "Plane", "Foreground"):
            self._seg_view_selector.add_item(item)
        self._seg_view_selector.set_on_selection_changed(self._show_seg_view)
        self._seg_section.add_child(self._seg_view_selector)
        self._seg_count = gui.Label("Plane: — | Foreground: —")
        self._seg_equation = gui.Label("Equation: —")
        self._seg_warning = gui.Label("Plane identity unverified: inspect all views.")
        for label in (self._seg_count, self._seg_equation, self._seg_warning):
            self._seg_section.add_child(label)
        self._use_foreground = gui.Checkbox("Use inspected foreground for Locate Part")
        self._use_foreground.set_on_checked(self._set_foreground_choice)
        self._seg_section.add_child(self._use_foreground)
        self._seg_section.add_child(gui.Label(
            "Do not enable if RANSAC selected the coin face."))

        self._localisation_section = gui.CollapsableVert("SINGLE-PART LOCALISATION", spacing)
        self._sidebar.add_child(self._localisation_section)
        self._cad_label = gui.Label("CAD: none")
        self._localisation_section.add_child(self._cad_label)
        self._open_cad = self._button("Choose CAD (STEP / STL)", self._open_cad_dialog,
                                     self._localisation_section)
        self._localisation_section.add_child(gui.Label("Registration voxel (mm)"))
        self._voxel_input = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._voxel_input.set_value(1.0)
        self._localisation_section.add_child(self._voxel_input)
        self._viewpoint_origin = gui.Checkbox("Scan origin is camera optical centre")
        self._viewpoint_origin.checked = True
        self._localisation_section.add_child(self._viewpoint_origin)
        locate_row = gui.Horiz(int(0.4 * em))
        self._locate = self._button("Locate Part", self._locate_part, locate_row)
        self._clear_overlay = self._button("Clear Overlay", self._clear_localisation, locate_row)
        self._localisation_section.add_child(locate_row)
        self._loc_status = gui.Label("Status: not run")
        self._loc_metrics = gui.Label("Coverage / inliers / RMSE: —")
        self._loc_pose = gui.Label("Translation (mm): —")
        self._loc_runtime = gui.Label("Runtime: —")
        for label in (self._loc_status, self._loc_metrics,
                      self._loc_pose, self._loc_runtime):
            self._localisation_section.add_child(label)

        self._candidate_section = gui.CollapsableVert("CANDIDATE DIAGNOSTICS", spacing)
        self._sidebar.add_child(self._candidate_section)
        self._candidate_section.add_child(gui.Label("ORANGE = unverified hypothesis"))
        self._candidate_selector = gui.Combobox()
        self._candidate_selector.add_item("No candidates")
        self._candidate_selector.set_on_selection_changed(self._select_candidate)
        self._candidate_section.add_child(self._candidate_selector)
        self._candidate_metrics = gui.Label("No results")
        self._candidate_reason = gui.Label("Evidence: —")
        self._candidate_translation = gui.Label("CAD origin (mm): —")
        self._candidate_rotation = gui.Label("Rotation matrix R:")
        self._candidate_r0 = gui.Label("—")
        self._candidate_r1 = gui.Label("—")
        self._candidate_r2 = gui.Label("—")
        for label in (self._candidate_metrics, self._candidate_reason,
                      self._candidate_translation, self._candidate_rotation,
                      self._candidate_r0, self._candidate_r1, self._candidate_r2):
            self._candidate_section.add_child(label)
        self._preview_candidate = gui.Checkbox("Preview selected candidate (unverified)")
        self._preview_candidate.set_on_checked(self._toggle_candidate_preview)
        self._candidate_section.add_child(self._preview_candidate)
        self._export_diagnostics = self._button("Export Candidate JSON",
                                                self._export_candidate_json,
                                                self._candidate_section)

        self._status = gui.Label("Discover a scanner or open a point cloud.")
        self._status.background_color = gui.Color(0.075, 0.095, 0.125)
        for widget in (self._viewport.widget, self._sidebar, self._status):
            self.window.add_child(widget)
        self.window.set_on_layout(self._layout)
        self.window.set_on_tick_event(self._tick)
        self.window.set_on_close(self._close)
        self._update_controls()

    def _button(self, title, callback, parent=None):
        button = gui.Button(title)
        button.set_on_clicked(callback)
        (parent if parent is not None else self._sidebar).add_child(button)
        return button

    def _layout(self, _context):
        rect = self.window.content_rect
        sidebar_width = min(self._sidebar_width, rect.width // 2)
        status_height = int(self.window.theme.font_size * 1.9)
        height = max(1, rect.height - status_height)
        self._viewport.widget.frame = gui.Rect(rect.x, rect.y, rect.width - sidebar_width, height)
        self._sidebar.frame = gui.Rect(rect.get_right() - sidebar_width, rect.y, sidebar_width, height)
        self._status.frame = gui.Rect(rect.x, rect.y + height, rect.width, status_height)

    def _update_controls(self):
        ready = self._pending is None and not self._closing
        connected = self._device_id is not None
        self._refresh.enabled = self._devices.enabled = self._scanner_id.enabled = ready and not connected
        self._connect.enabled = ready and (connected or bool(self._scanner_id.text_value.strip()))
        self._connect.text = "Disconnect" if connected else "Connect"
        self._trigger.enabled = ready and connected
        self._profiles.enabled = ready and connected and bool(self._profile_names)
        self._reload_profiles.enabled = ready and connected
        self._apply_profile.enabled = ready and connected and bool(self._profile_names)

        self._open.enabled = ready
        self._open_cad.enabled = ready
        self._voxel_input.enabled = ready
        self._viewpoint_origin.enabled = ready
        self._locate.enabled = ready and self._current_frame is not None and self._cad_path is not None
        self._run_segmentation.enabled = ready and self._current_frame is not None
        self._plane_threshold.enabled = ready
        self._seg_view_selector.enabled = ready and self._segmentation is not None
        self._use_foreground.enabled = ready and self._segmentation is not None
        self._clear_overlay.enabled = ready and self._has_frame
        self._candidate_selector.enabled = ready and bool(self._diagnostic_rows)
        self._preview_candidate.enabled = ready and bool(self._diagnostic_rows)
        self._export_diagnostics.enabled = ready and self._diagnostic_result is not None
        self._reset.enabled = self._mode.enabled = self._has_frame
        for control in (self._screenshot, self._smart, self._full, self._manual_button,
                        self._min_depth, self._max_depth, self._gamma_slider, self._axes):
            control.enabled = self._has_frame

    def _submit(self, label, work, complete):
        self._status.text = label
        self._pending = (self._worker.submit(work), complete)
        self._update_controls()

    def _tick(self):
        if self._pending is None or not self._pending[0].done():
            return False
        future, complete = self._pending
        self._pending = None
        try:
            result = future.result()
            if not self._closing:
                complete(result)
        except Exception as exc:
            LOG.exception("Viewer command failed")
            if not self._closing:
                self._status.text = str(exc)
                self.window.show_message_box("Command failed", str(exc))
        if self._closing:
            if self._ready_to_close:
                self.window.close()
                return False
            self._begin_shutdown()
        else:
            self._update_controls()
        return True

    def discover(self):
        self._submit("Discovering scanners…", self._scanner.discover, self._show_devices)

    def _show_devices(self, devices):
        self._devices.clear_items()
        for name in devices:
            self._devices.add_item(name)
        if devices:
            self._devices.selected_index = 0
            self._choose_device(devices[0], 0)
        self._status.text = f"Scanners found: {len(devices)}" if devices else "No scanners discovered. Check PhoXi Control."

    def _choose_device(self, name, _index):
        self._scanner_id.text_value = name
        self._update_controls()

    def _toggle_connection(self):
        if self._device_id is not None:
            self._device_id = None
            self._profile_names = []
            self._active_profile = None
            self._profiles.clear_items()
            self._submit("Disconnecting…", self._scanner.disconnect,
                         lambda _: setattr(self._status, "text", "Scanner disconnected."))
        else:
            self._connect_device(self._scanner_id.text_value.strip())

    def _connect_device(self, device_id):
        def connected(_result):
            self._device_id = device_id
            self._status.text = f"Connected to {device_id}. Ready to scan."
            self._refresh_profiles()

        self._submit(f"Connecting to {device_id}…", lambda: self._scanner.connect(device_id), connected)

    def _refresh_profiles(self):
        self._submit("Reading scanner profiles…", self._scanner.get_profiles,
                     self._show_profiles)

    def _show_profiles(self, result):
        names, active = result
        self._profile_names = names
        self._active_profile = active
        self._profiles.clear_items()
        for name in names:
            self._profiles.add_item(name)
        if names:
            self._profiles.selected_index = names.index(active) if active in names else 0
        self._status.text = (f"Active profile: {active or 'none'} · {len(names)} available."
                             if names else "No scanner profiles available.")
        self._update_controls()

    def _change_profile(self):
        if not self._profile_names:
            return
        name = self._profile_names[self._profiles.selected_index]
        if name == self._active_profile:
            self._status.text = f"Profile {name} is already active."
            return
        self._submit(f"Applying scanner profile {name}…",
                     lambda: self._scanner.set_profile(name), self._show_profiles)

    def _detect_plane(self):
        if self._current_frame is None:
            return
        threshold = self._plane_threshold.double_value
        frame = self._current_frame
        self._use_foreground.checked = False
        self._segmentation = None
        self._clear_localisation()
        self._submit("Fitting largest plane with RANSAC...",
                     lambda: segment_plane(frame, threshold),
                     self._receive_segmentation)

    def _receive_segmentation(self, segmentation):
        self._segmentation = segmentation
        self._seg_count.text = (
            f"Plane {segmentation.plane_count:,} ({segmentation.plane_fraction:.1%})"
            f" | Foreground {segmentation.foreground_count:,}")
        a, b, c, d = segmentation.equation
        self._seg_equation.text = f"Plane: {a:+.3f}x {b:+.3f}y {c:+.3f}z {d:+.1f} = 0"
        ex, ey, ez = segmentation.plane_extent_mm
        self._seg_warning.text = (
            f"Plane extent XYZ: {ex:.1f}, {ey:.1f}, {ez:.1f} mm")
        self._seg_view_selector.selected_index = 1
        self._show_seg_view("Combined", 1)
        self._status.text = (
            "Review red/orange plane vs cyan foreground BEFORE using segmentation.")
        self._update_controls()
        self.window.set_needs_layout()

    def _show_seg_view(self, name, _index):
        self._seg_view = name
        if self._current_frame is None:
            return
        if name == "Raw" or self._segmentation is None:
            frame = self._current_frame
        elif name == "Combined":
            frame = preview_colors(self._segmentation)
        else:
            frame = self._segmentation.frame(name)
        # Preview only: don't change _current_frame or the camera viewpoint.
        self._viewport.show_frame(frame, fit=False)
        self._preview_candidate.checked = False
        self._status.text = f"Viewing {name.lower()} — source scan is unchanged."

    def _set_foreground_choice(self, enabled):
        if enabled and self._segmentation is None:
            self._use_foreground.checked = False
            return
        # Existing pose metrics/overlays belong to the previous input cloud.
        self._clear_localisation()
        self._status.text = (
            "Foreground chosen for next localisation; verify the plane is TABLE."
            if enabled else "Raw scan chosen for next localisation.")

    def _reset_segmentation(self):
        self._segmentation = None
        self._seg_view = "Raw"
        self._use_foreground.checked = False
        self._seg_view_selector.selected_index = 0
        self._seg_count.text = "Plane: — | Foreground: —"
        self._seg_equation.text = "Equation: —"
        self._seg_warning.text = "Plane identity unverified: inspect all views."

    def _open_cad_dialog(self):
        dialog = gui.FileDialog(gui.FileDialog.OPEN, "Choose CAD model", self.window.theme)
        dialog.add_filter(".step .stp .stl", "CAD files (.step, .stp, .stl)")
        dialog.set_on_cancel(self.window.close_dialog)

        def chosen(path):
            self.window.close_dialog()
            self._cad_path = Path(path)
            self._cad_label.text = f"CAD: {self._cad_path.name}"
            self._clear_localisation()
            self._update_controls()

        dialog.set_on_done(chosen)
        self.window.show_dialog(dialog)

    def _clear_localisation(self):
        self._viewport.clear_cad_overlay()
        self._diagnostic_result = None
        self._diagnostic_mesh = None
        self._diagnostic_rows = []
        self._preview_candidate.checked = False
        self._candidate_selector.clear_items()
        self._candidate_selector.add_item("No candidates")
        self._candidate_metrics.text = "No results"
        self._candidate_reason.text = "Evidence: —"
        self._candidate_translation.text = "CAD origin (mm): —"
        self._candidate_rotation.text = "Rotation matrix R:"
        for label in (self._candidate_r0, self._candidate_r1, self._candidate_r2):
            label.text = "—"
        self._loc_status.text = "Status: not run"
        self._loc_metrics.text = "Coverage / inliers / RMSE: —"
        self._loc_pose.text = "Translation (mm): —"
        self._loc_runtime.text = "Runtime: —"

    def _locate_part(self):
        if self._current_frame is None or self._cad_path is None:
            return
        voxel = self._voxel_input.double_value
        if not (0.05 <= voxel <= 100):
            self.window.show_message_box("Invalid voxel size", "Voxel must be between 0.05 and 100 mm.")
            return
        frame = (self._segmentation.frame("Foreground")
                 if self._segmentation is not None and self._use_foreground.checked
                 else self._current_frame)
        cad = self._cad_path
        viewpoint = self._viewpoint_origin.checked
        self._clear_localisation()
        self._loc_status.text = "Status: running…"
        self._submit("Localising single part…",
                     lambda: locate_frame(frame, cad, voxel=voxel,
                                          viewpoint_is_origin=viewpoint),
                     self._show_localisation)

    def _show_localisation(self, payload):
        result, pose, candidate, mesh = payload
        selection = result["assisted"] if result.get("assisted") is not None else result["mesh_only"]
        status = selection["status"]
        self._loc_status.text = f"Status: {status} ({result['candidate_count']} candidates)"
        self._loc_runtime.text = f"Runtime: {result['runtime_seconds']:.2f} s"
        self._diagnostic_result = result
        self._diagnostic_mesh = mesh
        self._diagnostic_rows = diagnostic_rows(result)
        self._candidate_selector.clear_items()
        for row in self._diagnostic_rows:
            verdict = "overlap OK" if row["passes_overlap"] else "overlap FAIL"
            self._candidate_selector.add_item(
                f"#{row['index'] + 1} | {verdict} | score {row['mesh_score']:.3f}")
        if self._diagnostic_rows:
            self._candidate_selector.selected_index = 0
            self._select_candidate("", 0)
        else:
            self._candidate_selector.add_item("No candidates")
            self._candidate_metrics.text = "No refined candidates."
            self._candidate_reason.text = "Check ROI, geometry and voxel resolution."
        if candidate is not None:
            rmse = candidate.get("inlier_rmse_mm")
            error = "—" if rmse is None else f"{rmse:.3f} mm"
            self._loc_metrics.text = (
                f"Coverage {candidate['model_coverage']:.1%} | "
                f"Inliers {candidate['scan_inlier_fraction']:.1%} | RMSE {error}")
        else:
            self._loc_metrics.text = "No accepted candidate; see diagnostics."
        if pose is not None and mesh is not None:
            self._viewport.show_cad_overlay(mesh, pose, diagnostic=False)
            x, y, z = pose[:3, 3]
            self._loc_pose.text = f"Translation: {x:.2f}, {y:.2f}, {z:.2f} mm"
            self._status.text = "GREEN: accepted 6D pose."
        else:
            self._loc_pose.text = "No accepted 6D pose."
            self._status.text = "Use candidate diagnostics to inspect unverified hypotheses."
        self._update_controls()
        self.window.set_needs_layout()

    def _select_candidate(self, _label, index):
        if not 0 <= index < len(self._diagnostic_rows):
            return
        row = self._diagnostic_rows[index]
        rmse = row["rmse"]
        self._candidate_metrics.text = (
            f"Coverage {row['coverage']:.1%} | Inliers {row['inliers']:.1%} | "
            f"RMSE {'—' if rmse is None else f'{rmse:.3f} mm'}")
        reasons = row["reasons"]
        self._candidate_reason.text = (
            "Evidence: " + (", ".join(reasons) if reasons
                             else "overlap OK; pose may still be ambiguous"))
        import numpy as np
        pose = np.asarray(row["pose"], dtype=float)
        x, y, z = pose[:3, 3]
        self._candidate_translation.text = f"CAD origin: {x:.2f}, {y:.2f}, {z:.2f} mm"
        for label, matrix_row in zip(
                (self._candidate_r0, self._candidate_r1, self._candidate_r2),
                pose[:3, :3]):
            label.text = "  ".join(f"{value:+.3f}" for value in matrix_row)
        if self._preview_candidate.checked:
            self._display_candidate(index)
        self.window.set_needs_layout()

    def _toggle_candidate_preview(self, checked):
        if checked and self._diagnostic_rows:
            self._display_candidate(self._candidate_selector.selected_index)
            return
        self._viewport.clear_cad_overlay()
        if self._diagnostic_result is not None and self._diagnostic_mesh is not None:
            selected = (self._diagnostic_result["assisted"]
                        if self._diagnostic_result.get("assisted") is not None
                        else self._diagnostic_result["mesh_only"])
            if selected["status"] == "match":
                import numpy as np
                self._viewport.show_cad_overlay(
                    self._diagnostic_mesh,
                    np.asarray(selected["pose_cad_to_scan"]), diagnostic=False)

    def _display_candidate(self, index):
        if self._diagnostic_mesh is None or not 0 <= index < len(self._diagnostic_rows):
            return
        import numpy as np
        pose = np.asarray(self._diagnostic_rows[index]["pose"], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            self._status.text = "Candidate matrix invalid."
            return
        self._viewport.show_cad_overlay(self._diagnostic_mesh, pose, diagnostic=True)
        self._status.text = f"ORANGE: candidate #{index + 1}, NOT VERIFIED."

    def _export_candidate_json(self):
        if self._diagnostic_result is None:
            return
        dialog = gui.FileDialog(gui.FileDialog.SAVE, "Export candidates", self.window.theme)
        dialog.add_filter(".json", "JSON file (.json)")
        dialog.set_on_cancel(self.window.close_dialog)

        def save(path):
            self.window.close_dialog()
            if not path.lower().endswith(".json"):
                path += ".json"
            import json
            try:
                Path(path).write_text(
                    json.dumps(self._diagnostic_result, indent=2, allow_nan=False),
                    encoding="utf-8")
                self._status.text = f"Candidate JSON saved: {path}"
            except OSError as exc:
                self.window.show_message_box("Export failed", str(exc))

        dialog.set_on_done(save)
        self.window.show_dialog(dialog)

    def _set_contrast(self, strategy):
        self._viewport.set_contrast(strategy)
        self._status.text = f"Depth contrast: {strategy}."

    def _apply_manual_range(self):
        try:
            self._viewport.set_manual_range(
                self._min_depth.double_value, self._max_depth.double_value)
            self._status.text = "Manual depth range applied."
        except ValueError as exc:
            self.window.show_message_box("Invalid depth range", str(exc))

    def _reset_gamma_value(self):
        self._gamma_slider.double_value = 1.0
        self._viewport.set_gamma(1.0)

    def _save_screenshot(self):
        dialog = gui.FileDialog(gui.FileDialog.SAVE, "Save screenshot", self.window.theme)
        dialog.add_filter(".png", "PNG image (.png)")
        dialog.set_on_cancel(self.window.close_dialog)

        def done(path):
            self.window.close_dialog()
            if not path.lower().endswith(".png"):
                path += ".png"

            def completed(ok):
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: setattr(
                        self._status, "text",
                        f"Screenshot saved: {path}" if ok else "Screenshot export failed."))

            self._viewport.save_screenshot(path, completed)
            self._status.text = "Capturing screenshot…"

        dialog.set_on_done(done)
        self.window.show_dialog(dialog)

    def _trigger_scan(self):
        self._submit("Capturing one frame…", self._scanner.trigger,
                     lambda frame: self._show_frame(frame, fit=not self._has_frame))

    def _open_dialog(self):
        dialog = gui.FileDialog(gui.FileDialog.OPEN, "Open point cloud", self.window.theme)
        dialog.add_filter(".ply .pcd", "Point clouds (.ply, .pcd)")
        dialog.set_on_cancel(self.window.close_dialog)

        def opened(path):
            self.window.close_dialog()
            self.open_file(Path(path))

        dialog.set_on_done(opened)
        self.window.show_dialog(dialog)

    def open_file(self, path):
        self._submit("Loading point cloud…", lambda: load_point_cloud(path),
                     lambda frame: self._show_frame(frame, fit=True))

    def _show_frame(self, frame, *, fit):
        self._reset_segmentation()
        self._current_frame = frame
        self._clear_localisation()
        self._viewport.show_frame(frame, fit=fit)
        self._has_frame = True
        self._source.text = frame.source if frame.frame_id is None else f"{frame.source} · frame {frame.frame_id}"
        self._points.text = f"Valid points: {len(frame.points):,}"
        self._time.text = "Capture: —" if frame.elapsed_ms is None else f"Capture: {frame.elapsed_ms:.0f} ms"
        z = frame.points[:, 2]
        self._zrange.text = f"Z range: {z.min():.1f} .. {z.max():.1f} mm"
        self._status.text = "Frame displayed. Drag to orbit · Shift-drag to pan · Scroll to zoom."
        self._update_controls()
        self.window.set_needs_layout()

    def _begin_shutdown(self):
        self._ready_to_close = True
        self._submit("Releasing scanner…", self._scanner.disconnect, lambda _: None)

    def _close(self):
        if self._ready_to_close and self._pending is None:
            self._worker.shutdown(wait=True)
            return True
        self._closing = True
        if self._pending is None:
            self._begin_shutdown()
        else:
            self._status.text = "Waiting for the current command…"
            self._update_controls()
        return False
