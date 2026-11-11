"""Desktop UI and serialized acquisition commands."""
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

from .frames import load_point_cloud
from .phoxi import PhoXiScanner
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
        self._viewport.show_frame(frame, fit=fit)
        self._has_frame = True
        self._source.text = frame.source if frame.frame_id is None else f"{frame.source} · frame {frame.frame_id}"
        self._points.text = f"Valid points: {len(frame.points):,}"
        self._time.text = "Capture: —" if frame.elapsed_ms is None else f"Capture: {frame.elapsed_ms:.0f} ms"
        z = frame.points[:, 2]
        self._zrange.text = f"Z range: {z.min():.1f} .. {z.max():.1f} mm"
        self._status.text = "Frame displayed. Drag to orbit · Shift-drag to pan · Scroll to zoom."
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
