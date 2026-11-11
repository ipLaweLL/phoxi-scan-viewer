"""Open3D viewport with deterministic display-range controls (XYZ is never modified)."""
import numpy as np
import open3d as o3d

gui = o3d.visualization.gui
rendering = o3d.visualization.rendering


def scalar_range(values, strategy="Smart", manual=None):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    if strategy == "Manual" and manual is not None:
        lo, hi = map(float, manual)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError("Minimum must be smaller than maximum.")
    elif strategy == "Full":
        lo, hi = float(finite.min()), float(finite.max())
    else:
        lo, hi = map(float, np.percentile(finite, (1, 99)))
    return lo, hi if hi > lo else lo + 1.0


def depth_colors(z, mode="Depth - Hue", strategy="Smart", manual=None, gamma=1.0):
    """Returns clipped RGB colors and the range used; no geometry alteration."""
    lo, hi = scalar_range(z, strategy, manual)
    fraction = np.nan_to_num((np.asarray(z, dtype=np.float64) - lo) / (hi - lo),
                             nan=0.0, posinf=1.0, neginf=0.0)
    fraction = np.clip(fraction, 0.0, 1.0) ** (1.0 / max(float(gamma), 0.05))
    if mode == "Depth - Grayscale":
        rgb = np.repeat(fraction[:, None], 3, axis=1)
    else:
        # Vectorized, perceptually readable blue -> cyan -> green -> yellow -> red.
        # Avoid Python per-point loops: PhoXi frames routinely exceed 1M points.
        stops = [0.0, 0.25, 0.5, 0.75, 1.0]
        palette = np.array([[0.12, 0.25, 0.94], [0.05, 0.85, 0.92],
                            [0.12, 0.79, 0.36], [0.98, 0.88, 0.17],
                            [0.92, 0.17, 0.12]])
        rgb = np.column_stack([np.interp(fraction, stops, palette[:, i])
                               for i in range(3)])
    return rgb, (lo, hi)


class PointCloudViewport:
    def __init__(self, renderer):
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(renderer)
        self.widget.scene.set_background([0.0, 0.0, 0.0, 1])
        self.widget.scene.show_skybox(False)
        self.widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self._material = rendering.MaterialRecord()
        self._material.shader = "defaultUnlit"
        self._material.point_size = 2.0
        self._frame = None
        self._geometry = None
        self._mode = "Texture"
        self._contrast = "Smart"
        self._manual = None
        self._gamma = 1.0
        self._show_axes = False
        self._display_range = None
        self._cad_mesh = None
        self._cad_material = rendering.MaterialRecord()
        self._cad_material.shader = "defaultLitTransparency"
        self._cad_material.base_color = [0.04, 0.95, 0.36, 0.45]

    def show_frame(self, frame, *, fit):
        self.clear_cad_overlay()
        self._frame = frame
        self._draw()
        if fit:
            self.reset_view()

    def show_cad_overlay(self, mesh, pose, *, diagnostic=False):
        """Keep source mesh untouched; render transformed CAD transparently."""
        if self._cad_mesh is not None:
            self.widget.scene.remove_geometry("cad_overlay")
            self._cad_mesh = None
        model = o3d.geometry.TriangleMesh(mesh)
        model.transform(np.asarray(pose, dtype=np.float64))
        model.compute_vertex_normals()
        material = rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        material.base_color = ([1.0, 0.55, 0.05, 0.52] if diagnostic
                               else [0.04, 0.95, 0.36, 0.45])
        self.widget.scene.add_geometry("cad_overlay", model, material)
        self._cad_mesh = model
        self.widget.force_redraw()

    def clear_cad_overlay(self):
        if self._cad_mesh is not None:
            self.widget.scene.remove_geometry("cad_overlay")
            self._cad_mesh = None
            self.widget.force_redraw()

    def set_mode(self, mode):
        self._mode = mode
        if self._frame is not None:
            self._draw()

    def set_point_size(self, size):
        self._material.point_size = float(size)
        if self._geometry is not None:
            self.widget.scene.modify_geometry_material("scan", self._material)
            self.widget.force_redraw()

    def set_contrast(self, strategy):
        if strategy not in ("Smart", "Full"):
            raise ValueError("Unknown auto contrast mode.")
        self._contrast = strategy
        self._draw_if_loaded()

    def set_manual_range(self, low, high):
        low, high = float(low), float(high)
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("Minimum must be smaller than maximum.")
        self._manual = (low, high)
        self._contrast = "Manual"
        self._draw_if_loaded()

    def set_gamma(self, gamma):
        self._gamma = max(0.2, min(3.0, float(gamma)))
        self._draw_if_loaded()

    def set_show_axes(self, enabled):
        self._show_axes = bool(enabled)
        self.widget.scene.show_axes(self._show_axes)
        self.widget.force_redraw()

    @property
    def display_range(self):
        return self._display_range

    def save_screenshot(self, path, on_complete=None):
        # Render callbacks are asynchronous; report completion only when written.
        def write_image(image):
            ok = o3d.io.write_image(str(path), image)
            if on_complete is not None:
                on_complete(bool(ok))
        self.widget.scene.scene.render_to_image(write_image)

    def _draw_if_loaded(self):
        if self._frame is not None:
            self._draw()

    def _draw(self):
        frame = self._frame
        if frame is None:
            return
        points = np.asarray(frame.points, dtype=np.float64)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        self._display_range = None
        if self._mode == "Texture" and frame.colors is not None:
            texture = np.asarray(frame.colors, dtype=np.float64)
            cloud.colors = o3d.utility.Vector3dVector(
                np.clip(texture, 0, 1) ** (1.0 / self._gamma))
        elif self._mode in ("Depth - Hue", "Depth - Grayscale", "Depth"):
            mode = "Depth - Grayscale" if self._mode == "Depth - Grayscale" else "Depth - Hue"
            colors, self._display_range = depth_colors(
                points[:, 2], mode, self._contrast, self._manual, self._gamma)
            cloud.colors = o3d.utility.Vector3dVector(colors)
        else:
            cloud.paint_uniform_color((0.92, 0.95, 1.0))
        if self._geometry is not None:
            self.widget.scene.remove_geometry("scan")
        self._geometry = cloud
        self.widget.scene.add_geometry("scan", cloud, self._material)
        self.widget.force_redraw()

    def reset_view(self):
        if self._geometry is None:
            return
        box = self._geometry.get_axis_aligned_bounding_box()
        center = box.get_center()
        offset = max(1.0, float(np.linalg.norm(box.get_extent())))
        self.widget.setup_camera(55, box, center)
        self.widget.look_at(center, center + offset * np.array([0, -0.32, -1.2]), [0, -1, 0])
        self.widget.force_redraw()
