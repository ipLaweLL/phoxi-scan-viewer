"""Open3D scene, point rendering and camera controls."""
import numpy as np
import open3d as o3d

gui = o3d.visualization.gui
rendering = o3d.visualization.rendering


class PointCloudViewport:
    def __init__(self, renderer):
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(renderer)
        self.widget.scene.set_background([0.012, 0.018, 0.028, 1])
        self.widget.scene.show_skybox(False)
        self.widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self._material = rendering.MaterialRecord()
        self._material.shader = "defaultUnlit"
        self._material.point_size = 2.0
        self._frame = None
        self._geometry = None
        self._mode = "Texture"

    def show_frame(self, frame, *, fit):
        self._frame = frame
        self._draw()
        if fit:
            self.reset_view()

    def set_mode(self, mode):
        self._mode = mode
        if self._frame is not None:
            self._draw()

    def set_point_size(self, size):
        self._material.point_size = float(size)
        if self._geometry is not None:
            self.widget.scene.modify_geometry_material("scan", self._material)
            self.widget.force_redraw()

    def _draw(self):
        frame = self._frame
        points = frame.points
        geometry = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
        if self._mode == "Texture" and frame.colors is not None:
            geometry.colors = o3d.utility.Vector3dVector(frame.colors.astype(np.float64))
        elif self._mode == "Depth":
            low, high = np.percentile(points[:, 2], (2, 98))
            fraction = np.clip((points[:, 2] - low) / max(high - low, 1e-6), 0, 1)
            colors = np.column_stack((0.17 + 0.83 * fraction,
                                      0.37 + 0.39 * fraction,
                                      0.92 - 0.60 * fraction))
            geometry.colors = o3d.utility.Vector3dVector(colors)
        else:
            geometry.paint_uniform_color((0.92, 0.95, 1.0))

        scene = self.widget.scene
        if self._geometry is not None:
            scene.remove_geometry("scan")
        self._geometry = geometry
        scene.add_geometry("scan", geometry, self._material)
        self.widget.force_redraw()

    def reset_view(self):
        if self._geometry is None:
            return
        box = self._geometry.get_axis_aligned_bounding_box()
        center = box.get_center()
        # Fit from the scanner side for the usual PhoXi camera coordinates (+Z away).
        offset = max(1.0, float(np.linalg.norm(box.get_extent())))
        self.widget.setup_camera(55, box, center)
        self.widget.look_at(center, center + offset * np.array([0, -0.32, -1.2]), [0, -1, 0])
        self.widget.force_redraw()
