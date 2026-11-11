import unittest
from unittest.mock import patch

import numpy as np
from phoxi_api import PhoXiDevice

from viewer.frames import ScanFrame
from viewer.phoxi import PhoXiScanner


class ScannerDevice:
    def __init__(self):
        from types import SimpleNamespace

        self.mode = PhoXiDevice.TriggerMode.FREERUN
        self.acquiring = True
        self.outputs = SimpleNamespace(
            PointCloud=SimpleNamespace(value=False), Texture=SimpleNamespace(value=False)
        )
        self.closed = False
        self.profile = "Default"
        self.profile_changes = []

    def profiles(self):
        return [{"name": "Default"}, {"name": "High Quality"}]

    def active_profile(self):
        return self.profile

    def set_active_profile(self, name):
        assert not self.acquiring
        self.profile_changes.append(name)
        self.profile = name

    def info(self):
        return {"file_camera": False}

    def trigger_mode(self):
        return self.mode

    def set_trigger_mode(self, mode):
        assert not self.acquiring
        self.mode = mode

    def frame_settings(self):
        return self.outputs

    def is_acquiring(self):
        return self.acquiring

    def start_acquisition(self):
        self.acquiring = True

    def stop_acquisition(self):
        self.acquiring = False

    def trigger_frame(self, *, wait_accept, wait_grabbing_end):
        assert self.acquiring and self.mode == PhoXiDevice.TriggerMode.SOFTWARE
        assert self.outputs.PointCloud.value and self.outputs.Texture.value
        return 17

    def get_frame(self, frame_id, timeout_ms):
        from types import SimpleNamespace

        assert frame_id == 17 and timeout_ms == 15000
        return SimpleNamespace(
            Successful=True, Messages=[], PointCloud=np.array([[[0, 0, 0], [1, 2, 300]]]),
            Texture=np.array([[[0, 0, 0], [255, 128, 0]]], dtype=np.uint8)
        )

    def disconnect(self, *, logout, stop_acquisition):
        assert not logout and not stop_acquisition
        self.closed = True


class ScanTests(unittest.TestCase):
    def test_discovers_triggers_and_restores_scanner(self):
        device = ScannerDevice()
        scanner = PhoXiScanner()
        with patch("phoxi_api.PhoXiControl") as control:
            control.return_value.is_phoxicontrol_running.return_value = True
            control.return_value.get_device_list.return_value = [{"device_id": "TEST-1"}]
            control.return_value.connect.return_value = device
            self.assertEqual(scanner.discover(), ["TEST-1"])
            scanner.connect("TEST-1")
            frame = scanner.trigger()
            self.assertEqual(frame.frame_id, 17)
            np.testing.assert_equal(frame.points, [[1, 2, 300]])
            np.testing.assert_allclose(frame.colors, [[1, 128 / 255, 0]])
            scanner.disconnect()
        self.assertTrue(device.closed)
        self.assertTrue(device.acquiring)
        self.assertEqual(device.mode, PhoXiDevice.TriggerMode.FREERUN)
        self.assertFalse(device.outputs.PointCloud.value)
        self.assertFalse(device.outputs.Texture.value)

    def test_profiles_discovery_and_apply(self):
        device = ScannerDevice()
        scanner = PhoXiScanner()
        with patch("phoxi_api.PhoXiControl") as control:
            control.return_value.is_phoxicontrol_running.return_value = True
            control.return_value.connect.return_value = device
            scanner.connect("TEST-1")
            self.assertEqual(scanner.get_profiles(), (["Default", "High Quality"], "Default"))
            self.assertEqual(scanner.set_profile("High Quality"),
                             (["Default", "High Quality"], "High Quality"))
            self.assertEqual(device.profile_changes, ["High Quality"])
            self.assertTrue(device.acquiring)
            self.assertEqual(device.mode, PhoXiDevice.TriggerMode.SOFTWARE)
            with self.assertRaises(ValueError):
                scanner.set_profile("not present")
            scanner.disconnect()

    def test_grayscale_filters_invalid_xyz_and_keeps_color_alignment(self):
        xyz = [[[0, 0, 0], [5, 6, 7]], [[np.nan, 3, 4], [8, 9, 10]]]
        gray = np.array([[0, 51], [0, 204]], dtype=np.uint8)
        frame = ScanFrame.from_arrays(xyz, gray, source="Test")
        np.testing.assert_equal(frame.points, [[5, 6, 7], [8, 9, 10]])
        np.testing.assert_allclose(frame.colors, [[0.2] * 3, [0.8] * 3])

    def test_phoxi_uint16_intensity_is_not_almost_black(self):
        # Typical PhoXi intensity range fits in uint16 but is far below 65535.
        xyz = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
        intensity = np.array([26, 200, 550, 972], dtype=np.uint16)
        frame = ScanFrame.from_arrays(xyz, intensity, source="PhoXi")
        self.assertLess(frame.colors[0, 0], 0.05)
        self.assertGreater(frame.colors[-1, 0], 0.95)
        self.assertGreater(frame.colors[2, 0], 0.4)
        self.assertLess(frame.colors[2, 0], 0.75)

    def test_unaligned_texture_and_empty_cloud(self):
        frame = ScanFrame.from_arrays([[5, 6, 7]], np.zeros((2, 2)), source="Test")
        self.assertIsNone(frame.colors)
        with self.assertRaisesRegex(ValueError, "no valid"):
            ScanFrame.from_arrays([[0, 0, 0]], source="Test")


if __name__ == "__main__":
    unittest.main()
