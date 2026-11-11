"""Checks for non-destructive plane segmentation on synthetic geometry."""
import unittest
import numpy as np
from viewer.frames import ScanFrame
from viewer.segmentation import segment_plane, preview_colors


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        xy = rng.uniform(-35, 35, size=(4000, 2))
        table = np.column_stack([xy, 500 + rng.normal(0, 0.035, len(xy))])
        # Raised cloud with enough points and spatial diversity.
        raised_xy = rng.uniform(-12, 12, size=(600, 2))
        raised = np.column_stack(
            [raised_xy, 497.5 + rng.normal(0, 0.06, len(raised_xy))])
        self.raw = np.concatenate([table, raised]).astype(np.float32)
        self.frame = ScanFrame(points=self.raw.copy(), colors=None, source="synthetic")

    def test_plane_does_not_modify_raw_frame(self):
        original = self.frame.points.copy()
        result = segment_plane(self.frame, threshold_mm=0.2)
        self.assertEqual(result.plane_count + result.foreground_count, len(original))
        self.assertGreater(result.plane_count, 3500)
        self.assertGreater(result.foreground_count, 500)
        np.testing.assert_array_equal(self.frame.points, original)

    def test_preview_and_foreground_preserve_xyz_membership(self):
        result = segment_plane(self.frame, threshold_mm=0.2)
        combined = preview_colors(result)
        foreground = result.frame("Foreground")
        plane = result.frame("Plane")
        np.testing.assert_allclose(combined.points, self.raw)
        np.testing.assert_array_equal(foreground.points, self.raw[~result.plane_mask])
        np.testing.assert_array_equal(plane.points, self.raw[result.plane_mask])
        self.assertEqual(combined.colors.shape, self.raw.shape)
        self.assertEqual(len(plane.points) + len(foreground.points), len(self.raw))

    def test_reject_invalid_thresholds(self):
        for threshold in (-1, 0, np.nan, 4):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    segment_plane(self.frame, threshold_mm=threshold)


if __name__ == "__main__":
    unittest.main()
