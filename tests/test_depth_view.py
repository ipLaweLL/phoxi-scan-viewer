"""Pure processing tests for depth color mapping (no scanner required)."""
import unittest
import numpy as np
from viewer.viewport import depth_colors, scalar_range


class DepthViewTests(unittest.TestCase):
    def test_grayscale_uses_full_manual_range(self):
        rgb, bounds = depth_colors(np.array([100., 150., 200.]),
                                   "Depth - Grayscale", "Manual", (100, 200))
        self.assertEqual(bounds, (100, 200))
        np.testing.assert_allclose(rgb[:, 0], [0, 0.5, 1])
        np.testing.assert_allclose(rgb[:, 0], rgb[:, 1])

    def test_smart_range_rejects_outliers(self):
        z = np.r_[np.full(100, 500.), np.linspace(500, 510, 100), 100000.]
        lo, hi = scalar_range(z, "Smart")
        self.assertLess(hi, 100000)
        self.assertGreater(hi, lo)

    def test_constant_depth_is_safe(self):
        rgb, bounds = depth_colors(np.full(10, 22), "Depth - Hue")
        self.assertEqual(rgb.shape, (10, 3))
        self.assertTrue(np.isfinite(rgb).all())
        self.assertGreater(bounds[1], bounds[0])

    def test_manual_bounds_validation(self):
        with self.assertRaises(ValueError):
            scalar_range([1, 2], "Manual", (20, 10))

    def test_hue_rgb_range(self):
        rgb, _ = depth_colors(np.linspace(0, 1, 20))
        self.assertTrue(((rgb >= 0) & (rgb <= 1)).all())


if __name__ == "__main__":
    unittest.main()
