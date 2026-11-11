"""Regression checks for viewer localisation decision logic."""
import unittest

import numpy as np

from viewer.localisation import select_pose


class LocalisationBridgeTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.eye(4).tolist()
        self.candidate = {"model_coverage": 0.99, "scan_inlier_fraction": 0.9,
                          "inlier_rmse_mm": 0.1}

    def test_accepted_pose_uses_assisted_when_available(self):
        result = {"mesh_only": {"status": "match", "pose_cad_to_scan": self.pose,
                                "best_candidate": self.candidate},
                  "assisted": {"status": "no_match", "pose_cad_to_scan": None,
                               "best_candidate": None}}
        pose, status, _ = select_pose(result)
        self.assertIsNone(pose)
        self.assertEqual(status, "no_match")

    def test_mesh_only_fallback(self):
        result = {"mesh_only": {"status": "match", "pose_cad_to_scan": self.pose,
                                "best_candidate": self.candidate},
                  "assisted": None}
        pose, status, candidate = select_pose(result)
        np.testing.assert_allclose(pose, np.eye(4))
        self.assertEqual(status, "match")
        self.assertEqual(candidate, self.candidate)

    def test_ambiguous_candidate_not_displayed_as_pose(self):
        result = {"mesh_only": {"status": "ambiguous", "pose_cad_to_scan": None,
                                "best_candidate": self.candidate},
                  "assisted": None}
        pose, status, candidate = select_pose(result)
        self.assertIsNone(pose)
        self.assertEqual(status, "ambiguous")
        self.assertEqual(candidate, self.candidate)

    def test_nonfinite_pose_rejected(self):
        malformed = np.eye(4)
        malformed[0, 3] = np.nan
        result = {"mesh_only": {"status": "match", "pose_cad_to_scan": malformed.tolist(),
                                "best_candidate": self.candidate}, "assisted": None}
        with self.assertRaises(ValueError):
            select_pose(result)


if __name__ == "__main__":
    unittest.main()
