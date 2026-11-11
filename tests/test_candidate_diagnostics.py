"""Candidate diagnostics are preserved even for rejected and ambiguous results."""
import unittest

import numpy as np

from viewer.localisation import diagnostic_rows, select_pose


class CandidateDiagnosticsTests(unittest.TestCase):
    def sample(self, *, coverage=0.95, inliers=0.02, passes=False):
        return {
            "pose_cad_to_scan": np.eye(4).tolist(),
            "model_coverage": coverage,
            "scan_inlier_fraction": inliers,
            "inlier_rmse_mm": 0.2,
            "mesh_score": 0.45,
            "passes_overlap": passes,
            "feature_checks": [],
        }

    def test_rejected_candidate_is_visible_with_reason(self):
        result = {"candidates": [self.sample()]}
        rows = diagnostic_rows(result)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["passes_overlap"])
        self.assertIn("low global scan-inlier fraction", rows[0]["reasons"])
        self.assertEqual(rows[0]["coverage"], 0.95)

    def test_passing_candidate_is_still_not_necessarily_unique(self):
        candidate = self.sample(coverage=0.98, inliers=0.8, passes=True)
        result = {
            "candidates": [candidate, candidate],
            "mesh_only": {"status": "ambiguous", "pose_cad_to_scan": None,
                          "best_candidate": candidate},
            "assisted": None,
        }
        self.assertEqual(len(diagnostic_rows(result)), 2)
        self.assertEqual(diagnostic_rows(result)[0]["reasons"], [])
        pose, status, _ = select_pose(result)
        self.assertEqual(status, "ambiguous")
        self.assertIsNone(pose)

    def test_low_coverage_and_inliers_both_explained(self):
        result = {"candidates": [self.sample(coverage=0.10, inliers=0.08)]}
        reasons = diagnostic_rows(result)[0]["reasons"]
        self.assertIn("low CAD coverage", reasons)
        self.assertIn("low global scan-inlier fraction", reasons)

    def test_contradicted_hole_is_identified(self):
        candidate = self.sample(coverage=0.98, inliers=0.90, passes=True)
        candidate["feature_checks"] = [{"id": "hole/1", "state": "contradicted"}]
        reasons = diagnostic_rows({"candidates": [candidate]})[0]["reasons"]
        self.assertIn("CAD feature contradicted", reasons)


if __name__ == "__main__":
    unittest.main()
