import pathlib
import sys
import unittest

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from profiling.benchmark_droid_update_trt import build_report, parity_metrics


class DroidUpdateTensorRTBenchmarkTests(unittest.TestCase):
    def test_parity_metrics_reports_shape_finite_cosine_and_relative_error(self):
        expected = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        actual = np.array([1.001, -1.998, 0.499], dtype=np.float32)

        metrics = parity_metrics(expected, actual)

        self.assertEqual(metrics["shape"], [3])
        self.assertTrue(metrics["finite"])
        self.assertGreater(metrics["cosine"], 0.999)
        self.assertLess(metrics["mean_relative_error"], 0.01)

    def test_parity_metrics_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            parity_metrics(np.zeros((2, 3)), np.zeros((6,)))

    def test_report_schema_and_recommendation_use_e16_improvement(self):
        passing_outputs = {
            name: {
                "shape": [16, 2, 43, 77],
                "finite": True,
                "cosine": 0.9999,
                "mean_relative_error": 0.001,
            }
            for name in ("updated_net", "delta", "weight")
        }
        cases = [
            {
                "edge_count": edge_count,
                "outputs": passing_outputs,
                "pytorch_ms": 10.0,
                "tensorrt_ms": 7.0 if edge_count == 16 else 8.0,
            }
            for edge_count in (1, 4, 16, 32, 48)
        ]

        report = build_report(cases, engine="engine.plan", checkpoint="droid.pth")

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["edge_counts"], [1, 4, 16, 32, 48])
        self.assertEqual(report["thresholds"]["minimum_improvement"], 0.20)
        self.assertTrue(report["recommendation"]["integrate"])
        representative = next(case for case in report["cases"] if case["edge_count"] == 16)
        self.assertAlmostEqual(representative["latency"]["improvement"], 0.30)
        self.assertAlmostEqual(representative["latency"]["speedup"], 10.0 / 7.0)

    def test_report_requires_full_dynamic_range_and_all_outputs(self):
        passing = {
            "shape": [16, 2, 43, 77],
            "finite": True,
            "cosine": 0.9999,
            "mean_relative_error": 0.001,
        }
        cases = [{
            "edge_count": 16,
            "outputs": {"updated_net": passing, "delta": passing},
            "pytorch_ms": 10.0,
            "tensorrt_ms": 5.0,
        }]

        report = build_report(cases, engine="engine.plan", checkpoint="droid.pth")

        self.assertFalse(report["recommendation"]["integrate"])
        self.assertIn("coverage", report["recommendation"]["reason"])

    def test_report_rejects_bad_parity_even_when_latency_is_fast(self):
        passing = {
            "shape": [16, 2, 43, 77],
            "finite": True,
            "cosine": 0.9999,
            "mean_relative_error": 0.001,
        }
        failing = dict(passing, cosine=0.998)
        cases = [
            {
                "edge_count": edge_count,
                "outputs": {
                    "updated_net": passing,
                    "delta": failing,
                    "weight": passing,
                },
                "pytorch_ms": 10.0,
                "tensorrt_ms": 5.0,
            }
            for edge_count in (1, 4, 16, 32, 48)
        ]

        report = build_report(cases, engine="engine.plan", checkpoint="droid.pth")

        self.assertFalse(report["recommendation"]["integrate"])
        self.assertIn("parity", report["recommendation"]["reason"])


if __name__ == "__main__":
    unittest.main()
