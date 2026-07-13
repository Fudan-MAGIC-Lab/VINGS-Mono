import pathlib
import sys
import unittest

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from profiling.benchmark_metric3d_trt import (
    build_report,
    latency_statistics,
    prepare_input_tensor,
)


FRAME_IDS = tuple(range(0, 50, 5))


def passing_metrics():
    return {
        "shape": [1, 1, 448, 784],
        "valid_count": 351232,
        "finite": True,
        "cosine": 0.9999,
        "mean_relative_error": 0.001,
    }


def passing_cases():
    return [
        {
            "frame_id": frame_id,
            "raw": passing_metrics(),
            "final": passing_metrics(),
            "preprocess_ms": 20.0,
            "pytorch_model_ms": 100.0 + index,
            "tensorrt_model_ms": 60.0 + index,
            "pytorch_postprocess_ms": 2.0,
            "tensorrt_postprocess_ms": 2.0,
        }
        for index, frame_id in enumerate(FRAME_IDS)
    ]


class Metric3DTensorRTReportTests(unittest.TestCase):
    def test_prepare_input_tensor_makes_noncontiguous_numpy_contiguous(self):
        array = np.zeros((1, 3, 4, 6), dtype=np.float32)[..., ::2]
        self.assertFalse(array.flags.c_contiguous)

        tensor = prepare_input_tensor(torch, array, device="cpu")

        self.assertTrue(tensor.is_contiguous())
        self.assertEqual(tuple(tensor.shape), (1, 3, 4, 3))

    def test_latency_statistics(self):
        result = latency_statistics([10.0, 12.0, 14.0])

        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean_ms"], 12.0)
        self.assertEqual(result["median_ms"], 12.0)
        self.assertAlmostEqual(result["p90_ms"], 13.6)

    def test_report_accepts_complete_fast_parity_result(self):
        report = build_report(
            passing_cases(), engine="metric.plan", checkpoint="metric.pth"
        )

        self.assertTrue(report["recommendation"]["integrate"])
        self.assertEqual(report["frame_ids"], list(FRAME_IDS))
        self.assertGreaterEqual(report["latency"]["model_improvement"], 0.20)
        self.assertGreater(report["latency"]["model_speedup"], 1.0)

    def test_report_rejects_parity_failure(self):
        cases = passing_cases()
        cases[3]["final"] = dict(passing_metrics(), mean_relative_error=0.0201)

        report = build_report(cases, engine="metric.plan", checkpoint="metric.pth")

        self.assertFalse(report["recommendation"]["integrate"])
        self.assertIn("parity", report["recommendation"]["reason"])

    def test_report_rejects_speed_failure(self):
        cases = passing_cases()
        for case in cases:
            case["tensorrt_model_ms"] = case["pytorch_model_ms"] * 0.81

        report = build_report(cases, engine="metric.plan", checkpoint="metric.pth")

        self.assertFalse(report["recommendation"]["integrate"])
        self.assertIn("20%", report["recommendation"]["reason"])

    def test_report_rejects_incomplete_frame_coverage(self):
        report = build_report(
            passing_cases()[:-1], engine="metric.plan", checkpoint="metric.pth"
        )

        self.assertFalse(report["recommendation"]["integrate"])
        self.assertIn("coverage", report["recommendation"]["reason"])


if __name__ == "__main__":
    unittest.main()
