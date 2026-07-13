import pathlib
import sys
import unittest

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from profiling.benchmark_droid_encoders_trt import (
    build_report,
    latency_statistics,
    validate_encoder_tensors,
)


FRAME_IDS = [0, 8, 11, 15, 19, 28, 38, 44, 56, 72, 82, 104, 112, 129, 144, 158, 165, 170, 179, 195]


def passing_metrics(channels=128):
    return {
        "shape": [1, channels, 43, 77],
        "dtype": "float16",
        "value_count": channels * 43 * 77,
        "finite": True,
        "cosine": 0.9999,
        "mean_relative_error": 0.001,
    }


def passing_cases(channels=128):
    return [
        {
            "frame_id": frame_id,
            "metrics": passing_metrics(channels),
            "pytorch_ms": 10.0 + index / 100.0,
            "tensorrt_ms": 7.0 + index / 100.0,
            "tensorrt_executed": True,
        }
        for index, frame_id in enumerate(FRAME_IDS)
    ]


class FakeTensor:
    def __init__(self, shape, dtype, is_cuda=True, contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


class DroidEncoderTensorRTTests(unittest.TestCase):
    def test_latency_statistics(self):
        stats = latency_statistics([10.0, 12.0, 14.0])

        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["mean_ms"], 12.0)
        self.assertEqual(stats["median_ms"], 12.0)
        self.assertAlmostEqual(stats["p90_ms"], 13.6)

    def test_report_accepts_complete_fast_fnet_result(self):
        report = build_report(
            "fnet", passing_cases(), "fnet.plan", "droid.pth", FRAME_IDS
        )

        self.assertTrue(report["recommendation"]["integrate"])
        self.assertEqual(report["frame_ids"], FRAME_IDS)
        self.assertGreaterEqual(report["latency"]["improvement"], 0.20)

    def test_report_rejects_incomplete_or_wrong_frame_coverage(self):
        report = build_report(
            "fnet", passing_cases()[:-1], "fnet.plan", "droid.pth", FRAME_IDS
        )

        self.assertFalse(report["recommendation"]["integrate"])
        self.assertIn("coverage", report["recommendation"]["reason"])

    def test_report_rejects_parity_actual_backend_and_speed_failures(self):
        variants = []
        bad_parity = passing_cases()
        bad_parity[0]["metrics"] = dict(passing_metrics(), cosine=0.998)
        variants.append((bad_parity, "parity"))
        fake_backend = passing_cases()
        fake_backend[0]["tensorrt_executed"] = False
        variants.append((fake_backend, "TensorRT"))
        slow = passing_cases()
        for case in slow:
            case["tensorrt_ms"] = case["pytorch_ms"] * 0.81
        variants.append((slow, "20%"))

        for cases, reason in variants:
            with self.subTest(reason=reason):
                report = build_report(
                    "fnet", cases, "fnet.plan", "droid.pth", FRAME_IDS
                )
                self.assertFalse(report["recommendation"]["integrate"])
                self.assertIn(reason, report["recommendation"]["reason"])

    def test_cnet_uses_exact_output_contract(self):
        report = build_report(
            "cnet", passing_cases(channels=256), "cnet.plan", "droid.pth", FRAME_IDS
        )

        self.assertTrue(report["recommendation"]["integrate"])

    def test_tensor_contract_accepts_exact_cuda_contiguous_dtypes(self):
        image = FakeTensor((1, 3, 344, 616), "float32")
        features = FakeTensor((1, 128, 43, 77), "float16")

        validate_encoder_tensors("fnet", image, features, "float32", "float16")

    def test_tensor_contract_rejects_cpu_noncontiguous_dtype_and_shape(self):
        valid_image = FakeTensor((1, 3, 344, 616), "float32")
        valid_output = FakeTensor((1, 128, 43, 77), "float16")
        cases = (
            (FakeTensor(valid_image.shape, "float32", is_cuda=False), valid_output, "CUDA"),
            (FakeTensor(valid_image.shape, "float32", contiguous=False), valid_output, "contiguous"),
            (FakeTensor(valid_image.shape, "float16"), valid_output, "dtype"),
            (FakeTensor((1, 3, 343, 616), "float32"), valid_output, "shape"),
            (valid_image, FakeTensor(valid_output.shape, "float32"), "dtype"),
            (valid_image, FakeTensor((1, 127, 43, 77), "float16"), "shape"),
        )

        for image, output, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_encoder_tensors(
                        "fnet", image, output, "float32", "float16"
                    )


if __name__ == "__main__":
    unittest.main()
