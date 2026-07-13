import pathlib
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from acceleration.export_metric3d_onnx import (
    Metric3DDepthExportWrapper,
    apply_depth_scale,
    build_report,
    load_runtime_rgb,
    onnx_export_options,
    parity_metrics,
    require_provider,
    sample_frame_ids,
    scaled_forward_size,
    smallcity_intrinsics,
)


class Metric3DOnnxUtilityTests(unittest.TestCase):
    def test_export_options_disable_broken_jetson_constant_fold(self):
        options = onnx_export_options()

        self.assertEqual(options["opset_version"], 17)
        self.assertEqual(options["input_names"], ["rgb"])
        self.assertEqual(options["output_names"], ["depth"])
        self.assertFalse(options["do_constant_folding"])

    def test_export_options_support_tensor_rt_compatible_opset16(self):
        options = onnx_export_options(opset_version=16)

        self.assertEqual(options["opset_version"], 16)
        self.assertFalse(options["do_constant_folding"])

    def test_sample_frames_are_evenly_spaced(self):
        self.assertEqual(sample_frame_ids(), (0, 5, 10, 15, 20, 25, 30, 35, 40, 45))

    def test_scaled_forward_size_is_static_balanced_fast_shape(self):
        self.assertEqual(scaled_forward_size((616, 1064), 0.75), (448, 784))

    def test_parity_metrics_use_positive_finite_expected_mask(self):
        expected = np.array([0.0, 1.0, 2.0, np.nan], dtype=np.float32)
        actual = np.array([9.0, 1.001, 1.998, 7.0], dtype=np.float32)

        result = parity_metrics(expected, actual)

        self.assertEqual(result["valid_count"], 2)
        self.assertTrue(result["finite"])
        self.assertGreaterEqual(result["cosine"], 0.999)
        self.assertLessEqual(result["mean_relative_error"], 0.02)

    def test_parity_metrics_reject_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            parity_metrics(np.zeros((2, 2)), np.zeros((4,)))

    def test_parity_metrics_reject_empty_valid_mask(self):
        with self.assertRaisesRegex(ValueError, "valid"):
            parity_metrics(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_cuda_provider_is_required(self):
        with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"):
            require_provider(["CPUExecutionProvider"], "CUDAExecutionProvider")


class Metric3DOnnxWrapperAndReportTests(unittest.TestCase):
    def test_wrapper_returns_only_depth_prediction(self):
        class FakeDepthModel(torch.nn.Module):
            def forward(self, input):
                return {
                    "prediction": input[:, :1] + 2.0,
                    "confidence": input[:, :1] - 2.0,
                }

        wrapper = Metric3DDepthExportWrapper(FakeDepthModel())
        rgb = torch.ones(1, 3, 4, 5)

        depth = wrapper(rgb)

        self.assertEqual(tuple(depth.shape), (1, 1, 4, 5))
        self.assertTrue(torch.equal(depth, torch.full((1, 1, 4, 5), 3.0)))

    @staticmethod
    def passing_metrics():
        return {
            "shape": [1, 1, 448, 784],
            "valid_count": 100,
            "finite": True,
            "cosine": 0.9999,
            "mean_relative_error": 0.001,
        }

    def test_report_requires_all_frames_and_both_boundaries(self):
        cases = [
            {
                "frame_id": frame_id,
                "raw": self.passing_metrics(),
                "final": self.passing_metrics(),
            }
            for frame_id in sample_frame_ids()
        ]

        report = build_report(cases, onnx_path="model.onnx", provider="CUDAExecutionProvider")

        self.assertEqual(report["frame_ids"], list(sample_frame_ids()))
        self.assertTrue(report["accepted"])
        self.assertEqual(report["thresholds"]["minimum_cosine"], 0.999)
        self.assertEqual(report["thresholds"]["maximum_mean_relative_error"], 0.02)
        self.assertEqual(report["aggregates"]["minimum_cosine"], 0.9999)
        self.assertEqual(report["aggregates"]["maximum_mean_relative_error"], 0.001)

    def test_report_rejects_threshold_failure(self):
        cases = [
            {
                "frame_id": frame_id,
                "raw": self.passing_metrics(),
                "final": self.passing_metrics(),
            }
            for frame_id in sample_frame_ids()
        ]
        cases[4]["final"] = dict(self.passing_metrics(), mean_relative_error=0.0201)

        report = build_report(cases, onnx_path="model.onnx", provider="CUDAExecutionProvider")

        self.assertFalse(report["accepted"])

    def test_report_rejects_incomplete_coverage(self):
        cases = [
            {
                "frame_id": frame_id,
                "raw": self.passing_metrics(),
                "final": self.passing_metrics(),
            }
            for frame_id in sample_frame_ids()[:-1]
        ]

        report = build_report(cases, onnx_path="model.onnx", provider="CUDAExecutionProvider")

        self.assertFalse(report["accepted"])


class Metric3DRealInputPreparationTests(unittest.TestCase):
    def test_apply_depth_scale_sets_static_crop_and_vit_size(self):
        predictor = type(
            "Predictor",
            (),
            {
                "cfg_": type(
                    "Config",
                    (),
                    {
                        "data_basic": {
                            "crop_size": (616, 1064),
                            "vit_size": (616, 1064),
                        }
                    },
                )()
            },
        )()

        shape = apply_depth_scale(predictor, 0.75)

        self.assertEqual(shape, (448, 784))
        self.assertEqual(predictor.cfg_.data_basic["crop_size"], (448, 784))
        self.assertEqual(predictor.cfg_.data_basic["vit_size"], (448, 784))

    def test_runtime_image_loader_matches_smallcity_rgb_chw_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "frame.png"
            bgr = np.zeros((2, 3, 3), dtype=np.uint8)
            bgr[..., 0] = 10
            bgr[..., 1] = 20
            bgr[..., 2] = 30
            self.assertTrue(cv2.imwrite(str(path), bgr))

            rgb = load_runtime_rgb(path, image_size=(344, 616))

            self.assertEqual(tuple(rgb.shape), (3, 344, 616))
            self.assertEqual(rgb.dtype, torch.float32)
            self.assertTrue(torch.all(rgb[0] == 30))
            self.assertTrue(torch.all(rgb[1] == 20))
            self.assertTrue(torch.all(rgb[2] == 10))

    def test_smallcity_intrinsics_match_metric_model_order(self):
        config = {
            "intrinsic": {"fv": 486.9, "fu": 487.1, "cv": 512.0, "cu": 345.0}
        }

        intrinsic = smallcity_intrinsics(config)

        np.testing.assert_allclose(intrinsic, [486.9, 487.1, 512.0, 345.0])

if __name__ == "__main__":
    unittest.main()
