import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.acceleration.export_droid_encoders_onnx import (
    EncoderCoreExportWrapper,
    admitted_frame_ids,
    build_report,
    encoder_spec,
    load_droid_state_dict,
    load_encoder,
    onnx_export_options,
    parity_metrics,
    prepare_encoder_input,
    require_provider,
    select_validation_frame_ids,
)
from scripts.frontend.modules.extractor import BasicEncoder


class FakeEncoder:
    def __init__(self, name):
        self.name = name
        self.device = None
        self.evaluating = False

    def to(self, device):
        self.device = str(device)
        return self

    def eval(self):
        self.evaluating = True
        return self


class FakeDroidNet:
    instances = []

    def __init__(self):
        self.fnet = FakeEncoder("fnet")
        self.cnet = FakeEncoder("cnet")
        self.loaded_state = None
        self.strict = None
        type(self).instances.append(self)

    def load_state_dict(self, state, strict=True):
        self.loaded_state = state
        self.strict = strict


class DroidEncoderExportUtilityTests(unittest.TestCase):
    def test_encoder_specs_are_independent_and_validated(self):
        fnet = encoder_spec("fnet")
        cnet = encoder_spec("cnet")

        self.assertEqual(fnet["output_channels"], 128)
        self.assertEqual(fnet["norm_fn"], "instance")
        self.assertEqual(cnet["output_channels"], 256)
        self.assertEqual(cnet["norm_fn"], "none")
        fnet["output_channels"] = 1
        self.assertEqual(encoder_spec("fnet")["output_channels"], 128)
        with self.assertRaisesRegex(ValueError, "unsupported DROID encoder"):
            encoder_spec("update")

    def test_checkpoint_loader_strips_prefix_and_crops_update_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "droid.pth"
            torch.save(
                {
                    "module.fnet.conv.weight": torch.ones(2, 2),
                    "module.cnet.conv.weight": torch.ones(3, 3),
                    "module.update.weight.2.weight": torch.arange(16).view(4, 4),
                    "module.update.weight.2.bias": torch.arange(4),
                    "module.update.delta.2.weight": torch.arange(20).view(4, 5),
                    "module.update.delta.2.bias": torch.arange(4),
                },
                checkpoint,
            )

            state = load_droid_state_dict(checkpoint)

        self.assertIn("fnet.conv.weight", state)
        self.assertIn("cnet.conv.weight", state)
        self.assertNotIn("module.fnet.conv.weight", state)
        self.assertEqual(tuple(state["update.weight.2.weight"].shape), (2, 4))
        self.assertEqual(tuple(state["update.weight.2.bias"].shape), (2,))
        self.assertEqual(tuple(state["update.delta.2.weight"].shape), (2, 5))
        self.assertEqual(tuple(state["update.delta.2.bias"].shape), (2,))

    def test_load_encoder_strictly_validates_full_net_then_selects_one(self):
        FakeDroidNet.instances = []
        state = {
            "update.weight.2.weight": torch.ones(2, 2),
            "update.weight.2.bias": torch.ones(2),
            "update.delta.2.weight": torch.ones(2, 2),
            "update.delta.2.bias": torch.ones(2),
        }

        encoder = load_encoder(
            "cnet",
            checkpoint="unused.pth",
            device="cuda:0",
            state_loader=lambda path: state,
            droid_net_factory=FakeDroidNet,
        )

        instance = FakeDroidNet.instances[0]
        self.assertIs(instance.loaded_state, state)
        self.assertTrue(instance.strict)
        self.assertIs(encoder, instance.cnet)
        self.assertEqual(encoder.device, "cuda:0")
        self.assertTrue(encoder.evaluating)

    def test_admitted_frame_reader_filters_orders_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["frame_idx", "stage", "elapsed_s"]
                )
                writer.writeheader()
                for frame_id in list(range(25)) + [3, 7]:
                    writer.writerow(
                        {
                            "frame_idx": frame_id,
                            "stage": "motion_filter_feature_encoder",
                            "elapsed_s": 0.01,
                        }
                    )
                    writer.writerow(
                        {
                            "frame_idx": frame_id,
                            "stage": "frame_total",
                            "elapsed_s": 1.0,
                        }
                    )

            frame_ids = admitted_frame_ids(path, minimum=20)

        self.assertEqual(frame_ids, list(range(25)))

    def test_admitted_frame_reader_rejects_insufficient_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            path.write_text(
                "frame_idx,stage,elapsed_s\n"
                "0,motion_filter_feature_encoder,0.1\n"
            )
            with self.assertRaisesRegex(ValueError, "at least 20"):
                admitted_frame_ids(path, minimum=20)

    def test_validation_selection_is_distributed_and_includes_endpoints(self):
        selected = select_validation_frame_ids(list(range(69)), count=20)

        self.assertEqual(len(selected), 20)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 68)
        self.assertEqual(selected, sorted(set(selected)))

    def test_prepare_input_reproduces_motion_filter_bgr_normalization(self):
        image = np.asarray(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            self.assertTrue(cv2.imwrite(str(path), image))

            actual = prepare_encoder_input(path, image_size=(2, 2), device="cpu")

        expected = torch.from_numpy(image).permute(2, 0, 1).float()[None] / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        expected = expected.sub(mean).div(std)
        self.assertEqual(tuple(actual.shape), (1, 3, 2, 2))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, expected)


class DroidEncoderOnnxBoundaryTests(unittest.TestCase):
    def test_onnx_export_options_use_static_named_opset17_graph(self):
        self.assertEqual(
            onnx_export_options(),
            {
                "opset_version": 17,
                "input_names": ["image"],
                "output_names": ["features"],
                "do_constant_folding": False,
            },
        )

    def test_cuda_provider_is_required(self):
        require_provider(["CUDAExecutionProvider", "CPUExecutionProvider"])
        with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"):
            require_provider(["CPUExecutionProvider"])

    def test_export_wrapper_uses_4d_core_for_both_encoders(self):
        for output_dim, norm_fn in ((128, "instance"), (256, "none")):
            with self.subTest(output_dim=output_dim, norm_fn=norm_fn):
                encoder = BasicEncoder(output_dim=output_dim, norm_fn=norm_fn).eval()
                wrapper = EncoderCoreExportWrapper(encoder).eval()
                image = torch.randn(1, 3, 32, 48)

                with torch.no_grad():
                    output = wrapper(image)

                self.assertEqual(tuple(output.shape), (1, output_dim, 4, 6))
                self.assertTrue(torch.isfinite(output).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA autocast")
    def test_export_wrapper_preserves_online_fp16_output_on_cuda(self):
        encoder = BasicEncoder(output_dim=128, norm_fn="instance").cuda().eval()
        wrapper = EncoderCoreExportWrapper(encoder).eval()

        with torch.no_grad():
            output = wrapper(torch.randn(1, 3, 32, 48, device="cuda"))

        self.assertEqual(output.dtype, torch.float16)

    def test_parity_metrics_require_matching_shapes(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            parity_metrics(np.ones((1, 2)), np.ones((2, 1)))

    def test_parity_metrics_cover_all_activation_values(self):
        expected = np.asarray([[-2.0, 0.0, 4.0]], dtype=np.float32)
        actual = np.asarray([[-2.01, 0.0, 3.99]], dtype=np.float16)

        metrics = parity_metrics(expected, actual)

        self.assertEqual(metrics["shape"], [1, 3])
        self.assertEqual(metrics["dtype"], "float16")
        self.assertEqual(metrics["value_count"], 3)
        self.assertTrue(metrics["finite"])
        self.assertGreater(metrics["cosine"], 0.999)
        self.assertLess(metrics["mean_relative_error"], 0.01)

    @staticmethod
    def passing_cases(encoder="fnet"):
        channels = encoder_spec(encoder)["output_channels"]
        return [
            {
                "frame_id": frame_id,
                "metrics": {
                    "shape": [1, channels, 43, 77],
                    "dtype": "float16",
                    "value_count": channels * 43 * 77,
                    "finite": True,
                    "cosine": 0.9995,
                    "mean_relative_error": 0.009,
                },
            }
            for frame_id in range(20)
        ]

    def test_report_accepts_exact_cuda_coverage_and_thresholds(self):
        report = build_report(
            encoder="fnet",
            cases=self.passing_cases("fnet"),
            onnx_path="fnet.onnx",
            provider="CUDAExecutionProvider",
        )

        self.assertTrue(report["accepted"])
        self.assertEqual(report["frame_ids"], list(range(20)))
        self.assertEqual(report["aggregates"]["minimum_cosine"], 0.9995)
        self.assertEqual(report["aggregates"]["maximum_mean_relative_error"], 0.009)

    def test_report_rejects_missing_frames_cpu_or_bad_parity(self):
        cases = self.passing_cases("cnet")
        self.assertFalse(
            build_report("cnet", cases[:-1], "cnet.onnx", "CUDAExecutionProvider")[
                "accepted"
            ]
        )
        self.assertFalse(
            build_report("cnet", cases, "cnet.onnx", "CPUExecutionProvider")[
                "accepted"
            ]
        )
        cases[4]["metrics"]["mean_relative_error"] = 0.0101
        self.assertFalse(
            build_report("cnet", cases, "cnet.onnx", "CUDAExecutionProvider")[
                "accepted"
            ]
        )


if __name__ == "__main__":
    unittest.main()
    parity_metrics,
