import importlib
import io
import pathlib
import sys
import unittest
from contextlib import redirect_stdout

import torch
import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class FakeRunner:
    def __init__(self, output=None):
        self.output = output
        self.inputs = []
        self.output_cache = {"features": object()}
        self.metadata = {"engine_sha256": "engine-sha"}

    def infer(self, inputs):
        self.inputs.append(inputs)
        output = self.output
        if output is None:
            output = torch.ones(
                1, 256, 43, 77, device="cuda", dtype=torch.float16
            )
        return {"features": output}


class FakeTorchCNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return torch.full(
            (1, 1, 256, 43, 77),
            2.0,
            device=image.device,
            dtype=torch.float16,
        )


class FakeProfiler:
    def __init__(self):
        self.metadata = {}

    def set_metadata(self, key, value):
        self.metadata[key] = value


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DroidCNetTensorRTAdapterTests(unittest.TestCase):
    def setUp(self):
        sys.modules.pop("frontend.droid_cnet_backends", None)

    def test_module_import_does_not_import_tensorrt_or_shared_runner(self):
        sys.modules.pop("tensorrt", None)
        sys.modules.pop("acceleration.trt_engine", None)

        importlib.import_module("frontend.droid_cnet_backends")

        self.assertNotIn("tensorrt", sys.modules)
        self.assertNotIn("acceleration.trt_engine", sys.modules)

    def test_adapter_preserves_5d_contract_and_executes_static_binding(self):
        from frontend.droid_cnet_backends import DroidCNetTensorRTAdapter

        runner = FakeRunner()
        adapter = DroidCNetTensorRTAdapter(
            "cnet.plan", runner_factory=lambda *args, **kwargs: runner
        )
        image = torch.zeros(
            1, 1, 3, 344, 616, device="cuda", dtype=torch.float32
        )

        features = adapter(image)

        self.assertEqual(tuple(features.shape), (1, 1, 256, 43, 77))
        self.assertEqual(features.dtype, torch.float16)
        self.assertTrue(features.is_cuda)
        self.assertTrue(features.is_contiguous())
        recorded = runner.inputs[0]["image"]
        self.assertEqual(tuple(recorded.shape), (1, 3, 344, 616))
        self.assertEqual(recorded.dtype, torch.float32)
        self.assertTrue(recorded.is_contiguous())

    def test_adapter_rejects_invalid_input_before_inference(self):
        from frontend.droid_cnet_backends import DroidCNetTensorRTAdapter

        cases = (
            (torch.zeros(1, 1, 3, 344, 616), "CUDA"),
            (
                torch.zeros(1, 1, 3, 344, 1232, device="cuda")[..., ::2],
                "contiguous",
            ),
            (
                torch.zeros(
                    1, 1, 3, 344, 616, device="cuda", dtype=torch.float16
                ),
                "dtype",
            ),
            (torch.zeros(1, 2, 3, 344, 616, device="cuda"), "shape"),
        )
        for image, message in cases:
            runner = FakeRunner()
            adapter = DroidCNetTensorRTAdapter(
                "cnet.plan", runner_factory=lambda *args, **kwargs: runner
            )
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    adapter(image)
                self.assertEqual(runner.inputs, [])

    def test_adapter_rejects_invalid_output_contract(self):
        from frontend.droid_cnet_backends import DroidCNetTensorRTAdapter

        image = torch.zeros(1, 1, 3, 344, 616, device="cuda")
        outputs = (
            (
                torch.zeros(
                    1, 256, 43, 77, device="cuda", dtype=torch.float32
                ),
                "dtype",
            ),
            (
                torch.zeros(
                    1, 128, 43, 77, device="cuda", dtype=torch.float16
                ),
                "shape",
            ),
        )
        for output, message in outputs:
            adapter = DroidCNetTensorRTAdapter(
                "cnet.plan",
                runner_factory=lambda *args, output=output, **kwargs: FakeRunner(
                    output
                ),
            )
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    adapter(image)

    def test_close_clears_runner_cache(self):
        from frontend.droid_cnet_backends import DroidCNetTensorRTAdapter

        runner = FakeRunner()
        adapter = DroidCNetTensorRTAdapter(
            "cnet.plan", runner_factory=lambda *args, **kwargs: runner
        )

        adapter.close()

        self.assertEqual(runner.output_cache, {})
        self.assertIsNone(adapter.runner)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DroidCNetBackendSelectionTests(unittest.TestCase):
    def setUp(self):
        sys.modules.pop("frontend.droid_cnet_backends", None)
        self.image = torch.zeros(
            1, 1, 3, 344, 616, device="cuda", dtype=torch.float32
        )

    def test_strict_load_failure_raises_without_fallback(self):
        from frontend.droid_cnet_backends import DroidCNetBackend

        def fail_adapter(*args, **kwargs):
            raise RuntimeError("engine load exploded")

        with self.assertRaisesRegex(RuntimeError, "engine load exploded"):
            DroidCNetBackend(
                "missing.plan",
                strict=True,
                fallback=None,
                adapter_factory=fail_adapter,
            )

    def test_non_strict_load_failure_uses_pytorch_and_records_reason_once(self):
        from frontend.droid_cnet_backends import DroidCNetBackend

        fallback = FakeTorchCNet()

        def fail_adapter(*args, **kwargs):
            raise RuntimeError("engine load exploded")

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            backend = DroidCNetBackend(
                "missing.plan",
                strict=False,
                fallback=fallback,
                adapter_factory=fail_adapter,
            )
            first = backend(self.image)
            second = backend(self.image)

        self.assertEqual(fallback.calls, 2)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(backend.actual_backend, "torch")
        self.assertIn("engine load exploded", backend.fallback_reason)
        self.assertEqual(stdout.getvalue().count("actual_backend=torch"), 1)

    def test_non_strict_inference_failure_switches_once_and_updates_profiler(self):
        from frontend.droid_cnet_backends import DroidCNetBackend

        class FailingAdapter(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.close_calls = 0
                self.runner = type(
                    "Runner", (), {"metadata": {"engine_sha256": "sha"}}
                )()

            def forward(self, image):
                raise RuntimeError("inference exploded")

            def close(self):
                self.close_calls += 1

        fallback = FakeTorchCNet()
        backend = DroidCNetBackend(
            "engine.plan",
            strict=False,
            fallback=fallback,
            adapter_factory=FailingAdapter,
        )
        adapter = backend.tensorrt
        profiler = FakeProfiler()
        backend.set_profiler(profiler)

        result = backend(self.image)
        backend(self.image)

        self.assertEqual(tuple(result.shape), (1, 1, 256, 43, 77))
        self.assertEqual(adapter.close_calls, 1)
        self.assertEqual(fallback.calls, 2)
        self.assertEqual(backend.actual_backend, "torch")
        self.assertIn("inference exploded", backend.fallback_reason)
        self.assertEqual(
            profiler.metadata["droid_cnet"]["actual_backend"], "torch"
        )

    def test_backend_status_proves_strict_tensorrt_engine(self):
        from frontend.droid_cnet_backends import DroidCNetBackend

        class PassingAdapter(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.runner = type(
                    "Runner", (), {"metadata": {"engine_sha256": "sha"}}
                )()

            def forward(self, image):
                return torch.ones(
                    1, 1, 256, 43, 77, device="cuda", dtype=torch.float16
                )

            def close(self):
                pass

        backend = DroidCNetBackend(
            "engine.plan",
            metadata_path="engine.json",
            strict=True,
            fallback=None,
            adapter_factory=PassingAdapter,
        )

        status = backend.backend_status()

        self.assertEqual(status["requested_backend"], "tensorrt")
        self.assertEqual(status["actual_backend"], "tensorrt")
        self.assertTrue(status["strict"])
        self.assertEqual(status["engine_path"], "engine.plan")
        self.assertEqual(status["engine_sha256"], "sha")
        self.assertIsNone(status["fallback_reason"])

    def test_strict_install_releases_torch_cnet_before_adapter_construction(self):
        from frontend.droid_cnet_backends import install_droid_cnet_backend

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cnet = FakeTorchCNet().cuda()

        net = Net()
        construction_state = []

        class PassingAdapter(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                construction_state.append(net.cnet)
                self.runner = type(
                    "Runner", (), {"metadata": {"engine_sha256": "sha"}}
                )()

            def forward(self, image):
                return torch.ones(
                    1, 1, 256, 43, 77, device="cuda", dtype=torch.float16
                )

            def close(self):
                pass

        backend = install_droid_cnet_backend(
            net,
            {
                "droid_cnet_backend": "tensorrt",
                "droid_cnet_engine": "engine.plan",
                "tensorrt_strict": True,
            },
            adapter_factory=PassingAdapter,
        )

        self.assertEqual(construction_state, [None])
        self.assertIs(net.cnet, backend)
        self.assertIsNone(backend.fallback)

    def test_torch_install_is_a_noop(self):
        from frontend.droid_cnet_backends import install_droid_cnet_backend

        net = torch.nn.Module()
        net.cnet = FakeTorchCNet()
        original = net.cnet

        backend = install_droid_cnet_backend(
            net, {"droid_cnet_backend": "torch"}
        )

        self.assertIsNone(backend)
        self.assertIs(net.cnet, original)

    def test_smallcity_config_defaults_cnet_to_pytorch(self):
        config = yaml.safe_load(
            (REPO_ROOT / "configs" / "hierarchical" / "smallcity.yaml").read_text()
        )

        self.assertEqual(config["inference"]["droid_cnet_backend"], "torch")
        self.assertIsNone(config["inference"]["droid_cnet_engine"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DroidCNetRealTensorRTIntegrationTest(unittest.TestCase):
    engine = (
        REPO_ROOT
        / "engines"
        / "tensorrt"
        / "droid"
        / "droid_cnet_b1_344x616_fp16.plan"
    )
    frame = (
        REPO_ROOT
        / "data"
        / "smallcity_subset_200"
        / "small_city"
        / "color"
        / "00000.png"
    )

    @unittest.skipUnless(engine.is_file() and frame.is_file(), "real artifacts missing")
    def test_real_engine_preserves_motion_filter_context_contract(self):
        from acceleration.export_droid_encoders_onnx import prepare_encoder_input
        from frontend.droid_cnet_backends import DroidCNetBackend
        from frontend.motion_filter import MotionFilter

        backend = DroidCNetBackend(
            str(self.engine), strict=True, fallback=None
        )
        net = type(
            "Net",
            (),
            {"cnet": backend, "fnet": object(), "update": object()},
        )()
        motion_filter = MotionFilter(net, video=None, device="cuda:0")
        image = prepare_encoder_input(self.frame, device="cuda").unsqueeze(1)

        context_net, context_inp = motion_filter.context_encoder(image)

        self.assertEqual(tuple(context_net.shape), (1, 128, 43, 77))
        self.assertEqual(tuple(context_inp.shape), (1, 128, 43, 77))
        self.assertEqual(context_net.dtype, torch.float16)
        self.assertEqual(context_inp.dtype, torch.float16)
        self.assertTrue(torch.isfinite(context_net).all().item())
        self.assertTrue(torch.isfinite(context_inp).all().item())
        self.assertEqual(backend.actual_backend, "tensorrt")
        self.assertIsNone(backend.fallback_reason)
        backend.close()


if __name__ == "__main__":
    unittest.main()
