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
    def __init__(self, output_override=None):
        self.output_override = output_override
        self.inputs = []
        self.output_cache = {"updated_net": object()}
        self.metadata = {
            "engine_sha256": "update-sha",
            "precision": "fp16_delta_fp32",
        }

    def infer(self, inputs):
        self.inputs.append(inputs)
        if self.output_override is not None:
            return self.output_override
        edges = inputs["net"].shape[0]
        return {
            "updated_net": torch.ones(
                edges, 128, 43, 77, device="cuda", dtype=torch.float32
            ),
            "delta": torch.ones(
                edges, 2, 43, 77, device="cuda", dtype=torch.float32
            ),
            "weight": torch.ones(
                edges, 2, 43, 77, device="cuda", dtype=torch.float32
            ),
        }


class FakeProfiler:
    def __init__(self):
        self.metadata = {}

    def set_metadata(self, key, value):
        self.metadata[key] = value


def valid_inputs(edges=1, device="cuda", dtype=torch.float32):
    return (
        torch.zeros(edges, 128, 43, 77, device=device, dtype=dtype),
        torch.zeros(edges, 128, 43, 77, device=device, dtype=dtype),
        torch.zeros(edges, 196, 43, 77, device=device, dtype=dtype),
        torch.zeros(edges, 4, 43, 77, device=device, dtype=dtype),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DroidUpdateTensorRTAdapterTests(unittest.TestCase):
    def setUp(self):
        sys.modules.pop("frontend.droid_update_backends", None)

    def test_import_does_not_load_tensorrt_or_shared_runner(self):
        sys.modules.pop("tensorrt", None)
        sys.modules.pop("acceleration.trt_engine", None)

        importlib.import_module("frontend.droid_update_backends")

        self.assertNotIn("tensorrt", sys.modules)
        self.assertNotIn("acceleration.trt_engine", sys.modules)

    def test_adapter_executes_exact_bindings_and_returns_tuple(self):
        from frontend.droid_update_backends import DroidUpdateTensorRTAdapter

        runner = FakeRunner()
        adapter = DroidUpdateTensorRTAdapter(
            "update.plan", runner_factory=lambda *args, **kwargs: runner
        )
        inputs = valid_inputs(edges=4)

        outputs = adapter(*inputs)

        self.assertEqual(
            [tuple(value.shape) for value in outputs],
            [(4, 128, 43, 77), (4, 2, 43, 77), (4, 2, 43, 77)],
        )
        self.assertTrue(all(value.dtype == torch.float32 for value in outputs))
        self.assertTrue(all(value.is_contiguous() for value in outputs))
        self.assertEqual(
            set(runner.inputs[0]), {"net", "inp", "corr", "flow"}
        )

    def test_adapter_preserves_amp_fp16_interface_around_fp32_bindings(self):
        from frontend.droid_update_backends import DroidUpdateTensorRTAdapter

        runner = FakeRunner()
        adapter = DroidUpdateTensorRTAdapter(
            "update.plan", runner_factory=lambda *args, **kwargs: runner
        )
        inputs = valid_inputs(edges=4, dtype=torch.float16)

        outputs = adapter(*inputs)

        self.assertTrue(all(value.dtype == torch.float16 for value in outputs))
        self.assertTrue(
            all(value.dtype == torch.float32 for value in runner.inputs[0].values())
        )
        self.assertTrue(
            all(value.is_contiguous() for value in runner.inputs[0].values())
        )

    def test_adapter_accepts_amp_state_with_fp32_default_flow(self):
        from frontend.droid_update_backends import DroidUpdateTensorRTAdapter

        runner = FakeRunner()
        adapter = DroidUpdateTensorRTAdapter(
            "update.plan", runner_factory=lambda *args, **kwargs: runner
        )
        net, inp, corr, flow = valid_inputs(dtype=torch.float16)
        flow = flow.float()

        outputs = adapter(net, inp, corr, flow)

        self.assertTrue(all(value.dtype == torch.float16 for value in outputs))
        self.assertTrue(
            all(value.dtype == torch.float32 for value in runner.inputs[0].values())
        )

    def test_adapter_normalizes_covisible_noncontiguous_flow(self):
        from frontend.droid_update_backends import DroidUpdateTensorRTAdapter

        runner = FakeRunner()
        adapter = DroidUpdateTensorRTAdapter(
            "update.plan", runner_factory=lambda *args, **kwargs: runner
        )
        net, inp, corr, _ = valid_inputs(dtype=torch.float16)
        flow = torch.zeros(1, 4, 43, 154, device="cuda")[..., ::2]
        self.assertFalse(flow.is_contiguous())

        outputs = adapter(net, inp, corr, flow)

        self.assertTrue(runner.inputs[0]["flow"].is_contiguous())
        self.assertEqual(runner.inputs[0]["flow"].dtype, torch.float32)
        self.assertTrue(all(value.dtype == torch.float16 for value in outputs))

    def test_adapter_rejects_invalid_input_before_enqueue(self):
        from frontend.droid_update_backends import DroidUpdateTensorRTAdapter

        cases = []
        cpu = list(valid_inputs(device="cpu"))
        cases.append((cpu, "CUDA"))
        wrong_dtype = list(valid_inputs())
        wrong_dtype[0] = wrong_dtype[0].double()
        cases.append((wrong_dtype, "dtype"))
        wrong_shape = list(valid_inputs())
        wrong_shape[3] = torch.zeros(1, 5, 43, 77, device="cuda")
        cases.append((wrong_shape, "shape"))
        zero_edges = list(valid_inputs(edges=0))
        cases.append((zero_edges, "edge count"))

        for inputs, message in cases:
            runner = FakeRunner()
            adapter = DroidUpdateTensorRTAdapter(
                "update.plan", runner_factory=lambda *args, **kwargs: runner
            )
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    adapter(*inputs)
                self.assertEqual(runner.inputs, [])

    def test_adapter_rejects_mismatched_edge_counts(self):
        from frontend.droid_update_backends import DroidUpdateTensorRTAdapter

        inputs = list(valid_inputs(edges=1))
        inputs[1] = torch.zeros(2, 128, 43, 77, device="cuda")
        runner = FakeRunner()
        adapter = DroidUpdateTensorRTAdapter(
            "update.plan", runner_factory=lambda *args, **kwargs: runner
        )

        with self.assertRaisesRegex(ValueError, "edge count"):
            adapter(*inputs)

        self.assertEqual(runner.inputs, [])

    def test_adapter_rejects_invalid_output_contract(self):
        from frontend.droid_update_backends import DroidUpdateTensorRTAdapter

        inputs = valid_inputs()
        invalid_outputs = (
            {
                "updated_net": torch.zeros(1, 128, 43, 77, device="cuda"),
                "delta": torch.zeros(
                    1, 2, 43, 77, device="cuda", dtype=torch.float16
                ),
                "weight": torch.zeros(1, 2, 43, 77, device="cuda"),
            },
            {
                "updated_net": torch.zeros(1, 127, 43, 77, device="cuda"),
                "delta": torch.zeros(1, 2, 43, 77, device="cuda"),
                "weight": torch.zeros(1, 2, 43, 77, device="cuda"),
            },
        )
        for outputs in invalid_outputs:
            adapter = DroidUpdateTensorRTAdapter(
                "update.plan",
                runner_factory=lambda *args, outputs=outputs, **kwargs: FakeRunner(
                    outputs
                ),
            )
            with self.assertRaises(ValueError):
                adapter(*inputs)

    def test_close_clears_output_cache(self):
        from frontend.droid_update_backends import DroidUpdateTensorRTAdapter

        runner = FakeRunner()
        adapter = DroidUpdateTensorRTAdapter(
            "update.plan", runner_factory=lambda *args, **kwargs: runner
        )

        adapter.close()

        self.assertEqual(runner.output_cache, {})
        self.assertIsNone(adapter.runner)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DroidUpdateBackendSelectionTests(unittest.TestCase):
    def test_forward_core_torch_preserves_default_computation(self):
        from frontend.droid_net import UpdateModule

        update = UpdateModule().cuda().eval()
        torch.manual_seed(41)
        inputs = tuple(torch.randn_like(value) for value in valid_inputs())

        with torch.inference_mode():
            expected = update.forward_core_torch(*inputs)
            actual = update.forward_core(*inputs)

        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=0, atol=0
            )

    def test_strict_load_failure_propagates_without_fallback(self):
        from frontend.droid_update_backends import DroidUpdateBackend

        def fail_adapter(*args, **kwargs):
            raise RuntimeError("update engine load exploded")

        with self.assertRaisesRegex(RuntimeError, "load exploded"):
            DroidUpdateBackend(
                "missing.plan",
                strict=True,
                fallback=None,
                adapter_factory=fail_adapter,
            )

    def test_non_strict_inference_failure_falls_back_once_and_updates_metadata(self):
        from frontend.droid_update_backends import DroidUpdateBackend

        class FailingAdapter(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.close_calls = 0
                self.runner = type(
                    "Runner",
                    (),
                    {
                        "metadata": {
                            "engine_sha256": "sha",
                            "precision": "fp16_delta_fp32",
                        }
                    },
                )()

            def forward(self, *inputs):
                raise RuntimeError("update inference exploded")

            def close(self):
                self.close_calls += 1

        calls = []

        def fallback(*inputs):
            calls.append(inputs)
            edges = inputs[0].shape[0]
            return (
                torch.ones(edges, 128, 43, 77, device="cuda"),
                torch.ones(edges, 2, 43, 77, device="cuda"),
                torch.ones(edges, 2, 43, 77, device="cuda"),
            )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            backend = DroidUpdateBackend(
                "update.plan",
                strict=False,
                fallback=fallback,
                adapter_factory=FailingAdapter,
            )
            adapter = backend.tensorrt
            profiler = FakeProfiler()
            backend.set_profiler(profiler)
            backend(*valid_inputs())
            backend(*valid_inputs())

        self.assertEqual(adapter.close_calls, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(backend.actual_backend, "torch")
        self.assertIn("inference exploded", backend.fallback_reason)
        self.assertEqual(
            profiler.metadata["droid_update"]["actual_backend"], "torch"
        )
        self.assertEqual(stdout.getvalue().count("actual_backend=torch"), 1)

    def test_strict_install_releases_core_modules_but_preserves_graph_agg(self):
        from frontend.droid_net import UpdateModule
        from frontend.droid_update_backends import install_droid_update_backend

        update = UpdateModule().cuda().eval()
        agg = update.agg
        construction_state = []

        class PassingAdapter(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                construction_state.append(
                    (
                        update.corr_encoder,
                        update.flow_encoder,
                        update.gru,
                        update.delta,
                        update.weight,
                        update.agg,
                    )
                )
                self.runner = type(
                    "Runner",
                    (),
                    {
                        "metadata": {
                            "engine_sha256": "sha",
                            "precision": "fp16_delta_fp32",
                        }
                    },
                )()

            def forward(self, net, inp, corr, flow):
                edges = net.shape[0]
                return (
                    torch.ones(edges, 128, 43, 77, device="cuda"),
                    torch.ones(edges, 2, 43, 77, device="cuda"),
                    torch.ones(edges, 2, 43, 77, device="cuda"),
                )

            def close(self):
                pass

        backend = install_droid_update_backend(
            update,
            {
                "droid_update_backend": "tensorrt",
                "droid_update_engine": "update.plan",
                "tensorrt_strict": True,
            },
            adapter_factory=PassingAdapter,
        )

        self.assertEqual(
            construction_state, [(None, None, None, None, None, agg)]
        )
        self.assertIs(update.core_backend, backend)
        self.assertIs(update.agg, agg)
        self.assertIsNone(backend.fallback)

    def test_torch_install_keeps_default_core_unchanged(self):
        from frontend.droid_net import UpdateModule
        from frontend.droid_update_backends import install_droid_update_backend

        update = UpdateModule().cuda().eval()
        core_modules = (
            update.corr_encoder,
            update.flow_encoder,
            update.gru,
            update.delta,
            update.weight,
        )

        backend = install_droid_update_backend(
            update, {"droid_update_backend": "torch"}
        )

        self.assertIsNone(backend)
        self.assertEqual(
            core_modules,
            (
                update.corr_encoder,
                update.flow_encoder,
                update.gru,
                update.delta,
                update.weight,
            ),
        )

    def test_smallcity_config_defaults_update_to_pytorch(self):
        config = yaml.safe_load(
            (REPO_ROOT / "configs/hierarchical/smallcity.yaml").read_text()
        )

        self.assertEqual(config["inference"]["droid_update_backend"], "torch")
        self.assertIsNone(config["inference"]["droid_update_engine"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DroidUpdateRealTensorRTIntegrationTest(unittest.TestCase):
    engine = (
        REPO_ROOT
        / "engines/tensorrt/droid/droid_update_core_43x77_deltafp32.plan"
    )

    @unittest.skipUnless(engine.is_file(), "real update engine missing")
    def test_real_engine_dispatches_all_supported_edge_counts(self):
        from frontend.droid_net import UpdateModule
        from frontend.droid_update_backends import install_droid_update_backend

        update = UpdateModule().cuda().eval()
        agg = update.agg
        backend = install_droid_update_backend(
            update,
            {
                "droid_update_backend": "tensorrt",
                "droid_update_engine": str(self.engine),
                "tensorrt_strict": True,
            },
        )

        with torch.inference_mode():
            for edges in (1, 4, 16, 32, 48):
                inputs = valid_inputs(edges=edges)
                outputs = update.forward_core(*inputs)
                self.assertEqual(
                    [tuple(value.shape) for value in outputs],
                    [
                        (edges, 128, 43, 77),
                        (edges, 2, 43, 77),
                        (edges, 2, 43, 77),
                    ],
                )
                self.assertTrue(
                    all(torch.isfinite(value).all().item() for value in outputs)
                )
                del inputs, outputs

        self.assertIs(update.agg, agg)
        self.assertEqual(backend.actual_backend, "tensorrt")
        self.assertEqual(backend.precision, "fp16_delta_fp32")
        self.assertIsNone(backend.fallback_reason)
        backend.close()

    @unittest.skipUnless(engine.is_file(), "real update engine missing")
    def test_real_engine_handles_motion_filter_amp_and_default_flow(self):
        from frontend.droid_net import UpdateModule
        from frontend.droid_update_backends import install_droid_update_backend

        update = UpdateModule().cuda().eval()
        backend = install_droid_update_backend(
            update,
            {
                "droid_update_backend": "tensorrt",
                "droid_update_engine": str(self.engine),
                "tensorrt_strict": True,
            },
        )
        net = torch.zeros(1, 1, 128, 43, 77, device="cuda", dtype=torch.float16)
        inp = torch.zeros_like(net)
        corr = torch.zeros(
            1, 1, 196, 43, 77, device="cuda", dtype=torch.float16
        )

        with torch.inference_mode():
            updated_net, delta, weight = update(net, inp, corr, flow=None)

        self.assertEqual(tuple(updated_net.shape), (1, 1, 128, 43, 77))
        self.assertEqual(tuple(delta.shape), (1, 1, 43, 77, 2))
        self.assertEqual(tuple(weight.shape), (1, 1, 43, 77, 2))
        self.assertEqual(updated_net.dtype, torch.float16)
        self.assertEqual(delta.dtype, torch.float16)
        self.assertEqual(weight.dtype, torch.float16)
        self.assertEqual(backend.actual_backend, "tensorrt")
        backend.close()

    @unittest.skipUnless(engine.is_file(), "real update engine missing")
    def test_real_engine_handles_covisible_strided_flow(self):
        from frontend.droid_net import UpdateModule
        from frontend.droid_update_backends import install_droid_update_backend

        update = UpdateModule().cuda().eval()
        backend = install_droid_update_backend(
            update,
            {
                "droid_update_backend": "tensorrt",
                "droid_update_engine": str(self.engine),
                "tensorrt_strict": True,
            },
        )
        net, inp, corr, _ = valid_inputs(edges=4, dtype=torch.float16)
        flow = torch.zeros(4, 4, 43, 154, device="cuda")[..., ::2]
        self.assertFalse(flow.is_contiguous())

        with torch.inference_mode():
            outputs = update.forward_core(net, inp, corr, flow)

        self.assertEqual(tuple(outputs[0].shape), (4, 128, 43, 77))
        self.assertTrue(all(value.dtype == torch.float16 for value in outputs))
        self.assertEqual(backend.actual_backend, "tensorrt")
        backend.close()


if __name__ == "__main__":
    unittest.main()
