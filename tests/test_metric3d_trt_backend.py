import importlib
import io
import sys
import types
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch

from scripts.metric.metric3d_backends import (
    Metric3DPyTorchBackend,
    Metric3DTensorRTBackend,
)


class FakeMetric:
    init_calls = 0
    load_config_calls = 0

    def __init__(self, checkpoint, model_name):
        type(self).init_calls += 1
        self.cfg_ = self._make_config()

    @staticmethod
    def _make_config():
        return types.SimpleNamespace(
            data_basic={
                "crop_size": (616, 1064),
                "vit_size": (616, 1064),
                "depth_range": (0.0, 300.0),
            }
        )

    def _load_config_(self, model_name, checkpoint):
        type(self).load_config_calls += 1
        return self._make_config()

    def preprocess(self, image, intrinsic):
        return (
            torch.zeros(1, 3, 448, 784),
            None,
            None,
            None,
            tuple(image.shape[:2]),
        )

    def forward_depth(self, *prepared):
        return torch.ones(1, 1, 448, 784)

    def postprocess(self, prediction, d_max, d_min):
        depth = prediction.squeeze().cpu().numpy().copy()
        depth[depth > d_max] = 0
        depth[depth < d_min] = 0
        return depth


class FakeRunner:
    instances = []

    def __init__(self, plan_path, metadata_path=None):
        self.plan_path = plan_path
        self.metadata_path = metadata_path
        self.output_cache = {"depth": object()}
        self.inputs = []
        type(self).instances.append(self)

    def infer(self, inputs):
        self.inputs.append(inputs)
        return {"depth": torch.ones(1, 1, 448, 784)}


class Metric3DBackendConstructionTests(unittest.TestCase):
    def setUp(self):
        FakeMetric.init_calls = 0
        FakeMetric.load_config_calls = 0
        FakeRunner.instances = []

    def test_pytorch_backend_constructs_full_metric_model(self):
        backend = Metric3DPyTorchBackend(
            checkpoint="checkpoint.pth",
            depth_scale=0.75,
            metric_class=FakeMetric,
        )

        self.assertEqual(FakeMetric.init_calls, 1)
        self.assertEqual(FakeMetric.load_config_calls, 0)
        self.assertEqual(backend.predictor.cfg_.data_basic["crop_size"], (448, 784))
        self.assertEqual(backend.predictor.cfg_.data_basic["vit_size"], (448, 784))

    def test_tensorrt_backend_loads_config_without_constructing_model(self):
        backend = Metric3DTensorRTBackend(
            engine_path="engine.plan",
            metadata_path="engine.json",
            checkpoint="checkpoint.pth",
            depth_scale=0.75,
            metric_class=FakeMetric,
            runner_factory=FakeRunner,
        )

        self.assertEqual(FakeMetric.init_calls, 0)
        self.assertEqual(FakeMetric.load_config_calls, 1)
        self.assertEqual(backend.predictor.cfg_.data_basic["crop_size"], (448, 784))
        self.assertEqual(backend.predictor.cfg_.data_basic["vit_size"], (448, 784))
        self.assertEqual(FakeRunner.instances[0].plan_path, "engine.plan")
        self.assertEqual(FakeRunner.instances[0].metadata_path, "engine.json")

    def test_tensorrt_infer_enforces_static_input_contract(self):
        backend = Metric3DTensorRTBackend(
            engine_path="engine.plan",
            metadata_path="engine.json",
            checkpoint="checkpoint.pth",
            depth_scale=0.75,
            metric_class=FakeMetric,
            runner_factory=FakeRunner,
        )
        noncontiguous = torch.zeros(1, 3, 448, 1568)[:, :, :, ::2]

        output = backend.infer((noncontiguous, None, None, None, (344, 616)))

        recorded = FakeRunner.instances[0].inputs[0]["rgb"]
        self.assertEqual(tuple(output.shape), (1, 1, 448, 784))
        self.assertEqual(recorded.dtype, torch.float32)
        self.assertTrue(recorded.is_contiguous())

    def test_tensorrt_infer_rejects_wrong_static_shape(self):
        backend = Metric3DTensorRTBackend(
            engine_path="engine.plan",
            metadata_path="engine.json",
            checkpoint="checkpoint.pth",
            depth_scale=0.75,
            metric_class=FakeMetric,
            runner_factory=FakeRunner,
        )

        with self.assertRaisesRegex(ValueError, "must have shape"):
            backend.infer((torch.zeros(1, 3, 224, 392), None, None, None, (344, 616)))

        self.assertEqual(FakeRunner.instances[0].inputs, [])

    def test_tensorrt_infer_moves_input_to_configured_device(self):
        backend = Metric3DTensorRTBackend(
            engine_path="engine.plan",
            metadata_path="engine.json",
            checkpoint="checkpoint.pth",
            depth_scale=0.75,
            device="meta",
            metric_class=FakeMetric,
            runner_factory=FakeRunner,
        )

        backend.infer(
            (torch.zeros(1, 3, 448, 784), None, None, None, (344, 616))
        )

        recorded = FakeRunner.instances[0].inputs[0]["rgb"]
        self.assertEqual(recorded.device.type, "meta")

    def test_tensorrt_rejects_non_static_depth_scale(self):
        with self.assertRaisesRegex(ValueError, "requires forward shape"):
            Metric3DTensorRTBackend(
                engine_path="engine.plan",
                metadata_path="engine.json",
                checkpoint="checkpoint.pth",
                depth_scale=1.0,
                metric_class=FakeMetric,
                runner_factory=FakeRunner,
            )

        self.assertEqual(FakeMetric.init_calls, 0)
        self.assertEqual(FakeRunner.instances, [])

    def test_tensorrt_close_clears_output_cache_and_releases_runner(self):
        backend = Metric3DTensorRTBackend(
            engine_path="engine.plan",
            metadata_path="engine.json",
            checkpoint="checkpoint.pth",
            depth_scale=0.75,
            metric_class=FakeMetric,
            runner_factory=FakeRunner,
        )
        runner = backend.runner

        backend.close()

        self.assertEqual(runner.output_cache, {})
        self.assertIsNone(backend.runner)
        self.assertIsNone(backend.predictor)


class FakeFacadeBackend:
    def __init__(self, name, fail_infer=False):
        self.name = name
        self.fail_infer = fail_infer
        self.preprocess_calls = 0
        self.infer_calls = 0
        self.postprocess_calls = 0
        self.close_calls = 0

    def preprocess(self, image, intrinsic):
        self.preprocess_calls += 1
        return (torch.zeros(1, 3, 448, 784), None, None, None, image.shape[:2])

    def infer(self, prepared):
        self.infer_calls += 1
        if self.fail_infer:
            raise RuntimeError("inference exploded")
        return torch.ones(1, 1, 4, 4)

    def postprocess(self, prediction, d_max, d_min):
        self.postprocess_calls += 1
        return prediction.squeeze().cpu().numpy().astype(np.float32)

    def close(self):
        self.close_calls += 1


def load_metric_model_module():
    previous_metric_modules = sys.modules.get("metric_modules")
    previous_model = sys.modules.pop("scripts.metric.metric_model", None)
    sys.modules["metric_modules"] = types.SimpleNamespace(Metric=FakeMetric)
    try:
        return importlib.import_module("scripts.metric.metric_model")
    finally:
        if previous_metric_modules is None:
            sys.modules.pop("metric_modules", None)
        else:
            sys.modules["metric_modules"] = previous_metric_modules
        if previous_model is not None:
            sys.modules["scripts.metric.metric_model"] = previous_model


def metric_config(backend="torch", strict=False, engine=None):
    return {
        "device": {"tracker": "cpu"},
        "intrinsic": {"fv": 80.0, "fu": 100.0, "cv": 40.0, "cu": 50.0},
        "metric_depth_scale": 0.75,
        "inference": {
            "metric3d_backend": backend,
            "metric3d_engine": engine,
            "tensorrt_strict": strict,
        },
    }


class MetricModelFacadeTests(unittest.TestCase):
    def setUp(self):
        self.module = load_metric_model_module()

    def test_missing_inference_config_defaults_to_pytorch_only(self):
        pytorch = FakeFacadeBackend("torch")
        cfg = metric_config()
        cfg.pop("inference")
        with mock.patch.object(
            self.module, "Metric3DPyTorchBackend", return_value=pytorch
        ) as pytorch_factory, mock.patch.object(
            self.module, "Metric3DTensorRTBackend"
        ) as tensorrt_factory:
            model = self.module.Metric_Model(cfg)

        self.assertEqual(model.actual_backend, "torch")
        pytorch_factory.assert_called_once()
        tensorrt_factory.assert_not_called()

    def test_tensorrt_selection_does_not_construct_pytorch(self):
        tensorrt = FakeFacadeBackend("tensorrt")
        with mock.patch.object(
            self.module, "Metric3DPyTorchBackend"
        ) as pytorch_factory, mock.patch.object(
            self.module, "Metric3DTensorRTBackend", return_value=tensorrt
        ) as tensorrt_factory:
            model = self.module.Metric_Model(
                metric_config(backend="tensorrt", strict=True, engine="engine.plan")
            )

        self.assertEqual(model.actual_backend, "tensorrt")
        pytorch_factory.assert_not_called()
        tensorrt_factory.assert_called_once()

    def test_strict_initialization_failure_never_constructs_pytorch(self):
        with mock.patch.object(
            self.module, "Metric3DPyTorchBackend"
        ) as pytorch_factory, mock.patch.object(
            self.module,
            "Metric3DTensorRTBackend",
            side_effect=ValueError("bad engine"),
        ):
            with self.assertRaisesRegex(RuntimeError, "strict mode"):
                self.module.Metric_Model(
                    metric_config(backend="tensorrt", strict=True, engine="engine.plan")
                )

        pytorch_factory.assert_not_called()

    def test_strict_inference_failure_never_constructs_pytorch(self):
        tensorrt = FakeFacadeBackend("tensorrt", fail_infer=True)
        with mock.patch.object(
            self.module, "Metric3DPyTorchBackend"
        ) as pytorch_factory, mock.patch.object(
            self.module, "Metric3DTensorRTBackend", return_value=tensorrt
        ):
            model = self.module.Metric_Model(
                metric_config(backend="tensorrt", strict=True, engine="engine.plan")
            )
            with self.assertRaisesRegex(RuntimeError, "strict mode"):
                model.predict(torch.zeros(3, 4, 4))

        pytorch_factory.assert_not_called()

    def test_non_strict_initialization_failure_falls_back_once(self):
        pytorch = FakeFacadeBackend("torch")
        output = io.StringIO()
        with mock.patch.object(
            self.module, "Metric3DPyTorchBackend", return_value=pytorch
        ) as pytorch_factory, mock.patch.object(
            self.module,
            "Metric3DTensorRTBackend",
            side_effect=ValueError("bad engine"),
        ), redirect_stdout(output):
            model = self.module.Metric_Model(
                metric_config(backend="tensorrt", strict=False, engine="engine.plan")
            )

        self.assertEqual(model.actual_backend, "torch")
        self.assertIn("ValueError: bad engine", model.fallback_reason)
        self.assertEqual(output.getvalue().count("fallback_from=tensorrt"), 1)
        pytorch_factory.assert_called_once()

    def test_non_strict_inference_failure_releases_and_stays_on_pytorch(self):
        tensorrt = FakeFacadeBackend("tensorrt", fail_infer=True)
        pytorch = FakeFacadeBackend("torch")
        output = io.StringIO()
        with mock.patch.object(
            self.module, "Metric3DPyTorchBackend", return_value=pytorch
        ) as pytorch_factory, mock.patch.object(
            self.module, "Metric3DTensorRTBackend", return_value=tensorrt
        ), redirect_stdout(output):
            model = self.module.Metric_Model(
                metric_config(backend="tensorrt", strict=False, engine="engine.plan")
            )
            model._cleanup_cuda = mock.Mock()
            first = model.predict(torch.zeros(3, 4, 4))
            second = model.predict(torch.zeros(3, 4, 4))

        self.assertEqual(tuple(first.shape), (4, 4))
        self.assertEqual(tuple(second.shape), (4, 4))
        self.assertEqual(tensorrt.infer_calls, 1)
        self.assertEqual(tensorrt.close_calls, 1)
        self.assertEqual(pytorch.preprocess_calls, 2)
        self.assertEqual(pytorch.infer_calls, 2)
        self.assertEqual(model.actual_backend, "torch")
        self.assertEqual(output.getvalue().count("fallback_from=tensorrt"), 1)
        model._cleanup_cuda.assert_called_once()
        pytorch_factory.assert_called_once()

    def test_predict_records_actual_backend_in_profiler_metadata(self):
        pytorch = FakeFacadeBackend("torch")

        class RecordingProfiler:
            def __init__(self):
                self.metadata = {}

            @contextmanager
            def time(self, stage, frame_idx=None):
                yield

            def set_metadata(self, key, value):
                self.metadata[key] = value

        profiler = RecordingProfiler()
        with mock.patch.object(
            self.module, "Metric3DPyTorchBackend", return_value=pytorch
        ):
            model = self.module.Metric_Model(metric_config())
            model.predict(torch.zeros(3, 4, 4), profiler=profiler, frame_idx=3)

        self.assertEqual(
            profiler.metadata["metric_depth"],
            {
                "requested_backend": "torch",
                "actual_backend": "torch",
                "strict": False,
                "engine_path": None,
                "fallback_reason": None,
            },
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class Metric3DRealTensorRTIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.engine = (
            cls.repo_root
            / "engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan"
        )
        cls.metadata = cls.engine.with_suffix(".json")
        cls.checkpoint = cls.repo_root / "ckpts/metric_depth_vit_small_800k.pth"
        cls.frame = (
            cls.repo_root
            / "data/smallcity_subset_50/small_city/color/00000.png"
        )
        for path in (cls.engine, cls.metadata, cls.checkpoint, cls.frame):
            if not path.is_file():
                raise unittest.SkipTest(f"required integration artifact is missing: {path}")

    def config(self, engine):
        return {
            "device": {"tracker": "cuda:0"},
            "intrinsic": {"fv": 693.0, "fu": 693.0, "cv": 172.0, "cu": 308.0},
            "metric_depth_scale": 0.75,
            "inference": {
                "metric3d_backend": "tensorrt",
                "metric3d_engine": str(engine),
                "tensorrt_strict": True,
            },
        }

    def test_strict_missing_engine_fails_without_constructing_metric_model(self):
        module = load_metric_model_module()
        missing = self.engine.with_name("missing_metric3d.plan")
        with mock.patch.object(
            module.Metric,
            "__init__",
            side_effect=AssertionError("heavy Metric3D constructor was called"),
        ):
            with self.assertRaisesRegex(RuntimeError, "strict mode"):
                module.Metric_Model(self.config(missing))

    def test_real_engine_predicts_without_constructing_metric_model(self):
        module = load_metric_model_module()
        image = cv2.imread(str(self.frame), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        image = cv2.resize(image, (616, 344), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(image.transpose(2, 0, 1)).float().cuda()

        with mock.patch.object(
            module.Metric,
            "__init__",
            side_effect=AssertionError("heavy Metric3D constructor was called"),
        ):
            model = module.Metric_Model(self.config(self.engine))
            depth = model.predict(rgb)

        self.assertEqual(model.actual_backend, "tensorrt")
        self.assertEqual(tuple(depth.shape), (344, 616))
        self.assertEqual(depth.dtype, torch.float32)
        self.assertEqual(depth.device.type, "cuda")
        self.assertTrue(torch.isfinite(depth).all().item())
        self.assertIsNone(model.backend_status()["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
