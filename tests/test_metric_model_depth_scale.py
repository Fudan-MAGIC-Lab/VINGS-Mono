import importlib
import sys
import types
import unittest

import numpy as np
import torch


class MetricModelDepthScaleTests(unittest.TestCase):
    def load_metric_model(self):
        previous_metric_modules = sys.modules.get("metric_modules")
        previous_model = sys.modules.pop("scripts.metric.metric_model", None)

        class DummyMetric:
            calls = []
            instances = []

            def __init__(self, *args, **kwargs):
                self.cfg_ = types.SimpleNamespace(
                    data_basic={"crop_size": (616, 1064), "vit_size": (616, 1064)}
                )
                DummyMetric.instances.append(self)

            def __call__(self, rgb_image, intrinsic, d_max):
                DummyMetric.calls.append(
                    {
                        "shape": rgb_image.shape,
                        "intrinsic": np.array(intrinsic, copy=True),
                        "d_max": d_max,
                    }
                )
                return np.ones(rgb_image.shape[:2], dtype=np.float32)

        sys.modules["metric_modules"] = types.SimpleNamespace(Metric=DummyMetric)
        try:
            module = importlib.import_module("scripts.metric.metric_model")
            return module.Metric_Model, DummyMetric
        finally:
            if previous_metric_modules is None:
                sys.modules.pop("metric_modules", None)
            else:
                sys.modules["metric_modules"] = previous_metric_modules
            if previous_model is not None:
                sys.modules["scripts.metric.metric_model"] = previous_model

    def test_metric_depth_scale_shrinks_internal_forward_size_and_restores_depth_size(self):
        MetricModel, DummyMetric = self.load_metric_model()
        cfg = {
            "device": {"tracker": "cpu"},
            "intrinsic": {"fv": 80.0, "fu": 100.0, "cv": 40.0, "cu": 50.0},
            "metric_depth_scale": 0.5,
        }
        model = MetricModel(cfg)

        image = torch.zeros(3, 80, 100)
        depth = model.predict(image)

        self.assertEqual(tuple(depth.shape), (80, 100))
        self.assertEqual(DummyMetric.instances[0].cfg_.data_basic["crop_size"], (308, 532))
        self.assertEqual(DummyMetric.instances[0].cfg_.data_basic["vit_size"], (308, 532))
        self.assertEqual(DummyMetric.calls[0]["shape"], (80, 100, 3))
        np.testing.assert_allclose(DummyMetric.calls[0]["intrinsic"], np.array([80.0, 100.0, 40.0, 50.0]))


if __name__ == "__main__":
    unittest.main()
