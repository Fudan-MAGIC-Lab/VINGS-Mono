import sys
from pathlib import Path

import torch


def _config_get(config, key):
    return config.get(key) if isinstance(config, dict) else getattr(config, key, None)


def _config_set(config, key, value):
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


def scaled_forward_size(size, scale, multiple=28):
    height, width = map(int, size)
    return (
        max(multiple, int(round(height * scale / multiple)) * multiple),
        max(multiple, int(round(width * scale / multiple)) * multiple),
    )


def apply_depth_scale(predictor, depth_scale):
    data_basic = predictor.cfg_.data_basic
    crop_size = _config_get(data_basic, "crop_size")
    shape = scaled_forward_size(crop_size, depth_scale)
    _config_set(data_basic, "crop_size", shape)
    if _config_get(data_basic, "vit_size") is not None:
        _config_set(data_basic, "vit_size", shape)
    return shape


def _load_metric_class():
    from metric_modules import Metric

    return Metric


class Metric3DPyTorchBackend:
    name = "torch"

    def __init__(self, checkpoint, depth_scale, metric_class=None):
        metric_class = metric_class or _load_metric_class()
        self.predictor = metric_class(checkpoint=checkpoint, model_name="v2-S")
        apply_depth_scale(self.predictor, depth_scale)

    def preprocess(self, image, intrinsic):
        return self.predictor.preprocess(image, intrinsic)

    def infer(self, prepared):
        return self.predictor.forward_depth(*prepared)

    def postprocess(self, prediction, d_max, d_min):
        return self.predictor.postprocess(prediction, d_max=d_max, d_min=d_min)

    def close(self):
        self.predictor = None


class Metric3DTensorRTBackend:
    name = "tensorrt"
    expected_shape = (1, 3, 448, 784)

    def __init__(
        self,
        engine_path,
        metadata_path,
        checkpoint,
        depth_scale,
        device="cpu",
        metric_class=None,
        runner_factory=None,
    ):
        metric_class = metric_class or _load_metric_class()
        self.device = device
        self.predictor = metric_class.__new__(metric_class)
        self.predictor.cfg_ = self.predictor._load_config_("v2-S", checkpoint)
        shape = apply_depth_scale(self.predictor, depth_scale)
        if (1, 3, *shape) != self.expected_shape:
            raise ValueError(
                f"Metric3D TensorRT requires forward shape {self.expected_shape}, "
                f"got {(1, 3, *shape)}"
            )
        if runner_factory is None:
            scripts_root = Path(__file__).resolve().parents[1]
            if str(scripts_root) not in sys.path:
                sys.path.insert(0, str(scripts_root))
            from acceleration.trt_engine import TensorRTEngine

            runner_factory = TensorRTEngine
        self.runner = runner_factory(engine_path, metadata_path=metadata_path)

    def preprocess(self, image, intrinsic):
        return self.predictor.preprocess(image, intrinsic)

    def infer(self, prepared):
        rgb = prepared[0].to(device=self.device, dtype=torch.float32).contiguous()
        if tuple(rgb.shape) != self.expected_shape:
            raise ValueError(
                f"Metric3D TensorRT input must have shape {self.expected_shape}, "
                f"got {tuple(rgb.shape)}"
            )
        return self.runner.infer({"rgb": rgb})["depth"]

    def postprocess(self, prediction, d_max, d_min):
        return self.predictor.postprocess(prediction, d_max=d_max, d_min=d_min)

    def close(self):
        if self.runner is not None:
            self.runner.output_cache.clear()
        self.runner = None
        self.predictor = None
