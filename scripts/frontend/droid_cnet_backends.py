import gc
import sys
from pathlib import Path

import torch


class DroidCNetTensorRTAdapter(torch.nn.Module):
    input_shape = (1, 1, 3, 344, 616)
    binding_input_shape = (1, 3, 344, 616)
    binding_output_shape = (1, 256, 43, 77)
    output_shape = (1, 1, 256, 43, 77)

    def __init__(
        self, engine_path, metadata_path=None, runner_factory=None
    ):
        super().__init__()
        if runner_factory is None:
            scripts_root = Path(__file__).resolve().parents[1]
            if str(scripts_root) not in sys.path:
                sys.path.insert(0, str(scripts_root))
            from acceleration.trt_engine import TensorRTEngine

            runner_factory = TensorRTEngine
        self.runner = runner_factory(engine_path, metadata_path=metadata_path)

    @staticmethod
    def _validate_tensor(name, tensor, shape, dtype):
        if not tensor.is_cuda:
            raise ValueError(f"DROID cnet {name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"DROID cnet {name} must be contiguous")
        if tensor.dtype != dtype:
            raise ValueError(
                f"DROID cnet {name} dtype mismatch: "
                f"expected={dtype} actual={tensor.dtype}"
            )
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"DROID cnet {name} shape mismatch: "
                f"expected={tuple(shape)} actual={tuple(tensor.shape)}"
            )

    def forward(self, image):
        self._validate_tensor(
            "input", image, self.input_shape, torch.float32
        )
        flat = image.view(self.binding_input_shape)
        features = self.runner.infer({"image": flat})["features"]
        self._validate_tensor(
            "output", features, self.binding_output_shape, torch.float16
        )
        return features.view(self.output_shape)

    def close(self):
        if self.runner is not None:
            self.runner.output_cache.clear()
        self.runner = None


class DroidCNetBackend(torch.nn.Module):
    def __init__(
        self,
        engine_path,
        metadata_path=None,
        strict=False,
        fallback=None,
        adapter_factory=DroidCNetTensorRTAdapter,
    ):
        super().__init__()
        self.requested_backend = "tensorrt"
        self.actual_backend = None
        self.strict = bool(strict)
        self.engine_path = str(engine_path)
        self.metadata_path = (
            None if metadata_path is None else str(metadata_path)
        )
        self.fallback = fallback
        self.fallback_reason = None
        self.engine_sha256 = None
        self.profiler = None
        self.tensorrt = None
        try:
            self.tensorrt = adapter_factory(
                engine_path, metadata_path=metadata_path
            )
            self.engine_sha256 = self.tensorrt.runner.metadata.get(
                "engine_sha256"
            )
            self.actual_backend = "tensorrt"
            self._log_status()
        except Exception as error:
            if self.strict or self.fallback is None:
                raise
            self._activate_fallback(error)

    def _log_status(self):
        print(
            "[droid_cnet] "
            f"requested_backend={self.requested_backend} "
            f"actual_backend={self.actual_backend} "
            f"strict={str(self.strict).lower()}"
        )

    def _record_status(self):
        if self.profiler is not None and hasattr(self.profiler, "set_metadata"):
            self.profiler.set_metadata("droid_cnet", self.backend_status())

    def _activate_fallback(self, error):
        if self.tensorrt is not None:
            self.tensorrt.close()
            self.tensorrt = None
        self.actual_backend = "torch"
        self.fallback_reason = f"{type(error).__name__}: {error}"
        self._log_status()
        self._record_status()

    def set_profiler(self, profiler):
        self.profiler = profiler
        self._record_status()

    def backend_status(self):
        return {
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "strict": self.strict,
            "engine_path": self.engine_path,
            "metadata_path": self.metadata_path,
            "engine_sha256": self.engine_sha256,
            "fallback_reason": self.fallback_reason,
        }

    def forward(self, image):
        if self.actual_backend == "tensorrt":
            try:
                return self.tensorrt(image)
            except Exception as error:
                if self.strict or self.fallback is None:
                    raise
                self._activate_fallback(error)
        return self.fallback(image)

    def close(self):
        if self.tensorrt is not None:
            self.tensorrt.close()
        self.tensorrt = None
        self.fallback = None


def install_droid_cnet_backend(
    net, inference_config, adapter_factory=DroidCNetTensorRTAdapter
):
    requested = inference_config.get("droid_cnet_backend", "torch")
    if requested == "torch":
        return None
    if requested != "tensorrt":
        raise ValueError(f"unsupported DROID cnet backend: {requested}")

    engine_path = inference_config.get("droid_cnet_engine")
    if not engine_path:
        raise ValueError("DROID cnet TensorRT backend requires an engine path")
    metadata_path = inference_config.get("droid_cnet_metadata")
    strict = bool(inference_config.get("tensorrt_strict", False))
    original = net.cnet
    fallback = original
    if strict:
        net.cnet = None
        fallback = None
        del original
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    backend = DroidCNetBackend(
        engine_path,
        metadata_path=metadata_path,
        strict=strict,
        fallback=fallback,
        adapter_factory=adapter_factory,
    )
    net.cnet = backend
    return backend
