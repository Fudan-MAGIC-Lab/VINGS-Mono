import gc
import sys
from pathlib import Path

import torch


class DroidUpdateTensorRTAdapter(torch.nn.Module):
    input_channels = {"net": 128, "inp": 128, "corr": 196, "flow": 4}
    output_channels = {"updated_net": 128, "delta": 2, "weight": 2}
    spatial_shape = (43, 77)
    minimum_edges = 1
    maximum_edges = 48

    def __init__(self, engine_path, metadata_path=None, runner_factory=None):
        super().__init__()
        if runner_factory is None:
            scripts_root = Path(__file__).resolve().parents[1]
            if str(scripts_root) not in sys.path:
                sys.path.insert(0, str(scripts_root))
            from acceleration.trt_engine import TensorRTEngine

            runner_factory = TensorRTEngine
        self.runner = runner_factory(engine_path, metadata_path=metadata_path)

    @staticmethod
    def _validate_basic(name, tensor, allowed_dtypes, require_contiguous):
        if not tensor.is_cuda:
            raise ValueError(f"DROID update {name} must be a CUDA tensor")
        if require_contiguous and not tensor.is_contiguous():
            raise ValueError(f"DROID update {name} must be contiguous")
        if tensor.dtype not in allowed_dtypes:
            raise ValueError(
                f"DROID update {name} dtype mismatch: "
                f"expected={allowed_dtypes} actual={tensor.dtype}"
            )

    @classmethod
    def _validate_shape(cls, name, tensor, edges, channels):
        expected = (edges, channels, *cls.spatial_shape)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"DROID update {name} shape mismatch: "
                f"expected={expected} actual={tuple(tensor.shape)}"
            )

    def forward(self, net, inp, corr, flow):
        inputs = {"net": net, "inp": inp, "corr": corr, "flow": flow}
        for name, tensor in inputs.items():
            self._validate_basic(
                name,
                tensor,
                (torch.float16, torch.float32),
                require_contiguous=False,
            )
        runtime_dtype = net.dtype
        edges = int(net.shape[0])
        if not self.minimum_edges <= edges <= self.maximum_edges:
            raise ValueError(
                f"DROID update edge count must be in "
                f"[{self.minimum_edges}, {self.maximum_edges}], got {edges}"
            )
        if any(int(tensor.shape[0]) != edges for tensor in inputs.values()):
            raise ValueError("DROID update inputs must have the same edge count")
        for name, tensor in inputs.items():
            self._validate_shape(
                name, tensor, edges, self.input_channels[name]
            )

        binding_inputs = {
            name: tensor.to(dtype=torch.float32).contiguous()
            for name, tensor in inputs.items()
        }
        outputs = self.runner.infer(binding_inputs)
        expected_names = set(self.output_channels)
        if set(outputs) != expected_names:
            raise ValueError(
                f"DROID update output names mismatch: "
                f"expected={sorted(expected_names)} actual={sorted(outputs)}"
            )
        for name, tensor in outputs.items():
            self._validate_basic(
                name, tensor, (torch.float32,), require_contiguous=True
            )
            self._validate_shape(
                name, tensor, edges, self.output_channels[name]
            )
        return tuple(
            outputs[name].to(dtype=runtime_dtype).contiguous()
            for name in self.output_channels
        )

    def close(self):
        if self.runner is not None:
            self.runner.output_cache.clear()
        self.runner = None


class DroidUpdateBackend(torch.nn.Module):
    def __init__(
        self,
        engine_path,
        metadata_path=None,
        strict=False,
        fallback=None,
        adapter_factory=DroidUpdateTensorRTAdapter,
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
        self.precision = None
        self.profiler = None
        self.tensorrt = None
        try:
            self.tensorrt = adapter_factory(
                engine_path, metadata_path=metadata_path
            )
            metadata = self.tensorrt.runner.metadata
            self.engine_sha256 = metadata.get("engine_sha256")
            self.precision = metadata.get("precision")
            self.actual_backend = "tensorrt"
            self._log_status()
        except Exception as error:
            if self.strict or self.fallback is None:
                raise
            self._activate_fallback(error)

    def _log_status(self):
        print(
            "[droid_update] "
            f"requested_backend={self.requested_backend} "
            f"actual_backend={self.actual_backend} "
            f"strict={str(self.strict).lower()}"
        )

    def _record_status(self):
        if self.profiler is not None and hasattr(self.profiler, "set_metadata"):
            self.profiler.set_metadata("droid_update", self.backend_status())

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
            "precision": self.precision,
            "fallback_reason": self.fallback_reason,
        }

    def forward(self, net, inp, corr, flow):
        if self.actual_backend == "tensorrt":
            try:
                return self.tensorrt(net, inp, corr, flow)
            except Exception as error:
                if self.strict or self.fallback is None:
                    raise
                self._activate_fallback(error)
        return self.fallback(net, inp, corr, flow)

    def close(self):
        if self.tensorrt is not None:
            self.tensorrt.close()
        self.tensorrt = None
        self.fallback = None


def install_droid_update_backend(
    update, inference_config, adapter_factory=DroidUpdateTensorRTAdapter
):
    requested = inference_config.get("droid_update_backend", "torch")
    if requested == "torch":
        return None
    if requested != "tensorrt":
        raise ValueError(f"unsupported DROID update backend: {requested}")
    engine_path = inference_config.get("droid_update_engine")
    if not engine_path:
        raise ValueError("DROID update TensorRT backend requires an engine path")

    strict = bool(inference_config.get("tensorrt_strict", False))
    metadata_path = inference_config.get("droid_update_metadata")
    fallback = update.forward_core_torch
    if strict:
        fallback = None
        update.corr_encoder = None
        update.flow_encoder = None
        update.gru = None
        update.delta = None
        update.weight = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    backend = DroidUpdateBackend(
        engine_path,
        metadata_path=metadata_path,
        strict=strict,
        fallback=fallback,
        adapter_factory=adapter_factory,
    )
    update.core_backend = backend
    return backend
