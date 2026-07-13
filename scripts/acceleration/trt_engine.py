import importlib
import sys
from pathlib import Path

from acceleration.engine_metadata import (
    initialize_tensorrt_plugins,
    load_engine_metadata,
    validate_engine_metadata,
)


SYSTEM_TENSORRT_PATH = "/usr/lib/python3.8/dist-packages"


def import_tensorrt():
    try:
        return importlib.import_module("tensorrt")
    except ModuleNotFoundError:
        if SYSTEM_TENSORRT_PATH not in sys.path:
            sys.path.append(SYSTEM_TENSORRT_PATH)
        return importlib.import_module("tensorrt")


class TensorRTEngine:
    def __init__(self, plan_path, metadata_path=None, trt_module=None, torch_module=None):
        self.plan_path = Path(plan_path)
        if not self.plan_path.is_file():
            raise FileNotFoundError(f"TensorRT plan not found: {self.plan_path}")
        self.metadata_path = (
            Path(metadata_path) if metadata_path is not None else self.plan_path.with_suffix(".json")
        )
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"TensorRT metadata not found: {self.metadata_path}")

        self.trt = trt_module or import_tensorrt()
        self.torch = torch_module or importlib.import_module("torch")
        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        initialize_tensorrt_plugins(self.trt, self.logger)
        self.runtime = self.trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.plan_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT plan: {self.plan_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"failed to create TensorRT execution context: {self.plan_path}")

        self.bindings = [self._binding_info(index) for index in range(self.engine.num_bindings)]
        self.input_bindings = [binding for binding in self.bindings if binding["is_input"]]
        self.output_bindings = [binding for binding in self.bindings if not binding["is_input"]]
        metadata = load_engine_metadata(self.metadata_path)
        validate_engine_metadata(
            metadata,
            engine_path=self.plan_path,
            actual_bindings=[self._metadata_binding(binding) for binding in self.bindings],
            runtime_info={
                "tensorrt_version": self.trt.__version__,
                "cuda_version": self.torch.version.cuda,
                "compute_capability": ".".join(
                    str(value) for value in self.torch.cuda.get_device_capability()
                ),
            },
        )
        self.metadata = metadata
        self.output_cache = {}

    def _dtype_info(self, trt_dtype):
        mappings = (
            (self.trt.DataType.FLOAT, "float32", self.torch.float32),
            (self.trt.DataType.HALF, "float16", self.torch.float16),
            (self.trt.DataType.INT32, "int32", self.torch.int32),
            (self.trt.DataType.INT8, "int8", self.torch.int8),
            (self.trt.DataType.BOOL, "bool", self.torch.bool),
        )
        for candidate, name, torch_dtype in mappings:
            if trt_dtype == candidate:
                return name, torch_dtype
        raise TypeError(f"unsupported TensorRT binding dtype: {trt_dtype}")

    def _binding_info(self, index):
        trt_dtype = self.engine.get_binding_dtype(index)
        dtype_name, torch_dtype = self._dtype_info(trt_dtype)
        return {
            "index": index,
            "name": self.engine.get_binding_name(index),
            "is_input": bool(self.engine.binding_is_input(index)),
            "dtype_name": dtype_name,
            "torch_dtype": torch_dtype,
            "shape": tuple(int(value) for value in self.engine.get_binding_shape(index)),
        }

    @staticmethod
    def _metadata_binding(binding):
        return {
            "name": binding["name"],
            "is_input": binding["is_input"],
            "dtype": binding["dtype_name"],
            "shape": list(binding["shape"]),
        }

    def _validate_inputs(self, inputs):
        expected_names = {binding["name"] for binding in self.input_bindings}
        actual_names = set(inputs)
        if actual_names != expected_names:
            raise ValueError(
                f"TensorRT input names mismatch: expected={sorted(expected_names)} "
                f"actual={sorted(actual_names)}"
            )

        device = None
        for binding in self.input_bindings:
            tensor = inputs[binding["name"]]
            if not tensor.is_cuda:
                raise ValueError(f"TensorRT input {binding['name']} must be a CUDA tensor")
            if not tensor.is_contiguous():
                raise ValueError(f"TensorRT input {binding['name']} must be contiguous")
            if tensor.dtype != binding["torch_dtype"]:
                raise ValueError(
                    f"TensorRT input {binding['name']} dtype mismatch: "
                    f"expected={binding['torch_dtype']} actual={tensor.dtype}"
                )
            if device is None:
                device = tensor.device
            elif tensor.device != device:
                raise ValueError("all TensorRT inputs must be on the same CUDA device")
        return device

    def infer(self, inputs):
        device = self._validate_inputs(inputs)
        addresses = [0] * len(self.bindings)

        for binding in self.input_bindings:
            tensor = inputs[binding["name"]]
            if -1 in binding["shape"]:
                if not self.context.set_binding_shape(binding["index"], tuple(tensor.shape)):
                    raise ValueError(
                        f"TensorRT rejected shape for {binding['name']}: {tuple(tensor.shape)}"
                    )
            elif tuple(tensor.shape) != binding["shape"]:
                raise ValueError(
                    f"TensorRT input {binding['name']} shape mismatch: "
                    f"expected={binding['shape']} actual={tuple(tensor.shape)}"
                )
            addresses[binding["index"]] = tensor.data_ptr()

        if not self.context.all_binding_shapes_specified:
            raise ValueError("not all TensorRT dynamic binding shapes were specified")

        outputs = {}
        for binding in self.output_bindings:
            shape = tuple(int(value) for value in self.context.get_binding_shape(binding["index"]))
            if any(value < 0 for value in shape):
                raise ValueError(f"TensorRT output {binding['name']} has unresolved shape {shape}")
            cached = self.output_cache.get(binding["name"])
            if (
                cached is None
                or tuple(cached.shape) != shape
                or cached.dtype != binding["torch_dtype"]
                or cached.device != device
            ):
                cached = self.torch.empty(shape, dtype=binding["torch_dtype"], device=device)
                self.output_cache[binding["name"]] = cached
            outputs[binding["name"]] = cached
            addresses[binding["index"]] = cached.data_ptr()

        stream = self.torch.cuda.current_stream(device=device)
        if not self.context.execute_async_v2(addresses, stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        return outputs
