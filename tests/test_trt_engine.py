import json
import pathlib
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from acceleration.engine_metadata import (
    build_engine_metadata,
    load_engine_metadata,
    sha256_file,
    validate_engine_metadata,
    write_engine_metadata,
)
from acceleration.trt_engine import TensorRTEngine


def valid_metadata(engine_sha256):
    return {
        "engine_sha256": engine_sha256,
        "source_onnx_sha256": "onnx-sha",
        "checkpoint_sha256": "checkpoint-sha",
        "tensorrt_version": "8.5.2.2",
        "cuda_version": "11.4",
        "compute_capability": "8.7",
        "precision": "fp16",
        "build_command": "trtexec --fp16",
        "bindings": [
            {"name": "net", "is_input": True, "dtype": "float32", "shape": [-1, 128, 43, 77]},
            {"name": "updated_net", "is_input": False, "dtype": "float32", "shape": [-1, 128, 43, 77]},
        ],
        "profiles": {"net": {"min": [1, 128, 43, 77], "opt": [16, 128, 43, 77], "max": [48, 128, 43, 77]}},
    }


class EngineMetadataTests(unittest.TestCase):
    def test_round_trip_and_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            engine = root / "engine.plan"
            engine.write_bytes(b"tensor-rt-engine")
            metadata_path = root / "engine.json"
            metadata = valid_metadata(sha256_file(engine))

            write_engine_metadata(metadata_path, metadata)

            self.assertEqual(load_engine_metadata(metadata_path), metadata)
            self.assertEqual(
                sha256_file(engine),
                "3e900e5fc0873b645661b41aa87cc6221dd2f6dbf8bbb44d30f01f6e5d1b9d03",
            )

    def test_missing_required_key_is_rejected(self):
        metadata = valid_metadata("sha")
        del metadata["checkpoint_sha256"]

        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            validate_engine_metadata(metadata, engine_path=None, actual_bindings=None, runtime_info=None)

    def test_engine_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = pathlib.Path(tmp) / "engine.plan"
            engine.write_bytes(b"actual")

            with self.assertRaisesRegex(ValueError, "engine SHA256"):
                validate_engine_metadata(
                    valid_metadata("wrong"), engine_path=engine, actual_bindings=None, runtime_info=None
                )

    def test_binding_mismatch_is_rejected(self):
        metadata = valid_metadata("sha")
        actual = [
            {"name": "net", "is_input": True, "dtype": "float16", "shape": [-1, 128, 43, 77]},
            {"name": "updated_net", "is_input": False, "dtype": "float32", "shape": [-1, 128, 43, 77]},
        ]

        with self.assertRaisesRegex(ValueError, "binding metadata"):
            validate_engine_metadata(
                metadata, engine_path=None, actual_bindings=actual, runtime_info=None
            )


class FakeTensor:
    _next_pointer = 1000

    def __init__(self, shape, dtype="float32", device="cuda:0", contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = str(device).startswith("cuda")
        self._contiguous = contiguous
        self._pointer = FakeTensor._next_pointer
        FakeTensor._next_pointer += 1

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._pointer


class FakeStream:
    cuda_stream = 4321


class FakeCuda:
    @staticmethod
    def current_stream(device=None):
        return FakeStream()

    @staticmethod
    def get_device_capability(device=None):
        return (8, 7)


class FakeTorch:
    float32 = "float32"
    float16 = "float16"
    int32 = "int32"
    int8 = "int8"
    bool = "bool"
    cuda = FakeCuda()
    version = type("Version", (), {"cuda": "11.4"})()
    allocations = []

    @classmethod
    def empty(cls, shape, dtype, device):
        tensor = FakeTensor(shape, dtype=dtype, device=device)
        cls.allocations.append(tensor)
        return tensor


class FakeContext:
    def __init__(self, enqueue_success=True):
        self.enqueue_success = enqueue_success
        self.input_shape = None
        self.set_calls = []
        self.execute_calls = []

    def set_binding_shape(self, index, shape):
        self.input_shape = tuple(shape)
        self.set_calls.append((index, tuple(shape)))
        return True

    @property
    def all_binding_shapes_specified(self):
        return self.input_shape is not None

    def get_binding_shape(self, index):
        if index == 0:
            return self.input_shape or (-1, 128, 43, 77)
        edge_count = self.input_shape[0]
        return (edge_count, 128, 43, 77)

    def execute_async_v2(self, bindings, stream_handle):
        self.execute_calls.append((list(bindings), stream_handle))
        return self.enqueue_success


class FakeEngine:
    def __init__(self, context):
        self.context = context
        self.num_bindings = 2

    def get_binding_name(self, index):
        return ("net", "updated_net")[index]

    def binding_is_input(self, index):
        return index == 0

    def get_binding_dtype(self, index):
        return "trt_float32"

    def get_binding_shape(self, index):
        return (-1, 128, 43, 77)

    def create_execution_context(self):
        return self.context


class FakeRuntime:
    def __init__(self, logger, engine):
        self.engine = engine

    def deserialize_cuda_engine(self, serialized):
        return self.engine


class FakeTRT:
    __version__ = "8.5.2.2"

    class Logger:
        WARNING = 1

        def __init__(self, severity):
            self.severity = severity

    class DataType:
        FLOAT = "trt_float32"
        HALF = "trt_float16"
        INT32 = "trt_int32"
        INT8 = "trt_int8"
        BOOL = "trt_bool"

    def __init__(self, engine):
        self.engine = engine
        self.plugin_init_calls = []

    def init_libnvinfer_plugins(self, logger, namespace):
        self.plugin_init_calls.append((logger, namespace))
        return True

    def Runtime(self, logger):
        return FakeRuntime(logger, self.engine)


def make_runner(root, enqueue_success=True):
    FakeTorch.allocations = []
    plan = root / "engine.plan"
    plan.write_bytes(b"fake-engine")
    context = FakeContext(enqueue_success=enqueue_success)
    engine = FakeEngine(context)
    metadata = valid_metadata(sha256_file(plan))
    metadata["bindings"] = [
        {"name": "net", "is_input": True, "dtype": "float32", "shape": [-1, 128, 43, 77]},
        {"name": "updated_net", "is_input": False, "dtype": "float32", "shape": [-1, 128, 43, 77]},
    ]
    metadata_path = root / "engine.json"
    write_engine_metadata(metadata_path, metadata)
    trt = FakeTRT(engine)
    runner = TensorRTEngine(
        plan,
        metadata_path=metadata_path,
        trt_module=trt,
        torch_module=FakeTorch,
    )
    return runner, context, trt


class TensorRTEngineTests(unittest.TestCase):
    def test_build_metadata_introspects_plan_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            plan = root / "engine.plan"
            onnx = root / "model.onnx"
            checkpoint = root / "model.pth"
            plan.write_bytes(b"fake-engine")
            onnx.write_bytes(b"onnx")
            checkpoint.write_bytes(b"weights")
            engine = FakeEngine(FakeContext())

            trt = FakeTRT(engine)
            metadata = build_engine_metadata(
                engine_path=plan,
                onnx_path=onnx,
                checkpoint_path=checkpoint,
                build_command="trtexec --fp16",
                profiles={"net": {"min": [1, 128, 43, 77], "opt": [16, 128, 43, 77], "max": [48, 128, 43, 77]}},
                precision="fp16",
                trt_module=trt,
                torch_module=FakeTorch,
            )

            self.assertEqual(len(trt.plugin_init_calls), 1)
            self.assertEqual(trt.plugin_init_calls[0][1], "")
            self.assertEqual(metadata["engine_sha256"], sha256_file(plan))
            self.assertEqual(metadata["source_onnx_sha256"], sha256_file(onnx))
            self.assertEqual(metadata["checkpoint_sha256"], sha256_file(checkpoint))
            self.assertEqual(metadata["tensorrt_version"], "8.5.2.2")
            self.assertEqual(metadata["cuda_version"], "11.4")
            self.assertEqual(metadata["compute_capability"], "8.7")
            self.assertEqual(metadata["bindings"], [
                {"name": "net", "is_input": True, "dtype": "float32", "shape": [-1, 128, 43, 77]},
                {"name": "updated_net", "is_input": False, "dtype": "float32", "shape": [-1, 128, 43, 77]},
            ])

    def test_inputs_are_validated_before_enqueue(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, context, trt = make_runner(pathlib.Path(tmp))

            self.assertEqual(len(trt.plugin_init_calls), 1)
            self.assertEqual(trt.plugin_init_calls[0][1], "")

            cases = (
                ({}, "input names"),
                ({"net": FakeTensor((4, 128, 43, 77), device="cpu")}, "CUDA"),
                ({"net": FakeTensor((4, 128, 43, 77), contiguous=False)}, "contiguous"),
                ({"net": FakeTensor((4, 128, 43, 77), dtype="float16")}, "dtype"),
            )
            for inputs, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    runner.infer(inputs)
            self.assertEqual(context.execute_calls, [])

    def test_dynamic_shape_enqueue_uses_current_stream_and_reuses_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, context, _ = make_runner(pathlib.Path(tmp))
            input_tensor = FakeTensor((4, 128, 43, 77))

            first = runner.infer({"net": input_tensor})
            second = runner.infer({"net": input_tensor})

            self.assertEqual(context.set_calls, [(0, input_tensor.shape), (0, input_tensor.shape)])
            self.assertIs(first["updated_net"], second["updated_net"])
            self.assertEqual(len(FakeTorch.allocations), 1)
            pointers, stream = context.execute_calls[-1]
            self.assertEqual(pointers, [input_tensor.data_ptr(), first["updated_net"].data_ptr()])
            self.assertEqual(stream, 4321)

    def test_enqueue_failure_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, _, _ = make_runner(pathlib.Path(tmp), enqueue_success=False)

            with self.assertRaisesRegex(RuntimeError, "enqueue"):
                runner.infer({"net": FakeTensor((4, 128, 43, 77))})


if __name__ == "__main__":
    unittest.main()
