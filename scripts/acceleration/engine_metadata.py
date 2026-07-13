import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


REQUIRED_KEYS = (
    "engine_sha256",
    "source_onnx_sha256",
    "checkpoint_sha256",
    "tensorrt_version",
    "cuda_version",
    "compute_capability",
    "precision",
    "build_command",
    "bindings",
    "profiles",
)

SYSTEM_TENSORRT_PATH = "/usr/lib/python3.8/dist-packages"


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_engine_metadata(path):
    with Path(path).open() as handle:
        return json.load(handle)


def write_engine_metadata(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _import_tensorrt():
    try:
        return importlib.import_module("tensorrt")
    except ModuleNotFoundError:
        if SYSTEM_TENSORRT_PATH not in sys.path:
            sys.path.append(SYSTEM_TENSORRT_PATH)
        return importlib.import_module("tensorrt")


def initialize_tensorrt_plugins(trt, logger):
    initializer = getattr(trt, "init_libnvinfer_plugins", None)
    if initializer is None:
        raise RuntimeError("TensorRT does not expose init_libnvinfer_plugins")
    if not initializer(logger, ""):
        raise RuntimeError("failed to initialize TensorRT built-in plugins")


def _dtype_name(trt, dtype):
    mappings = (
        (trt.DataType.FLOAT, "float32"),
        (trt.DataType.HALF, "float16"),
        (trt.DataType.INT32, "int32"),
        (trt.DataType.INT8, "int8"),
        (trt.DataType.BOOL, "bool"),
    )
    for candidate, name in mappings:
        if dtype == candidate:
            return name
    raise TypeError(f"unsupported TensorRT binding dtype: {dtype}")


def build_engine_metadata(
    engine_path,
    onnx_path,
    checkpoint_path,
    build_command,
    profiles,
    precision="fp16",
    trt_module=None,
    torch_module=None,
):
    engine_path = Path(engine_path)
    onnx_path = Path(onnx_path)
    checkpoint_path = Path(checkpoint_path)
    trt = trt_module or _import_tensorrt()
    torch = torch_module or importlib.import_module("torch")
    logger = trt.Logger(trt.Logger.WARNING)
    initialize_tensorrt_plugins(trt, logger)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT plan: {engine_path}")

    bindings = []
    for index in range(engine.num_bindings):
        bindings.append(
            {
                "name": engine.get_binding_name(index),
                "is_input": bool(engine.binding_is_input(index)),
                "dtype": _dtype_name(trt, engine.get_binding_dtype(index)),
                "shape": [int(value) for value in engine.get_binding_shape(index)],
            }
        )

    return {
        "engine_sha256": sha256_file(engine_path),
        "source_onnx_sha256": sha256_file(onnx_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "tensorrt_version": trt.__version__,
        "cuda_version": torch.version.cuda,
        "compute_capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability()
        ),
        "precision": precision,
        "build_command": build_command,
        "bindings": bindings,
        "profiles": profiles,
    }


def validate_engine_metadata(metadata, engine_path, actual_bindings, runtime_info):
    missing = [key for key in REQUIRED_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"engine metadata missing required keys: {', '.join(missing)}")

    if engine_path is not None and sha256_file(engine_path) != metadata["engine_sha256"]:
        raise ValueError("engine SHA256 does not match metadata")

    if actual_bindings is not None and actual_bindings != metadata["bindings"]:
        raise ValueError("binding metadata does not match deserialized engine")

    if runtime_info is not None:
        for key in ("tensorrt_version", "cuda_version", "compute_capability"):
            if str(runtime_info[key]) != str(metadata[key]):
                raise ValueError(
                    f"{key} mismatch: runtime={runtime_info[key]} metadata={metadata[key]}"
                )

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate a strict TensorRT engine sidecar.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-command", required=True)
    parser.add_argument("--profiles", required=True, help="JSON object keyed by input name")
    parser.add_argument("--precision", default="fp16")
    args = parser.parse_args()
    metadata = build_engine_metadata(
        engine_path=args.engine,
        onnx_path=args.onnx,
        checkpoint_path=args.checkpoint,
        build_command=args.build_command,
        profiles=json.loads(args.profiles),
        precision=args.precision,
    )
    validate_engine_metadata(
        metadata,
        engine_path=args.engine,
        actual_bindings=metadata["bindings"],
        runtime_info={
            "tensorrt_version": metadata["tensorrt_version"],
            "cuda_version": metadata["cuda_version"],
            "compute_capability": metadata["compute_capability"],
        },
    )
    write_engine_metadata(args.output, metadata)
    print(f"wrote={args.output} engine_sha256={metadata['engine_sha256']}")


if __name__ == "__main__":
    main()
