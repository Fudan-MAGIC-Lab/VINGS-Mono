import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


FRAME_COUNT = 20
MINIMUM_COSINE = 0.999
MAXIMUM_MEAN_RELATIVE_ERROR = 0.01
MINIMUM_IMPROVEMENT = 0.20
ENCODER_SHAPES = {
    "fnet": ((1, 3, 344, 616), (1, 128, 43, 77)),
    "cnet": ((1, 3, 344, 616), (1, 256, 43, 77)),
}


def _encoder_shapes(encoder):
    try:
        return ENCODER_SHAPES[encoder]
    except KeyError as error:
        raise ValueError(f"unsupported DROID encoder: {encoder}") from error


def latency_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("latency statistics require at least one value")
    return {
        "count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
    }


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def validate_encoder_tensors(
    encoder, image, features, input_dtype="float32", output_dtype="float16"
):
    input_shape, output_shape = _encoder_shapes(encoder)
    for name, tensor, shape, dtype in (
        ("image", image, input_shape, input_dtype),
        ("features", features, output_shape, output_dtype),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if _dtype_name(tensor.dtype) != _dtype_name(dtype):
            raise ValueError(
                f"{name} dtype mismatch: expected={_dtype_name(dtype)} "
                f"actual={_dtype_name(tensor.dtype)}"
            )
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"{name} shape mismatch: expected={tuple(shape)} "
                f"actual={tuple(tensor.shape)}"
            )


def _metrics_pass(metrics, output_shape):
    return (
        list(metrics.get("shape", ())) == list(output_shape)
        and metrics.get("dtype") == "float16"
        and int(metrics.get("value_count", 0)) == int(np.prod(output_shape))
        and bool(metrics.get("finite"))
        and float(metrics.get("cosine", 0.0)) >= MINIMUM_COSINE
        and float(metrics.get("mean_relative_error", float("inf")))
        <= MAXIMUM_MEAN_RELATIVE_ERROR
    )


def build_report(
    encoder, cases, engine, checkpoint, expected_frame_ids, metadata=None
):
    _, output_shape = _encoder_shapes(encoder)
    frame_ids = [int(case["frame_id"]) for case in cases]
    expected_frame_ids = [int(value) for value in expected_frame_ids]
    coverage_ok = (
        len(cases) == FRAME_COUNT
        and len(set(frame_ids)) == FRAME_COUNT
        and len(expected_frame_ids) == FRAME_COUNT
        and frame_ids == expected_frame_ids
        and all(
            set(case)
            >= {
                "frame_id",
                "metrics",
                "pytorch_ms",
                "tensorrt_ms",
                "tensorrt_executed",
            }
            for case in cases
        )
    )
    parity_ok = coverage_ok and all(
        _metrics_pass(case["metrics"], output_shape) for case in cases
    )
    actual_backend_ok = coverage_ok and all(
        case["tensorrt_executed"] is True for case in cases
    )
    pytorch = latency_statistics([case["pytorch_ms"] for case in cases])
    tensorrt = latency_statistics([case["tensorrt_ms"] for case in cases])
    improvement = (pytorch["mean_ms"] - tensorrt["mean_ms"]) / pytorch[
        "mean_ms"
    ]
    speedup = pytorch["mean_ms"] / tensorrt["mean_ms"]

    if not coverage_ok:
        integrate = False
        reason = "do not integrate: benchmark coverage is incomplete"
    elif not actual_backend_ok:
        integrate = False
        reason = "do not integrate: actual TensorRT execution was not proven"
    elif not parity_ok:
        integrate = False
        reason = "do not integrate: module parity thresholds were not met"
    elif improvement < MINIMUM_IMPROVEMENT:
        integrate = False
        reason = "do not integrate: mean latency improvement is below 20%"
    else:
        integrate = True
        reason = (
            "integration candidate: parity passed and mean latency improved by at "
            "least 20%"
        )

    metrics = [case["metrics"] for case in cases if "metrics" in case]
    return {
        "schema_version": 1,
        "encoder": encoder,
        "engine": str(engine),
        "metadata": None if metadata is None else str(metadata),
        "checkpoint": str(checkpoint),
        "frame_ids": frame_ids,
        "thresholds": {
            "minimum_cosine": MINIMUM_COSINE,
            "maximum_mean_relative_error": MAXIMUM_MEAN_RELATIVE_ERROR,
            "minimum_latency_improvement": MINIMUM_IMPROVEMENT,
        },
        "cases": cases,
        "aggregates": {
            "minimum_cosine": min(
                (float(item["cosine"]) for item in metrics), default=None
            ),
            "maximum_mean_relative_error": max(
                (float(item["mean_relative_error"]) for item in metrics),
                default=None,
            ),
        },
        "latency": {
            "pytorch": pytorch,
            "tensorrt": tensorrt,
            "speedup": speedup,
            "improvement": improvement,
        },
        "recommendation": {"integrate": integrate, "reason": reason},
    }


def _time_cuda(torch, function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _load_frame_ids(validation_report):
    report = json.loads(Path(validation_report).read_text(encoding="utf-8"))
    frame_ids = [int(value) for value in report.get("frame_ids", ())]
    if not report.get("accepted") or len(frame_ids) != FRAME_COUNT:
        raise ValueError(
            "benchmark requires an accepted ONNX report with exactly 20 frames"
        )
    return frame_ids


def run_benchmark(
    encoder,
    engine,
    metadata,
    checkpoint,
    validation_report,
    dataset_root,
    warmup,
    iterations,
):
    import torch

    from acceleration.export_droid_encoders_onnx import (
        EncoderCoreExportWrapper,
        load_encoder,
        parity_metrics,
        prepare_encoder_input,
    )
    from acceleration.trt_engine import TensorRTEngine

    frame_ids = _load_frame_ids(validation_report)
    model = load_encoder(encoder, checkpoint=checkpoint, device="cuda")
    wrapper = EncoderCoreExportWrapper(model).cuda().eval()
    references = []
    for frame_id in frame_ids:
        image_path = Path(dataset_root) / "color" / f"{frame_id:05d}.png"
        image = prepare_encoder_input(image_path, device="cuda")
        with torch.inference_mode():
            expected = wrapper(image)
            pytorch_ms = _time_cuda(
                torch, lambda: wrapper(image), warmup=warmup, iterations=iterations
            )
        references.append(
            {
                "frame_id": frame_id,
                "image": image.detach().cpu().numpy().copy(),
                "expected": expected.detach().cpu().numpy().copy(),
                "pytorch_ms": pytorch_ms,
            }
        )
        del image, expected

    del wrapper, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    memory_after_pytorch_release = int(torch.cuda.memory_allocated())

    runner = TensorRTEngine(engine, metadata_path=metadata)
    cases = []
    for reference in references:
        image = (
            torch.from_numpy(reference["image"])
            .to(device="cuda", dtype=torch.float32)
            .contiguous()
        )
        with torch.inference_mode():
            outputs = runner.infer({"image": image})
            features = outputs["features"]
            torch.cuda.synchronize()
            validate_encoder_tensors(
                encoder, image, features, torch.float32, torch.float16
            )
            actual = features.detach().cpu().numpy().copy()
            tensorrt_ms = _time_cuda(
                torch,
                lambda: runner.infer({"image": image}),
                warmup=warmup,
                iterations=iterations,
            )
        cases.append(
            {
                "frame_id": reference["frame_id"],
                "metrics": parity_metrics(reference["expected"], actual),
                "pytorch_ms": reference["pytorch_ms"],
                "tensorrt_ms": tensorrt_ms,
                "tensorrt_executed": True,
            }
        )
        del image, outputs, features

    report = build_report(
        encoder,
        cases,
        engine=engine,
        checkpoint=checkpoint,
        expected_frame_ids=frame_ids,
        metadata=metadata,
    )
    report.update(
        {
            "validation_report": str(validation_report),
            "dataset_root": str(dataset_root),
            "warmup": int(warmup),
            "iterations": int(iterations),
            "memory_architecture": (
                "PyTorch encoder released before TensorRT engine load"
            ),
            "cuda_memory_after_pytorch_release_bytes": (
                memory_after_pytorch_release
            ),
        }
    )
    return report


def write_markdown(path, report):
    latency = report["latency"]
    lines = [
        f"# DROID {report['encoder']} TensorRT FP16 Module Report",
        "",
        f"- Engine: `{report['engine']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Timing: {report['warmup']} warmup, {report['iterations']} measured iterations per frame",
        f"- Backend separation: {report['memory_architecture']}",
        "",
        "| Frame | PyTorch (ms) | TensorRT (ms) | Cosine | MRE | Finite |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for case in report["cases"]:
        metrics = case["metrics"]
        lines.append(
            f"| {case['frame_id']} | {case['pytorch_ms']:.4f} | "
            f"{case['tensorrt_ms']:.4f} | {metrics['cosine']:.9f} | "
            f"{metrics['mean_relative_error']:.6f} | "
            f"{'yes' if metrics['finite'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Aggregates",
            "",
            f"- PyTorch mean/median/p90: {latency['pytorch']['mean_ms']:.4f} / "
            f"{latency['pytorch']['median_ms']:.4f} / {latency['pytorch']['p90_ms']:.4f} ms",
            f"- TensorRT mean/median/p90: {latency['tensorrt']['mean_ms']:.4f} / "
            f"{latency['tensorrt']['median_ms']:.4f} / {latency['tensorrt']['p90_ms']:.4f} ms",
            f"- Speedup: {latency['speedup']:.3f}x",
            f"- Improvement: {latency['improvement'] * 100:.2f}%",
            f"- Minimum cosine: {report['aggregates']['minimum_cosine']:.9f}",
            "- Maximum mean relative error: "
            f"{report['aggregates']['maximum_mean_relative_error']:.6f}",
            "",
            "## Decision",
            "",
            report["recommendation"]["reason"],
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark static DROID fnet/cnet TensorRT FP16 engines."
    )
    parser.add_argument("--encoder", choices=("fnet", "cnet"), required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--checkpoint", default="ckpts/droid.pth")
    parser.add_argument("--validation-report", required=True)
    parser.add_argument(
        "--dataset-root", default="data/smallcity_subset_200/small_city"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()
    report = run_benchmark(
        encoder=args.encoder,
        engine=args.engine,
        metadata=args.metadata,
        checkpoint=args.checkpoint,
        validation_report=args.validation_report,
        dataset_root=args.dataset_root,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(args.markdown_output, report)
    print(json.dumps(report["recommendation"], sort_keys=True))
    if not report["recommendation"]["integrate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
