import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


FRAME_IDS = tuple(range(0, 50, 5))
MINIMUM_COSINE = 0.999
MAXIMUM_MEAN_RELATIVE_ERROR = 0.02
MINIMUM_IMPROVEMENT = 0.20


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


def prepare_input_tensor(torch_module, array, device="cuda"):
    return (
        torch_module.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        .to(device=device, dtype=torch_module.float32)
        .contiguous()
    )


def _parity_passes(metrics):
    return (
        bool(metrics.get("shape"))
        and int(metrics.get("valid_count", 0)) > 0
        and bool(metrics.get("finite"))
        and float(metrics.get("cosine", 0.0)) >= MINIMUM_COSINE
        and float(metrics.get("mean_relative_error", float("inf")))
        <= MAXIMUM_MEAN_RELATIVE_ERROR
    )


def build_report(cases, engine, checkpoint):
    frame_ids = [int(case["frame_id"]) for case in cases]
    coverage_ok = frame_ids == list(FRAME_IDS) and all(
        set(case) >= {"frame_id", "raw", "final"} for case in cases
    )
    parity_ok = coverage_ok and all(
        _parity_passes(case[boundary])
        for case in cases
        for boundary in ("raw", "final")
    )
    pytorch = latency_statistics([case["pytorch_model_ms"] for case in cases])
    tensorrt = latency_statistics([case["tensorrt_model_ms"] for case in cases])
    preprocess = latency_statistics([case["preprocess_ms"] for case in cases])
    pytorch_postprocess = latency_statistics(
        [case["pytorch_postprocess_ms"] for case in cases]
    )
    tensorrt_postprocess = latency_statistics(
        [case["tensorrt_postprocess_ms"] for case in cases]
    )
    model_speedup = pytorch["mean_ms"] / tensorrt["mean_ms"]
    model_improvement = (
        pytorch["mean_ms"] - tensorrt["mean_ms"]
    ) / pytorch["mean_ms"]

    if not coverage_ok:
        integrate = False
        reason = "do not integrate: benchmark coverage is incomplete"
    elif not parity_ok:
        integrate = False
        reason = "do not integrate: module parity thresholds were not met"
    elif model_improvement < MINIMUM_IMPROVEMENT:
        integrate = False
        reason = "do not integrate: mean model latency improvement is below 20%"
    else:
        integrate = True
        reason = "integration candidate: parity passed and mean model latency improved by at least 20%"

    return {
        "schema_version": 1,
        "engine": str(engine),
        "checkpoint": str(checkpoint),
        "frame_ids": frame_ids,
        "thresholds": {
            "minimum_cosine": MINIMUM_COSINE,
            "maximum_mean_relative_error": MAXIMUM_MEAN_RELATIVE_ERROR,
            "minimum_model_improvement": MINIMUM_IMPROVEMENT,
        },
        "cases": cases,
        "latency": {
            "preprocess": preprocess,
            "pytorch_model": pytorch,
            "tensorrt_model": tensorrt,
            "pytorch_postprocess": pytorch_postprocess,
            "tensorrt_postprocess": tensorrt_postprocess,
            "model_speedup": model_speedup,
            "model_improvement": model_improvement,
            "estimated_pytorch_pipeline_ms": (
                preprocess["mean_ms"]
                + pytorch["mean_ms"]
                + pytorch_postprocess["mean_ms"]
            ),
            "estimated_tensorrt_pipeline_ms": (
                preprocess["mean_ms"]
                + tensorrt["mean_ms"]
                + tensorrt_postprocess["mean_ms"]
            ),
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


def _time_synchronized(torch, function):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def run_benchmark(
    engine,
    metadata,
    checkpoint,
    config_path,
    dataset_root,
    warmup,
    iterations,
):
    import cv2
    import torch

    from acceleration.export_metric3d_onnx import (
        Metric3DDepthExportWrapper,
        load_config,
        load_predictor,
        load_runtime_rgb,
        parity_metrics,
        smallcity_intrinsics,
    )
    from acceleration.trt_engine import TensorRTEngine

    config = load_config(config_path)
    predictor = load_predictor(checkpoint, depth_scale=0.75)
    wrapper = Metric3DDepthExportWrapper(predictor.model_.module.depth_model).eval()
    image_size = tuple(config["frontend"]["image_size"])
    intrinsic = smallcity_intrinsics(config)
    references = []

    for frame_id in FRAME_IDS:
        path = Path(dataset_root) / "color" / f"{frame_id:05d}.png"
        runtime_rgb = load_runtime_rgb(path, image_size=image_size)
        image_numpy = runtime_rgb.permute(1, 2, 0).numpy()
        prepared, preprocess_ms = _time_synchronized(
            torch, lambda: predictor.preprocess(image_numpy, intrinsic)
        )
        rgb_input = prepared[0]
        with torch.no_grad():
            expected = wrapper(rgb_input)
            pytorch_model_ms = _time_cuda(
                torch,
                lambda: wrapper(rgb_input),
                warmup=warmup,
                iterations=iterations,
            )
        expected_raw = expected.detach().float().cpu().numpy()

        def pytorch_postprocess():
            depth = predictor.postprocess(expected.detach().clone(), d_max=300, d_min=0)
            return cv2.resize(
                depth,
                (image_numpy.shape[1], image_numpy.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        expected_final, pytorch_postprocess_ms = _time_synchronized(
            torch, pytorch_postprocess
        )
        references.append(
            {
                "frame_id": frame_id,
                "rgb_input": rgb_input.detach().float().cpu().numpy(),
                "expected_raw": expected_raw,
                "expected_final": expected_final,
                "final_size": (image_numpy.shape[1], image_numpy.shape[0]),
                "preprocess_ms": preprocess_ms,
                "pytorch_model_ms": pytorch_model_ms,
                "pytorch_postprocess_ms": pytorch_postprocess_ms,
            }
        )
        del prepared, rgb_input, expected

    del wrapper
    del predictor.model_
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    memory_after_pytorch_release = int(torch.cuda.memory_allocated())

    runner = TensorRTEngine(engine, metadata_path=metadata)
    cases = []
    for reference in references:
        rgb_input = prepare_input_tensor(torch, reference["rgb_input"], device="cuda")
        with torch.no_grad():
            outputs = runner.infer({"rgb": rgb_input})
            torch.cuda.synchronize()
            actual_raw = outputs["depth"].detach().float().cpu().numpy().copy()
            tensorrt_model_ms = _time_cuda(
                torch,
                lambda: runner.infer({"rgb": rgb_input}),
                warmup=warmup,
                iterations=iterations,
            )

        def tensorrt_postprocess():
            depth = predictor.postprocess(
                torch.from_numpy(actual_raw.copy()), d_max=300, d_min=0
            )
            return cv2.resize(
                depth,
                reference["final_size"],
                interpolation=cv2.INTER_LINEAR,
            )

        actual_final, tensorrt_postprocess_ms = _time_synchronized(
            torch, tensorrt_postprocess
        )
        cases.append(
            {
                "frame_id": reference["frame_id"],
                "raw": parity_metrics(reference["expected_raw"], actual_raw),
                "final": parity_metrics(reference["expected_final"], actual_final),
                "preprocess_ms": reference["preprocess_ms"],
                "pytorch_model_ms": reference["pytorch_model_ms"],
                "tensorrt_model_ms": tensorrt_model_ms,
                "pytorch_postprocess_ms": reference["pytorch_postprocess_ms"],
                "tensorrt_postprocess_ms": tensorrt_postprocess_ms,
            }
        )
        del rgb_input, outputs

    report = build_report(cases, engine=engine, checkpoint=checkpoint)
    report.update(
        {
            "metadata": str(metadata),
            "config": str(config_path),
            "dataset_root": str(dataset_root),
            "warmup": int(warmup),
            "iterations": int(iterations),
            "memory_architecture": "PyTorch model released before TensorRT engine load",
            "cuda_memory_after_pytorch_release_bytes": memory_after_pytorch_release,
        }
    )
    return report


def write_markdown(path, report):
    latency = report["latency"]
    lines = [
        "# Metric3D TensorRT FP16 Module Report",
        "",
        f"- Engine: `{report['engine']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Timing: {report['warmup']} warmup, {report['iterations']} measured iterations per frame",
        f"- Backend separation: {report['memory_architecture']}",
        "",
        "| Frame | PyTorch model (ms) | TensorRT model (ms) | Raw cosine | Raw MRE | Final cosine | Final MRE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['frame_id']} | {case['pytorch_model_ms']:.4f} | "
            f"{case['tensorrt_model_ms']:.4f} | {case['raw']['cosine']:.9f} | "
            f"{case['raw']['mean_relative_error']:.6f} | "
            f"{case['final']['cosine']:.9f} | "
            f"{case['final']['mean_relative_error']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Latency aggregates",
            "",
            f"- PyTorch model mean/median/p90: {latency['pytorch_model']['mean_ms']:.4f} / "
            f"{latency['pytorch_model']['median_ms']:.4f} / {latency['pytorch_model']['p90_ms']:.4f} ms",
            f"- TensorRT model mean/median/p90: {latency['tensorrt_model']['mean_ms']:.4f} / "
            f"{latency['tensorrt_model']['median_ms']:.4f} / {latency['tensorrt_model']['p90_ms']:.4f} ms",
            f"- Model speedup: {latency['model_speedup']:.3f}x",
            f"- Model improvement: {latency['model_improvement'] * 100:.2f}%",
            f"- Estimated PyTorch/TensorRT pipeline: "
            f"{latency['estimated_pytorch_pipeline_ms']:.4f} / "
            f"{latency['estimated_tensorrt_pipeline_ms']:.4f} ms",
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
    parser = argparse.ArgumentParser(description="Benchmark static Metric3D TensorRT FP16.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--checkpoint", default="ckpts/metric_depth_vit_small_800k.pth")
    parser.add_argument("--config", default="configs/hierarchical/smallcity.yaml")
    parser.add_argument("--dataset-root", default="data/smallcity_subset_50/small_city")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()
    report = run_benchmark(
        engine=args.engine,
        metadata=args.metadata,
        checkpoint=args.checkpoint,
        config_path=args.config,
        dataset_root=args.dataset_root,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.markdown_output, report)
    print(json.dumps(report["recommendation"], sort_keys=True))


if __name__ == "__main__":
    main()
