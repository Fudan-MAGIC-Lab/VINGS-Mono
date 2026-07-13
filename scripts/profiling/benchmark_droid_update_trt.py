import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


OUTPUT_NAMES = ("updated_net", "delta", "weight")
REQUIRED_EDGE_COUNTS = (1, 4, 16, 32, 48)
COSINE_THRESHOLD = 0.999
RELATIVE_ERROR_THRESHOLD = 0.01
MINIMUM_IMPROVEMENT = 0.20


def parity_metrics(expected, actual, relative_floor=1e-3):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape:
        raise ValueError(f"output shape mismatch: expected={expected.shape} actual={actual.shape}")
    expected_flat = expected.astype(np.float64, copy=False).reshape(-1)
    actual_flat = actual.astype(np.float64, copy=False).reshape(-1)
    denominator = max(
        float(np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat)),
        1e-12,
    )
    return {
        "shape": list(actual.shape),
        "finite": bool(np.isfinite(expected_flat).all() and np.isfinite(actual_flat).all()),
        "cosine": float(np.dot(expected_flat, actual_flat) / denominator),
        "mean_relative_error": float(
            np.mean(
                np.abs(expected_flat - actual_flat)
                / np.maximum(np.abs(expected_flat), relative_floor)
            )
        ),
    }


def _parity_passes(outputs):
    return set(outputs) == set(OUTPUT_NAMES) and all(
        metrics["finite"]
        and metrics["cosine"] >= COSINE_THRESHOLD
        and metrics["mean_relative_error"] <= RELATIVE_ERROR_THRESHOLD
        for metrics in outputs.values()
    )


def build_report(cases, engine, checkpoint):
    normalized_cases = []
    parity_ok = True
    for case in cases:
        pytorch_ms = float(case["pytorch_ms"])
        tensorrt_ms = float(case["tensorrt_ms"])
        improvement = (pytorch_ms - tensorrt_ms) / pytorch_ms
        outputs = case["outputs"]
        parity_ok = parity_ok and _parity_passes(outputs)
        normalized_cases.append(
            {
                "edge_count": int(case["edge_count"]),
                "outputs": outputs,
                "latency": {
                    "pytorch_ms": pytorch_ms,
                    "tensorrt_ms": tensorrt_ms,
                    "speedup": pytorch_ms / tensorrt_ms,
                    "improvement": improvement,
                },
            }
        )

    representative = next(
        (case for case in normalized_cases if case["edge_count"] == 16), None
    )
    coverage_ok = (
        {case["edge_count"] for case in normalized_cases} == set(REQUIRED_EDGE_COUNTS)
        and all(set(case["outputs"]) == set(OUTPUT_NAMES) for case in normalized_cases)
    )
    if not coverage_ok:
        integrate = False
        reason = "do not integrate: benchmark coverage is incomplete"
    elif not parity_ok:
        integrate = False
        reason = "do not integrate: module parity thresholds were not met"
    elif representative is None:
        integrate = False
        reason = "do not integrate: representative E=16 result is missing"
    elif representative["latency"]["improvement"] < MINIMUM_IMPROVEMENT:
        integrate = False
        reason = "do not integrate: E=16 latency improvement is below 20%"
    else:
        integrate = True
        reason = "integration candidate: parity passed and E=16 latency improved by at least 20%"

    return {
        "schema_version": 1,
        "engine": str(engine),
        "checkpoint": str(checkpoint),
        "edge_counts": [case["edge_count"] for case in normalized_cases],
        "thresholds": {
            "minimum_cosine": COSINE_THRESHOLD,
            "maximum_mean_relative_error": RELATIVE_ERROR_THRESHOLD,
            "minimum_improvement": MINIMUM_IMPROVEMENT,
            "representative_edge_count": 16,
        },
        "cases": normalized_cases,
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


def run_benchmark(engine, metadata, checkpoint, edge_counts, warmup, iterations):
    import torch

    from acceleration.export_droid_onnx import load_update_module
    from acceleration.trt_engine import TensorRTEngine

    update = load_update_module(checkpoint, device="cuda")
    runner = TensorRTEngine(engine, metadata_path=metadata)
    cases = []
    for edge_count in edge_counts:
        torch.manual_seed(2000 + edge_count)
        inputs = {
            "net": torch.randn(edge_count, 128, 43, 77, device="cuda"),
            "inp": torch.randn(edge_count, 128, 43, 77, device="cuda"),
            "corr": torch.randn(edge_count, 196, 43, 77, device="cuda"),
            "flow": torch.randn(edge_count, 4, 43, 77, device="cuda").clamp(-64.0, 64.0),
        }
        ordered_inputs = tuple(inputs[name] for name in ("net", "inp", "corr", "flow"))
        with torch.inference_mode():
            expected = update.forward_core(*ordered_inputs)
            actual = runner.infer(inputs)
            torch.cuda.synchronize()
            output_metrics = {
                name: parity_metrics(
                    expected_tensor.detach().float().cpu().numpy(),
                    actual[name].detach().float().cpu().numpy(),
                )
                for name, expected_tensor in zip(OUTPUT_NAMES, expected)
            }
            pytorch_ms = _time_cuda(
                torch,
                lambda: update.forward_core(*ordered_inputs),
                warmup,
                iterations,
            )
            tensorrt_ms = _time_cuda(
                torch,
                lambda: runner.infer(inputs),
                warmup,
                iterations,
            )
        cases.append(
            {
                "edge_count": edge_count,
                "outputs": output_metrics,
                "pytorch_ms": pytorch_ms,
                "tensorrt_ms": tensorrt_ms,
            }
        )
        del inputs, ordered_inputs, expected, actual
    return build_report(cases, engine=engine, checkpoint=checkpoint)


def write_markdown(path, report, warmup, iterations):
    lines = [
        "# DROID Update-Core TensorRT FP16 Module Report",
        "",
        f"- Engine: `{report['engine']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Timing: {warmup} warmup, {iterations} measured iterations",
        "",
        "| E | PyTorch (ms) | TensorRT (ms) | Speedup | Improvement | Parity |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for case in report["cases"]:
        latency = case["latency"]
        parity = "PASS" if _parity_passes(case["outputs"]) else "FAIL"
        lines.append(
            f"| {case['edge_count']} | {latency['pytorch_ms']:.4f} | "
            f"{latency['tensorrt_ms']:.4f} | {latency['speedup']:.3f}x | "
            f"{latency['improvement'] * 100:.2f}% | {parity} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["recommendation"]["reason"],
            "",
            "## Output parity",
            "",
        ]
    )
    for case in report["cases"]:
        lines.append(f"### E={case['edge_count']}")
        lines.append("")
        for name, metrics in case["outputs"].items():
            lines.append(
                f"- `{name}`: shape={metrics['shape']}, finite={metrics['finite']}, "
                f"cosine={metrics['cosine']:.9f}, "
                f"mean_relative_error={metrics['mean_relative_error']:.6f}"
            )
        lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark DROID update-core TensorRT FP16.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--checkpoint", default="ckpts/droid.pth")
    parser.add_argument("--edge-counts", default="1,4,16,32,48")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args()
    edge_counts = tuple(int(value) for value in args.edge_counts.split(","))
    report = run_benchmark(
        engine=args.engine,
        metadata=args.metadata,
        checkpoint=args.checkpoint,
        edge_counts=edge_counts,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.markdown_output, report, args.warmup, args.iterations)
    print(json.dumps(report["recommendation"], sort_keys=True))


if __name__ == "__main__":
    main()
