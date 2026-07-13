import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from acceleration.engine_metadata import sha256_file


FRAME_IDS = tuple(range(0, 50, 5))
MINIMUM_COSINE = 0.999
MAXIMUM_MEAN_RELATIVE_ERROR = 0.02


class Metric3DDepthExportWrapper(torch.nn.Module):
    def __init__(self, depth_model):
        super().__init__()
        self.depth_model = depth_model

    def forward(self, rgb):
        return self.depth_model(input=rgb)["prediction"]


def sample_frame_ids():
    return FRAME_IDS


def onnx_export_options(opset_version=17):
    return {
        "opset_version": int(opset_version),
        "input_names": ["rgb"],
        "output_names": ["depth"],
        "do_constant_folding": False,
    }


def scaled_forward_size(size, scale, multiple=28):
    h, w = map(int, size)
    return (
        max(multiple, int(round(h * scale / multiple)) * multiple),
        max(multiple, int(round(w * scale / multiple)) * multiple),
    )


def _config_get(config, key):
    return config.get(key) if isinstance(config, dict) else getattr(config, key)


def _config_set(config, key, value):
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


def apply_depth_scale(predictor, scale):
    data_basic = predictor.cfg_.data_basic
    shape = scaled_forward_size(_config_get(data_basic, "crop_size"), scale)
    _config_set(data_basic, "crop_size", shape)
    if (isinstance(data_basic, dict) and "vit_size" in data_basic) or hasattr(
        data_basic, "vit_size"
    ):
        _config_set(data_basic, "vit_size", shape)
    return shape


def require_provider(providers, required="CUDAExecutionProvider"):
    if required not in providers:
        raise RuntimeError(f"required ONNX Runtime provider is unavailable: {required}")


def load_runtime_rgb(path, image_size=(344, 616)):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"SmallCity color frame not found or unreadable: {path}")
    height, width = map(int, image_size)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


def smallcity_intrinsics(config):
    intrinsic = config["intrinsic"]
    return np.asarray(
        [intrinsic["fv"], intrinsic["fu"], intrinsic["cv"], intrinsic["cu"]],
        dtype=np.float32,
    )


def parity_metrics(expected, actual, relative_floor=1e-3):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape:
        raise ValueError(f"output shape mismatch: {expected.shape} != {actual.shape}")
    mask = np.isfinite(expected) & (expected > 0)
    if not mask.any():
        raise ValueError("expected depth has no valid positive values")
    expected_valid = expected[mask].astype(np.float64, copy=False)
    actual_valid = actual[mask].astype(np.float64, copy=False)
    finite = bool(np.isfinite(actual_valid).all())
    denominator = max(
        float(np.linalg.norm(expected_valid) * np.linalg.norm(actual_valid)),
        1e-12,
    )
    return {
        "shape": list(actual.shape),
        "valid_count": int(mask.sum()),
        "finite": finite,
        "cosine": float(np.dot(expected_valid, actual_valid) / denominator),
        "mean_relative_error": float(
            np.mean(
                np.abs(expected_valid - actual_valid)
                / np.maximum(np.abs(expected_valid), relative_floor)
            )
        ),
    }


def _metrics_pass(metrics):
    return (
        bool(metrics.get("shape"))
        and int(metrics.get("valid_count", 0)) > 0
        and bool(metrics.get("finite"))
        and float(metrics.get("cosine", 0.0)) >= MINIMUM_COSINE
        and float(metrics.get("mean_relative_error", float("inf")))
        <= MAXIMUM_MEAN_RELATIVE_ERROR
    )


def build_report(cases, onnx_path, provider):
    frame_ids = [int(case["frame_id"]) for case in cases]
    coverage_ok = frame_ids == list(FRAME_IDS) and all(
        set(case) >= {"frame_id", "raw", "final"} for case in cases
    )
    metrics = [
        case[boundary]
        for case in cases
        for boundary in ("raw", "final")
        if boundary in case
    ]
    accepted = (
        coverage_ok
        and provider == "CUDAExecutionProvider"
        and len(metrics) == 2 * len(FRAME_IDS)
        and all(_metrics_pass(result) for result in metrics)
    )
    return {
        "schema_version": 1,
        "onnx_path": str(onnx_path),
        "provider": provider,
        "frame_ids": frame_ids,
        "thresholds": {
            "minimum_cosine": MINIMUM_COSINE,
            "maximum_mean_relative_error": MAXIMUM_MEAN_RELATIVE_ERROR,
        },
        "cases": cases,
        "aggregates": {
            "minimum_cosine": min(
                (float(result["cosine"]) for result in metrics), default=None
            ),
            "maximum_mean_relative_error": max(
                (float(result["mean_relative_error"]) for result in metrics),
                default=None,
            ),
        },
        "accepted": accepted,
    }


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_predictor(checkpoint, depth_scale=0.75):
    submodules_root = REPO_ROOT / "submodules"
    if str(submodules_root) not in sys.path:
        sys.path.append(str(submodules_root))
    from metric_modules import Metric

    predictor = Metric(checkpoint=checkpoint, model_name="v2-S")
    shape = apply_depth_scale(predictor, depth_scale)
    if shape != (448, 784):
        raise ValueError(f"unexpected Metric3D static shape: {shape}")
    return predictor


def collect_pytorch_references(predictor, wrapper, dataset_root, config):
    dataset_root = Path(dataset_root)
    image_size = tuple(config["frontend"]["image_size"])
    intrinsic = smallcity_intrinsics(config)
    references = []
    for frame_id in FRAME_IDS:
        path = dataset_root / "color" / f"{frame_id:05d}.png"
        runtime_rgb = load_runtime_rgb(path, image_size=image_size)
        image_numpy = runtime_rgb.permute(1, 2, 0).numpy()
        prepared = predictor.preprocess(image_numpy, intrinsic)
        rgb_input = prepared[0]
        if tuple(rgb_input.shape) != (1, 3, 448, 784):
            raise ValueError(
                f"unexpected preprocessed shape for frame {frame_id}: {tuple(rgb_input.shape)}"
            )
        with torch.no_grad():
            expected = wrapper(rgb_input)
        expected_raw = expected.detach().float().cpu().numpy()
        expected_final = predictor.postprocess(expected.detach().clone(), d_max=300, d_min=0)
        expected_final = cv2.resize(
            expected_final,
            (image_numpy.shape[1], image_numpy.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        references.append(
            {
                "frame_id": frame_id,
                "rgb_input": rgb_input.detach().float().cpu().numpy(),
                "expected_raw": expected_raw,
                "expected_final": expected_final,
                "final_size": (image_numpy.shape[1], image_numpy.shape[0]),
            }
        )
        del prepared, rgb_input, expected
    return references


def export_static_onnx(wrapper, example_input, output, opset_version=17):
    import onnx

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing ONNX artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = next(wrapper.parameters()).device
    example_input = torch.from_numpy(example_input).to(device=device, dtype=torch.float32)
    wrapper.eval()
    with torch.no_grad():
        wrapper(example_input)
        torch.onnx.export(
            wrapper,
            example_input,
            str(output),
            **onnx_export_options(opset_version=opset_version),
        )
    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred)
    onnx.save(inferred, str(output))


def validate_onnx(onnx_path, predictor, references, provider="CUDAExecutionProvider"):
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    require_provider(available_providers, provider)
    session = ort.InferenceSession(str(onnx_path), providers=[provider])
    session_providers = session.get_providers()
    require_provider(session_providers, provider)
    cases = []
    for reference in references:
        actual_raw = session.run(
            ["depth"],
            {"rgb": np.ascontiguousarray(reference["rgb_input"], dtype=np.float32)},
        )[0]
        actual_final = predictor.postprocess(
            torch.from_numpy(actual_raw.copy()), d_max=300, d_min=0
        )
        actual_final = cv2.resize(
            actual_final,
            reference["final_size"],
            interpolation=cv2.INTER_LINEAR,
        )
        cases.append(
            {
                "frame_id": reference["frame_id"],
                "raw": parity_metrics(reference["expected_raw"], actual_raw),
                "final": parity_metrics(reference["expected_final"], actual_final),
            }
        )
    return cases, available_providers, session_providers


def write_markdown(path, report):
    lines = [
        "# Metric3D Static ONNX Real-Input Parity",
        "",
        f"- ONNX: `{report['onnx_path']}`",
        f"- SHA256: `{report['onnx_sha256']}`",
        f"- Provider: `{report['provider']}`",
        f"- Static input: `{report['input']['shape']}`",
        f"- Accepted: `{report['accepted']}`",
        "",
        "| Frame | Boundary | Cosine | Mean relative error | Valid | Finite |",
        "|---:|:---|---:|---:|---:|:---:|",
    ]
    for case in report["cases"]:
        for boundary in ("raw", "final"):
            result = case[boundary]
            lines.append(
                f"| {case['frame_id']} | {boundary} | {result['cosine']:.9f} | "
                f"{result['mean_relative_error']:.6f} | {result['valid_count']} | "
                f"{result['finite']} |"
            )
    lines.extend(
        [
            "",
            "## Aggregates",
            "",
            f"- Minimum cosine: `{report['aggregates']['minimum_cosine']:.9f}`",
            "- Maximum mean relative error: "
            f"`{report['aggregates']['maximum_mean_relative_error']:.6f}`",
            "",
            "## Decision",
            "",
            (
                "Static Metric3D ONNX parity passed; TensorRT build is authorized."
                if report["accepted"]
                else "Static Metric3D ONNX parity failed; do not build TensorRT."
            ),
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Export and validate static Metric3D ONNX.")
    parser.add_argument("--checkpoint", default="ckpts/metric_depth_vit_small_800k.pth")
    parser.add_argument("--config", default="configs/hierarchical/smallcity.yaml")
    parser.add_argument("--dataset-root", default="data/smallcity_subset_50/small_city")
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--opset-version", type=int, default=17)
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    predictor = load_predictor(args.checkpoint, depth_scale=0.75)
    wrapper = Metric3DDepthExportWrapper(predictor.model_.module.depth_model).eval()
    references = collect_pytorch_references(
        predictor, wrapper, args.dataset_root, config
    )
    export_started = time.time()
    export_static_onnx(
        wrapper,
        references[0]["rgb_input"],
        args.output,
        opset_version=args.opset_version,
    )
    export_elapsed = time.time() - export_started

    del wrapper
    del predictor.model_
    torch.cuda.empty_cache()

    validation_started = time.time()
    cases, available_providers, session_providers = validate_onnx(
        args.output, predictor, references, provider=args.provider
    )
    validation_elapsed = time.time() - validation_started
    report = build_report(cases, onnx_path=args.output, provider=args.provider)
    report.update(
        {
            "onnx_sha256": sha256_file(args.output),
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "dataset_root": str(args.dataset_root),
            "available_providers": available_providers,
            "session_providers": session_providers,
            "input": {"name": "rgb", "shape": [1, 3, 448, 784], "dtype": "float32"},
            "output": {"name": "depth", "dtype": "float32"},
            "opset_version": args.opset_version,
            "timing_seconds": {
                "export": export_elapsed,
                "validation": validation_elapsed,
                "total": time.time() - started,
            },
        }
    )
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.markdown_output, report)
    print(json.dumps({"accepted": report["accepted"], "onnx_sha256": report["onnx_sha256"]}))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
