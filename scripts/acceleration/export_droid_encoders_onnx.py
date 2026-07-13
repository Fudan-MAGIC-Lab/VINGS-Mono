import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


ENCODER_SPECS = {
    "fnet": {
        "output_channels": 128,
        "norm_fn": "instance",
        "input_shape": (1, 3, 344, 616),
        "output_shape": (1, 128, 43, 77),
    },
    "cnet": {
        "output_channels": 256,
        "norm_fn": "none",
        "input_shape": (1, 3, 344, 616),
        "output_shape": (1, 256, 43, 77),
    },
}

MINIMUM_COSINE = 0.999
MAXIMUM_MEAN_RELATIVE_ERROR = 0.01
VALIDATION_FRAME_COUNT = 20


class EncoderCoreExportWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image):
        with torch.cuda.amp.autocast(enabled=image.is_cuda):
            return self.encoder.forward_core(image)


def onnx_export_options():
    return {
        "opset_version": 17,
        "input_names": ["image"],
        "output_names": ["features"],
        "do_constant_folding": False,
    }


def require_provider(providers, required="CUDAExecutionProvider"):
    if required not in providers:
        raise RuntimeError(f"required ONNX Runtime provider is unavailable: {required}")


def encoder_spec(name):
    if name not in ENCODER_SPECS:
        raise ValueError(f"unsupported DROID encoder: {name}")
    return dict(ENCODER_SPECS[name])


def load_droid_state_dict(checkpoint, torch_module=torch):
    state = OrderedDict(
        (key.replace("module.", ""), value)
        for key, value in torch_module.load(checkpoint, map_location="cpu").items()
    )
    state["update.weight.2.weight"] = state["update.weight.2.weight"][:2]
    state["update.weight.2.bias"] = state["update.weight.2.bias"][:2]
    state["update.delta.2.weight"] = state["update.delta.2.weight"][:2]
    state["update.delta.2.bias"] = state["update.delta.2.bias"][:2]
    return state


def load_encoder(
    name,
    checkpoint,
    device="cuda",
    state_loader=None,
    droid_net_factory=None,
):
    encoder_spec(name)
    if state_loader is None:
        state_loader = load_droid_state_dict
    if droid_net_factory is None:
        from frontend.droid_net import DroidNet

        droid_net_factory = DroidNet
    state = state_loader(checkpoint)
    net = droid_net_factory()
    net.load_state_dict(state, strict=True)
    return getattr(net, name).to(device).eval()


def admitted_frame_ids(runtime_events_csv, minimum=20):
    frame_ids = []
    seen = set()
    with Path(runtime_events_csv).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("stage") != "motion_filter_feature_encoder":
                continue
            frame_id = int(row["frame_idx"])
            if frame_id not in seen:
                seen.add(frame_id)
                frame_ids.append(frame_id)
    if len(frame_ids) < int(minimum):
        raise ValueError(
            f"DROID encoder parity requires at least {minimum} admitted frames; "
            f"found {len(frame_ids)}"
        )
    return frame_ids


def select_validation_frame_ids(frame_ids, count=20):
    frame_ids = list(frame_ids)
    count = int(count)
    if len(frame_ids) < count:
        raise ValueError(
            f"cannot select {count} validation frames from {len(frame_ids)} inputs"
        )
    indices = np.linspace(0, len(frame_ids) - 1, count).round().astype(int)
    selected = [frame_ids[index] for index in indices]
    if len(set(selected)) != count:
        raise ValueError("distributed validation frame selection produced duplicates")
    return selected


def prepare_encoder_input(image_path, image_size=(344, 616), device="cuda"):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(
            f"SmallCity color frame not found or unreadable: {image_path}"
        )
    height, width = map(int, image_size)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32).contiguous().div_(255.0)
    mean = torch.as_tensor(
        [0.485, 0.456, 0.406], device=device, dtype=torch.float32
    )[:, None, None]
    std = torch.as_tensor(
        [0.229, 0.224, 0.225], device=device, dtype=torch.float32
    )[:, None, None]
    return tensor.sub_(mean).div_(std).contiguous()


def parity_metrics(expected, actual, relative_floor=1e-3):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape:
        raise ValueError(f"output shape mismatch: {expected.shape} != {actual.shape}")
    expected_flat = expected.astype(np.float64, copy=False).reshape(-1)
    actual_flat = actual.astype(np.float64, copy=False).reshape(-1)
    finite = bool(np.isfinite(actual_flat).all())
    denominator = max(
        float(np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat)),
        1e-12,
    )
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "value_count": int(actual.size),
        "finite": finite,
        "cosine": float(np.dot(expected_flat, actual_flat) / denominator),
        "mean_relative_error": float(
            np.mean(
                np.abs(expected_flat - actual_flat)
                / np.maximum(np.abs(expected_flat), float(relative_floor))
            )
        ),
    }


def _metrics_pass(metrics, expected_shape):
    return (
        list(metrics.get("shape", [])) == list(expected_shape)
        and metrics.get("dtype") == "float16"
        and int(metrics.get("value_count", 0)) == int(np.prod(expected_shape))
        and bool(metrics.get("finite"))
        and float(metrics.get("cosine", 0.0)) >= MINIMUM_COSINE
        and float(metrics.get("mean_relative_error", float("inf")))
        <= MAXIMUM_MEAN_RELATIVE_ERROR
    )


def build_report(encoder, cases, onnx_path, provider):
    spec = encoder_spec(encoder)
    frame_ids = [int(case["frame_id"]) for case in cases]
    coverage_ok = (
        len(cases) == VALIDATION_FRAME_COUNT
        and len(set(frame_ids)) == VALIDATION_FRAME_COUNT
        and all(set(case) >= {"frame_id", "metrics"} for case in cases)
    )
    metrics = [case["metrics"] for case in cases if "metrics" in case]
    accepted = (
        coverage_ok
        and provider == "CUDAExecutionProvider"
        and len(metrics) == VALIDATION_FRAME_COUNT
        and all(_metrics_pass(result, spec["output_shape"]) for result in metrics)
    )
    return {
        "schema_version": 1,
        "encoder": encoder,
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


def export_and_validate(
    encoder,
    checkpoint,
    runtime_events_csv,
    dataset_root,
    onnx_output,
    frame_count=VALIDATION_FRAME_COUNT,
):
    import onnx
    import onnxruntime as ort

    from acceleration.engine_metadata import sha256_file

    onnx_output = Path(onnx_output)
    if onnx_output.exists():
        raise FileExistsError(f"refusing to overwrite ONNX artifact: {onnx_output}")
    frame_ids = select_validation_frame_ids(
        admitted_frame_ids(runtime_events_csv, minimum=frame_count),
        count=frame_count,
    )
    model = load_encoder(encoder, checkpoint=checkpoint, device="cuda")
    wrapper = EncoderCoreExportWrapper(model).cuda().eval()
    first_path = Path(dataset_root) / "color" / f"{frame_ids[0]:05d}.png"
    example = prepare_encoder_input(first_path, device="cuda")
    onnx_output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, example, str(onnx_output), **onnx_export_options())

    graph = onnx.load(str(onnx_output))
    onnx.checker.check_model(graph)
    inferred = onnx.shape_inference.infer_shapes(graph)
    onnx.checker.check_model(inferred)
    onnx.save(inferred, str(onnx_output))

    session = ort.InferenceSession(
        str(onnx_output), providers=["CUDAExecutionProvider"]
    )
    require_provider(session.get_providers())
    cases = []
    for frame_id in frame_ids:
        image_path = Path(dataset_root) / "color" / f"{frame_id:05d}.png"
        image = prepare_encoder_input(image_path, device="cuda")
        with torch.no_grad():
            expected = wrapper(image).detach().cpu().numpy()
        actual = session.run(
            ["features"], {"image": image.detach().cpu().numpy()}
        )[0]
        cases.append(
            {
                "frame_id": int(frame_id),
                "metrics": parity_metrics(expected, actual),
            }
        )

    report = build_report(
        encoder=encoder,
        cases=cases,
        onnx_path=onnx_output,
        provider="CUDAExecutionProvider",
    )
    report.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "onnx_sha256": sha256_file(onnx_output),
            "opset_version": 17,
            "input_shape": list(encoder_spec(encoder)["input_shape"]),
            "output_shape": list(encoder_spec(encoder)["output_shape"]),
        }
    )
    return report


def write_markdown(path, report):
    lines = [
        f"# DROID {report['encoder']} Static ONNX Parity",
        "",
        f"- ONNX: `{report['onnx_path']}`",
        f"- ONNX SHA-256: `{report['onnx_sha256']}`",
        f"- Checkpoint SHA-256: `{report['checkpoint_sha256']}`",
        f"- Provider: `{report['provider']}`",
        f"- Accepted: `{str(report['accepted']).lower()}`",
        "",
        "| Frame | Shape | Dtype | Cosine | Mean relative error | Finite |",
        "|---:|---|---|---:|---:|:---:|",
    ]
    for case in report["cases"]:
        metrics = case["metrics"]
        lines.append(
            "| {frame} | {shape} | {dtype} | {cosine:.9f} | {mre:.6f} | {finite} |".format(
                frame=case["frame_id"],
                shape="x".join(str(value) for value in metrics["shape"]),
                dtype=metrics["dtype"],
                cosine=metrics["cosine"],
                mre=metrics["mean_relative_error"],
                finite="yes" if metrics["finite"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Aggregates",
            "",
            f"- Minimum cosine: {report['aggregates']['minimum_cosine']:.9f}",
            "- Maximum mean relative error: "
            f"{report['aggregates']['maximum_mean_relative_error']:.6f}",
            "",
            "TensorRT build is authorized."
            if report["accepted"]
            else "TensorRT build is not authorized.",
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Export and validate static DROID fnet/cnet ONNX graphs."
    )
    parser.add_argument("--encoder", choices=sorted(ENCODER_SPECS), required=True)
    parser.add_argument("--checkpoint", default="ckpts/droid.pth")
    parser.add_argument("--runtime-events", required=True)
    parser.add_argument(
        "--dataset-root", default="data/smallcity_subset_200/small_city"
    )
    parser.add_argument("--onnx-output", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--report-markdown", required=True)
    parser.add_argument("--frame-count", type=int, default=VALIDATION_FRAME_COUNT)
    args = parser.parse_args()

    report = export_and_validate(
        encoder=args.encoder,
        checkpoint=args.checkpoint,
        runtime_events_csv=args.runtime_events,
        dataset_root=args.dataset_root,
        onnx_output=args.onnx_output,
        frame_count=args.frame_count,
    )
    json_path = Path(args.report_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(args.report_markdown, report)
    print(
        json.dumps(
            {
                "encoder": report["encoder"],
                "accepted": report["accepted"],
                "minimum_cosine": report["aggregates"]["minimum_cosine"],
                "maximum_mean_relative_error": report["aggregates"][
                    "maximum_mean_relative_error"
                ],
            },
            sort_keys=True,
        )
    )
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
