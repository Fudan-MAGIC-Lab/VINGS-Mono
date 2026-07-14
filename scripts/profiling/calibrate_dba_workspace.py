import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from profiling.dba_memory_calibration import (
    CAPTURE_BOUNDARIES,
    calibrate,
    load_memory_samples,
)


def _source_hash(path):
    path = Path(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _environment_metadata():
    result = {
        "host": platform.platform(),
        "python": platform.python_version(),
        "torch_version": None,
        "cuda_version": None,
        "cuda_device": None,
    }
    try:
        import torch
    except ImportError:
        return result
    result["torch_version"] = str(torch.__version__)
    result["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        result["cuda_device"] = torch.cuda.get_device_name(0)
    return result


def render_report(payload):
    coverage = payload["coverage"]
    validation = payload["validation"]
    margin = payload["safety_margin"]
    lines = [
        "# DBA Memory Calibration",
        "",
        "## Environment and evidence",
        "",
        f"- Capture boundary: `{payload['capture_boundary']}`",
        f"- Host platform: `{payload['platform']['host']}`",
        f"- Torch: `{payload['platform']['torch_version']}`",
        f"- CUDA: `{payload['platform']['cuda_version']}`",
        f"- CUDA device: `{payload['platform']['cuda_device']}`",
        f"- Feature shape: `{payload['platform']['feature_shape']}`",
        f"- Frontend image size: `{payload['platform']['frontend_image_size']}`",
    ]
    for source in payload["source_hashes"]:
        lines.append(
            f"- Source: `{source['path']}` (`{source['sha256']}`)"
        )
    lines.extend([
        "",
        "## Signature coverage",
        "",
        f"- Sampled signatures: {coverage['sampled_signature_count']}",
        f"- Memory samples: {coverage['sample_count']}",
        (
            "- Weighted observed-call coverage: "
            f"{coverage['weighted_sample_coverage']:.2%}"
        ),
        "",
        "## Boundary model",
        "",
    ])
    term_names = payload["signatures"][0]["shadow_terms"]
    for name in term_names:
        values = [
            row["shadow_terms"][name]
            for row in payload["signatures"]
        ]
        lines.append(
            f"- `{name}`: {min(values)} to {max(values)} bytes"
        )
    lines.extend([
        "",
        "## Coefficients and safety margin",
        "",
    ])
    for name, value in payload["coefficients"].items():
        lines.append(f"- `{name}`: {value:.6f}")
    lines.extend([
        (
            "- Margin: max("
            f"{margin['minimum_bytes']} bytes, "
            f"{margin['ratio']:.0%} of modeled workspace)"
        ),
        "",
        "## Validation",
        "",
        f"- Valid: **{validation['valid']}**",
        f"- Weighted MAPE: {validation['weighted_mape']:.2%}",
        (
            "- Training weighted MAPE: "
            f"{validation['training_weighted_mape']:.2%}"
        ),
        (
            "- Held-out weighted MAPE: "
            f"{validation['held_out_weighted_mape']:.2%}"
        ),
        (
            "- Underpredicted signatures: "
            f"{validation['underpredicted_signatures'] or 'none'}"
        ),
        (
            "- Unstable signatures: "
            f"{validation['unstable_signatures'] or 'none'}"
        ),
        "",
    ])
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Calibrate boundary-specific DBA CUDA workspace"
    )
    parser.add_argument(
        "--details",
        action="append",
        required=True,
        help="Runtime detail JSONL file; repeat for multiple evidence runs",
    )
    parser.add_argument(
        "--capture-boundary",
        choices=CAPTURE_BOUNDARIES,
        required=True,
    )
    parser.add_argument("--output-calibration", required=True)
    parser.add_argument("--output-report", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    samples, observed_counts = load_memory_samples(args.details)
    payload = calibrate(
        samples,
        observed_counts,
        args.capture_boundary,
    )
    payload["platform"].update(_environment_metadata())
    payload["source_hashes"] = [
        _source_hash(path) for path in args.details
    ]

    calibration_path = Path(args.output_calibration)
    report_path = Path(args.output_report)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    report_path.write_text(render_report(payload))
    return 0 if payload["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
