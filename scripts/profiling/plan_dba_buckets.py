import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from frontend.dba_bucket_manifest import DBABucketSpec, DBACallSignature
from profiling.dba_memory_calibration import (
    CAPTURE_BOUNDARIES,
    validate_calibration,
    workspace_breakdown,
)


class CalibratedWorkspaceEstimator:
    def __init__(self, calibration, provenance):
        self.calibration = calibration
        self.provenance = provenance
        self.supported_variants = {
            (
                bool(row["use_inactive"]),
                bool(row["upsample"]),
                str(row["dtype"]),
            )
            for row in calibration["signatures"]
        }

    def _breakdown_for_call(self, call):
        variant = (call.use_inactive, call.upsample, call.dtype)
        if variant not in self.supported_variants:
            raise ValueError(
                "DBA memory calibration does not support call variant "
                f"{variant!r}"
            )
        result = workspace_breakdown(self.calibration, call)
        result["persistent_shadow_bytes"] = (
            result["modeled_workspace_bytes"]
            - result["calibrated_transient_bytes"]
        )
        return result

    def estimate_call(self, call):
        return self._breakdown_for_call(call)[
            "estimated_workspace_bytes"
        ]

    def breakdown_for_bucket(self, bucket):
        call = DBACallSignature(
            active_edges=bucket.active_edges,
            ba_edges=bucket.ba_edges,
            source_poses=bucket.source_poses,
            pose_window=bucket.pose_window,
            use_inactive=bucket.use_inactive,
            upsample=bucket.upsample,
            dtype=bucket.dtype,
            feature_shape=(43, 77),
            frontend_image_size=(344, 616),
            mode="vo",
            backend="torch",
        )
        return self._breakdown_for_call(call)

    def __call__(
        self,
        active_edges,
        ba_edges,
        source_poses,
        pose_window,
        dtype,
    ):
        variants = [
            variant
            for variant in self.supported_variants
            if variant[2] == dtype
        ]
        if len(variants) != 1:
            raise ValueError(
                "calibrated estimator requires an unambiguous call variant"
            )
        use_inactive, upsample, _ = variants[0]
        call = DBACallSignature(
            active_edges=active_edges,
            ba_edges=ba_edges,
            source_poses=source_poses,
            pose_window=pose_window,
            use_inactive=use_inactive,
            upsample=upsample,
            dtype=dtype,
            feature_shape=(43, 77),
            frontend_image_size=(344, 616),
            mode="vo",
            backend="torch",
        )
        return self.estimate_call(call)


def load_calibrated_estimator(path, boundary):
    path = Path(path)
    raw = path.read_bytes()
    calibration = validate_calibration(json.loads(raw))
    if calibration.get("capture_boundary") != boundary:
        raise ValueError("DBA memory calibration boundary mismatch")
    platform = calibration.get("platform", {})
    if (
        tuple(platform.get("frontend_image_size", ())) != (344, 616)
        or tuple(platform.get("feature_shape", ())) != (43, 77)
    ):
        raise ValueError(
            "DBA memory calibration must target 344x616 -> 43x77"
        )
    if platform.get("mode") != "vo" or platform.get("backend") != "torch":
        raise ValueError(
            "DBA memory calibration must target VO with the torch backend"
        )
    dtypes = set(platform.get("dtypes", ()))
    if not dtypes or not dtypes <= {"float16", "float32"}:
        raise ValueError("DBA memory calibration has unsupported dtypes")
    validation = calibration["validation"]
    provenance = {
        "kind": "calibrated",
        "capture_boundary": boundary,
        "calibration_sha256": hashlib.sha256(raw).hexdigest(),
        "weighted_mape": validation["weighted_mape"],
        "held_out_weighted_mape": validation[
            "held_out_weighted_mape"
        ],
        "underprediction_count": len(
            validation["underpredicted_signatures"]
        ),
    }
    return CalibratedWorkspaceEstimator(calibration, provenance)


def load_calls(paths):
    calls = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row.get("kind") != "dba_signature":
                continue
            payload = dict(row["payload"])
            payload["feature_shape"] = tuple(payload["feature_shape"])
            payload["frontend_image_size"] = tuple(
                payload["frontend_image_size"]
            )
            calls.append(
                DBACallSignature(
                    **{
                        key: payload[key]
                        for key in DBACallSignature.__dataclass_fields__
                    }
                )
            )
    return calls


def estimate_workspace_bytes(
    active_edges,
    ba_edges,
    source_poses,
    pose_window,
    dtype,
):
    height, width = 43, 77
    pixels = height * width
    value_bytes = 2 if dtype == "float16" else 4
    pyramid_pixels = sum(
        (height // (2**level)) * (width // (2**level))
        for level in range(4)
    )
    corr_volume = active_edges * pixels * pyramid_pixels * value_bytes
    recurrent = active_edges * pixels * (128 + 128) * value_bytes
    sampled_corr = active_edges * pixels * 196 * value_bytes
    motion_and_heads = active_edges * pixels * (4 + 2 + 2) * 4
    graph_agg = source_poses * pixels * (128 + 1 + 576) * value_bytes
    ba_inputs = ba_edges * pixels * (2 + 2) * 4
    ba_pose_terms = max(pose_window, 1) * 6 * 6 * 8
    fixed_state_margin = 16 * 1024 * 1024
    return int(
        corr_volume
        + recurrent
        + sampled_corr
        + motion_and_heads
        + graph_agg
        + ba_inputs
        + ba_pose_terms
        + fixed_state_margin
    )


def _call_key(call):
    return (
        call.active_edges,
        call.ba_edges,
        call.source_poses,
        call.pose_window,
        call.use_inactive,
        call.upsample,
        call.dtype,
    )


def _bucket_from_call(call, max_padding_ratio, workspace_estimator):
    name = (
        f"e{call.active_edges}_ba{call.ba_edges}_s{call.source_poses}_"
        f"p{call.pose_window}_{call.dtype}_"
        f"{'inactive' if call.use_inactive else 'active'}_"
        f"{'up' if call.upsample else 'native'}"
    )
    return DBABucketSpec(
        name=name,
        active_edges=call.active_edges,
        ba_edges=call.ba_edges,
        source_poses=call.source_poses,
        pose_window=call.pose_window,
        use_inactive=call.use_inactive,
        upsample=call.upsample,
        dtype=call.dtype,
        max_padding_ratio=max_padding_ratio,
        estimated_workspace_bytes=(
            workspace_estimator.estimate_call(call)
            if hasattr(workspace_estimator, "estimate_call")
            else workspace_estimator(
                call.active_edges,
                call.ba_edges,
                call.source_poses,
                call.pose_window,
                call.dtype,
            )
        ),
    )


def _as_signature(item):
    if isinstance(item, DBACallSignature):
        return item
    return DBACallSignature(
        **{
            **item,
            "feature_shape": tuple(item["feature_shape"]),
            "frontend_image_size": tuple(item["frontend_image_size"]),
        }
    )


def plan_buckets(
    raw_calls,
    target_coverage,
    max_buckets,
    max_padding_ratio,
    max_workspace_bytes,
    workspace_estimator=None,
    memory_model=None,
):
    workspace_estimator = workspace_estimator or estimate_workspace_bytes
    memory_model = memory_model or {"kind": "phase0_conservative"}
    calls = [_as_signature(item) for item in raw_calls]
    supported = [call for call in calls if call.support_error() is None]
    counts = Counter(_call_key(call) for call in supported)
    unique = {_call_key(call): call for call in supported}
    candidates = [
        _bucket_from_call(
            unique[key],
            max_padding_ratio,
            workspace_estimator,
        )
        for key in sorted(unique)
    ]
    candidates = [
        bucket
        for bucket in candidates
        if bucket.estimated_workspace_bytes <= max_workspace_bytes
    ]

    uncovered = set(range(len(supported)))
    selected = []
    target_calls = int(len(supported) * target_coverage + 0.999999)
    while uncovered and len(selected) < max_buckets:
        ranked = []
        for bucket in candidates:
            if bucket in selected:
                continue
            covered = {
                index
                for index in uncovered
                if bucket.fits(supported[index])
            }
            if covered:
                ranked.append(
                    (
                        len(covered) / bucket.estimated_workspace_bytes,
                        len(covered),
                        -bucket.estimated_workspace_bytes,
                        bucket.name,
                        bucket,
                        covered,
                    )
                )
        if not ranked:
            break
        _, _, _, _, bucket, covered = max(ranked)
        selected.append(bucket)
        uncovered -= covered
        if len(supported) - len(uncovered) >= target_calls:
            break

    selected.sort(
        key=lambda bucket: (bucket.estimated_workspace_bytes, bucket.name)
    )
    covered_calls = len(supported) - len(uncovered)
    workspace_breakdowns = {}
    if hasattr(workspace_estimator, "breakdown_for_bucket"):
        workspace_breakdowns = {
            bucket.name: workspace_estimator.breakdown_for_bucket(bucket)
            for bucket in selected
        }
    return {
        "schema_version": 1,
        "frontend_image_size": [344, 616],
        "feature_shape": [43, 77],
        "mode": "vo",
        "backend": "torch",
        "total_calls": len(calls),
        "supported_calls": len(supported),
        "covered_calls": covered_calls,
        "coverage": covered_calls / len(supported) if supported else 0.0,
        "unsupported_calls": len(calls) - len(supported),
        "memory_model": dict(memory_model),
        "workspace_breakdowns": workspace_breakdowns,
        "buckets": [asdict(bucket) for bucket in selected],
        "observed_signature_counts": {
            repr(key): count for key, count in sorted(counts.items())
        },
    }


def render_report(result):
    lines = [
        "# DBA CUDA Graph Bucket Plan",
        "",
        f"- Total calls: {result['total_calls']}",
        f"- Supported calls: {result['supported_calls']}",
        f"- Covered calls: {result['covered_calls']}",
        f"- Coverage: {result['coverage']:.2%}",
        f"- Unsupported calls: {result['unsupported_calls']}",
        "",
        "## Memory model",
        "",
        f"- Kind: `{result['memory_model']['kind']}`",
    ]
    memory_model = result["memory_model"]
    if memory_model["kind"] == "calibrated":
        lines.extend([
            f"- Capture boundary: `{memory_model['capture_boundary']}`",
            (
                "- Calibration SHA-256: "
                f"`{memory_model['calibration_sha256']}`"
            ),
            f"- Weighted MAPE: {memory_model['weighted_mape']:.2%}",
            (
                "- Underpredicted signatures: "
                f"{memory_model['underprediction_count']}"
            ),
        ])
    lines.extend([
        "",
        "## Buckets",
        "",
        "| Bucket | Active edges | BA edges | Source poses | Pose window | "
        "Padding limit | Shadow MiB | Transient MiB | Margin MiB | "
        "Estimated MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for bucket in result["buckets"]:
        breakdown = result["workspace_breakdowns"].get(bucket["name"])
        if breakdown is None:
            shadow = transient = margin = "-"
        else:
            shadow = f"{breakdown['persistent_shadow_bytes'] / 1024**2:.2f}"
            transient = (
                f"{breakdown['calibrated_transient_bytes'] / 1024**2:.2f}"
            )
            margin = f"{breakdown['safety_margin_bytes'] / 1024**2:.2f}"
        lines.append(
            "| {name} | {active_edges} | {ba_edges} | {source_poses} | "
            "{pose_window} | {max_padding_ratio:.2%} | {shadow} | "
            "{transient} | {margin} | {mib:.2f} |".format(
                **bucket,
                shadow=shadow,
                transient=transient,
                margin=margin,
                mib=bucket["estimated_workspace_bytes"] / (1024**2),
            )
        )
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plan fixed DBA CUDA Graph buckets from runtime details."
    )
    parser.add_argument("--details", action="append", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--max-buckets", type=int, default=6)
    parser.add_argument("--max-padding-ratio", type=float, default=0.35)
    parser.add_argument("--max-workspace-mb", type=float, default=1024.0)
    parser.add_argument("--memory-calibration")
    parser.add_argument(
        "--capture-boundary",
        choices=CAPTURE_BOUNDARIES,
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if bool(args.memory_calibration) != bool(args.capture_boundary):
        raise ValueError(
            "--memory-calibration and --capture-boundary must be used together"
        )
    estimator = None
    memory_model = None
    if args.memory_calibration:
        estimator = load_calibrated_estimator(
            args.memory_calibration,
            args.capture_boundary,
        )
        memory_model = estimator.provenance
    result = plan_buckets(
        load_calls(args.details),
        target_coverage=args.target_coverage,
        max_buckets=args.max_buckets,
        max_padding_ratio=args.max_padding_ratio,
        max_workspace_bytes=int(args.max_workspace_mb * 1024**2),
        workspace_estimator=estimator,
        memory_model=memory_model,
    )
    manifest_path = Path(args.output_manifest)
    report_path = Path(args.output_report)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_path.write_text(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
