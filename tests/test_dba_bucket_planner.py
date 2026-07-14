import json
from pathlib import Path

import pytest

from scripts.frontend.dba_bucket_manifest import DBABucketManifest
from scripts.profiling.dba_memory_calibration import (
    calibrate,
    load_memory_samples,
)
from scripts.profiling.plan_dba_buckets import (
    estimate_workspace_bytes,
    load_calibrated_estimator,
    load_calls,
    main,
    plan_buckets,
)


FIXTURE = Path(__file__).parent / "fixtures" / "dba_memory_samples_43x77.jsonl"


def call(active, ba, sources, window, frame, dtype="float16"):
    return {
        "kind": "dba_signature",
        "frame_idx": frame,
        "payload": {
            "active_edges": active,
            "ba_edges": ba,
            "source_poses": sources,
            "pose_window": window,
            "use_inactive": True,
            "upsample": True,
            "dtype": dtype,
            "feature_shape": [43, 77],
            "frontend_image_size": [344, 616],
            "mode": "vo",
            "backend": "torch",
        },
    }


def write_valid_calibration(tmp_path, boundary):
    samples, observed_counts = load_memory_samples([FIXTURE])
    payload = calibrate(samples, observed_counts, boundary)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def test_load_calls_ignores_other_detail_kinds(tmp_path):
    path = tmp_path / "details.jsonl"
    rows = [call(8, 12, 4, 7, 1), {"kind": "other", "payload": {}}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    calls = load_calls([path])

    assert len(calls) == 1
    assert calls[0].active_edges == 8


def test_planner_reaches_target_with_deterministic_bucket_order():
    rows = []
    rows.extend([call(8, 12, 4, 7, frame) for frame in range(70)])
    rows.extend([call(12, 18, 6, 10, frame) for frame in range(70, 95)])
    rows.extend([call(24, 32, 10, 16, frame) for frame in range(95, 100)])
    signatures = [row["payload"] for row in rows]

    result = plan_buckets(
        signatures,
        target_coverage=0.90,
        max_buckets=4,
        max_padding_ratio=0.35,
        max_workspace_bytes=1_073_741_824,
    )

    assert result["covered_calls"] >= 90
    assert result["total_calls"] == 100
    assert result["coverage"] >= 0.90
    assert [bucket["active_edges"] for bucket in result["buckets"]] == [8, 12]
    assert result["buckets"] == sorted(
        result["buckets"],
        key=lambda item: (item["estimated_workspace_bytes"], item["name"]),
    )


def test_legacy_planner_result_is_unchanged_without_calibration():
    rows = [call(8, 12, 4, 7, 1), call(12, 18, 6, 10, 2)]

    result = plan_buckets(
        [row["payload"] for row in rows],
        0.90,
        6,
        0.35,
        1024 * 1024**2,
    )

    assert result["memory_model"] == {"kind": "phase0_conservative"}
    assert [
        bucket["estimated_workspace_bytes"]
        for bucket in result["buckets"]
    ] == [
        estimate_workspace_bytes(12, 18, 6, 10, "float16"),
    ]


def test_calibrated_planner_records_boundary_hash_and_breakdown(tmp_path):
    calibration = write_valid_calibration(
        tmp_path,
        "update_aggregation_only",
    )
    details = tmp_path / "details.jsonl"
    details.write_text(json.dumps(call(48, 72, 12, 11, 1)) + "\n")
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "manifest.md"

    status = main(
        [
            "--details",
            str(details),
            "--memory-calibration",
            str(calibration),
            "--capture-boundary",
            "update_aggregation_only",
            "--output-manifest",
            str(manifest),
            "--output-report",
            str(report),
        ]
    )

    assert status == 0
    payload = json.loads(manifest.read_text())
    assert payload["memory_model"]["kind"] == "calibrated"
    assert (
        payload["memory_model"]["capture_boundary"]
        == "update_aggregation_only"
    )
    assert len(payload["memory_model"]["calibration_sha256"]) == 64
    bucket = payload["buckets"][0]
    breakdown = payload["workspace_breakdowns"][bucket["name"]]
    assert breakdown["estimated_workspace_bytes"] == (
        bucket["estimated_workspace_bytes"]
    )
    assert breakdown["ba_inputs_bytes"] == 0
    assert "## Memory model" in report.read_text()


def test_planner_rejects_boundary_mismatch(tmp_path):
    calibration = write_valid_calibration(
        tmp_path,
        "update_aggregation_only",
    )

    with pytest.raises(ValueError, match="boundary"):
        load_calibrated_estimator(
            calibration,
            "corr_update_aggregation",
        )


def test_cli_writes_loadable_manifest_and_markdown_report(tmp_path):
    details = tmp_path / "details.jsonl"
    details.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                call(8, 12, 4, 7, 1),
                call(12, 18, 6, 10, 2),
                call(24, 32, 10, 16, 3),
            )
        )
        + "\n"
    )
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "manifest.md"

    status = main(
        [
            "--details",
            str(details),
            "--output-manifest",
            str(manifest_path),
            "--output-report",
            str(report_path),
            "--target-coverage",
            "0.90",
            "--max-buckets",
            "6",
            "--max-padding-ratio",
            "0.35",
            "--max-workspace-mb",
            "1024",
        ]
    )

    assert status == 0
    assert DBABucketManifest.load(manifest_path).buckets
    report = report_path.read_text()
    assert "# DBA CUDA Graph Bucket Plan" in report
    assert "| Bucket | Active edges | BA edges |" in report
