import json

from scripts.frontend.dba_bucket_manifest import DBABucketManifest
from scripts.profiling.plan_dba_buckets import load_calls, main, plan_buckets


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
