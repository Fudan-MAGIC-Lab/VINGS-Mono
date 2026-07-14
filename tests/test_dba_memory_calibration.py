import json
from pathlib import Path

import pytest

from scripts.frontend.dba_bucket_manifest import DBACallSignature
from scripts.profiling.calibrate_dba_workspace import main
from scripts.profiling.dba_memory_calibration import (
    boundary_shadow_terms,
    calibrate,
    deterministic_split,
    load_memory_samples,
    validate_calibration,
)


FIXTURE = Path(__file__).parent / "fixtures" / "dba_memory_samples_43x77.jsonl"


def signature(active=48, ba=72, sources=12, window=11):
    return DBACallSignature(
        active_edges=active,
        ba_edges=ba,
        source_poses=sources,
        pose_window=window,
        use_inactive=True,
        upsample=True,
        dtype="float16",
        feature_shape=(43, 77),
        frontend_image_size=(344, 616),
        mode="vo",
        backend="torch",
    )


def test_update_only_boundary_excludes_full_corr_and_eager_ba():
    terms = boundary_shadow_terms(
        signature(48, 72, 12, 11),
        "update_aggregation_only",
    )

    assert terms["corr_pyramid_bytes"] == 0
    assert terms["ba_inputs_bytes"] == 0
    assert terms["sampled_corr_bytes"] > 0
    assert terms["recurrent_bytes"] > 0
    assert terms["aggregation_bytes"] > 0


def test_full_corr_boundary_includes_exact_four_level_pyramid():
    call = signature(48, 72, 12, 11)
    terms = boundary_shadow_terms(call, "corr_update_aggregation")
    pixels = 43 * 77
    pyramid_pixels = sum(
        (43 // (2**level)) * (77 // (2**level))
        for level in range(4)
    )

    assert terms["corr_pyramid_bytes"] == (
        48 * pixels * pyramid_pixels * 2
    )


def test_boundary_terms_reject_unsupported_input():
    with pytest.raises(ValueError, match="capture boundary"):
        boundary_shadow_terms(signature(), "ba")
    with pytest.raises(ValueError, match="unsupported DBA signature"):
        boundary_shadow_terms(
            DBACallSignature(
                **{
                    **signature().to_dict(),
                    "feature_shape": (22, 39),
                }
            ),
            "update_aggregation_only",
        )


def test_calibration_holds_out_every_fifth_sorted_signature():
    samples, observed_counts = load_memory_samples([FIXTURE])

    result = calibrate(
        samples,
        observed_counts,
        "update_aggregation_only",
    )

    keys = sorted(observed_counts)
    assert result["validation"]["held_out_keys"] == [
        repr(keys[4]),
        repr(keys[9]),
    ]


def test_calibration_requires_weighted_sample_coverage():
    samples, observed_counts = load_memory_samples([FIXTURE])

    with pytest.raises(ValueError, match="95%"):
        calibrate(
            samples[:2],
            observed_counts,
            "update_aggregation_only",
        )


def test_calibration_adds_margin_and_never_underpredicts_holdout():
    samples, observed_counts = load_memory_samples([FIXTURE])

    result = calibrate(
        samples,
        observed_counts,
        "update_aggregation_only",
    )

    assert result["validation"]["valid"] is True
    assert result["validation"]["weighted_mape"] <= 0.10
    assert result["validation"]["underpredicted_signatures"] == []
    assert result["safety_margin"]["minimum_bytes"] == 64 * 1024**2
    assert result["coverage"]["weighted_sample_coverage"] == 1.0


def test_deterministic_split_requires_five_signatures():
    with pytest.raises(ValueError, match="at least five"):
        deterministic_split([(1,), (2,), (3,), (4,)])


def test_validate_calibration_rejects_schema_and_invalid_result():
    with pytest.raises(ValueError, match="unsupported"):
        validate_calibration({"schema_version": 2})
    with pytest.raises(ValueError, match="did not pass"):
        validate_calibration(
            {"schema_version": 1, "validation": {"valid": False}}
        )


def test_calibration_cli_writes_json_and_markdown(tmp_path):
    calibration_path = tmp_path / "calibration.json"
    report_path = tmp_path / "calibration.md"

    status = main(
        [
            "--details",
            str(FIXTURE),
            "--details",
            str(FIXTURE),
            "--capture-boundary",
            "update_aggregation_only",
            "--output-calibration",
            str(calibration_path),
            "--output-report",
            str(report_path),
        ]
    )

    assert status == 0
    payload = json.loads(calibration_path.read_text())
    assert payload["validation"]["valid"] is True
    assert payload["platform"]["host"]
    assert payload["platform"]["python"]
    assert "torch_version" in payload["platform"]
    assert "cuda_version" in payload["platform"]
    report = report_path.read_text()
    assert "# DBA Memory Calibration" in report
    assert "`corr_pyramid_bytes`" in report
    assert "## Validation" in report
