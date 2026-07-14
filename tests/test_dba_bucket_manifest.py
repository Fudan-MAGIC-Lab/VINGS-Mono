import json

import pytest

from scripts.frontend.dba_bucket_manifest import (
    DBABucketManifest,
    DBABucketSpec,
    DBACallSignature,
)


def signature(**overrides):
    values = {
        "active_edges": 11,
        "ba_edges": 17,
        "source_poses": 5,
        "pose_window": 9,
        "use_inactive": True,
        "upsample": True,
        "dtype": "float16",
        "feature_shape": (43, 77),
        "frontend_image_size": (344, 616),
        "mode": "vo",
        "backend": "torch",
    }
    values.update(overrides)
    return DBACallSignature(**values)


def test_support_contract_rejects_wrong_shape_mode_and_backend():
    assert signature().support_error() is None
    assert signature(feature_shape=(32, 56)).support_error() == "feature_shape"
    assert signature(mode="vio").support_error() == "mode"
    assert signature(backend="tensorrt").support_error() == "backend"


def test_bucket_requires_all_capacities_and_control_fields():
    bucket = DBABucketSpec(
        name="e16_ba24_s8_p12_fp16_inactive_up",
        active_edges=16,
        ba_edges=24,
        source_poses=8,
        pose_window=12,
        use_inactive=True,
        upsample=True,
        dtype="float16",
        max_padding_ratio=0.40,
        estimated_workspace_bytes=64_000_000,
    )

    assert bucket.fits(signature())
    assert not bucket.fits(signature(active_edges=17))
    assert not bucket.fits(signature(use_inactive=False))
    assert not bucket.fits(signature(dtype="float32"))


def test_manifest_selects_smallest_fitting_bucket(tmp_path):
    payload = {
        "schema_version": 1,
        "frontend_image_size": [344, 616],
        "feature_shape": [43, 77],
        "mode": "vo",
        "backend": "torch",
        "buckets": [
            {
                "name": "large",
                "active_edges": 24,
                "ba_edges": 32,
                "source_poses": 10,
                "pose_window": 16,
                "use_inactive": True,
                "upsample": True,
                "dtype": "float16",
                "max_padding_ratio": 0.70,
                "estimated_workspace_bytes": 96_000_000,
            },
            {
                "name": "small",
                "active_edges": 16,
                "ba_edges": 24,
                "source_poses": 8,
                "pose_window": 12,
                "use_inactive": True,
                "upsample": True,
                "dtype": "float16",
                "max_padding_ratio": 0.40,
                "estimated_workspace_bytes": 64_000_000,
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))

    manifest = DBABucketManifest.load(path)

    assert manifest.select(signature()).name == "small"
    assert manifest.select(signature(active_edges=30)) is None


def test_manifest_rejects_non_43x77_contract(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frontend_image_size": [256, 448],
                "feature_shape": [32, 56],
                "mode": "vo",
                "backend": "torch",
                "buckets": [],
            }
        )
    )

    with pytest.raises(ValueError, match="344x616.*43x77"):
        DBABucketManifest.load(path)
