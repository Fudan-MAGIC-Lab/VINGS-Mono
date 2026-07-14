import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from frontend.dba_bucket_manifest import DBACallSignature


CAPTURE_BOUNDARIES = (
    "corr_update_aggregation",
    "update_aggregation_only",
)
FEATURE_NAMES = (
    "intercept",
    "active_edges",
    "ba_edges",
    "source_poses",
    "pose_window",
)
MINIMUM_MARGIN_BYTES = 64 * 1024**2
MARGIN_RATIO = 0.10


def _as_signature(value):
    if isinstance(value, DBACallSignature) or (
        hasattr(value, "support_error")
        and hasattr(value, "feature_shape")
    ):
        return value
    payload = dict(value)
    payload["feature_shape"] = tuple(payload["feature_shape"])
    payload["frontend_image_size"] = tuple(
        payload["frontend_image_size"]
    )
    return DBACallSignature(**payload)


def boundary_shadow_terms(call, boundary):
    if boundary not in CAPTURE_BOUNDARIES:
        raise ValueError(f"unsupported DBA capture boundary: {boundary}")
    call = _as_signature(call)
    support_error = call.support_error()
    if support_error is not None:
        raise ValueError(
            f"unsupported DBA signature field: {support_error}"
        )

    height, width = call.feature_shape
    pixels = height * width
    value_bytes = 2 if call.dtype == "float16" else 4
    pyramid_pixels = sum(
        (height // (2**level)) * (width // (2**level))
        for level in range(4)
    )
    corr_pyramid = 0
    if boundary == "corr_update_aggregation":
        corr_pyramid = (
            call.active_edges
            * pixels
            * pyramid_pixels
            * value_bytes
        )

    return {
        "corr_pyramid_bytes": int(corr_pyramid),
        "sampled_corr_bytes": int(
            call.active_edges * pixels * 196 * value_bytes
        ),
        "recurrent_bytes": int(
            call.active_edges * pixels * (128 + 128) * value_bytes
        ),
        "motion_bytes": int(call.active_edges * pixels * 4 * 4),
        "delta_bytes": int(call.active_edges * pixels * 2 * 4),
        "weight_bytes": int(call.active_edges * pixels * 2 * 4),
        "damping_bytes": int(
            call.source_poses * pixels * value_bytes
        ),
        "upmask_bytes": int(
            call.source_poses
            * pixels
            * 576
            * value_bytes
            if call.upsample
            else 0
        ),
        "aggregation_bytes": int(
            call.source_poses * pixels * 128 * value_bytes
        ),
        "ba_inputs_bytes": 0,
    }


def signature_key(payload):
    return (
        int(payload["active_edges"]),
        int(payload["ba_edges"]),
        int(payload["source_poses"]),
        int(payload["pose_window"]),
        bool(payload["use_inactive"]),
        bool(payload["upsample"]),
        str(payload["dtype"]),
    )


def validate_sample(payload):
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported DBA memory sample schema")
    if "signature" not in payload:
        raise ValueError("DBA memory sample is missing its signature")
    call = _as_signature(payload["signature"])
    support_error = call.support_error()
    if support_error is not None:
        raise ValueError(
            f"unsupported DBA memory sample signature: {support_error}"
        )
    stages = payload.get("stages")
    if not stages:
        raise ValueError("DBA memory sample must contain stage snapshots")
    for stage in stages:
        peak = int(stage.get("peak_delta_bytes", -1))
        if peak < 0:
            raise ValueError(
                "DBA memory sample peak_delta_bytes must be nonnegative"
            )
    return payload


def load_memory_samples(paths):
    samples = []
    observed_counts = Counter()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "dba_signature":
                observed_counts[signature_key(row["payload"])] += 1
            elif row.get("kind") == "dba_memory_sample":
                payload = row["payload"]
                validate_sample(payload)
                samples.append(payload)
    return samples, observed_counts


def deterministic_split(signature_keys):
    keys = sorted(set(signature_keys))
    if len(keys) < 5:
        raise ValueError("at least five sampled signatures are required")
    held_out = [
        key for index, key in enumerate(keys, 1) if index % 5 == 0
    ]
    held_out_set = set(held_out)
    training = [key for key in keys if key not in held_out_set]
    return training, held_out


def _feature_row(key):
    return [1.0, key[0], key[1], key[2], key[3]]


def _weighted_mean(values, keys, observed_counts):
    weights = [int(observed_counts.get(key, 0)) for key in keys]
    if not any(weights):
        weights = [1] * len(keys)
    return sum(
        value * weight for value, weight in zip(values, weights)
    ) / sum(weights)


def _signature_from_sample(sample):
    return _as_signature(sample["signature"])


def _resident_distributions(samples):
    result = {}
    fields = (
        "corr_resident_storage_bytes",
        "recurrent_resident_storage_bytes",
        "state_resident_storage_bytes",
        "excluded_tensor_count",
    )
    for field in fields:
        values = [
            int(sample.get("entry", {}).get(field, 0))
            for sample in samples
        ]
        result[field] = {
            "minimum": min(values, default=0),
            "maximum": max(values, default=0),
        }
    return result


def _predict_transient(coefficients, key):
    return max(
        0.0,
        sum(
            float(coefficients[name]) * value
            for name, value in zip(FEATURE_NAMES, _feature_row(key))
        ),
    )


def workspace_breakdown(calibration, call):
    validate_calibration(calibration)
    call = _as_signature(call)
    terms = boundary_shadow_terms(
        call,
        calibration["capture_boundary"],
    )
    key = signature_key(call.to_dict())
    transient = int(
        math.ceil(
            _predict_transient(calibration["coefficients"], key)
        )
    )
    modeled = sum(terms.values()) + transient
    margin = max(
        int(calibration["safety_margin"]["minimum_bytes"]),
        int(math.ceil(
            float(calibration["safety_margin"]["ratio"]) * modeled
        )),
    )
    return {
        **terms,
        "calibrated_transient_bytes": transient,
        "modeled_workspace_bytes": modeled,
        "safety_margin_bytes": margin,
        "estimated_workspace_bytes": modeled + margin,
    }


def calibrate(samples, observed_counts, boundary):
    if boundary not in CAPTURE_BOUNDARIES:
        raise ValueError(f"unsupported DBA capture boundary: {boundary}")
    grouped = defaultdict(list)
    calls = {}
    for sample in samples:
        validate_sample(sample)
        key = signature_key(sample["signature"])
        grouped[key].append(sample)
        calls[key] = _signature_from_sample(sample)

    if not grouped:
        raise ValueError("no DBA memory samples were found")
    observed_total = sum(observed_counts.values())
    if observed_total <= 0:
        raise ValueError("no observed DBA signatures were found")
    sampled_observations = sum(
        count
        for key, count in observed_counts.items()
        if key in grouped
    )
    weighted_coverage = sampled_observations / observed_total
    if weighted_coverage < 0.95:
        raise ValueError(
            "sampled signatures must cover at least 95% of observed calls"
        )

    grouped_metrics = {}
    for key, signature_samples in grouped.items():
        peaks = [
            max(
                int(stage["peak_delta_bytes"])
                for stage in sample["stages"]
            )
            for sample in signature_samples
        ]
        maximum = max(peaks)
        spread = (maximum - min(peaks)) / maximum if maximum else 0.0
        grouped_metrics[key] = {
            "measured_transient_peak_bytes": maximum,
            "peak_spread": spread,
            "sample_count": len(signature_samples),
        }

    training_keys, held_out_keys = deterministic_split(grouped)
    matrix = np.asarray(
        [_feature_row(key) for key in training_keys],
        dtype=np.float64,
    )
    targets = np.asarray(
        [
            grouped_metrics[key]["measured_transient_peak_bytes"]
            for key in training_keys
        ],
        dtype=np.float64,
    )
    coefficients = np.maximum(
        np.linalg.lstsq(matrix, targets, rcond=None)[0],
        0.0,
    )
    raw_training = matrix @ coefficients
    if any(
        target > 0 and prediction <= 0
        for target, prediction in zip(targets, raw_training)
    ):
        raise ValueError(
            "positive transient target has zero model prediction"
        )
    scale = max(
        1.0,
        max(
            (
                target / prediction
                for target, prediction in zip(targets, raw_training)
                if prediction > 0
            ),
            default=1.0,
        ),
    )
    coefficients *= scale
    coefficient_payload = {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, coefficients)
    }

    rows = []
    errors = {}
    underpredicted = []
    unstable = []
    held_out_set = set(held_out_keys)
    for key in sorted(grouped):
        call = calls[key]
        terms = boundary_shadow_terms(call, boundary)
        shadow_bytes = sum(terms.values())
        transient = int(math.ceil(
            _predict_transient(coefficient_payload, key)
        ))
        measured = (
            shadow_bytes
            + grouped_metrics[key]["measured_transient_peak_bytes"]
        )
        modeled = shadow_bytes + transient
        margin = max(
            MINIMUM_MARGIN_BYTES,
            int(math.ceil(MARGIN_RATIO * modeled)),
        )
        final = modeled + margin
        error = abs(modeled - measured) / measured if measured else 0.0
        errors[key] = error
        if final < measured:
            underpredicted.append(repr(key))
        if grouped_metrics[key]["peak_spread"] > 0.15:
            unstable.append(repr(key))
        rows.append({
            "signature_key": repr(key),
            **call.to_dict(),
            "split": "held_out" if key in held_out_set else "training",
            **grouped_metrics[key],
            "shadow_terms": terms,
            "persistent_shadow_bytes": shadow_bytes,
            "predicted_transient_bytes": transient,
            "measured_target_bytes": measured,
            "modeled_workspace_bytes": modeled,
            "safety_margin_bytes": margin,
            "estimated_workspace_bytes": final,
            "absolute_percentage_error": error,
            "observed_count": int(observed_counts.get(key, 0)),
        })

    all_keys = sorted(grouped)
    weighted_mape = _weighted_mean(
        [errors[key] for key in all_keys],
        all_keys,
        observed_counts,
    )
    training_mape = _weighted_mean(
        [errors[key] for key in training_keys],
        training_keys,
        observed_counts,
    )
    held_out_mape = _weighted_mean(
        [errors[key] for key in held_out_keys],
        held_out_keys,
        observed_counts,
    )
    valid = (
        weighted_mape <= 0.10
        and not underpredicted
        and not unstable
        and weighted_coverage >= 0.95
    )
    representative = calls[sorted(calls)[0]]
    return {
        "schema_version": 1,
        "capture_boundary": boundary,
        "platform": {
            "frontend_image_size": list(
                representative.frontend_image_size
            ),
            "feature_shape": list(representative.feature_shape),
            "mode": representative.mode,
            "backend": representative.backend,
            "dtypes": sorted({call.dtype for call in calls.values()}),
        },
        "source_hashes": [],
        "coverage": {
            "sampled_signature_count": len(grouped),
            "observed_signature_count": len(observed_counts),
            "sample_count": len(samples),
            "weighted_sample_coverage": weighted_coverage,
        },
        "resident_storage": _resident_distributions(samples),
        "coefficients": coefficient_payload,
        "coefficient_scale": float(scale),
        "safety_margin": {
            "minimum_bytes": MINIMUM_MARGIN_BYTES,
            "ratio": MARGIN_RATIO,
        },
        "validation": {
            "valid": valid,
            "weighted_mape": weighted_mape,
            "training_weighted_mape": training_mape,
            "held_out_weighted_mape": held_out_mape,
            "held_out_keys": [repr(key) for key in held_out_keys],
            "underpredicted_signatures": underpredicted,
            "unstable_signatures": unstable,
            "maximum_peak_spread": max(
                item["peak_spread"] for item in grouped_metrics.values()
            ),
        },
        "signatures": rows,
    }


def validate_calibration(payload):
    validation = payload.get("validation", {})
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported DBA memory calibration schema")
    if not validation.get("valid", False):
        raise ValueError("DBA memory calibration did not pass validation")
    return payload
