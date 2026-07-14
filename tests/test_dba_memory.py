import torch

from scripts.profiling.dba_memory import (
    DBAMemorySampleBuilder,
    MemorySnapshot,
    SignatureSampleLimiter,
    unique_storage_bytes,
)


def test_signature_limiter_reserves_only_configured_samples():
    limiter = SignatureSampleLimiter(samples_per_signature=2)

    assert limiter.reserve((48, 72, 12, 11)) == 1
    assert limiter.reserve((48, 72, 12, 11)) == 2
    assert limiter.reserve((48, 72, 12, 11)) is None
    assert limiter.reserve((31, 38, 6, 5)) == 1


def test_signature_limiter_rejects_nonpositive_limit():
    try:
        SignatureSampleLimiter(samples_per_signature=0)
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_unique_storage_bytes_deduplicates_views_and_nested_values():
    base = torch.zeros(32, dtype=torch.float32)
    view = base[4:20]
    other = torch.zeros(8, dtype=torch.float16)

    result = unique_storage_bytes(
        {"corr": [base, view], "other": (other,)},
        required_device_type="cpu",
    )

    expected_bytes = (
        base.untyped_storage().nbytes()
        + other.untyped_storage().nbytes()
    )
    assert result.storage_bytes == expected_bytes
    assert result.tensor_count == 3
    assert result.unique_storage_count == 2
    assert result.excluded_tensor_count == 0


def test_unique_storage_bytes_reports_excluded_devices():
    tensor = torch.zeros(4)

    result = unique_storage_bytes(tensor, required_device_type="cuda")

    assert result.storage_bytes == 0
    assert result.tensor_count == 1
    assert result.unique_storage_count == 0
    assert result.excluded_tensor_count == 1


class FakeBackend:
    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)
        self.reset_calls = 0
        self.sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1

    def reset_peak(self):
        self.reset_calls += 1

    def snapshot(self):
        return next(self.snapshots)


def snap(allocated, reserved, peak, free=10_000, total=20_000):
    return MemorySnapshot(allocated, reserved, peak, free, total)


def test_sample_builder_records_entry_relative_retained_and_stage_peak():
    backend = FakeBackend([
        snap(1_000, 2_000, 1_000),
        snap(1_300, 2_500, 1_700, free=9_400),
        snap(1_100, 2_500, 1_450, free=9_300),
    ])
    builder = DBAMemorySampleBuilder(backend)

    builder.begin(
        signature={"active_edges": 48},
        sample_ordinal=1,
        storage={"corr_resident_storage_bytes": 800},
    )
    builder.mark("corr_sample_complete")
    builder.mark("update_op_complete")
    payload = builder.finish()

    assert payload["entry"]["allocated_bytes"] == 1_000
    assert payload["entry"]["corr_resident_storage_bytes"] == 800
    assert payload["stages"][0]["allocated_delta_bytes"] == 300
    assert payload["stages"][0]["retained_delta_bytes"] == 300
    assert payload["stages"][0]["peak_delta_bytes"] == 700
    assert payload["stages"][0]["free_delta_bytes"] == -600
    assert payload["stages"][1]["allocated_delta_bytes"] == -200
    assert payload["stages"][1]["retained_delta_bytes"] == 100
    assert payload["stages"][1]["peak_delta_bytes"] == 150
    assert backend.reset_calls == 3
    assert backend.sync_calls == 3


def test_sample_builder_requires_begin_before_mark_or_finish():
    builder = DBAMemorySampleBuilder(FakeBackend([]))

    for action in (
        lambda: builder.mark("corr_sample_complete"),
        builder.finish,
    ):
        try:
            action()
        except RuntimeError as exc:
            assert "has not started" in str(exc)
        else:
            raise AssertionError("expected a RuntimeError")
