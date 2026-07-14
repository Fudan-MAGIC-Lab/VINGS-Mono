# DBA Memory Calibration Phase 0.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure real incremental DBA CUDA memory on Jetson, calibrate boundary-specific workspace estimates, and produce a revised evidence gate without implementing CUDA Graph replay or changing tracking behavior.

**Architecture:** Add an opt-in sampled memory probe that records synchronized allocator snapshots and exact resident tensor storage at existing eager DBA stage boundaries. An offline CPU-safe calibration module converts those samples into validated models for `corr_update_aggregation` and `update_aggregation_only`; the existing bucket planner accepts a validated model through an explicit estimator interface while retaining the Phase 0 conservative estimator by default.

**Tech Stack:** Python 3.8, PyTorch 2.1 Jetson CUDA APIs, NumPy, pytest/unittest, existing runtime profiler JSONL, Bash evidence drivers, `tegrastats`.

**Design reference:** `docs/superpowers/specs/2026-07-14-dba-memory-calibration-design.md`

**Scope boundary:** This plan ends with measurements, calibration artifacts, revised manifests, and a GO/NO-GO report. It must not add `torch.cuda.CUDAGraph`, change `frontend.max_factors`, change factor/keyframe scheduling, modify `submodules/dbaf`, or implement GPU BA.

---

## File map

- Create `scripts/profiling/dba_memory.py`: CPU-testable sampling limiter, storage accounting, CUDA backend adapter, and sample builder.
- Modify `scripts/run.py`: add and validate memory-profiling CLI/config overrides.
- Modify `scripts/frontend/covisible_graph.py`: build exact signatures once and mark memory stages around the unchanged eager update.
- Modify `scripts/frontend/dbaf.py`: publish memory-profiling metadata through the existing runtime profiler attachment.
- Create `scripts/profiling/dba_memory_calibration.py`: calibration schema, analytical boundary terms, deterministic split, fit, validation, and estimator.
- Create `scripts/profiling/calibrate_dba_workspace.py`: calibration CLI and Markdown renderer.
- Modify `scripts/profiling/plan_dba_buckets.py`: optional calibrated estimator and manifest provenance.
- Create `reports/cuda_graph_dba/run_phase05_memory_profiles.sh`: guarded SmallCity-200 and full-Hotel evidence driver.
- Create `tests/test_dba_memory.py`: limiter, storage, snapshot, and sample-builder tests.
- Modify `tests/test_jetson_runtime_overrides.py`: CLI/config override tests.
- Create `tests/test_dba_memory_instrumentation.py`: disabled path, signature reuse, stage ordering, and structured-record tests.
- Create `tests/test_dba_memory_calibration.py`: calibration schema, boundary, fitting, split, and validation tests.
- Modify `tests/test_dba_bucket_planner.py`: calibrated planner and legacy compatibility tests.
- Create `tests/test_dba_phase05_profile_script.py`: evidence-driver contract tests.
- Create `tests/test_dba_memory_cuda.py`: executable Jetson CUDA smoke tests.
- Generate `reports/cuda_graph_dba/phase05_*`: small calibration, manifest, report, and decision artifacts only; raw evidence stays ignored.

---

### Task 1: Build the CPU-testable memory sampling primitives

**Files:**
- Create: `scripts/profiling/dba_memory.py`
- Create: `tests/test_dba_memory.py`

- [ ] **Step 1: Write failing tests for deterministic sample limiting and unique storage accounting**

Create `tests/test_dba_memory.py` with these first tests:

```python
import torch

from scripts.profiling.dba_memory import (
    SignatureSampleLimiter,
    unique_storage_bytes,
)


def test_signature_limiter_reserves_only_configured_samples():
    limiter = SignatureSampleLimiter(samples_per_signature=2)

    assert limiter.reserve((48, 72, 12, 11)) == 1
    assert limiter.reserve((48, 72, 12, 11)) == 2
    assert limiter.reserve((48, 72, 12, 11)) is None
    assert limiter.reserve((31, 38, 6, 5)) == 1


def test_unique_storage_bytes_deduplicates_views_and_nested_values():
    base = torch.zeros(32, dtype=torch.float32)
    view = base[4:20]
    other = torch.zeros(8, dtype=torch.float16)

    result = unique_storage_bytes(
        {"corr": [base, view], "other": (other,)},
        required_device_type="cpu",
    )

    assert result.storage_bytes == base.untyped_storage().nbytes() + other.untyped_storage().nbytes()
    assert result.tensor_count == 3
    assert result.unique_storage_count == 2
    assert result.excluded_tensor_count == 0
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest --import-mode=importlib tests/test_dba_memory.py -q
```

Expected: collection error because `scripts.profiling.dba_memory` does not exist.

- [ ] **Step 3: Implement the limiter and storage accounting**

Create `scripts/profiling/dba_memory.py` with these public types and helpers:

```python
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class StorageSummary:
    storage_bytes: int
    tensor_count: int
    unique_storage_count: int
    excluded_tensor_count: int


class SignatureSampleLimiter:
    def __init__(self, samples_per_signature):
        samples_per_signature = int(samples_per_signature)
        if samples_per_signature < 1:
            raise ValueError("samples_per_signature must be at least 1")
        self.samples_per_signature = samples_per_signature
        self.counts = {}

    def reserve(self, signature_key):
        count = self.counts.get(signature_key, 0)
        if count >= self.samples_per_signature:
            return None
        count += 1
        self.counts[signature_key] = count
        return count


def _walk_tensors(value):
    if hasattr(value, "untyped_storage") and hasattr(value, "device"):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_tensors(item)


def unique_storage_bytes(value, required_device_type="cuda"):
    seen = set()
    total = 0
    tensor_count = 0
    excluded = 0
    for tensor in _walk_tensors(value):
        tensor_count += 1
        if tensor.device.type != required_device_type:
            excluded += 1
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()))
        if key in seen:
            continue
        seen.add(key)
        total += int(storage.nbytes())
    return StorageSummary(total, tensor_count, len(seen), excluded)
```

Keep this module free of top-level `torch` imports so CPU-only planner tests can import it without CUDA extension modules.

- [ ] **Step 4: Add failing tests for snapshot deltas and structured payloads**

Append tests using an injected fake backend:

```python
from scripts.profiling.dba_memory import DBAMemorySampleBuilder, MemorySnapshot


class FakeBackend:
    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)
        self.reset_calls = 0

    def synchronize(self):
        pass

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
    assert payload["stages"][0]["retained_delta_bytes"] == 300
    assert payload["stages"][0]["peak_delta_bytes"] == 700
    assert payload["stages"][1]["retained_delta_bytes"] == 100
    assert payload["stages"][1]["peak_delta_bytes"] == 150
    assert backend.reset_calls == 3
```

- [ ] **Step 5: Run the new test and verify RED**

Run:

```powershell
python -m pytest --import-mode=importlib tests/test_dba_memory.py -q
```

Expected: import error for `DBAMemorySampleBuilder` or `MemorySnapshot`.

- [ ] **Step 6: Implement the backend adapter and sample builder**

Add:

```python
@dataclass(frozen=True)
class MemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    free_bytes: int
    total_bytes: int


class TorchCUDAMemoryBackend:
    def __init__(self, device):
        import torch
        self.torch = torch
        self.device = device

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def reset_peak(self):
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self):
        free_bytes, total_bytes = self.torch.cuda.mem_get_info(self.device)
        return MemorySnapshot(
            allocated_bytes=int(self.torch.cuda.memory_allocated(self.device)),
            reserved_bytes=int(self.torch.cuda.memory_reserved(self.device)),
            peak_allocated_bytes=int(self.torch.cuda.max_memory_allocated(self.device)),
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
        )


class DBAMemorySampleBuilder:
    def __init__(self, backend):
        self.backend = backend
        self.payload = None
        self.entry = None
        self.stage_baseline = None

    def _snapshot_and_reset(self):
        self.backend.synchronize()
        snapshot = self.backend.snapshot()
        self.backend.reset_peak()
        return snapshot

    def begin(self, signature, sample_ordinal, storage):
        entry = self._snapshot_and_reset()
        self.entry = entry
        self.stage_baseline = entry
        self.payload = {
            "schema_version": 1,
            "sample_ordinal": int(sample_ordinal),
            "signature": dict(signature),
            "entry": {**asdict(entry), **storage},
            "stages": [],
        }

    def mark(self, name):
        current = self._snapshot_and_reset()
        self.payload["stages"].append({
            "name": str(name),
            "allocated_delta_bytes": current.allocated_bytes - self.stage_baseline.allocated_bytes,
            "retained_delta_bytes": current.allocated_bytes - self.entry.allocated_bytes,
            "peak_delta_bytes": max(0, current.peak_allocated_bytes - self.stage_baseline.allocated_bytes),
            "free_delta_bytes": current.free_bytes - self.entry.free_bytes,
        })
        self.stage_baseline = current

    def finish(self):
        if self.payload is None:
            raise RuntimeError("DBA memory sample has not started")
        return self.payload
```

- [ ] **Step 7: Run Task 1 tests**

Run:

```powershell
python -m pytest --import-mode=importlib tests/test_dba_memory.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 1**

```powershell
git add scripts/profiling/dba_memory.py tests/test_dba_memory.py
git commit -m "feat: add sampled DBA CUDA memory probes"
```

---

### Task 2: Add explicit CLI/config plumbing with a zero-cost disabled path

**Files:**
- Modify: `scripts/run.py`
- Modify: `tests/test_jetson_runtime_overrides.py`

- [ ] **Step 1: Add failing run.py override tests using the existing `_load_run_globals` helper**

Append to the existing test class:

```python
def test_run_py_can_enable_sampled_dba_memory_profiling(self):
    globals_dict = _load_run_globals([
        "run.py",
        "dummy.yaml",
        "--profile-runtime",
        "--profile-dba-memory",
        "--profile-dba-memory-samples-per-signature",
        "3",
    ])

    cfg = {
        "dataset": {}, "output": {}, "frontend": {}, "device": {},
        "looper": {}, "profiling": {"runtime": {"enabled": False}},
        "training_args": {"iters": 30},
    }
    updated = globals_dict["apply_overrides"](cfg)

    self.assertTrue(updated["profiling"]["dba_memory"]["enabled"])
    self.assertEqual(updated["profiling"]["dba_memory"]["samples_per_signature"], 3)


def test_run_py_rejects_dba_memory_without_runtime_profiler(self):
    with self.assertRaisesRegex(SystemExit, "2"):
        _load_run_globals([
            "run.py",
            "dummy.yaml",
            "--profile-dba-memory",
        ])
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m pytest --import-mode=importlib tests/test_jetson_runtime_overrides.py -q
```

Expected: parser rejects unknown `--profile-dba-memory`.

- [ ] **Step 3: Add parser arguments, validation, and config values**

In `scripts/run.py`, add next to `--profile-runtime`:

```python
parser.add_argument(
    "--profile-dba-memory",
    action="store_true",
    help="Sample synchronized CUDA memory at DBA stage boundaries; requires --profile-runtime",
)
parser.add_argument(
    "--profile-dba-memory-samples-per-signature",
    type=int,
    default=2,
    help="Maximum synchronized memory samples for one exact DBA signature",
)
```

Immediately after `parse_args()` add:

```python
if args.profile_dba_memory and not args.profile_runtime:
    parser.error("--profile-dba-memory requires --profile-runtime")
if args.profile_dba_memory_samples_per_signature < 1:
    parser.error("--profile-dba-memory-samples-per-signature must be at least 1")
```

Initialize and override configuration with:

```python
cfg.setdefault("profiling", {})
cfg["profiling"].setdefault("runtime", {})
cfg["profiling"].setdefault("dba_memory", {})
cfg["profiling"]["dba_memory"]["enabled"] = bool(args.profile_dba_memory)
cfg["profiling"]["dba_memory"]["samples_per_signature"] = int(
    args.profile_dba_memory_samples_per_signature
)
```

- [ ] **Step 4: Run override and runtime-profiler tests**

Run:

```powershell
python -m pytest --import-mode=importlib tests/test_jetson_runtime_overrides.py tests/test_runtime_profiler.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add scripts/run.py tests/test_jetson_runtime_overrides.py
git commit -m "feat: configure sampled DBA memory profiling"
```

---

### Task 3: Instrument the unchanged eager CovisibleGraph update

**Files:**
- Modify: `scripts/frontend/covisible_graph.py`
- Modify: `scripts/frontend/dbaf.py`
- Create: `tests/test_dba_memory_instrumentation.py`
- Modify: `tests/test_dba_signature_profiling.py`

- [ ] **Step 1: Refactor signature creation under a failing equivalence test**

Add a test asserting `_build_dba_signature()` returns the same fields currently
recorded by `_record_dba_signature()`. Reuse the stubs in
`tests/test_dba_signature_profiling.py`:

```python
call = graph._build_dba_signature(
    t0=2,
    observed_t1=6,
    use_inactive=True,
    ba_edges=6,
)

assert call.active_edges == 4
assert call.ba_edges == 6
assert call.source_poses == 3
assert call.pose_window == 4
assert call.feature_shape == (43, 77)
assert call.frontend_image_size == (344, 616)
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
python -m pytest --import-mode=importlib tests/test_dba_signature_profiling.py -q
```

Expected: `CovisibleGraph` has no `_build_dba_signature`.

- [ ] **Step 3: Extract signature construction without changing recording behavior**

In `covisible_graph.py`, move the existing `DBACallSignature` construction into:

```python
def _build_dba_signature(self, t0, observed_t1, use_inactive, ba_edges):
    return DBACallSignature(
        active_edges=int(self.ii.numel()),
        ba_edges=int(ba_edges),
        source_poses=int(torch.unique(self.ii).numel()),
        pose_window=int(observed_t1 - t0),
        use_inactive=bool(use_inactive),
        upsample=bool(self.upsample),
        dtype=str(self.net.dtype).replace("torch.", ""),
        feature_shape=(int(self.ht), int(self.wd)),
        frontend_image_size=(int(self.video.ht), int(self.video.wd)),
        mode=str(self.video.cfg["mode"]),
        backend=self._update_backend_name(),
    )
```

Make `_record_dba_signature(call)` accept the constructed object, add current
allocated/reserved fields exactly as before, and update the call site. Run the
existing signature test to prove the JSON contract did not change.

- [ ] **Step 4: Write failing instrumentation tests**

Create `tests/test_dba_memory_instrumentation.py` with module stubs matching
`test_dba_signature_profiling.py`. Test these behaviors with a fake sampler:

```python
class RecordingBuilder:
    def finish(self):
        return {
            "schema_version": 1,
            "sample_ordinal": 1,
            "signature": {"active_edges": 4},
            "entry": {},
            "stages": [
                {"name": "corr_sample_complete"},
                {"name": "update_op_complete"},
                {"name": "ba_inputs_complete"},
                {"name": "ba_complete"},
                {"name": "upsample_complete"},
            ],
        }


class DetailProfiler:
    def __init__(self):
        self.details = []

    def record_detail(self, kind, payload, frame_idx=None):
        self.details.append((kind, payload, frame_idx))


def test_disabled_memory_profile_does_not_create_sampler(graph):
    graph.video.cfg["profiling"] = {"dba_memory": {"enabled": False}}
    graph._configure_dba_memory_sampler()
    assert graph.dba_memory_limiter is None


def test_sampler_records_only_reserved_signatures(graph, monkeypatch):
    graph.video.cfg["profiling"] = {
        "dba_memory": {"enabled": True, "samples_per_signature": 1}
    }
    graph._configure_dba_memory_sampler(backend=FakeMemoryBackend())
    call = graph._build_dba_signature(2, 6, True, 6)

    first = graph._begin_dba_memory_sample(call)
    second = graph._begin_dba_memory_sample(call)

    assert first is not None
    assert second is None


def test_finished_sample_records_required_stage_order(graph):
    builder = RecordingBuilder()
    graph.profiler = DetailProfiler()
    graph.profiler_frame_idx = 17

    graph._finish_dba_memory_sample(builder)

    kind, payload, frame_idx = graph.profiler.details[0]
    assert kind == "dba_memory_sample"
    assert frame_idx == 17
    assert [row["name"] for row in payload["stages"]] == [
        "corr_sample_complete",
        "update_op_complete",
        "ba_inputs_complete",
        "ba_complete",
        "upsample_complete",
    ]
```

The fixture must set `corr.corr_pyramid`, `net`, `inp`, indices, target, weight,
and damping to CPU tensors and call storage accounting with
`required_device_type="cpu"`; no real CUDA allocation belongs in this unit test.
Build the fixture with `CovisibleGraph.__new__(CovisibleGraph)` so its normal
CUDA constructor is not invoked, matching the existing signature test pattern.

- [ ] **Step 5: Run instrumentation tests and verify RED**

Run:

```powershell
python -m pytest --import-mode=importlib tests/test_dba_memory_instrumentation.py -q
```

Expected: missing sampler methods.

- [ ] **Step 6: Implement opt-in sampler configuration and exact resident accounting**

Import the Task 1 helpers. In `CovisibleGraph.__init__`, set both sampler fields
to `None`, then call `_configure_dba_memory_sampler()` after config is available.
Implement:

```python
def _configure_dba_memory_sampler(self, backend=None):
    cfg = self.video.cfg.get("profiling", {}).get("dba_memory", {})
    if not cfg.get("enabled", False):
        self.dba_memory_limiter = None
        self.dba_memory_backend = None
        return
    self.dba_memory_limiter = SignatureSampleLimiter(
        cfg.get("samples_per_signature", 2)
    )
    self.dba_memory_backend = backend or TorchCUDAMemoryBackend(self.device)

def _dba_memory_storage(self):
    corr_pyramid = [] if self.corr is None else self.corr.corr_pyramid
    corr = unique_storage_bytes(corr_pyramid)
    recurrent = unique_storage_bytes([self.net, self.inp])
    state = unique_storage_bytes({
        "coords0": self.coords0,
        "target": self.target,
        "weight": self.weight,
        "damping": self.damping,
        "active_indices": [self.ii, self.jj],
        "inactive_indices": [self.ii_inac, self.jj_inac],
    })
    return {
        "corr_resident_storage_bytes": corr.storage_bytes,
        "recurrent_resident_storage_bytes": recurrent.storage_bytes,
        "state_resident_storage_bytes": state.storage_bytes,
        "excluded_tensor_count": (
            corr.excluded_tensor_count
            + recurrent.excluded_tensor_count
            + state.excluded_tensor_count
        ),
    }
```

`_preview_dba_memory_signature(t0, t1, use_inactive)` computes the exact key
before corr using only active/inactive index tensors. It resolves default `t0`,
the inactive mask, BA-edge count, observed `t1`, source poses, and pose window
without constructing or mutating target/weight tensors. `_begin_dba_memory_sample(call)`
then reserves that key, creates a `DBAMemorySampleBuilder`, and calls
`begin(call.to_dict(), ordinal, storage)`. Return `None` immediately when
disabled or exhausted.

- [ ] **Step 7: Mark the five stage boundaries without changing tensor order or lifetime**

In `update()`, preview the call only when memory sampling is enabled and begin
sampling immediately before `self.corr(coords1)`. Construct the authoritative
call after `ii/jj/target/weight` are finalized, as today, and compare its key to
the preview. Raise `RuntimeError("DBA memory signature preview mismatch")` on a
mismatch so malformed evidence cannot be accepted. Call `builder.mark()`
immediately after each existing operation:

1. after `corr = self.corr(coords1)`;
2. after the existing `self.update_op` call returns;
3. after contiguous `target`, `weight`, and `damping` BA inputs are ready;
4. after the existing `self.video.ba` call returns;
5. after the existing `self.video.upsample` call returns, or mark `update_exit`
   when upsample is off.

Finally call:

```python
if memory_builder is not None:
    self.profiler.record_detail(
        "dba_memory_sample",
        memory_builder.finish(),
        frame_idx=self.profiler_frame_idx,
    )
```

Do not wrap sampling exceptions around the eager operations; a sampler failure
must stop the evidence run instead of swallowing an incomplete sample.

- [ ] **Step 8: Publish memory profiling metadata when attaching the profiler**

In `DBAFusion.set_runtime_profiler`, add:

```python
memory_cfg = self.cfg.get("profiling", {}).get("dba_memory", {})
if hasattr(profiler, "set_metadata"):
    profiler.set_metadata("dba_memory", {
        "enabled": bool(memory_cfg.get("enabled", False)),
        "samples_per_signature": int(memory_cfg.get("samples_per_signature", 2)),
        "tracker_device": str(self.cfg["device"]["tracker"]),
    })
```

- [ ] **Step 9: Run signature, instrumentation, and profiler regressions**

Run:

```powershell
python -m pytest --import-mode=importlib tests/test_dba_signature_profiling.py tests/test_dba_memory_instrumentation.py tests/test_runtime_profiler.py -q
```

Expected: all tests pass and the existing `dba_signature` payload remains
unchanged.

- [ ] **Step 10: Commit Task 3**

```powershell
git add scripts/frontend/covisible_graph.py scripts/frontend/dbaf.py tests/test_dba_signature_profiling.py tests/test_dba_memory_instrumentation.py
git commit -m "feat: sample real DBA stage memory"
```

---

### Task 4: Implement the boundary-specific calibration model and CLI

**Files:**
- Create: `scripts/profiling/dba_memory_calibration.py`
- Create: `scripts/profiling/calibrate_dba_workspace.py`
- Create: `tests/test_dba_memory_calibration.py`
- Create: `tests/fixtures/dba_memory_samples_43x77.jsonl`

- [ ] **Step 1: Write failing tests for analytical boundary terms**

Create calls with the existing `DBACallSignature` helper and assert:

```python
from scripts.frontend.dba_bucket_manifest import DBACallSignature
from scripts.profiling.dba_memory_calibration import boundary_shadow_terms


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
    terms = boundary_shadow_terms(signature(48, 72, 12, 11), "update_aggregation_only")

    assert terms["corr_pyramid_bytes"] == 0
    assert terms["ba_inputs_bytes"] == 0
    assert terms["sampled_corr_bytes"] > 0
    assert terms["recurrent_bytes"] > 0
    assert terms["aggregation_bytes"] > 0


def test_full_corr_boundary_includes_exact_four_level_pyramid():
    call = signature(48, 72, 12, 11)
    terms = boundary_shadow_terms(call, "corr_update_aggregation")
    pixels = 43 * 77
    pyramid_pixels = sum((43 // (2**level)) * (77 // (2**level)) for level in range(4))

    assert terms["corr_pyramid_bytes"] == 48 * pixels * pyramid_pixels * 2
```

- [ ] **Step 2: Run and verify RED**

```powershell
python -m pytest --import-mode=importlib tests/test_dba_memory_calibration.py -q
```

Expected: calibration module missing.

- [ ] **Step 3: Implement exact boundary shadow terms**

Implement `boundary_shadow_terms(call, boundary)` with the same `43x77`
dimensions and dtype byte rules as Phase 0. Return named integer terms:

- `corr_pyramid_bytes` only for `corr_update_aggregation`;
- `sampled_corr_bytes = active_edges * pixels * 196 * value_bytes`;
- recurrent `net` and `inp` shadows;
- motion, delta, weight, damping, and upmask shadows;
- aggregation storage based on `source_poses`;
- zero `ba_inputs_bytes` for both approved boundaries because BA stays eager.

Reject unknown boundaries and unsupported signatures with `ValueError`.

- [ ] **Step 4: Add failing tests for deterministic holdout and validation gates**

Use `tests/fixtures/dba_memory_samples_43x77.jsonl` containing at least ten exact
signatures, two memory samples each, plus repeated `dba_signature` rows that
define observed counts. Load it through `load_memory_samples([FIXTURE])` with:

```python
from pathlib import Path

import pytest

from scripts.profiling.dba_memory_calibration import (
    calibrate,
    load_memory_samples,
)

FIXTURE = Path(__file__).parent / "fixtures" / "dba_memory_samples_43x77.jsonl"
```

Assert:

```python
from scripts.profiling.dba_memory_calibration import calibrate


def test_calibration_holds_out_every_fifth_sorted_signature():
    samples, observed_counts = load_memory_samples([FIXTURE])
    result = calibrate(samples, observed_counts, "update_aggregation_only")
    keys = sorted(observed_counts)
    assert result["validation"]["held_out_keys"] == [
        repr(keys[4]), repr(keys[9])
    ]


def test_calibration_requires_weighted_sample_coverage():
    samples, observed_counts = load_memory_samples([FIXTURE])
    with pytest.raises(ValueError, match="95%"):
        calibrate(samples[:2], observed_counts, "update_aggregation_only")


def test_calibration_adds_margin_and_never_underpredicts_holdout():
    samples, observed_counts = load_memory_samples([FIXTURE])
    result = calibrate(samples, observed_counts, "update_aggregation_only")
    assert result["validation"]["weighted_mape"] <= 0.10
    assert result["validation"]["underpredicted_signatures"] == []
    assert result["safety_margin"]["minimum_bytes"] == 64 * 1024 * 1024
```

- [ ] **Step 5: Implement loading, fitting, and validation**

Implement these public functions and deterministic key helper:

```python
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


def load_memory_samples(paths):
    samples = []
    observed_counts = Counter()
    for path in paths:
        for line in Path(path).read_text().splitlines():
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
    held_out = [key for index, key in enumerate(keys, 1) if index % 5 == 0]
    training = [key for key in keys if key not in set(held_out)]
    return training, held_out


def validate_calibration(payload):
    validation = payload.get("validation", {})
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported DBA memory calibration schema")
    if not validation.get("valid", False):
        raise ValueError("DBA memory calibration did not pass validation")
    return payload
```

Use NumPy least squares for the transient model with features
`[1, active_edges, ba_edges, source_poses, pose_window]`. Clamp negative fitted
coefficients to zero, then multiply the transient prediction by the smallest
global scale that covers every training sample. For each signature, define the
measured transient as the maximum `peak_delta_bytes` across its samples and the
pre-margin measured target as `sum(boundary_shadow_terms) + measured_transient`.
Fit and compute weighted MAPE against that pre-margin target. Then add
`max(64 * 1024**2, ceil(0.10 * modeled_workspace))`; only this final estimate is
used by the no-underprediction gate. Compute within-signature peak spread as
`(max_peak - min_peak) / max_peak`. Mark `valid` only when every design gate
passes.

The central fit must follow this exact sequence:

```python
training_keys, held_out_keys = deterministic_split(grouped_samples)
matrix = np.asarray([
    [1.0, key[0], key[1], key[2], key[3]]
    for key in training_keys
], dtype=np.float64)
targets = np.asarray([
    grouped_samples[key]["measured_transient_peak_bytes"]
    for key in training_keys
], dtype=np.float64)
coefficients = np.maximum(np.linalg.lstsq(matrix, targets, rcond=None)[0], 0.0)
raw_training = matrix @ coefficients
scale = max(
    1.0,
    max(
        target / prediction
        for target, prediction in zip(targets, raw_training)
        if prediction > 0
    ),
)
coefficients *= scale
```

If any positive target has zero prediction, fail calibration instead of
dividing by zero. Serialize signature keys with `repr(key)` and preserve the
original numeric fields in every per-signature row.

The emitted schema must contain `schema_version`, `capture_boundary`, platform
metadata, source hashes, coverage, coefficients, safety margin, validation,
and per-signature measured/predicted rows.

- [ ] **Step 6: Write the CLI test, then implement the CLI**

Test `main()` with this complete argument list and assert valid JSON and report
headings:

```python
status = main([
    "--details", str(details_a),
    "--details", str(details_b),
    "--capture-boundary", "update_aggregation_only",
    "--output-calibration", str(calibration_path),
    "--output-report", str(report_path),
])
assert status == 0
assert json.loads(calibration_path.read_text())["validation"]["valid"] is True
assert "# DBA Memory Calibration" in report_path.read_text()
```

Implement `calibrate_dba_workspace.py` as a thin wrapper around the module. It
must exit nonzero when validation is invalid and render:

- environment and source hashes;
- signature sample coverage;
- boundary term summary;
- coefficients and safety margin;
- train/held-out errors;
- unstable and underpredicted signatures.

- [ ] **Step 7: Run calibration tests**

```powershell
python -m pytest --import-mode=importlib tests/test_dba_memory_calibration.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 4**

```powershell
git add scripts/profiling/dba_memory_calibration.py scripts/profiling/calibrate_dba_workspace.py tests/test_dba_memory_calibration.py tests/fixtures/dba_memory_samples_43x77.jsonl
git commit -m "feat: calibrate boundary-specific DBA workspaces"
```

---

### Task 5: Integrate validated calibration with the bucket planner

**Files:**
- Modify: `scripts/profiling/plan_dba_buckets.py`
- Modify: `tests/test_dba_bucket_planner.py`

- [ ] **Step 1: Write failing legacy-compatibility and calibrated-estimator tests**

Add:

```python
def write_valid_calibration(tmp_path, boundary):
    samples, observed_counts = load_memory_samples([
        Path("tests/fixtures/dba_memory_samples_43x77.jsonl")
    ])
    payload = calibrate(samples, observed_counts, boundary)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def test_legacy_planner_result_is_unchanged_without_calibration():
    rows = [call(8, 12, 4, 7, 1), call(12, 18, 6, 10, 2)]
    result = plan_buckets([row["payload"] for row in rows], 0.90, 6, 0.35, 1024 * 1024**2)
    assert result["memory_model"]["kind"] == "phase0_conservative"


def test_calibrated_planner_records_boundary_and_hash(tmp_path):
    calibration = write_valid_calibration(tmp_path, "update_aggregation_only")
    details = tmp_path / "details.jsonl"
    details.write_text(json.dumps(call(48, 72, 12, 11, 1)) + "\n")
    result = main([
        "--details", str(details),
        "--memory-calibration", str(calibration),
        "--capture-boundary", "update_aggregation_only",
        "--output-manifest", str(tmp_path / "manifest.json"),
        "--output-report", str(tmp_path / "manifest.md"),
    ])
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["memory_model"]["kind"] == "calibrated"
    assert payload["memory_model"]["capture_boundary"] == "update_aggregation_only"
    assert len(payload["memory_model"]["calibration_sha256"]) == 64


def test_planner_rejects_boundary_mismatch(tmp_path):
    calibration = write_valid_calibration(tmp_path, "update_aggregation_only")
    with pytest.raises(ValueError, match="boundary"):
        load_calibrated_estimator(calibration, "corr_update_aggregation")
```

- [ ] **Step 2: Run planner tests and verify RED**

```powershell
python -m pytest --import-mode=importlib tests/test_dba_bucket_planner.py -q
```

Expected: missing calibrated estimator or `memory_model`.

- [ ] **Step 3: Inject a workspace estimator without changing selection logic**

Change `plan_buckets` to accept `workspace_estimator=None`. Resolve it once:

```python
workspace_estimator = workspace_estimator or estimate_workspace_bytes
```

Pass it to `_bucket_from_call`; the callable signature remains
`(active_edges, ba_edges, source_poses, pose_window, dtype) -> int`. Keep sorting,
fits/padding checks, and greedy coverage ranking unchanged.

Add `memory_model` to the result. Use
`{"kind": "phase0_conservative"}` by default; accept a supplied provenance
dictionary for calibrated runs.

- [ ] **Step 4: Add CLI calibration loading and strict compatibility checks**

Add both arguments:

```python
parser.add_argument("--memory-calibration")
parser.add_argument(
    "--capture-boundary",
    choices=("corr_update_aggregation", "update_aggregation_only"),
)
```

Require both or neither. Load and validate the calibration through
`dba_memory_calibration.py`; reject invalid validation state, resolution, mode,
backend, dtype support, or boundary mismatch. Build the estimator closure from
the calibrated coefficients and analytical terms. Include SHA-256 and validation
metrics in manifest provenance.

- [ ] **Step 5: Update report rendering with workspace breakdown provenance**

Add a `Memory model` section before the bucket table. For calibrated manifests,
include boundary, calibration hash, weighted MAPE, and underprediction count.
Add a top-level `workspace_breakdowns` mapping keyed by bucket name; do not add
unknown fields to `DBABucketSpec` dictionaries because the CPU-safe manifest
loader constructs that dataclass strictly. Render shadow, transient, and
safety-margin MiB columns from the top-level mapping while preserving the
existing total estimated MiB column.

- [ ] **Step 6: Run planner, manifest, and calibration tests**

```powershell
python -m pytest --import-mode=importlib tests/test_dba_bucket_planner.py tests/test_dba_bucket_manifest.py tests/test_dba_memory_calibration.py -q
```

Expected: all tests pass, including unchanged legacy fixture coverage.

- [ ] **Step 7: Commit Task 5**

```powershell
git add scripts/profiling/plan_dba_buckets.py tests/test_dba_bucket_planner.py
git commit -m "feat: plan buckets with calibrated DBA memory"
```

---

### Task 6: Add the guarded Phase 0.5 evidence driver and CUDA smoke test

**Files:**
- Create: `reports/cuda_graph_dba/run_phase05_memory_profiles.sh`
- Create: `tests/test_dba_phase05_profile_script.py`
- Create: `tests/test_dba_memory_cuda.py`

- [ ] **Step 1: Write failing content tests for the evidence driver**

Create tests asserting the new script:

- uses `set -uo pipefail`;
- refuses existing evidence and output directories;
- snapshots Git, Python, Torch/CUDA, and `tegrastats`;
- rejects residual `run.py`/`tegrastats` processes;
- runs only SmallCity-200 and complete Hotel;
- uses `344,616`, Torch update, runtime profile, DBA memory profile, and two
  samples per signature;
- uses save buffer 64 for SmallCity and 512 for Hotel;
- verifies both `dba_signature` and `dba_memory_sample` records;
- invokes calibration for both boundaries and the planner at 1024 MiB;
- contains no `rm -rf` or `rm -f`.

Run and expect failure because the script does not exist.

- [ ] **Step 2: Implement the guarded driver**

Use defaults:

```bash
EVIDENCE_DIR="${VINGS_PHASE05_EVIDENCE_DIR:-$shared_root/reports/cuda_graph_dba/phase05_20260714}"
output_parent="${VINGS_PHASE05_OUTPUT_DIR:-$shared_root/output/cuda_graph_dba_phase05_20260714}"
```

Reuse the Phase 0 snapshot, residual-process, `tegrastats`, single-run-directory,
and cleanup functions. Do not source or mutate the Phase 0 evidence directory.

Run SmallCity-200 with its Phase 0 options plus:

```bash
--profile-runtime \
--profile-dba-memory \
--profile-dba-memory-samples-per-signature 2
```

Run complete Hotel with the same memory options and
`--frontend-save-buffer 512`. After each run require nonempty JSONL and both
detail kinds. Compute sampled weighted signature coverage with a read-only
Python check and require at least 0.95.

Run `calibrate_dba_workspace.py` separately for both boundaries. Then run
`plan_dba_buckets.py` only for each valid calibration with target 0.90, max
buckets 6, padding 0.35, and workspace 1024 MiB. Keep invalid calibration
artifacts, mark that boundary NO-GO, and do not generate a misleading manifest.

- [ ] **Step 3: Run content and shell syntax tests**

```powershell
python -m pytest --import-mode=importlib tests/test_dba_phase05_profile_script.py -q
ssh -i C:\Users\x\.ssh\id_ed25519_jetson_codex -p 2222 jetson@10.201.133.102 "cd /home/jetson/VINGS-Mono && bash -n reports/cuda_graph_dba/run_phase05_memory_profiles.sh"
```

Expected: tests pass and `bash -n` exits 0.

- [ ] **Step 4: Write an executable Jetson CUDA smoke test**

Create `tests/test_dba_memory_cuda.py` as `unittest.TestCase`. Skip unless CUDA
is available. Allocate a baseline tensor, begin a real sample, allocate a larger
temporary tensor, mark `corr_sample_complete`, and assert:

- device total/free are positive;
- peak delta is positive;
- retained and peak fields are integers;
- stage ordering is preserved;
- `TorchCUDAMemoryBackend` accepts `cuda:0`.

Include `if __name__ == "__main__": unittest.main()` because the Jetson Conda
environment does not include pytest.

- [ ] **Step 5: Run the CUDA smoke test on Jetson**

```powershell
ssh -i C:\Users\x\.ssh\id_ed25519_jetson_codex -p 2222 jetson@10.201.133.102 "cd /home/jetson/VINGS-Mono && export PYTHONPATH=/home/jetson/VINGS-Mono:/home/jetson/VINGS-Mono/scripts && /home/jetson/miniconda3/envs/vings_jetson/bin/python tests/test_dba_memory_cuda.py"
```

Expected: all CUDA smoke tests pass with no residual process.

- [ ] **Step 6: Run the complete CPU-safe Phase 0/0.5 suite**

Run with the repository `tests` package injected if the Windows user-site
package shadows it:

```powershell
python -m pytest --import-mode=importlib tests/test_runtime_profiler.py tests/test_dba_signature_profiling.py tests/test_dba_memory.py tests/test_dba_memory_instrumentation.py tests/test_dba_memory_calibration.py tests/test_dba_bucket_manifest.py tests/test_dba_bucket_planner.py tests/test_dba_phase05_profile_script.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 6**

```powershell
git add -f reports/cuda_graph_dba/run_phase05_memory_profiles.sh
git add tests/test_dba_phase05_profile_script.py tests/test_dba_memory_cuda.py
git commit -m "test: add guarded DBA memory calibration runs"
```

---

### Task 7: Run Jetson evidence and record the revised gate

**Files:**
- Generate: `reports/cuda_graph_dba/phase05_corr_memory_calibration.json`
- Generate: `reports/cuda_graph_dba/phase05_corr_memory_calibration.md`
- Generate: `reports/cuda_graph_dba/phase05_update_memory_calibration.json`
- Generate: `reports/cuda_graph_dba/phase05_update_memory_calibration.md`
- Generate when the matching calibration validates: `reports/cuda_graph_dba/phase05_corr_bucket_manifest.json`
- Generate when the matching calibration validates: `reports/cuda_graph_dba/phase05_corr_bucket_manifest.md`
- Generate when the matching calibration validates: `reports/cuda_graph_dba/phase05_update_bucket_manifest.json`
- Generate when the matching calibration validates: `reports/cuda_graph_dba/phase05_update_bucket_manifest.md`
- Create: `reports/cuda_graph_dba/phase05_memory_decision.md`

- [ ] **Step 1: Verify hardware preconditions and clean process state**

On Jetson, verify CUDA availability, weights, datasets, LightGlue ONNX files,
and no `run.py` or `tegrastats` process. Record the exact HEAD. Do not run when
the evidence or output directory already exists.

- [ ] **Step 2: Run the guarded evidence driver**

```powershell
ssh -i C:\Users\x\.ssh\id_ed25519_jetson_codex -p 2222 jetson@10.201.133.102 "cd /home/jetson/VINGS-Mono && export PATH=/home/jetson/miniconda3/envs/vings_jetson/bin:\$PATH && export PYTHONPATH=/home/jetson/VINGS-Mono:/home/jetson/VINGS-Mono/scripts && bash reports/cuda_graph_dba/run_phase05_memory_profiles.sh"
```

Expected: both tracking runs exit 0, Hotel reaches 405/405, all configured
signature samples are present, and no residual process remains. Calibration may
validly conclude that one or both models fail; preserve those artifacts, write
NO-GO, and stop before graph runtime work rather than treating the scientific
result as an instrumentation failure.

- [ ] **Step 3: Copy only small generated artifacts to stable report paths**

Use the generated evidence outputs as sources. Do not copy raw JSONL, run logs,
images, or `tegrastats` into tracked paths. Stable files must include source
hashes and exact evidence paths so results remain auditable.

- [ ] **Step 4: Write the Phase 0.5 decision report**

Populate these sections with measured values:

```markdown
# DBA Memory Calibration Phase 0.5 Decision

## Environment and source commits
## Evidence runs and sample coverage
## Resident eager state
## Measured transient peaks
## Calibration validation
## Conservative versus calibrated boundaries
## Revised bucket coverage and padding
## Capture-time CUDA headroom
## Unsupported, unstable, or underpredicted signatures
## Phase 1 decision
```

The decision is GO only if the `update_aggregation_only` model is valid, sampled
weighted coverage is at least 95%, held-out weighted MAPE is at most 10%, no
held-out signature is underpredicted, repeated peak spread is at most 15%, and
the revised bucket plan covers at least 90% of Phase 0 calls with no more than
six buckets under both the 1024 MiB and 25%-free-CUDA-memory limits. Otherwise
record NO-GO. Do not use the full-correlation boundary to rescue a failed update-
only validation.

- [ ] **Step 5: Run final verification before claiming completion**

Run fresh:

```powershell
python -m json.tool reports\cuda_graph_dba\phase05_corr_memory_calibration.json *> $null
python -m json.tool reports\cuda_graph_dba\phase05_update_memory_calibration.json *> $null
if (Test-Path reports\cuda_graph_dba\phase05_corr_bucket_manifest.json) { python -m json.tool reports\cuda_graph_dba\phase05_corr_bucket_manifest.json *> $null }
if (Test-Path reports\cuda_graph_dba\phase05_update_bucket_manifest.json) { python -m json.tool reports\cuda_graph_dba\phase05_update_bucket_manifest.json *> $null }
python -m pytest --import-mode=importlib tests/test_runtime_profiler.py tests/test_dba_signature_profiling.py tests/test_dba_memory.py tests/test_dba_memory_instrumentation.py tests/test_dba_memory_calibration.py tests/test_dba_bucket_manifest.py tests/test_dba_bucket_planner.py tests/test_dba_phase05_profile_script.py -q
git diff --check
git status --short --untracked-files=no
```

Also rerun `tests/test_dba_memory_cuda.py` on Jetson and confirm
`pgrep -af '[s]cripts/run.py|[t]egrastats'` returns no process.

- [ ] **Step 6: Review repository scope**

Expected tracked changes are only the files named in this plan and the small
Phase 0.5 artifacts. Preserve and exclude the pre-existing dirty
`submodules/metric_modules` state and any unrelated commits that appear in the
shared workspace.

- [ ] **Step 7: Commit Task 7**

```powershell
git add scripts/run.py scripts/frontend/covisible_graph.py scripts/frontend/dbaf.py
git add scripts/profiling/dba_memory.py scripts/profiling/dba_memory_calibration.py scripts/profiling/calibrate_dba_workspace.py scripts/profiling/plan_dba_buckets.py
git add tests/test_jetson_runtime_overrides.py tests/test_dba_signature_profiling.py tests/test_dba_memory.py tests/test_dba_memory_instrumentation.py tests/test_dba_memory_calibration.py tests/test_dba_bucket_planner.py tests/test_dba_phase05_profile_script.py tests/test_dba_memory_cuda.py tests/fixtures/dba_memory_samples_43x77.jsonl
$requiredReports = @(
  "reports/cuda_graph_dba/run_phase05_memory_profiles.sh",
  "reports/cuda_graph_dba/phase05_corr_memory_calibration.json",
  "reports/cuda_graph_dba/phase05_corr_memory_calibration.md",
  "reports/cuda_graph_dba/phase05_update_memory_calibration.json",
  "reports/cuda_graph_dba/phase05_update_memory_calibration.md",
  "reports/cuda_graph_dba/phase05_memory_decision.md"
)
git add -f -- $requiredReports
foreach ($optional in @(
  "reports/cuda_graph_dba/phase05_corr_bucket_manifest.json",
  "reports/cuda_graph_dba/phase05_corr_bucket_manifest.md",
  "reports/cuda_graph_dba/phase05_update_bucket_manifest.json",
  "reports/cuda_graph_dba/phase05_update_bucket_manifest.md"
)) {
  if (Test-Path $optional) { git add -f -- $optional }
}
git commit -m "perf: calibrate real DBA capture workspace"
```

Expected: the commit contains Phase 0.5 reusable instrumentation/calibration
code and small evidence artifacts only. If earlier task commits already contain
the code/test files, this final commit naturally contains only generated reports.

---

## Stop condition and follow-up

Stop after the Phase 0.5 decision commit. A GO result authorizes writing a new,
separate Phase 1 design/plan for an `update_aggregation_only` CUDA Graph runtime;
it does not authorize implementing that runtime in this plan. A NO-GO result
keeps active-edge capacity unchanged and requires a separate design review
before any `max_factors` experiment.
