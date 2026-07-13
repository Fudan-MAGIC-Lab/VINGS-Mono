# CUDA Graph / Bucketed DBA Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add low-overhead DBA call-signature recording, generate a deterministic data-driven `43x77` bucket manifest from fresh Jetson runs, and produce the evidence gate required before implementing CUDA Graph replay.

**Architecture:** Keep `CovisibleGraph.update()` numerically unchanged and emit one structured detail record per eager DBA update through the existing runtime profiler. A CPU-safe planner aggregates SmallCity-50, SmallCity-200, and `344x616` Hotel signatures, estimates fixed workspace size, and greedily selects a bounded bucket set by uncovered-call coverage per byte. Phase 0 ends with a reviewed manifest and capture-boundary report; Graph A/B implementation is a separate plan based on those concrete signatures.

**Tech Stack:** Python 3.8, PyTorch 2.1, CUDA 11.4, Jetson Linux, pytest/unittest, JSON Lines, Bash, `tegrastats`.

**Design reference:** `docs/superpowers/specs/2026-07-13-cuda-graph-bucketed-dba-design.md`

**Worktree constraint:** Work in the current `codex/local-snapshot-2026-06-04` tree and preserve unrelated user changes. Stage and commit only files named by the active task. Do not reset, clean, delete, or overwrite prior evidence.

**Scope boundary:** This plan implements Phase 0 only. Phase 1 requires a follow-up implementation plan after the generated manifest identifies real bucket capacities and the evidence report confirms memory headroom. Phase 3 GPU BA remains conditional on the later Phase 1 online gate.

---

## File structure

- Create `scripts/frontend/dba_bucket_manifest.py`: CPU-safe signature, bucket, manifest, padding, support, and selection types.
- Modify `scripts/profiling/runtime_profiler.py`: append structured detail records and write `runtime_profile_details.jsonl`.
- Modify `scripts/frontend/covisible_graph.py`: derive and record one DBA signature without changing the eager computation.
- Create `scripts/profiling/plan_dba_buckets.py`: aggregate signature files, estimate workspace, select buckets, and write manifest/report files.
- Create `reports/cuda_graph_dba/run_phase0_dba_profiles.sh`: guarded Jetson evidence driver for the three required datasets.
- Create `tests/test_dba_bucket_manifest.py`: pure manifest and selector tests.
- Modify `tests/test_runtime_profiler.py`: detail-record persistence tests.
- Create `tests/test_dba_signature_profiling.py`: isolated `CovisibleGraph` signature tests.
- Create `tests/test_dba_bucket_planner.py`: deterministic planner tests.
- Create `tests/test_dba_phase0_profile_script.py`: evidence-driver content and safety tests.
- Create after Jetson runs `reports/cuda_graph_dba/phase0_bucket_decision.md`: measured coverage, memory, unsupported signatures, and Phase 1 go/no-go decision.

---

### Task 1: Add structured runtime-profiler detail records

**Files:**
- Modify: `scripts/profiling/runtime_profiler.py`
- Modify: `tests/test_runtime_profiler.py`

- [ ] **Step 1: Write the failing JSONL persistence test**

Append this test to `tests/test_runtime_profiler.py`:

```python
import json


def test_runtime_profiler_writes_structured_details(tmp_path):
    profiler = RuntimeProfiler(enabled=True, output_dir=tmp_path)

    profiler.record_detail(
        "dba_signature",
        {"active_edges": 12, "feature_shape": [43, 77]},
        frame_idx=9,
    )
    profiler.write_reports()

    detail_path = tmp_path / "runtime_profile_details.jsonl"
    rows = [json.loads(line) for line in detail_path.read_text().splitlines()]
    assert rows == [
        {
            "frame_idx": 9,
            "kind": "dba_signature",
            "payload": {"active_edges": 12, "feature_shape": [43, 77]},
        }
    ]
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest --import-mode=importlib tests/test_runtime_profiler.py::test_runtime_profiler_writes_structured_details -q
```

Expected: FAIL with `AttributeError: 'RuntimeProfiler' object has no attribute 'record_detail'`.

- [ ] **Step 3: Implement structured detail recording**

Make these exact additions in `scripts/profiling/runtime_profiler.py`:

```python
class RuntimeProfiler:
    def __init__(self, enabled=False, output_dir=None, clock=None, sync_callback=None):
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.clock = clock or time.perf_counter
        self.sync_callback = sync_callback
        self.events = []
        self.details = []
        self.metadata = {}

    def record_detail(self, kind, payload, frame_idx=None):
        if not self.enabled:
            return
        self.details.append(
            {
                "kind": str(kind),
                "frame_idx": frame_idx,
                "payload": dict(payload),
            }
        )
```

Add this block in `write_reports()` immediately after writing
`runtime_profile_metadata.json`:

```python
        with (self.output_dir / "runtime_profile_details.jsonl").open("w") as handle:
            for detail in self.details:
                handle.write(json.dumps(detail, sort_keys=True) + "\n")
```

- [ ] **Step 4: Run focused and existing profiler tests**

Run:

```bash
python -m pytest --import-mode=importlib tests/test_runtime_profiler.py tests/test_dbaf_runtime_profiling.py tests/test_dbaf_frontend_runtime_profiling.py -q
```

Expected: PASS; existing timing CSV/Markdown behavior remains unchanged.

- [ ] **Step 5: Commit the profiler extension**

```bash
git add scripts/profiling/runtime_profiler.py tests/test_runtime_profiler.py
git commit -m "feat: record structured runtime profile details"
```

---

### Task 2: Define the CPU-safe DBA signature and bucket contract

**Files:**
- Create: `scripts/frontend/dba_bucket_manifest.py`
- Create: `tests/test_dba_bucket_manifest.py`

- [ ] **Step 1: Write failing signature, padding, and selector tests**

Create `tests/test_dba_bucket_manifest.py`:

```python
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
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest --import-mode=importlib tests/test_dba_bucket_manifest.py -q
```

Expected: collection ERROR because `scripts.frontend.dba_bucket_manifest` does not exist.

- [ ] **Step 3: Implement the manifest module**

Create `scripts/frontend/dba_bucket_manifest.py`:

```python
import json
from dataclasses import asdict, dataclass
from pathlib import Path


SUPPORTED_FRONTEND_IMAGE_SIZE = (344, 616)
SUPPORTED_FEATURE_SHAPE = (43, 77)
SUPPORTED_MODE = "vo"
SUPPORTED_BACKEND = "torch"


@dataclass(frozen=True)
class DBACallSignature:
    active_edges: int
    ba_edges: int
    source_poses: int
    pose_window: int
    use_inactive: bool
    upsample: bool
    dtype: str
    feature_shape: tuple
    frontend_image_size: tuple
    mode: str
    backend: str

    def support_error(self):
        if tuple(self.frontend_image_size) != SUPPORTED_FRONTEND_IMAGE_SIZE:
            return "frontend_image_size"
        if tuple(self.feature_shape) != SUPPORTED_FEATURE_SHAPE:
            return "feature_shape"
        if self.mode != SUPPORTED_MODE:
            return "mode"
        if self.backend != SUPPORTED_BACKEND:
            return "backend"
        if self.dtype not in {"float16", "float32"}:
            return "dtype"
        return None

    def to_dict(self):
        result = asdict(self)
        result["feature_shape"] = list(self.feature_shape)
        result["frontend_image_size"] = list(self.frontend_image_size)
        return result


@dataclass(frozen=True)
class DBABucketSpec:
    name: str
    active_edges: int
    ba_edges: int
    source_poses: int
    pose_window: int
    use_inactive: bool
    upsample: bool
    dtype: str
    max_padding_ratio: float
    estimated_workspace_bytes: int

    def padding_ratio(self, call):
        capacity = (
            self.active_edges
            + self.ba_edges
            + self.source_poses
            + self.pose_window
        )
        padding = (
            self.active_edges - call.active_edges
            + self.ba_edges - call.ba_edges
            + self.source_poses - call.source_poses
            + self.pose_window - call.pose_window
        )
        return padding / capacity

    def fits(self, call):
        if call.support_error() is not None:
            return False
        if (
            self.use_inactive != call.use_inactive
            or self.upsample != call.upsample
            or self.dtype != call.dtype
        ):
            return False
        if (
            call.active_edges > self.active_edges
            or call.ba_edges > self.ba_edges
            or call.source_poses > self.source_poses
            or call.pose_window > self.pose_window
        ):
            return False
        return self.padding_ratio(call) <= self.max_padding_ratio


@dataclass(frozen=True)
class DBABucketManifest:
    schema_version: int
    frontend_image_size: tuple
    feature_shape: tuple
    mode: str
    backend: str
    buckets: tuple

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        image_size = tuple(payload["frontend_image_size"])
        feature_shape = tuple(payload["feature_shape"])
        if (
            image_size != SUPPORTED_FRONTEND_IMAGE_SIZE
            or feature_shape != SUPPORTED_FEATURE_SHAPE
        ):
            raise ValueError("DBA bucket manifest must target 344x616 -> 43x77")
        if payload["mode"] != SUPPORTED_MODE or payload["backend"] != SUPPORTED_BACKEND:
            raise ValueError("DBA bucket manifest must target VO with the torch backend")
        buckets = tuple(DBABucketSpec(**item) for item in payload["buckets"])
        return cls(
            schema_version=int(payload["schema_version"]),
            frontend_image_size=image_size,
            feature_shape=feature_shape,
            mode=payload["mode"],
            backend=payload["backend"],
            buckets=buckets,
        )

    def select(self, call):
        candidates = [bucket for bucket in self.buckets if bucket.fits(call)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda bucket: (
                bucket.estimated_workspace_bytes,
                bucket.active_edges,
                bucket.ba_edges,
                bucket.name,
            ),
        )
```

- [ ] **Step 4: Run manifest tests**

Run:

```bash
python -m pytest --import-mode=importlib tests/test_dba_bucket_manifest.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the pure bucket contract**

```bash
git add scripts/frontend/dba_bucket_manifest.py tests/test_dba_bucket_manifest.py
git commit -m "feat: define DBA bucket manifest contract"
```

---

### Task 3: Record one exact signature per eager `CovisibleGraph.update()`

**Files:**
- Modify: `scripts/frontend/covisible_graph.py`
- Create: `tests/test_dba_signature_profiling.py`
- Modify: `tests/test_droid_fine_profiling.py`

- [ ] **Step 1: Write the failing signature-record test**

Create `tests/test_dba_signature_profiling.py` with this complete header and
test:

```python
import pathlib
import sys
import types

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from frontend.covisible_graph import CovisibleGraph


class DetailProfiler:
    def __init__(self):
        self.details = []

    def record_detail(self, kind, payload, frame_idx=None):
        self.details.append((kind, payload, frame_idx))


def test_record_dba_signature_counts_selected_inactive_edges():
    graph = CovisibleGraph.__new__(CovisibleGraph)
    graph.ii = torch.tensor([1, 1, 2, 3], device="cpu")
    graph.jj = torch.tensor([2, 3, 3, 4], device="cpu")
    graph.ii_inac = torch.tensor([0, 1, 4], device="cpu")
    graph.jj_inac = torch.tensor([1, 2, 5], device="cpu")
    graph.inac_range = 3
    graph.ht = 43
    graph.wd = 77
    graph.upsample = True
    graph.video = types.SimpleNamespace(
        ht=344,
        wd=616,
        cfg={"mode": "vo"},
    )
    graph.update_op = types.SimpleNamespace(core_backend=None)
    graph.net = torch.zeros(1, 4, 128, 43, 77, dtype=torch.float16)
    graph.profiler = DetailProfiler()
    graph.profiler_frame_idx = 12

    graph._record_dba_signature(
        t0=2,
        observed_t1=6,
        use_inactive=True,
        ba_edges=6,
    )

    kind, payload, frame_idx = graph.profiler.details[0]
    assert kind == "dba_signature"
    assert frame_idx == 12
    assert payload["active_edges"] == 4
    assert payload["ba_edges"] == 6
    assert payload["source_poses"] == 3
    assert payload["pose_window"] == 4
    assert payload["frontend_image_size"] == [344, 616]
    assert payload["feature_shape"] == [43, 77]
    assert payload["backend"] == "torch"
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest --import-mode=importlib tests/test_dba_signature_profiling.py -q
```

Expected: FAIL because `_record_dba_signature` is missing.

- [ ] **Step 3: Implement signature derivation and recording**

Import the signature type in `scripts/frontend/covisible_graph.py`:

```python
from frontend.dba_bucket_manifest import DBACallSignature
```

Add these methods to `CovisibleGraph`:

```python
    def _update_backend_name(self):
        backend = getattr(self.update_op, "core_backend", None)
        if backend is None:
            return "torch"
        return getattr(backend, "actual_backend", "unknown")

    def _record_dba_signature(self, t0, observed_t1, use_inactive, ba_edges):
        profiler = getattr(self, "profiler", None)
        if profiler is None or not hasattr(profiler, "record_detail"):
            return

        dtype = str(self.net.dtype).replace("torch.", "")
        call = DBACallSignature(
            active_edges=int(self.ii.numel()),
            ba_edges=int(ba_edges),
            source_poses=int(torch.unique(self.ii).numel()),
            pose_window=int(observed_t1 - t0),
            use_inactive=bool(use_inactive),
            upsample=bool(self.upsample),
            dtype=dtype,
            feature_shape=(int(self.ht), int(self.wd)),
            frontend_image_size=(int(self.video.ht), int(self.video.wd)),
            mode=str(self.video.cfg["mode"]),
            backend=self._update_backend_name(),
        )
        payload = call.to_dict()
        if torch.cuda.is_available():
            payload["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
            payload["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
        profiler.record_detail(
            "dba_signature",
            payload,
            frame_idx=getattr(self, "profiler_frame_idx", None),
        )
```

Keep the existing `t0` resolution in its current location. After the existing
`use_inactive` branch has constructed the final local `ii` and `jj`, observe the
effective `t1` for profiling without changing the `t1` argument passed to BA:

```python
        observed_t1 = t1
        if observed_t1 is None:
            observed_t1 = max(ii.max().item(), jj.max().item()) + 1
        self._record_dba_signature(
            t0=t0,
            observed_t1=observed_t1,
            use_inactive=use_inactive,
            ba_edges=int(ii.numel()),
        )
```

Continue to call `self.video.ba(..., t0, t1, ...)` with the original `t1`
value. Do not move, rename, or change any tensor operation in the eager path.

- [ ] **Step 4: Verify signature and eager profiling tests**

Run:

```bash
python -m pytest --import-mode=importlib tests/test_dba_signature_profiling.py tests/test_droid_fine_profiling.py tests/test_dbaf_frontend_runtime_profiling.py -q
```

Expected: PASS; the existing ordered stage list is unchanged and one additional
detail record is produced only when the profiler supports it.

- [ ] **Step 5: Run a no-profiler behavior regression**

Run:

```bash
python -m pytest --import-mode=importlib tests/test_dbaf_frontend_rollup.py tests/test_dbaf_frontend_iteration_config.py tests/test_dbaf_runtime_profiling.py -q
```

Expected: PASS, demonstrating the disabled path remains inert.

- [ ] **Step 6: Commit signature recording**

```bash
git add scripts/frontend/covisible_graph.py tests/test_dba_signature_profiling.py tests/test_droid_fine_profiling.py
git commit -m "feat: profile dynamic DBA call signatures"
```

---

### Task 4: Build the deterministic offline bucket planner

**Files:**
- Create: `scripts/profiling/plan_dba_buckets.py`
- Create: `tests/test_dba_bucket_planner.py`

- [ ] **Step 1: Write failing planner tests**

Create `tests/test_dba_bucket_planner.py`:

```python
import json

from scripts.profiling.plan_dba_buckets import load_calls, plan_buckets


def call(active, ba, sources, window, count_frame, dtype="float16"):
    return {
        "kind": "dba_signature",
        "frame_idx": count_frame,
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
    calls = []
    calls.extend([call(8, 12, 4, 7, frame) for frame in range(70)])
    calls.extend([call(12, 18, 6, 10, frame) for frame in range(70, 95)])
    calls.extend([call(24, 32, 10, 16, frame) for frame in range(95, 100)])
    signatures = [row["payload"] for row in calls]

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
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest --import-mode=importlib tests/test_dba_bucket_planner.py -q
```

Expected: collection ERROR because `scripts.profiling.plan_dba_buckets` is missing.

- [ ] **Step 3: Implement input loading and workspace estimation**

Create `scripts/profiling/plan_dba_buckets.py` with these public functions:

```python
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from frontend.dba_bucket_manifest import DBABucketSpec, DBACallSignature


def load_calls(paths):
    calls = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row.get("kind") != "dba_signature":
                continue
            payload = dict(row["payload"])
            payload["feature_shape"] = tuple(payload["feature_shape"])
            payload["frontend_image_size"] = tuple(payload["frontend_image_size"])
            calls.append(DBACallSignature(**{
                key: payload[key]
                for key in DBACallSignature.__dataclass_fields__
            }))
    return calls


def estimate_workspace_bytes(active_edges, ba_edges, source_poses, pose_window, dtype):
    height, width = 43, 77
    pixels = height * width
    value_bytes = 2 if dtype == "float16" else 4
    pyramid_pixels = sum(
        (height // (2 ** level)) * (width // (2 ** level))
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
```

- [ ] **Step 4: Implement deterministic greedy selection**

Continue the same file with:

```python
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


def _bucket_from_call(call, max_padding_ratio):
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
        estimated_workspace_bytes=estimate_workspace_bytes(
            call.active_edges,
            call.ba_edges,
            call.source_poses,
            call.pose_window,
            call.dtype,
        ),
    )


def plan_buckets(
    raw_calls,
    target_coverage,
    max_buckets,
    max_padding_ratio,
    max_workspace_bytes,
):
    calls = [
        item if isinstance(item, DBACallSignature) else DBACallSignature(
            **{
                **item,
                "feature_shape": tuple(item["feature_shape"]),
                "frontend_image_size": tuple(item["frontend_image_size"]),
            }
        )
        for item in raw_calls
    ]
    supported = [call for call in calls if call.support_error() is None]
    counts = Counter(_call_key(call) for call in supported)
    unique = {key: call for key, call in zip(map(_call_key, supported), supported)}
    candidates = [
        _bucket_from_call(unique[key], max_padding_ratio)
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
            covered = {index for index in uncovered if bucket.fits(supported[index])}
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

    selected.sort(key=lambda bucket: (bucket.estimated_workspace_bytes, bucket.name))
    bucket_rows = [bucket.__dict__ for bucket in selected]
    covered_calls = len(supported) - len(uncovered)
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
        "buckets": bucket_rows,
        "observed_signature_counts": {
            repr(key): count for key, count in sorted(counts.items())
        },
    }
```

Add a CLI that accepts repeatable `--details`, `--output-manifest`, and
`--output-report`, plus numeric options matching the function arguments. Write
the manifest as sorted, indented JSON. Write a Markdown report containing total,
supported, covered, coverage, unsupported, per-bucket capacities, padding limit,
and estimated bytes.

- [ ] **Step 5: Run planner and manifest tests**

Run:

```bash
python -m pytest --import-mode=importlib tests/test_dba_bucket_planner.py tests/test_dba_bucket_manifest.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Exercise the CLI with a test fixture**

Run:

```bash
python scripts/profiling/plan_dba_buckets.py \
  --details tests/fixtures/dba_signatures_43x77.jsonl \
  --output-manifest /tmp/dba_bucket_manifest.json \
  --output-report /tmp/dba_bucket_manifest.md \
  --target-coverage 0.90 \
  --max-buckets 6 \
  --max-padding-ratio 0.35 \
  --max-workspace-mb 1024
```

Before running, create `tests/fixtures/dba_signatures_43x77.jsonl` with these
three JSON Lines:

```jsonl
{"frame_idx": 1, "kind": "dba_signature", "payload": {"active_edges": 8, "ba_edges": 12, "source_poses": 4, "pose_window": 7, "use_inactive": true, "upsample": true, "dtype": "float16", "feature_shape": [43, 77], "frontend_image_size": [344, 616], "mode": "vo", "backend": "torch"}}
{"frame_idx": 2, "kind": "dba_signature", "payload": {"active_edges": 12, "ba_edges": 18, "source_poses": 6, "pose_window": 10, "use_inactive": true, "upsample": true, "dtype": "float16", "feature_shape": [43, 77], "frontend_image_size": [344, 616], "mode": "vo", "backend": "torch"}}
{"frame_idx": 3, "kind": "dba_signature", "payload": {"active_edges": 24, "ba_edges": 32, "source_poses": 10, "pose_window": 16, "use_inactive": true, "upsample": true, "dtype": "float16", "feature_shape": [43, 77], "frontend_image_size": [344, 616], "mode": "vo", "backend": "torch"}}
```

Expected: exit 0; both output files exist; loading the JSON with
`DBABucketManifest.load()` succeeds.

- [ ] **Step 7: Commit the planner**

```bash
git add scripts/profiling/plan_dba_buckets.py tests/test_dba_bucket_planner.py tests/fixtures/dba_signatures_43x77.jsonl
git commit -m "feat: plan data-driven DBA CUDA Graph buckets"
```

---

### Task 5: Add a guarded Phase 0 Jetson evidence driver

**Files:**
- Create: `reports/cuda_graph_dba/run_phase0_dba_profiles.sh`
- Create: `tests/test_dba_phase0_profile_script.py`

- [ ] **Step 1: Write failing driver-content tests**

Create `tests/test_dba_phase0_profile_script.py`:

```python
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "reports/cuda_graph_dba/run_phase0_dba_profiles.sh"


def test_phase0_driver_is_guarded_and_uses_43x77_inputs():
    content = SCRIPT.read_text()
    for expected in (
        "set -uo pipefail",
        "--frontend-image-size 344,616",
        "--profile-runtime",
        "--droid-update-backend torch",
        "data/smallcity_subset_50/small_city",
        "data/smallcity_subset_200/small_city",
        "configs/rtg/hotel.yaml",
        "runtime_profile_details.jsonl",
        "tegrastats --interval 1000",
        "git status --short",
        "git submodule status --recursive",
        "pgrep -af '[s]cripts/run.py|[t]egrastats'",
    ):
        assert expected in content
    assert "rm -rf" not in content
    assert "rm -f" not in content


def test_phase0_driver_refuses_to_overwrite_evidence():
    content = SCRIPT.read_text()
    assert 'if [ -e "$EVIDENCE_DIR" ]' in content
    assert 'if [ -e "$output_parent" ]' in content
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest --import-mode=importlib tests/test_dba_phase0_profile_script.py -q
```

Expected: FAIL because the driver does not exist.

- [ ] **Step 3: Implement the guarded driver**

Create `reports/cuda_graph_dba/run_phase0_dba_profiles.sh` using the guard,
environment snapshot, `tegrastats`, cleanup trap, and unique-output structure
from `reports/tensorrt_fp16/run_droid_update_smallcity50_pair.sh`.

The three instrumented dataset commands must be exactly scoped as follows. Add
a fourth SmallCity-50 control with the same first command, a unique
`cuda_graph_dba_phase0_smallcity50_control` prefix/output directory, and no
`--profile-runtime`; this control detects profiling-induced behavior changes.

```bash
python scripts/run.py configs/hierarchical/smallcity.yaml \
  --prefix cuda_graph_dba_phase0_smallcity50 \
  --dataset-root data/smallcity_subset_50/small_city \
  --output-dir "$output_parent/smallcity50" \
  --frontend-weight ckpts/droid.pth \
  --frontend-image-size 344,616 \
  --frontend-save-buffer 64 \
  --droid-update-backend torch \
  --profile-runtime \
  --export-eval --export-eval-interval 1 \
  --no-vis --skip-save-ply \
  --training-iters 10 \
  --enable-mapping-budget --enable-jetson-pruning --enable-pixel-budget \
  --enable-metric-depth-schedule --metric-depth-mode keyframe \
  --metric-depth-keyframe-min-interval 3 \
  --metric-depth-keyframe-force-interval 10 \
  --metric-depth-high-motion-ratio 3.0 \
  --metric-depth-scale 0.75 \
  --enable-jetson-motion-gate --motion-gate-backend vpi_cpp \
  --motion-gate-threshold 12.0 --motion-gate-force-interval 8 \
  --motion-gate-resize 96,160

python scripts/run.py configs/hierarchical/smallcity.yaml \
  --prefix cuda_graph_dba_phase0_smallcity200 \
  --dataset-root data/smallcity_subset_200/small_city \
  --output-dir "$output_parent/smallcity200" \
  --frontend-weight ckpts/droid.pth \
  --frontend-image-size 344,616 \
  --frontend-save-buffer 64 \
  --droid-update-backend torch \
  --profile-runtime \
  --export-eval --export-eval-interval 1 \
  --no-vis --skip-save-ply \
  --training-iters 10 \
  --enable-mapping-budget --enable-jetson-pruning --enable-pixel-budget \
  --enable-metric-depth-schedule --metric-depth-mode keyframe \
  --metric-depth-keyframe-min-interval 3 \
  --metric-depth-keyframe-force-interval 10 \
  --metric-depth-high-motion-ratio 3.0 \
  --metric-depth-scale 0.75 \
  --enable-jetson-motion-gate --motion-gate-backend vpi_cpp \
  --motion-gate-threshold 12.0 --motion-gate-force-interval 8 \
  --motion-gate-resize 96,160

python scripts/run.py configs/rtg/hotel.yaml \
  --prefix cuda_graph_dba_phase0_hotel344 \
  --dataset-root data/hotel \
  --output-dir "$output_parent/hotel344" \
  --frontend-weight ckpts/droid.pth \
  --lightglue-weight-dir ckpts/lightglue \
  --loop-onnx-provider cpu \
  --frontend-image-size 344,616 \
  --frontend-save-buffer 64 \
  --frontend-iters1 3 --frontend-iters2 1 \
  --droid-update-backend torch \
  --profile-runtime \
  --export-eval --export-eval-interval 1 \
  --no-vis --skip-save-ply \
  --training-iters 30 \
  --adaptive-runtime \
  --enable-mapping-budget --enable-jetson-pruning --enable-pixel-budget \
  --enable-metric-depth-schedule --metric-depth-mode keyframe \
  --metric-depth-keyframe-min-interval 3 \
  --metric-depth-keyframe-force-interval 10 \
  --metric-depth-high-motion-ratio 3.0 \
  --metric-depth-scale 0.75
```

For each run, require exit status 0 and locate exactly one run directory. Copy
its path into `run_dir_<label>.txt`; verify
`runtime_profile_details.jsonl` exists and contains at least one
`"kind": "dba_signature"` row for the three instrumented runs. Verify the
control does not require that file. Stop on the first failed run without
deleting evidence.

- [ ] **Step 4: Verify script tests and syntax**

Run:

```bash
python -m pytest --import-mode=importlib tests/test_dba_phase0_profile_script.py -q
bash -n reports/cuda_graph_dba/run_phase0_dba_profiles.sh
```

Expected: PASS and `bash -n` exit 0.

- [ ] **Step 5: Commit the evidence driver**

```bash
git add reports/cuda_graph_dba/run_phase0_dba_profiles.sh tests/test_dba_phase0_profile_script.py
git commit -m "test: add guarded DBA bucket profiling runs"
```

---

### Task 6: Run Phase 0 and generate the evidence-backed manifest

**Files:**
- Generate: `reports/cuda_graph_dba/phase0_20260713/`
- Generate: `reports/cuda_graph_dba/phase0_bucket_manifest.json`
- Generate: `reports/cuda_graph_dba/phase0_bucket_manifest.md`
- Create: `reports/cuda_graph_dba/phase0_bucket_decision.md`

- [ ] **Step 1: Run the complete CPU-safe regression set before Jetson work**

Run:

```bash
python -m pytest --import-mode=importlib \
  tests/test_runtime_profiler.py \
  tests/test_dbaf_runtime_profiling.py \
  tests/test_dbaf_frontend_runtime_profiling.py \
  tests/test_droid_fine_profiling.py \
  tests/test_dba_bucket_manifest.py \
  tests/test_dba_signature_profiling.py \
  tests/test_dba_bucket_planner.py \
  tests/test_dba_phase0_profile_script.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Confirm Jetson preconditions**

Run:

```bash
test -f ckpts/droid.pth
test -d data/smallcity_subset_50/small_city
test -d data/smallcity_subset_200/small_city
test -d data/hotel/nosky_color
python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name())
PY
pgrep -af '[s]cripts/run.py|[t]egrastats' && exit 3 || true
```

Expected: all files and datasets exist, CUDA reports the Jetson device, and no
residual benchmark process is found.

- [ ] **Step 3: Run the guarded three-dataset profile**

Run:

```bash
bash reports/cuda_graph_dba/run_phase0_dba_profiles.sh
```

Expected: all four statuses are 0; each instrumented run has runtime CSV,
metadata, details JSONL, and `tegrastats` evidence, while the SmallCity-50
control has evaluator and `tegrastats` evidence.

- [ ] **Step 4: Generate the manifest from all three detail files**

Run, substituting the exact run paths written by the driver:

```bash
python scripts/profiling/plan_dba_buckets.py \
  --details "$(cat reports/cuda_graph_dba/phase0_20260713/run_dir_smallcity50.txt)/runtime_profile_details.jsonl" \
  --details "$(cat reports/cuda_graph_dba/phase0_20260713/run_dir_smallcity200.txt)/runtime_profile_details.jsonl" \
  --details "$(cat reports/cuda_graph_dba/phase0_20260713/run_dir_hotel344.txt)/runtime_profile_details.jsonl" \
  --output-manifest reports/cuda_graph_dba/phase0_bucket_manifest.json \
  --output-report reports/cuda_graph_dba/phase0_bucket_manifest.md \
  --target-coverage 0.90 \
  --max-buckets 6 \
  --max-padding-ratio 0.35 \
  --max-workspace-mb 1024
```

Expected: exit 0; `supported_calls == total_calls`; coverage is at least 0.90;
every bucket is `344x616 -> 43x77`, VO, and PyTorch.

- [ ] **Step 5: Write the Phase 0 decision report**

Create `reports/cuda_graph_dba/phase0_bucket_decision.md` with these concrete
sections populated from the generated evidence:

```markdown
# Bucketed DBA CUDA Graph Phase 0 Decision

## Environment

## Input runs and exact commands

## Observed signature distribution

## Proposed buckets and call coverage

## Padding distribution

## Estimated workspace and Jetson headroom

## Unsupported or uncovered calls

## Profiling overhead check

## Phase 1 decision
```

Compare the instrumented and control SmallCity-50 keyframe list, evaluator
metrics, and exported sample identifiers before making the decision.

The Phase 1 decision is GO only when all four runs succeed, every recorded call
matches the support contract, planned coverage is at least 90%, the largest
estimated workspace is within both 1024 MiB and 25% of capture-time free CUDA
memory, and signature profiling does not change keyframe count or evaluator
outputs relative to a fresh uninstrumented control. Otherwise record NO-GO with
the failed condition and stop before CUDA Graph runtime code.

- [ ] **Step 6: Verify reports and repository scope**

Run:

```bash
python -m json.tool reports/cuda_graph_dba/phase0_bucket_manifest.json > /dev/null
python -m pytest --import-mode=importlib tests/test_dba_bucket_manifest.py tests/test_dba_bucket_planner.py -q
git diff --check
git status --short
pgrep -af '[s]cripts/run.py|[t]egrastats' && exit 3 || true
```

Expected: JSON is valid, tests pass, no whitespace errors or residual processes
exist, and all unrelated dirty files remain untouched.

- [ ] **Step 7: Commit only reusable code, tests, manifest, and decision text**

Do not commit raw output directories or large `tegrastats` logs. Run:

```bash
git add \
  scripts/frontend/dba_bucket_manifest.py \
  scripts/frontend/covisible_graph.py \
  scripts/profiling/runtime_profiler.py \
  scripts/profiling/plan_dba_buckets.py \
  reports/cuda_graph_dba/run_phase0_dba_profiles.sh \
  reports/cuda_graph_dba/phase0_bucket_manifest.json \
  reports/cuda_graph_dba/phase0_bucket_manifest.md \
  reports/cuda_graph_dba/phase0_bucket_decision.md \
  tests/test_dba_bucket_manifest.py \
  tests/test_dba_signature_profiling.py \
  tests/test_dba_bucket_planner.py \
  tests/test_dba_phase0_profile_script.py \
  tests/test_runtime_profiler.py \
  tests/test_droid_fine_profiling.py \
  tests/fixtures/dba_signatures_43x77.jsonl
git commit -m "perf: profile and plan bucketed DBA CUDA Graphs"
```

Expected: commit contains only Phase 0 reusable artifacts and small decision
evidence. If Tasks 1–5 were already committed individually, this final commit
contains only the generated manifest and decision documents.

---

## Phase 1 handoff

When `phase0_bucket_decision.md` records GO, write a new implementation plan
against the exact manifest. That plan must cover, in order:

1. recorded-fixture capture probes for `lietorch` reprojection, custom
   correlation lookup, PyTorch update-core, masked GraphAgg, and upsample;
2. eager-path extraction with behavior-parity tests;
3. fixed-address shared workspace and bucket dispatcher;
4. transactional shadow state and eager BA between Graph A and Graph B;
5. strict/fallback/circuit-breaker behavior;
6. 100-replay recurrence and allocator-stability tests;
7. three alternating SmallCity-50 pairs, SmallCity-200, and complete
   `344x616` Hotel validation;
8. the conditional Phase 3 BA-rewrite decision using the measured 40% DBA and
   10% `frame_total` thresholds.

Do not select a GPU sparse solver or modify `submodules/dbaf` during Phase 0.
