# CUDA Graph / Bucketed DBA Design

Date: 2026-07-13

## Decision

Build an opt-in, Jetson-only bucketed CUDA Graph runtime for the VO
`CovisibleGraph.update()` path at the `43x77` feature resolution produced by a
`344x616` frontend image. The
long-term target is full-path replay across reprojection, correlation, the
PyTorch DROID update operator, GraphAgg, Dense BA, and upsampling. Delivery is
phased: first measure real runtime signatures, then capture all graph-safe GPU
work around the existing eager BA, and rewrite BA only if the captured version
has a reproducible end-to-end benefit and BA remains the dominant bottleneck.

The original eager implementation remains the authoritative path and the
default. VIO, other spatial resolutions, the TensorRT DROID update backend,
online capture, and automatic promotion to the default are outside the first
version.

## Context

The SmallCity-200 reference measured `frontend_dba_update` at about 2.6 seconds
per call. Existing fine-grained profiling separates correlation, DROID update,
GraphAgg, BA, and upsample. A TensorRT DROID update-core experiment reduced the
update operator by about 44% and `frame_total` by about 12%, but failed the
approved quality gate. The CUDA Graph work therefore keeps the PyTorch update
operator and targets launch, allocation, and dynamic-shape overhead without
introducing TensorRT numerical changes.

The existing Dense BA extension cannot be captured as written. It dynamically
allocates tensors, copies sparse matrices and indices to CPU, constructs and
solves an Eigen system, and copies results back to CUDA. A full replay path
requires a separate graph-safe GPU BA API rather than wrapping the current
function in `torch.cuda.CUDAGraph`.

An earlier motion-aware iteration scheduler improved short Hotel runs but
diverged late in the complete Hotel sequence. This design does not change DBA
iteration counts, edge scheduling, keyframe policy, or frontend control flow.
Full Hotel validation is mandatory.

## Selected Approach

Use a gated, three-phase approach.

### Phase 0: Signature and memory profiling

Record the real dynamic signature of every eager DBA call on SmallCity-50,
SmallCity-200, and Hotel:

- active, inactive, and total BA edge counts;
- the number of unique source poses;
- `t0`, `t1`, and optimization-window size;
- `use_inactive`, `upsample`, and input dtype;
- eager stage timings;
- CUDA allocated and reserved memory.

An offline planner chooses the smallest bucket set that maximizes observed call
coverage while respecting padding and memory limits. Bucket selection is data
driven; fixed equal-width buckets and one graph per exact edge count are not
used.

### Phase 1: Captured graph islands around eager BA

For a matching bucket, replay two graphs around an eager BA call:

```text
dynamic graph state
  -> select the smallest dominating bucket
  -> copy valid data into persistent shadow workspace
  -> Graph A:
       reproject
       correlation lookup
       PyTorch DROID update operator
       masked GraphAgg
       target/weight/damping and BA input preparation
  -> eager droid_backends.ba() on shadow state
  -> Graph B:
       disparity/depth-covariance upsample
       output preparation
  -> atomically commit valid state slices
```

Python factor management, locks, bucket selection, graph age bookkeeping, and
failure handling stay outside capture. Phase 1 is a complete bucket-aware DBA
execution path but not a single CUDA Graph because BA still performs CPU work.

### Phase 2: Online decision gate

Run repeated paired SmallCity-50 comparisons, then SmallCity-200 and full
Hotel. Retain the implementation as an experiment only if it produces a stable,
reproducible positive end-to-end result within the approved quality and memory
limits.

Proceed to a BA rewrite only when the Phase 1 candidate has a positive
end-to-end result, eager BA still represents at least 40% of DBA time, and
eager BA represents at least 10% of `frame_total`.

### Phase 3: Graph-safe GPU BA and full replay

Add a new GPU-resident BA backend with fixed workspace, fixed-capacity inputs,
precomputed bucket topology, padding masks, and no CPU/Eigen round trip. Capture
it between the Phase 1 pre- and post-BA operations to form the full replay path.
Keep the existing `droid_backends.ba()` unchanged as the eager reference and
fallback.

If Phase 1 has no reproducible end-to-end benefit, stop before Phase 3.

## Runtime Signature and Bucket Selection

A call signature contains:

```text
active_edges
ba_edges
unique_source_poses
pose_window = t1 - t0
use_inactive
upsample
dtype
feature_shape
backend
mode
```

The first version accepts only:

```text
mode = vo
feature_shape = 43x77
frontend_image_size = 344x616
backend = pytorch
device = supported Jetson CUDA device
```

A bucket fixes capacities for active edges, BA edges, source-pose slots, and
pose-window slots. Dispatch selects the smallest bucket that dominates every
capacity in the call signature and stays below its measured maximum padding
ratio. A mismatch never triggers online capture.

The generated manifest records the feature shape, capture environment, memory
budget, padding limit, and bucket specifications. A representative entry is:

```json
{
  "feature_shape": [43, 77],
  "buckets": [
    {
      "active_edges": 16,
      "ba_edges": 24,
      "source_poses": 8,
      "pose_window": 12,
      "max_padding_ratio": 0.35
    }
  ]
}
```

The planner writes concrete values from observed profiles; the representative
numbers above are explanatory and are not runtime defaults.

## Components and Boundaries

Add `scripts/frontend/dba_cuda_graph.py` with focused components:

- `DBACallSignature` describes one live update call.
- `DBABucketSpec` describes one fixed-capacity bucket.
- `DBAGraphWorkspace` owns fixed-address inputs, outputs, masks, state shadows,
  and intermediate buffers.
- `CapturedDBABucket` owns Graph A and Graph B in Phase 1 and the full graph in
  Phase 3.
- `BucketedDBAGraphRuntime` loads the manifest, validates support, dispatches,
  replays, commits results, handles fallback, and exports counters.

Add `scripts/profiling/plan_dba_buckets.py` to aggregate profiling events and
write the manifest. Bucket planning remains offline and deterministic.

Refactor `CovisibleGraph.update()` into a narrow dispatcher and the existing
eager implementation:

```python
if self.cuda_graph_runtime.try_update(...):
    return

self._update_eager(...)
```

`_update_eager()` remains the sole fallback and correctness oracle. Factor
addition/removal, keyframe removal, and frontend update iteration scheduling do
not move into the CUDA Graph module.

## Configuration

The feature is disabled by default:

```yaml
frontend:
  cuda_graph_dba_enabled: false
  cuda_graph_dba_manifest: null
  cuda_graph_dba_allow_fallback: true
  cuda_graph_dba_max_workspace_mb: 1024
```

CLI overrides expose the same settings. A benchmark-only strict option rejects
every bucket miss and fallback so an eager call cannot be reported as CUDA
Graph performance.

The effective capture memory limit is:

```text
min(cuda_graph_dba_max_workspace_mb,
    25% of free CUDA memory after model initialization)
```

Logically independent bucket workspaces use a shared CUDA Graph memory pool and
a backing arena sized for the largest loaded bucket because replay is serialized.
Buckets that exceed the total memory budget are ranked by observed-call coverage
per byte, and low-value buckets are not loaded.

## Padding and Numerical Isolation

Zero-filling edge tensors alone is incorrect. Current GraphAgg uses
`torch.unique(ii)` and `scatter_mean`; padded entries would change both the
reduction denominator and the output shape.

The graph path uses fixed slots and explicit validity masks:

- `edge_valid` marks active neural-update slots;
- `ba_edge_valid` marks active plus selected inactive BA slots;
- `source_valid` marks real source-pose slots;
- fixed edge-to-pose mappings replace dynamic `torch.unique` output shapes;
- masked scatter sum divided by a clamped valid count replaces unmasked
  `scatter_mean`;
- DROID outputs for padded neural edges are masked before aggregation;
- BA excludes invalid edges from Hessian, Schur complement, and state updates;
- upsample and result commit address only valid source-pose slots.

The neural convolutions may execute padded entries because edges are independent
in the flattened batch. Their results cannot enter a valid aggregation or BA
equation.

## Transactional State Handling

The optimized path must not partially mutate live SLAM state. Each replay uses
shadow copies of the state slices needed by the bucket, including recurrent
state, targets, weights, damping, poses, disparities, and upsample outputs.

Graph A, eager or graph-safe BA, and Graph B operate on shadow state. Only after
all required work has been successfully enqueued does the runtime copy valid
slices back to live state. Graph age and Python bookkeeping update only after
the commit.

This permits safe eager fallback after preflight, Graph A, eager BA, or Graph B
raises before commit. A CUDA asynchronous device fault is not recoverable: the
runtime surfaces the error and terminates rather than claiming that a possibly
damaged CUDA context can fall back safely.

## Capture Lifecycle and Fallback

All selected buckets warm up and capture after model initialization and before
online tracking begins. Online tracking never creates or recaptures a graph.

Fallback behavior is explicit:

- unsupported mode, shape, dtype, backend, or device: eager path;
- bucket miss or excessive padding: eager path;
- insufficient capture memory: skip that bucket and use eager for its calls;
- bucket capture failure: disable only that bucket;
- replay enqueue failure before commit: permanently disable that bucket and run
  the current call eagerly;
- asynchronous device fault: report and terminate;
- strict benchmark mode: every fallback is a benchmark failure.

Each fallback records a stable reason code. The runtime logs requested and
actual execution once and exports per-call hit, miss, padding, and replay data.

## Phase 3 BA API Requirements

Do not alter the behavior of the existing `ba()` API. Add a dedicated
graph-safe entry point in the DBA submodule with an interface equivalent to:

```text
ba_graphsafe(
  persistent video tensors,
  fixed-capacity target/weight/eta/ii/jj,
  edge and pose validity masks,
  precomputed topology,
  fixed workspace,
  t0/t1 scalars
)
```

The graph-safe backend must:

- launch on the PyTorch current CUDA stream;
- allocate no CPU or CUDA memory during replay;
- copy no tensor or index to CPU;
- call no Eigen or CPU sparse solver;
- use bucket-fixed workspace and topology;
- exclude padding from every reduction and solve;
- update only valid pose and disparity slots;
- expose workspace sizing separately from execution;
- leave the existing eager BA available for parity tests and fallback.

GPU solver selection is a Phase 3 design decision based on the observed pose
window and sparsity profiles. Phase 1 does not pre-commit to a solver library.

## Profiling and Observability

The existing runtime profiler synchronizes CUDA around nested stages. Such
synchronization cannot occur during capture. The CUDA Graph path suppresses
nested Python stage profiling inside capture and records coarse outer timing
plus explicit CUDA-event timings for Graph A, eager BA, and Graph B.

Runtime metadata includes:

- manifest path and hash;
- requested and actual mode;
- loaded and rejected buckets;
- capture time and memory by bucket;
- bucket hits, misses, and fallbacks by reason;
- padding ratios;
- Graph A, BA, Graph B, commit, and total timings;
- graph replay hit rate;
- whether strict mode was enabled.

## Testing

### CPU-safe tests

Test manifest validation, signature construction, smallest-dominating-bucket
selection, padding limits, memory-budget ranking, fallback reason codes,
bucket circuit breaking, and counters without importing CUDA-only modules.

### Jetson component tests

Test:

- eager and masked GraphAgg parity with no padding, partial padding, and missing
  source slots;
- padding isolation for update, aggregation, BA preparation, and commit;
- repeated replay with stable workspace addresses;
- no allocator growth during steady-state replay;
- transactional behavior when Graph A, BA, or Graph B raises;
- 100 recurrent eager and replay steps to detect accumulated drift;
- strict-mode rejection of every fallback;
- real custom correlation and upsample operation capture.

Phase 3 adds recorded-fixture parity tests between eager and graph-safe BA across
the observed edge and pose-window buckets.

### Jetson microbenchmark

For each candidate bucket, warm up 20 calls and time at least 100 calls using
CUDA events. Compare eager and replay on the same recorded signature. Only
buckets with a stable positive microbenchmark result enter the online manifest.

### End-to-end paired validation

Run three alternating-order SmallCity-50 pairs. At least two pairs and the
median must show positive improvement in both `frontend_dba_update` and
`frame_total`. If this passes, run paired SmallCity-200 and complete Hotel.
Every candidate and control run uses `--frontend-image-size 344,616`. In
particular, Hotel validation overrides `run_jetson_hotel.sh`'s current
`256,448` default; the existing `32x56` Hotel baselines are not valid controls
for this feature.

Every run records `tegrastats`, configuration, Git state, manifest hash,
fallback counters, replay hit rate, process exit, and evaluator output. A short
run cannot substitute for complete Hotel because earlier adaptive DBA work
diverged only after frame 300.

## Acceptance Gates

This is an exploratory performance feature. No fixed percentage speedup is
required, but improvement must be positive and reproducible under the paired
protocol.

Use the approved relaxed quality gates:

- no tracking failure, hang, residual process, or undeclared fallback;
- ATE Sim3 RMSE increase no greater than 20%;
- mean PSNR drop no greater than 0.5 dB;
- mean SSIM drop no greater than 0.03;
- exported sample-count change no greater than 10%;
- keyframe and export identifiers are recorded but need not be identical.

Phase 1 may remain an explicit experimental option when SmallCity improvement
is reproducible, SmallCity-200 and complete Hotel pass quality gates, and the
memory budget is respected. It does not become the default automatically.

Phase 3 is authorized only when Phase 1 passes those gates and the measured BA
share satisfies both the 40% of remaining DBA and 10% of `frame_total`
conditions.

## Non-Goals

- Changing DBA iteration counts or adaptive iteration policy.
- Changing factor, keyframe, or active-window scheduling.
- Supporting VIO in the first version.
- Supporting frontend image sizes other than `344x616` or feature resolutions
  other than `43x77`.
- Combining TensorRT update-core inference with CUDA Graph capture.
- Capturing graph-management Python code.
- Capturing new buckets during online tracking.
- Removing the eager DBA implementation.
- Enabling the feature by default without a separate decision.
