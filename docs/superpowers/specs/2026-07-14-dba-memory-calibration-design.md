# DBA Memory Calibration Phase 0.5 Design

## Objective

Measure the real incremental CUDA memory used by the eager DBA update on Jetson
and replace the Phase 0 conservative workspace estimate with a calibrated,
capture-boundary-specific model. This phase produces measurements, a calibration
artifact, a revised bucket manifest, and a GO/NO-GO report. It does not implement
CUDA Graph capture, change `max_factors`, or change tracking behavior.

The motivating Phase 0 result is that 90% observed-call coverage requires a
conservatively estimated 1531.73 MiB bucket, while the approved limit is 1024
MiB. The current estimate includes the full resident correlation pyramid. Phase
0.5 must determine which bytes are existing eager state, which bytes require a
fixed-address capture shadow, and which bytes are temporary peaks.

## Scope

In scope:

- opt-in, sampled CUDA memory instrumentation for DBA updates;
- exact tensor-storage accounting for resident correlation and recurrent state;
- allocator snapshots around DBA stages;
- a CPU-safe calibration schema and offline calibration tool;
- planner support for an explicit calibrated capture boundary;
- SmallCity-200 and complete Hotel evidence runs at `344x616 -> 43x77`;
- a revised workspace/coverage decision.

Out of scope:

- CUDA Graph creation, replay, or persistent graph workspaces;
- GPU bundle adjustment;
- changing factors, keyframes, active windows, update iterations, or precision;
- TensorRT update-core integration;
- reducing `max_factors` or testing the quality effect of doing so.

## Approaches Considered

### Online stage-delta instrumentation (selected)

Instrument the real eager path and sample allocator state at stage boundaries.
This observes the actual online tensor lifetimes and Jetson allocator behavior.
Synchronization perturbs timing, so runs with memory instrumentation are memory
evidence only and are never used for throughput or quality comparisons.

### Standalone synthetic microbenchmark (secondary validation)

Replay representative tensor shapes outside the full application. This is easy
to repeat and useful for checking accounting formulas, but it cannot reproduce
all online resident state and allocator fragmentation. It may validate selected
signatures but cannot be the authoritative source.

### Full PyTorch memory snapshots (rejected as the primary path)

Allocator snapshots contain richer stack information, but their overhead and
artifact size are unsuitable for repeated full Hotel runs on the target Jetson.

## Capture Boundaries

Calibration is boundary-specific. A byte estimate is invalid unless the
calibration and planner request name the same boundary.

Phase 0.5 evaluates two analytical boundaries:

1. `corr_update_aggregation`: correlation sampling, update operator, graph
   aggregation, and upsample are treated as captured. A fixed-address shadow of
   the correlation pyramid is included.
2. `update_aggregation_only`: correlation sampling remains eager. Only sampled
   correlation features and the update/aggregation inputs and outputs require
   fixed-address shadows. The resident full correlation pyramid is measured and
   reported, but excluded from incremental capture workspace.

The second boundary is the recommended candidate. Bundle adjustment remains
eager for both boundaries. No runtime capture code is authorized by this design.

## Runtime Instrumentation

### Configuration

Add these options to `scripts/run.py`:

- `--profile-dba-memory`: enable sampled DBA memory measurements; it is invalid
  unless `--profile-runtime` is also enabled.
- `--profile-dba-memory-samples-per-signature N`: maximum samples for one exact
  signature, default 2, minimum 1.

Configuration lives under `profiling.dba_memory`. The disabled path performs no
CUDA synchronization, peak reset, tensor walk, or record allocation.

### Signature and sample selection

The exact sample key is the existing DBA signature tuple:

`(active_edges, ba_edges, source_poses, pose_window, use_inactive, upsample, dtype, feature_shape, frontend_image_size, mode, backend)`.

The first `N` occurrences of each key are sampled. Counts are local to a run and
are recorded in profiler metadata. This deterministic rule ensures rare shapes
are observed while bounding synchronization overhead.

The exact key is previewed immediately before correlation sampling using only
the already available active/inactive index tensors and `t0/t1`. The preview
does not assemble or mutate target/weight tensors. After the normal eager path
constructs its final BA inputs, the previewed key must equal the recorded call
signature; a mismatch fails the evidence run.

### Measurements

Every sampled update synchronizes the tracker device before each snapshot and
records:

- `torch.cuda.memory_allocated()`;
- `torch.cuda.memory_reserved()`;
- `torch.cuda.max_memory_allocated()` after a stage-local peak reset;
- free and total bytes from `torch.cuda.mem_get_info()`;
- exact unique storage bytes for the tensors assigned to the stage;
- frame index, signature, stage name, and sample ordinal.

The stage sequence is:

1. `update_entry`;
2. `corr_sample_complete`;
3. `update_op_complete`;
4. `ba_inputs_complete`;
5. `ba_complete`;
6. `upsample_complete` or `update_exit`.

At `update_entry`, exact tensor accounting separately reports:

- the resident `CorrBlock.corr_pyramid` storage;
- recurrent `net` and `inp` storage;
- coordinates, targets, weights, damping, active indices, and inactive indices.

Storage accounting deduplicates views by CUDA storage identity and counts
storage capacity, not logical tensor elements. CPU tensors and tensors on a
different CUDA device are excluded and reported as excluded counts.

Allocator deltas are always relative to the synchronized `update_entry`
baseline. A sample records both the retained delta at each stage and the largest
stage-local peak delta. Negative retained deltas are preserved rather than
clamped, because they reveal deallocation; peak workspace is never negative.

### Output records

Memory samples use the existing structured profiler JSONL with kind
`dba_memory_sample`. Each payload contains:

```json
{
  "schema_version": 1,
  "sample_ordinal": 1,
  "signature": {
    "active_edges": 48,
    "ba_edges": 72,
    "source_poses": 12,
    "pose_window": 11,
    "dtype": "float16"
  },
  "entry": {
    "allocated_bytes": 0,
    "reserved_bytes": 0,
    "free_bytes": 0,
    "total_bytes": 0,
    "corr_resident_storage_bytes": 0,
    "recurrent_resident_storage_bytes": 0
  },
  "stages": [
    {
      "name": "corr_sample_complete",
      "allocated_delta_bytes": 0,
      "retained_delta_bytes": 0,
      "peak_delta_bytes": 0,
      "free_delta_bytes": 0
    }
  ]
}
```

Real values replace the zeros. Required fields are validated offline; malformed
or incomplete samples fail calibration instead of silently falling back.

## Calibration Model

Create `scripts/profiling/calibrate_dba_workspace.py`. It reads one or more
runtime detail JSONL files and emits `dba_memory_calibration.json` plus a
Markdown report.

The calibration artifact includes:

- schema version, device, Torch/CUDA versions, resolution, mode, backend, and
  capture boundary;
- source evidence paths and hashes;
- sampled and observed signature counts;
- exact resident-state distributions;
- measured peak incremental bytes by signature and stage;
- analytical persistent-shadow bytes by signature;
- safety margin and validation errors;
- the final coefficients/terms used by the planner.

The corrected estimate is:

`persistent shadow bytes for the selected boundary + calibrated transient peak bytes + safety margin`.

For validation, the measured target before safety margin is the analytical
persistent-shadow bytes for that boundary plus the measured eager transient
peak. Accuracy metrics compare the pre-margin modeled value to that target.
The no-underprediction gate compares the final value after safety margin to the
same target. This prevents the fixed margin from mechanically inflating MAPE.

For `corr_update_aggregation`, persistent shadow bytes include the exact
capacity formula for the full four-level correlation pyramid. For
`update_aggregation_only`, they include sampled correlation features and
update/aggregation inputs and outputs, but not the already resident full
correlation pyramid. BA inputs are reported separately and excluded because BA
remains eager.

The safety margin is the larger of 64 MiB and 10% of the modeled incremental
workspace. It is applied after calibration. The model must never predict below
the measured incremental peak for a validation signature.

## Validation

Samples are split deterministically by sorted exact signature: every fifth
signature is held out, with at least one held-out signature per input run.
Calibration uses the remaining signatures. The report must provide both
unweighted and observed-call-weighted errors.

The model is valid only when:

- all required sample fields are present;
- sampled signatures account for at least 95% of the Phase 0 observed calls by
  frequency;
- held-out weighted mean absolute percentage error is at most 10%;
- no held-out signature is underpredicted after the safety margin;
- repeated samples for a signature have peak spread at most 15%, or the
  signature is marked unstable and excluded from a GO result.

If validation fails, the planner refuses the calibration unless an explicit
diagnostic-only override is passed. Evidence reports must never use that
override for a GO decision.

## Planner Integration

Extend `plan_dba_buckets.py` with:

- `--memory-calibration PATH`;
- `--capture-boundary {corr_update_aggregation,update_aggregation_only}`.

Both options must be supplied together. Without them, the Phase 0 conservative
estimator remains unchanged for backward compatibility. A calibration is
rejected when its resolution, mode, backend, dtype support, or boundary does not
match the calls being planned.

The generated manifest records the estimator kind, calibration hash, capture
boundary, validation metrics, and per-bucket workspace breakdown. Bucket
selection and padding logic otherwise remain unchanged.

## Evidence Runs

Add a guarded Phase 0.5 driver following the Phase 0 overwrite and residual
process protections. Run:

1. SmallCity-200 with runtime and DBA memory profiling;
2. complete Hotel at `344x616`, Torch backend, and Hotel save buffer 512.

The driver records Git state, exact commands, `tegrastats`, run status, runtime
details, and output directories. It checks that memory samples exist and that
the sampled signatures reach the 95% weighted coverage requirement. Because
memory sampling synchronizes CUDA, these runs do not replace the Phase 0 quality
or performance evidence.

After calibration, generate manifests for both boundaries at 1024 MiB, six
buckets, and 35% padding. The decision report compares:

- conservative Phase 0 workspace;
- calibrated full-correlation boundary workspace;
- calibrated update/aggregation-only workspace;
- coverage and headroom for each boundary;
- calibration error and unstable signatures.

## Testing

CPU-safe unit tests cover:

- option validation and disabled-path behavior;
- deterministic per-signature sampling;
- storage deduplication for tensor views using CPU fakes;
- stage delta and peak calculations from fake CUDA snapshots;
- calibration schema validation and boundary mismatch rejection;
- analytical boundary terms;
- deterministic holdout selection, error metrics, and no-underprediction gate;
- calibrated planner output and legacy-estimator compatibility;
- guarded evidence-driver contents.

Jetson tests cover real CUDA snapshot fields, synchronization, peak reset, and
one representative synthetic `43x77` signature before full evidence runs.

## Failure Handling

- Unsupported Torch memory APIs disable the memory sampler with a recorded
  reason and make calibration fail.
- CUDA OOM, tracking failure, missing samples, unstable signatures, model
  underprediction, or insufficient weighted signature coverage produces NO-GO.
- Sampling errors never alter the eager DBA tensors or retry an update.
- All output paths are new and refuse overwrite; failed evidence is preserved.

## Completion Gate

Phase 0.5 is complete when the instrumentation and calibration tests pass, both
Jetson evidence runs finish, and the calibration artifacts and decision report
are committed. Revised manifests are required only for boundaries whose
calibration validates; an invalid calibration is itself sufficient NO-GO
evidence and must not be forced through the planner.

Phase 1 may be reconsidered only if the `update_aggregation_only` boundary
achieves at least 90% observed-call coverage with at most six buckets, each
workspace is at most 1024 MiB and at most 25% of measured capture-time free
CUDA memory, and all model-validation requirements above pass. Otherwise the
decision remains NO-GO without modifying active-edge capacity.
