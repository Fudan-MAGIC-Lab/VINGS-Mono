from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StorageSummary:
    storage_bytes: int
    tensor_count: int
    unique_storage_count: int
    excluded_tensor_count: int


@dataclass(frozen=True)
class MemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    free_bytes: int
    total_bytes: int


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
    return StorageSummary(
        storage_bytes=total,
        tensor_count=tensor_count,
        unique_storage_count=len(seen),
        excluded_tensor_count=excluded,
    )


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
            allocated_bytes=int(
                self.torch.cuda.memory_allocated(self.device)
            ),
            reserved_bytes=int(
                self.torch.cuda.memory_reserved(self.device)
            ),
            peak_allocated_bytes=int(
                self.torch.cuda.max_memory_allocated(self.device)
            ),
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
        if self.payload is None:
            raise RuntimeError("DBA memory sample has not started")
        current = self._snapshot_and_reset()
        self.payload["stages"].append({
            "name": str(name),
            "allocated_delta_bytes": (
                current.allocated_bytes
                - self.stage_baseline.allocated_bytes
            ),
            "retained_delta_bytes": (
                current.allocated_bytes - self.entry.allocated_bytes
            ),
            "peak_delta_bytes": max(
                0,
                current.peak_allocated_bytes
                - self.stage_baseline.allocated_bytes,
            ),
            "free_delta_bytes": (
                current.free_bytes - self.entry.free_bytes
            ),
        })
        self.stage_baseline = current

    def finish(self):
        if self.payload is None:
            raise RuntimeError("DBA memory sample has not started")
        return self.payload
