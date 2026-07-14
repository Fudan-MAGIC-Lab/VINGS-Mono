import csv
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


class RuntimeProfiler:
    def __init__(self, enabled=False, output_dir=None, clock=None, sync_callback=None):
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.clock = clock or time.perf_counter
        self.sync_callback = sync_callback
        self.events = []
        self.details = []
        self.metadata = {}

    @classmethod
    def from_config(cls, cfg, sync_callback=None):
        profiling_cfg = cfg.get("profiling", {}).get("runtime", {})
        return cls(
            enabled=profiling_cfg.get("enabled", False),
            output_dir=cfg.get("output", {}).get("save_dir"),
            sync_callback=sync_callback if profiling_cfg.get("sync_cuda", True) else None,
        )

    def _sync(self):
        if self.sync_callback is not None:
            self.sync_callback()

    @contextmanager
    def time(self, stage, frame_idx=None):
        if not self.enabled:
            yield
            return

        self._sync()
        start = self.clock()
        try:
            yield
        finally:
            self._sync()
            self.record(stage, self.clock() - start, frame_idx=frame_idx)

    def record(self, stage, elapsed_s, frame_idx=None):
        if not self.enabled:
            return
        self.events.append(
            {
                "stage": str(stage),
                "frame_idx": frame_idx,
                "elapsed_s": float(elapsed_s),
            }
        )

    def set_metadata(self, key, value):
        if not self.enabled:
            return
        self.metadata[str(key)] = value

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

    def summary_rows(self):
        grouped = defaultdict(list)
        for event in self.events:
            grouped[event["stage"]].append(event["elapsed_s"])

        denominator = sum(grouped.get("frame_total", []))
        if denominator <= 0:
            denominator = sum(sum(values) for values in grouped.values())

        rows = []
        for stage, values in grouped.items():
            total_s = sum(values)
            calls = len(values)
            rows.append(
                {
                    "stage": stage,
                    "calls": calls,
                    "total_s": total_s,
                    "mean_ms": (total_s / calls) * 1000.0 if calls else 0.0,
                    "percent_of_frame_total": (total_s / denominator) * 100.0 if denominator > 0 else 0.0,
                }
            )
        return sorted(rows, key=lambda row: row["total_s"], reverse=True)

    def write_reports(self):
        if not self.enabled or self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = self.summary_rows()

        with (self.output_dir / "runtime_profile_metadata.json").open("w") as handle:
            json.dump(self.metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

        with (self.output_dir / "runtime_profile_details.jsonl").open("w") as handle:
            for detail in self.details:
                handle.write(json.dumps(detail, sort_keys=True) + "\n")

        with (self.output_dir / "runtime_profile.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["stage", "calls", "total_s", "mean_ms", "percent_of_frame_total"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        with (self.output_dir / "runtime_profile_events.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame_idx", "stage", "elapsed_s"])
            writer.writeheader()
            writer.writerows(self.events)

        lines = [
            "# Runtime Profile Summary",
            "",
            "| Stage | Calls | Total s | Mean ms | % frame total |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in summary_rows:
            lines.append(
                "| {stage} | {calls} | {total_s:.3f} | {mean_ms:.2f} | {percent:.1f}% |".format(
                    stage=row["stage"],
                    calls=row["calls"],
                    total_s=row["total_s"],
                    mean_ms=row["mean_ms"],
                    percent=row["percent_of_frame_total"],
                )
            )
        (self.output_dir / "runtime_profile_summary.md").write_text("\n".join(lines) + "\n")
