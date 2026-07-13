import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.profiling.runtime_profiler import RuntimeProfiler


class RuntimeProfilerTests(unittest.TestCase):
    def test_disabled_profiler_does_not_record_events(self):
        profiler = RuntimeProfiler(enabled=False, output_dir=None, clock=lambda: 1.0)

        with profiler.time("tracking", frame_idx=3):
            pass

        self.assertEqual(profiler.events, [])

    def test_profiler_records_stage_elapsed_time(self):
        ticks = iter([10.0, 10.25])
        profiler = RuntimeProfiler(enabled=True, output_dir=None, clock=lambda: next(ticks))

        with profiler.time("mapping", frame_idx=7):
            pass

        self.assertEqual(len(profiler.events), 1)
        self.assertEqual(profiler.events[0]["stage"], "mapping")
        self.assertEqual(profiler.events[0]["frame_idx"], 7)
        self.assertAlmostEqual(profiler.events[0]["elapsed_s"], 0.25)

    def test_summary_uses_frame_total_as_percent_denominator(self):
        profiler = RuntimeProfiler(enabled=True, output_dir=None)
        profiler.record("frame_total", 1.0, frame_idx=0)
        profiler.record("tracking", 0.2, frame_idx=0)
        profiler.record("mapping", 0.3, frame_idx=0)
        profiler.record("mapping", 0.1, frame_idx=1)

        rows = profiler.summary_rows()

        mapping = next(row for row in rows if row["stage"] == "mapping")
        self.assertEqual(mapping["calls"], 2)
        self.assertAlmostEqual(mapping["total_s"], 0.4)
        self.assertAlmostEqual(mapping["mean_ms"], 200.0)
        self.assertAlmostEqual(mapping["percent_of_frame_total"], 40.0)

    def test_write_reports_creates_csv_and_markdown_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            profiler = RuntimeProfiler(enabled=True, output_dir=output_dir)
            profiler.record("frame_total", 1.0, frame_idx=0)
            profiler.record("tracking", 0.2, frame_idx=0)
            profiler.record("mapping", 0.3, frame_idx=0)

            profiler.write_reports()

            summary_csv = output_dir / "runtime_profile.csv"
            events_csv = output_dir / "runtime_profile_events.csv"
            summary_md = output_dir / "runtime_profile_summary.md"
            self.assertTrue(summary_csv.is_file())
            self.assertTrue(events_csv.is_file())
            self.assertTrue(summary_md.is_file())

            with summary_csv.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["stage"], "frame_total")
            self.assertIn("| mapping |", summary_md.read_text())

    def test_write_reports_includes_mutable_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            profiler = RuntimeProfiler(enabled=True, output_dir=output_dir)
            profiler.set_metadata(
                "metric_depth",
                {
                    "requested_backend": "tensorrt",
                    "actual_backend": "tensorrt",
                    "strict": False,
                    "fallback_reason": None,
                },
            )
            profiler.set_metadata(
                "metric_depth",
                {
                    "requested_backend": "tensorrt",
                    "actual_backend": "torch",
                    "strict": False,
                    "fallback_reason": "engine load failed",
                },
            )

            profiler.write_reports()

            metadata = json.loads(
                (output_dir / "runtime_profile_metadata.json").read_text()
            )
            self.assertEqual(
                metadata,
                {
                    "metric_depth": {
                        "requested_backend": "tensorrt",
                        "actual_backend": "torch",
                        "strict": False,
                        "fallback_reason": "engine load failed",
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
