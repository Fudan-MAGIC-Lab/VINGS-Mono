import unittest

from scripts.metric_depth_schedule import MetricDepthScheduler


class MetricDepthSchedulerTests(unittest.TestCase):
    def test_disabled_scheduler_predicts_missing_depth_every_frame(self):
        scheduler = MetricDepthScheduler({})

        decision = scheduler.decide(frame_idx=17, has_depth=False)

        self.assertTrue(decision["should_predict"])
        self.assertEqual(decision["reason"], "disabled")

    def test_scheduler_never_predicts_when_depth_already_exists(self):
        scheduler = MetricDepthScheduler({"metric_depth_schedule": {"enabled": True}})

        decision = scheduler.decide(frame_idx=0, has_depth=True)

        self.assertFalse(decision["should_predict"])
        self.assertEqual(decision["reason"], "already_has_depth")

    def test_enabled_scheduler_predicts_during_warmup(self):
        scheduler = MetricDepthScheduler(
            {"metric_depth_schedule": {"enabled": True, "warmup_frames": 3, "interval": 5}}
        )

        self.assertTrue(scheduler.decide(frame_idx=0, has_depth=False)["should_predict"])
        self.assertTrue(scheduler.decide(frame_idx=1, has_depth=False)["should_predict"])
        self.assertTrue(scheduler.decide(frame_idx=2, has_depth=False)["should_predict"])

    def test_enabled_scheduler_predicts_by_interval_after_warmup(self):
        scheduler = MetricDepthScheduler(
            {"metric_depth_schedule": {"enabled": True, "warmup_frames": 3, "interval": 5}}
        )

        self.assertTrue(scheduler.decide(frame_idx=3, has_depth=False)["should_predict"])
        self.assertFalse(scheduler.decide(frame_idx=4, has_depth=False)["should_predict"])
        self.assertFalse(scheduler.decide(frame_idx=7, has_depth=False)["should_predict"])
        self.assertTrue(scheduler.decide(frame_idx=8, has_depth=False)["should_predict"])

    def test_scheduler_counts_predictions_and_skips(self):
        scheduler = MetricDepthScheduler(
            {"metric_depth_schedule": {"enabled": True, "warmup_frames": 1, "interval": 3}}
        )

        for frame_idx in range(5):
            decision = scheduler.decide(frame_idx=frame_idx, has_depth=False)
            scheduler.record(decision)

        summary = scheduler.summary()
        self.assertEqual(summary["predicted"], 3)
        self.assertEqual(summary["skipped"], 2)

    def test_log_summary_prints_counts(self):
        import contextlib
        import io

        scheduler = MetricDepthScheduler(
            {"metric_depth_schedule": {"enabled": True, "warmup_frames": 2, "interval": 4}}
        )
        scheduler.record({"should_predict": True, "reason": "warmup"})
        scheduler.record({"should_predict": False, "reason": "scheduled_skip"})

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            scheduler.maybe_log_summary()

        text = output.getvalue()
        self.assertIn("warmup=2", text)
        self.assertIn("interval=4", text)
        self.assertIn("predicted=1", text)
        self.assertIn("skipped=1", text)


if __name__ == "__main__":
    unittest.main()
