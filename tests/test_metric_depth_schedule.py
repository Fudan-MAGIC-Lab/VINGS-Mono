import unittest

from scripts.metric_depth_schedule import MetricDepthScheduler


class MetricDepthSchedulerTests(unittest.TestCase):
    def test_disabled_scheduler_predicts_missing_depth_every_frame(self):
        scheduler = MetricDepthScheduler({})

        decision = scheduler.decide(frame_idx=17, has_depth=False)

        self.assertTrue(decision["should_predict"])
        self.assertEqual(decision["reason"], "disabled")

    def test_scheduler_never_predicts_when_depth_already_exists(self):
        scheduler = MetricDepthScheduler({"metric_depth_schedule": {"enabled": True, "mode": "interval"}})

        decision = scheduler.decide(frame_idx=0, has_depth=True)

        self.assertFalse(decision["should_predict"])
        self.assertEqual(decision["reason"], "already_has_depth")

    def test_enabled_scheduler_predicts_during_warmup(self):
        scheduler = MetricDepthScheduler(
            {"metric_depth_schedule": {"enabled": True, "mode": "interval", "warmup_frames": 3, "interval": 5}}
        )

        self.assertTrue(scheduler.decide(frame_idx=0, has_depth=False)["should_predict"])
        self.assertTrue(scheduler.decide(frame_idx=1, has_depth=False)["should_predict"])
        self.assertTrue(scheduler.decide(frame_idx=2, has_depth=False)["should_predict"])

    def test_enabled_scheduler_predicts_by_interval_after_warmup(self):
        scheduler = MetricDepthScheduler(
            {"metric_depth_schedule": {"enabled": True, "mode": "interval", "warmup_frames": 3, "interval": 5}}
        )

        self.assertTrue(scheduler.decide(frame_idx=3, has_depth=False)["should_predict"])
        self.assertFalse(scheduler.decide(frame_idx=4, has_depth=False)["should_predict"])
        self.assertFalse(scheduler.decide(frame_idx=7, has_depth=False)["should_predict"])
        self.assertTrue(scheduler.decide(frame_idx=8, has_depth=False)["should_predict"])

    def test_scheduler_counts_predictions_and_skips(self):
        scheduler = MetricDepthScheduler(
            {"metric_depth_schedule": {"enabled": True, "mode": "interval", "warmup_frames": 1, "interval": 3}}
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
            {"metric_depth_schedule": {"enabled": True, "mode": "interval", "warmup_frames": 2, "interval": 4}}
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

    def test_keyframe_mode_returns_lazy_provider(self):
        calls = []
        scheduler = MetricDepthScheduler({"metric_depth_schedule": {"enabled": True, "mode": "keyframe"}})

        depth = scheduler.make_lazy_depth_provider(lambda: calls.append("predict") or "depth-value")

        self.assertTrue(callable(depth))
        self.assertEqual(calls, [])
        self.assertEqual(depth(), "depth-value")
        self.assertEqual(calls, ["predict"])

    def test_keyframe_mode_counts_lazy_predictions(self):
        calls = []
        scheduler = MetricDepthScheduler({"metric_depth_schedule": {"enabled": True, "mode": "keyframe"}})
        depth = scheduler.make_lazy_depth_provider(lambda: calls.append("predict") or "depth-value")

        decision = scheduler.decide(frame_idx=42, has_depth=False)
        self.assertEqual(decision["reason"], "keyframe_lazy")
        scheduler.record(decision)
        self.assertEqual(scheduler.summary()["lazy_requested"], 1)

        self.assertEqual(depth(), "depth-value")
        self.assertEqual(scheduler.summary()["predicted"], 1)
        self.assertEqual(scheduler.summary()["lazy_realized"], 1)

    def test_keyframe_mode_suppresses_dense_keyframes_by_min_interval(self):
        calls = []
        scheduler = MetricDepthScheduler(
            {"metric_depth_schedule": {"enabled": True, "mode": "keyframe", "keyframe_min_interval": 3}}
        )

        first = scheduler.make_lazy_depth_provider(lambda: calls.append("first") or "depth-0", frame_idx=0)
        second = scheduler.make_lazy_depth_provider(lambda: calls.append("second") or "depth-1", frame_idx=1)
        fourth = scheduler.make_lazy_depth_provider(lambda: calls.append("fourth") or "depth-3", frame_idx=3)

        self.assertEqual(first(), "depth-0")
        self.assertIsNone(second())
        self.assertEqual(fourth(), "depth-3")
        self.assertEqual(calls, ["first", "fourth"])
        self.assertEqual(scheduler.summary()["predicted"], 2)
        self.assertEqual(scheduler.summary()["lazy_realized"], 3)
        self.assertEqual(scheduler.summary()["lazy_suppressed"], 1)

    def test_keyframe_mode_uses_high_motion_to_override_min_interval(self):
        calls = []
        scheduler = MetricDepthScheduler(
            {
                "metric_depth_schedule": {
                    "enabled": True,
                    "mode": "keyframe",
                    "keyframe_min_interval": 3,
                    "high_motion_ratio": 2.0,
                }
            }
        )

        first = scheduler.make_lazy_depth_provider(lambda: calls.append("first") or "depth-0", frame_idx=0)
        second = scheduler.make_lazy_depth_provider(lambda: calls.append("second") or "depth-1", frame_idx=1)

        self.assertEqual(first(motion_score=2.6, motion_threshold=2.5), "depth-0")
        self.assertEqual(second(motion_score=6.0, motion_threshold=2.5), "depth-1")
        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(scheduler.summary()["predicted_by_high_motion"], 1)

    def test_keyframe_mode_uses_force_interval_to_prevent_starvation(self):
        calls = []
        scheduler = MetricDepthScheduler(
            {
                "metric_depth_schedule": {
                    "enabled": True,
                    "mode": "keyframe",
                    "keyframe_min_interval": 10,
                    "keyframe_force_interval": 5,
                    "high_motion_ratio": 2.0,
                }
            }
        )

        first = scheduler.make_lazy_depth_provider(lambda: calls.append("first") or "depth-0", frame_idx=0)
        sixth = scheduler.make_lazy_depth_provider(lambda: calls.append("sixth") or "depth-5", frame_idx=5)

        self.assertEqual(first(motion_score=2.6, motion_threshold=2.5), "depth-0")
        self.assertEqual(sixth(motion_score=2.6, motion_threshold=2.5), "depth-5")
        self.assertEqual(calls, ["first", "sixth"])
        self.assertEqual(scheduler.summary()["predicted_by_force_interval"], 1)

    def test_keyframe_mode_keeps_existing_depth(self):
        scheduler = MetricDepthScheduler({"metric_depth_schedule": {"enabled": True, "mode": "keyframe"}})

        decision = scheduler.decide(frame_idx=0, has_depth=True)

        self.assertFalse(decision["should_predict"])
        self.assertEqual(decision["reason"], "already_has_depth")


if __name__ == "__main__":
    unittest.main()
