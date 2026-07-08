class LazyMetricDepth:
    def __init__(self, resolve_fn):
        self._resolve_fn = resolve_fn
        self._resolved = False
        self._value = None

    def __call__(self, **context):
        if not self._resolved:
            self._value = self._resolve_fn(**context)
            self._resolved = True
        return self._value


class MetricDepthScheduler:
    def __init__(self, cfg):
        schedule_cfg = cfg.get("metric_depth_schedule", {})
        self.enabled = bool(schedule_cfg.get("enabled", False))
        self.mode = str(schedule_cfg.get("mode", "keyframe"))
        self.warmup_frames = max(0, int(schedule_cfg.get("warmup_frames", 30)))
        self.interval = max(1, int(schedule_cfg.get("interval", 5)))
        self.keyframe_min_interval = max(1, int(schedule_cfg.get("keyframe_min_interval", 3)))
        self.keyframe_force_interval = max(1, int(schedule_cfg.get("keyframe_force_interval", 10)))
        self.high_motion_ratio = max(1.0, float(schedule_cfg.get("high_motion_ratio", 3.0)))
        self.log = bool(schedule_cfg.get("log", True))
        self.predicted = 0
        self.skipped = 0
        self.lazy_requested = 0
        self.lazy_realized = 0
        self.lazy_suppressed = 0
        self.predicted_by_initial = 0
        self.predicted_by_min_interval = 0
        self.predicted_by_high_motion = 0
        self.predicted_by_force_interval = 0
        self._last_keyframe_depth_frame = None

    def decide(self, frame_idx, has_depth):
        if has_depth:
            return {"should_predict": False, "reason": "already_has_depth"}
        if not self.enabled:
            return {"should_predict": True, "reason": "disabled"}
        if self.mode == "keyframe":
            return {"should_predict": False, "reason": "keyframe_lazy"}
        if int(frame_idx) < self.warmup_frames:
            return {"should_predict": True, "reason": "warmup"}
        if (int(frame_idx) - self.warmup_frames) % self.interval == 0:
            return {"should_predict": True, "reason": "interval"}
        return {"should_predict": False, "reason": "scheduled_skip"}

    def make_lazy_depth_provider(self, predict_fn, frame_idx=None):
        return LazyMetricDepth(lambda **context: self._resolve_lazy_depth(predict_fn, frame_idx, **context))

    def _resolve_lazy_depth(self, predict_fn, frame_idx, motion_score=None, motion_threshold=None):
        self.lazy_realized += 1
        reason = self._keyframe_prediction_reason(frame_idx, motion_score, motion_threshold)
        if reason is None:
            self.lazy_suppressed += 1
            self.skipped += 1
            return None
        depth = predict_fn()
        self.predicted += 1
        self._record_keyframe_prediction_reason(reason)
        if frame_idx is not None:
            self._last_keyframe_depth_frame = int(frame_idx)
        return depth

    def _keyframe_prediction_reason(self, frame_idx, motion_score=None, motion_threshold=None):
        if frame_idx is None or self._last_keyframe_depth_frame is None:
            return "initial"
        gap = int(frame_idx) - self._last_keyframe_depth_frame
        if self._is_high_motion(motion_score, motion_threshold):
            return "high_motion"
        if gap >= self.keyframe_force_interval:
            return "force_interval"
        if gap >= self.keyframe_min_interval:
            return "min_interval"
        return None

    def _is_high_motion(self, motion_score, motion_threshold):
        if motion_score is None or motion_threshold is None:
            return False
        return float(motion_score) >= float(motion_threshold) * self.high_motion_ratio

    def _record_keyframe_prediction_reason(self, reason):
        if reason == "initial":
            self.predicted_by_initial += 1
        elif reason == "high_motion":
            self.predicted_by_high_motion += 1
        elif reason == "force_interval":
            self.predicted_by_force_interval += 1
        elif reason == "min_interval":
            self.predicted_by_min_interval += 1

    def record(self, decision):
        if decision["should_predict"]:
            self.predicted += 1
        elif decision["reason"] == "scheduled_skip":
            self.skipped += 1
        elif decision["reason"] == "keyframe_lazy":
            self.lazy_requested += 1

    def summary(self):
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "warmup_frames": self.warmup_frames,
            "interval": self.interval,
            "keyframe_min_interval": self.keyframe_min_interval,
            "keyframe_force_interval": self.keyframe_force_interval,
            "high_motion_ratio": self.high_motion_ratio,
            "predicted": self.predicted,
            "skipped": self.skipped,
            "lazy_requested": self.lazy_requested,
            "lazy_realized": self.lazy_realized,
            "lazy_suppressed": self.lazy_suppressed,
            "predicted_by_initial": self.predicted_by_initial,
            "predicted_by_min_interval": self.predicted_by_min_interval,
            "predicted_by_high_motion": self.predicted_by_high_motion,
            "predicted_by_force_interval": self.predicted_by_force_interval,
        }

    def maybe_log_summary(self):
        if not self.enabled or not self.log:
            return
        summary = self.summary()
        print(
            "[metric_depth_schedule] "
            f"enabled=1 mode={summary['mode']} "
            f"warmup={summary['warmup_frames']} "
            f"interval={summary['interval']} "
            f"keyframe_min_interval={summary['keyframe_min_interval']} "
            f"keyframe_force_interval={summary['keyframe_force_interval']} "
            f"high_motion_ratio={summary['high_motion_ratio']:.2f} "
            f"predicted={summary['predicted']} skipped={summary['skipped']} "
            f"lazy_requested={summary['lazy_requested']} lazy_realized={summary['lazy_realized']} "
            f"lazy_suppressed={summary['lazy_suppressed']} "
            f"predicted_by_initial={summary['predicted_by_initial']} "
            f"predicted_by_min_interval={summary['predicted_by_min_interval']} "
            f"predicted_by_high_motion={summary['predicted_by_high_motion']} "
            f"predicted_by_force_interval={summary['predicted_by_force_interval']}"
        )
