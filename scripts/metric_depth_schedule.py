class MetricDepthScheduler:
    def __init__(self, cfg):
        schedule_cfg = cfg.get("metric_depth_schedule", {})
        self.enabled = bool(schedule_cfg.get("enabled", False))
        self.warmup_frames = max(0, int(schedule_cfg.get("warmup_frames", 30)))
        self.interval = max(1, int(schedule_cfg.get("interval", 5)))
        self.log = bool(schedule_cfg.get("log", True))
        self.predicted = 0
        self.skipped = 0

    def decide(self, frame_idx, has_depth):
        if has_depth:
            return {"should_predict": False, "reason": "already_has_depth"}
        if not self.enabled:
            return {"should_predict": True, "reason": "disabled"}
        if int(frame_idx) < self.warmup_frames:
            return {"should_predict": True, "reason": "warmup"}
        if (int(frame_idx) - self.warmup_frames) % self.interval == 0:
            return {"should_predict": True, "reason": "interval"}
        return {"should_predict": False, "reason": "scheduled_skip"}

    def record(self, decision):
        if decision["should_predict"]:
            self.predicted += 1
        elif decision["reason"] == "scheduled_skip":
            self.skipped += 1

    def summary(self):
        return {
            "enabled": self.enabled,
            "warmup_frames": self.warmup_frames,
            "interval": self.interval,
            "predicted": self.predicted,
            "skipped": self.skipped,
        }

    def maybe_log_summary(self):
        if not self.enabled or not self.log:
            return
        summary = self.summary()
        print(
            "[metric_depth_schedule] "
            f"enabled=1 warmup={summary['warmup_frames']} "
            f"interval={summary['interval']} "
            f"predicted={summary['predicted']} skipped={summary['skipped']}"
        )
