def _normalize_three_level_schedule(values, default_values, minimum):
    if values is None:
        values = default_values
    if len(values) != 3:
        raise ValueError("Three-level schedules must provide exactly 3 values.")
    return tuple(max(minimum, int(value)) for value in values)


def _normalize_thresholds(values, default_values):
    if values is None:
        values = default_values
    if len(values) != 2:
        raise ValueError("Adaptive runtime thresholds must provide exactly 2 values.")
    return tuple(values)


class RuntimeBudgetController:
    def __init__(self, cfg, dataset_length):
        runtime_cfg = cfg.get("runtime_budget", {})
        self.enabled = bool(runtime_cfg.get("enabled", False))
        self.dataset_length = max(1, int(dataset_length))
        self.base_mapper_iters = int(cfg.get("training_args", {}).get("iters", 50))

        if self.enabled:
            default_vis_intervals = [10, 20, 30]
            default_loop_intervals = [6, 9, 12]
            default_mapper_iters = [
                self.base_mapper_iters,
                max(6, min(self.base_mapper_iters, int(self.base_mapper_iters * 0.67))),
                max(4, min(self.base_mapper_iters, int(self.base_mapper_iters * 0.4))),
            ]
        else:
            default_vis_intervals = [1, 1, 1]
            default_loop_intervals = [3, 3, 3]
            default_mapper_iters = [self.base_mapper_iters] * 3

        self.vis_intervals = _normalize_three_level_schedule(
            runtime_cfg.get("vis_intervals"),
            default_vis_intervals,
            minimum=1,
        )
        self.loop_intervals = _normalize_three_level_schedule(
            runtime_cfg.get("loop_intervals"),
            default_loop_intervals,
            minimum=1,
        )
        self.mapper_iters = _normalize_three_level_schedule(
            runtime_cfg.get("mapper_iters"),
            default_mapper_iters,
            minimum=1,
        )
        self.progress_thresholds = _normalize_thresholds(
            runtime_cfg.get("progress_thresholds"),
            [0.45, 0.75],
        )
        self.keyframe_thresholds = _normalize_thresholds(
            runtime_cfg.get("keyframe_thresholds"),
            [120, 220],
        )
        mapping_thresholds = cfg.get("mapping_budget", {}).get("gaussian_thresholds")
        default_gaussian_thresholds = [150000, 300000]
        if mapping_thresholds is not None:
            default_gaussian_thresholds = list(mapping_thresholds[:2])
        self.gaussian_thresholds = _normalize_thresholds(
            runtime_cfg.get("gaussian_thresholds"),
            default_gaussian_thresholds,
        )

    def _resolve_stage(self, progress, keyframe_id, gaussian_count):
        if (
            progress >= self.progress_thresholds[1]
            or keyframe_id >= self.keyframe_thresholds[1]
            or gaussian_count >= self.gaussian_thresholds[1]
        ):
            return 2
        if (
            progress >= self.progress_thresholds[0]
            or keyframe_id >= self.keyframe_thresholds[0]
            or gaussian_count >= self.gaussian_thresholds[0]
        ):
            return 1
        return 0

    def evaluate(self, frame_idx, keyframe_id, gaussian_count):
        progress = float(frame_idx + 1) / float(self.dataset_length)
        stage = self._resolve_stage(progress, keyframe_id, gaussian_count) if self.enabled else 0
        vis_interval = self.vis_intervals[stage]
        loop_interval = self.loop_intervals[stage]
        mapper_iters = self.mapper_iters[stage]
        return {
            "enabled": self.enabled,
            "stage": stage,
            "progress": progress,
            "vis_interval": vis_interval,
            "loop_interval": loop_interval,
            "mapper_iters": mapper_iters,
            "should_run_vis": ((frame_idx + 1) % vis_interval) == 0,
            "should_run_loop": keyframe_id > 10 and (keyframe_id % loop_interval) == 0,
        }
