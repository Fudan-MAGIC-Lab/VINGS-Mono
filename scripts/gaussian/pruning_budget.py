import torch


DEFAULT_THRESHOLDS = (150000, 300000, 600000)
DEFAULT_INTERVALS = (4, 3, 2, 1)
DEFAULT_LOW_THRESHOLDS = (0.05, 0.05, 0.04, 0.03)
DEFAULT_HIGH_THRESHOLDS = (0.8, 0.75, 0.7, 0.65)
DEFAULT_MAX_PRUNE_RATIOS = (0.05, 0.08, 0.12, 0.16)


def _as_tuple(values, expected_len, default_values, cast):
    if values is None:
        values = default_values
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(values)}.")
    return tuple(cast(value) for value in values)


def _select_stage(current_gaussians, thresholds):
    if current_gaussians < thresholds[0]:
        return 0
    if current_gaussians < thresholds[1]:
        return 1
    if current_gaussians < thresholds[2]:
        return 2
    return 3


def select_pruning_budget(cfg, current_gaussians, time_idx):
    pruning_cfg = cfg.get("pruning_budget", {})
    fixed_interval = max(1, int(pruning_cfg.get("fixed_interval", 4)))
    if not pruning_cfg.get("enabled", False):
        return {
            "enabled": False,
            "should_prune": (time_idx + 1) % fixed_interval == 0,
            "stage": 0,
            "interval": fixed_interval,
            "low_threshold": float(pruning_cfg.get("fixed_low_threshold", 0.05)),
            "high_threshold": float(pruning_cfg.get("fixed_high_threshold", 0.8)),
            "max_prune_ratio": 1.0,
        }

    default_thresholds = cfg.get("mapping_budget", {}).get("gaussian_thresholds", DEFAULT_THRESHOLDS)
    thresholds = _as_tuple(pruning_cfg.get("gaussian_thresholds"), 3, default_thresholds, int)
    intervals = _as_tuple(pruning_cfg.get("intervals"), 4, DEFAULT_INTERVALS, int)
    low_thresholds = _as_tuple(
        pruning_cfg.get("low_thresholds"),
        4,
        DEFAULT_LOW_THRESHOLDS,
        float,
    )
    high_thresholds = _as_tuple(
        pruning_cfg.get("high_thresholds"),
        4,
        DEFAULT_HIGH_THRESHOLDS,
        float,
    )
    max_prune_ratios = _as_tuple(
        pruning_cfg.get("max_prune_ratios"),
        4,
        DEFAULT_MAX_PRUNE_RATIOS,
        float,
    )

    stage = _select_stage(int(current_gaussians), thresholds)
    interval = max(1, intervals[stage])
    return {
        "enabled": True,
        "should_prune": (time_idx + 1) % interval == 0,
        "stage": stage,
        "interval": interval,
        "low_threshold": low_thresholds[stage],
        "high_threshold": high_thresholds[stage],
        "max_prune_ratio": max(0.0, min(1.0, max_prune_ratios[stage])),
    }


def cap_prune_mask_by_score(prune_mask, scores, max_prune_count):
    if max_prune_count <= 0:
        return prune_mask & False

    prune_count = int(prune_mask.sum().item())
    if prune_count <= max_prune_count:
        return prune_mask

    candidate_indices = prune_mask.nonzero(as_tuple=False).reshape(-1)
    candidate_scores = scores[candidate_indices]
    selected = torch.topk(candidate_scores, k=max_prune_count, largest=False).indices
    capped_mask = prune_mask & False
    capped_mask[candidate_indices[selected]] = True
    return capped_mask
