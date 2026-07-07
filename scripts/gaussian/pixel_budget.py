import torch


DEFAULT_THRESHOLDS = (150000, 300000, 600000)
DEFAULT_KEEP_RATIOS = (1.0, 0.75, 0.5, 0.5)


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


def select_pixel_budget(cfg, current_gaussians):
    pixel_cfg = cfg.get("pixel_budget", {})
    if not pixel_cfg.get("enabled", False):
        return {"enabled": False, "stage": 0, "keep_ratio": 1.0}

    default_thresholds = cfg.get("mapping_budget", {}).get("gaussian_thresholds", DEFAULT_THRESHOLDS)
    thresholds = _as_tuple(pixel_cfg.get("gaussian_thresholds"), 3, default_thresholds, int)
    keep_ratios = _as_tuple(pixel_cfg.get("keep_ratios"), 4, DEFAULT_KEEP_RATIOS, float)
    stage = _select_stage(int(current_gaussians), thresholds)
    keep_ratio = max(0.0, min(1.0, keep_ratios[stage]))
    return {"enabled": True, "stage": stage, "keep_ratio": keep_ratio}


def make_pixel_mask(height, width, keep_ratio, device="cuda", generator=None):
    total_pixels = int(height) * int(width)
    ratio = max(0.0, min(1.0, float(keep_ratio)))
    keep_pixels = max(1, min(total_pixels, int(round(total_pixels * ratio))))
    if keep_pixels == total_pixels:
        return torch.ones((int(height), int(width)), dtype=torch.bool, device=device)

    order = torch.randperm(total_pixels, device=device, generator=generator)
    flat_mask = torch.zeros(total_pixels, dtype=torch.bool, device=device)
    flat_mask[order[:keep_pixels]] = True
    return flat_mask.reshape(int(height), int(width))
