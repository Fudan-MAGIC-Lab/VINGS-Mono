DEFAULT_FIXED_NEW_GAUSSIANS = 40000
DEFAULT_THRESHOLDS = (150000, 300000, 600000)
DEFAULT_BUDGETS = (40000, 25000, 12000, 6000)
DEFAULT_MIN_NEW_GAUSSIANS = 2000


def _as_int_tuple(values, expected_len, default_values):
    if values is None:
        values = default_values
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(values)}.")
    return tuple(int(value) for value in values)


def select_new_gaussian_budget(cfg, current_gaussians, uncovered_ratio):
    budget_cfg = cfg.get("mapping_budget", {})
    if not budget_cfg.get("enabled", False):
        return int(budget_cfg.get("fixed_new_gaussians", DEFAULT_FIXED_NEW_GAUSSIANS))

    thresholds = _as_int_tuple(
        budget_cfg.get("gaussian_thresholds"),
        3,
        DEFAULT_THRESHOLDS,
    )
    budgets = _as_int_tuple(
        budget_cfg.get("new_gaussian_budgets"),
        4,
        DEFAULT_BUDGETS,
    )
    min_budget = int(budget_cfg.get("min_new_gaussians", DEFAULT_MIN_NEW_GAUSSIANS))

    if current_gaussians < thresholds[0]:
        stage_budget = budgets[0]
    elif current_gaussians < thresholds[1]:
        stage_budget = budgets[1]
    elif current_gaussians < thresholds[2]:
        stage_budget = budgets[2]
    else:
        stage_budget = budgets[3]

    ratio = max(0.0, min(1.0, float(uncovered_ratio)))
    scaled_budget = int(stage_budget * ratio)
    return max(min_budget, min(stage_budget, scaled_budget))
