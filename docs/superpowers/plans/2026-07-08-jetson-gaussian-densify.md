# Jetson Gaussian Densification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable adaptive Gaussian addition budget for the 2D Gaussian mapping stage on Jetson.

**Architecture:** Put budget selection in a pure Python helper so it can be unit-tested without CUDA. `GaussianModel.add_new_frame` will call the helper instead of using the fixed `40000` sample count, then log the selected budget and actual added Gaussians.

**Tech Stack:** Python, PyTorch-adjacent VINGS-Mono mapping code, pytest/unittest.

---

### Task 1: Budget Selection Helper

**Files:**
- Create: `scripts/gaussian/mapping_budget.py`
- Create: `tests/test_mapping_budget.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_mapping_budget.py`:

```python
import unittest

from scripts.gaussian.mapping_budget import select_new_gaussian_budget


class MappingBudgetTests(unittest.TestCase):
    def test_disabled_budget_preserves_fixed_default(self):
        budget = select_new_gaussian_budget({}, current_gaussians=500000, uncovered_ratio=0.05)
        self.assertEqual(budget, 40000)

    def test_enabled_budget_uses_gaussian_count_schedule(self):
        cfg = {"mapping_budget": {"enabled": True}}
        self.assertEqual(select_new_gaussian_budget(cfg, 100000, 1.0), 40000)
        self.assertEqual(select_new_gaussian_budget(cfg, 200000, 1.0), 25000)
        self.assertEqual(select_new_gaussian_budget(cfg, 400000, 1.0), 12000)
        self.assertEqual(select_new_gaussian_budget(cfg, 700000, 1.0), 6000)

    def test_enabled_budget_scales_down_when_current_view_is_covered(self):
        cfg = {"mapping_budget": {"enabled": True}}
        high_coverage = select_new_gaussian_budget(cfg, 100000, 0.10)
        low_coverage = select_new_gaussian_budget(cfg, 100000, 0.90)
        self.assertLess(high_coverage, low_coverage)
        self.assertGreaterEqual(high_coverage, 2000)

    def test_custom_schedule_and_minimum_are_supported(self):
        cfg = {
            "mapping_budget": {
                "enabled": True,
                "gaussian_thresholds": [10, 20, 30],
                "new_gaussian_budgets": [100, 80, 40, 20],
                "min_new_gaussians": 8,
            }
        }
        self.assertEqual(select_new_gaussian_budget(cfg, 25, 1.0), 40)
        self.assertEqual(select_new_gaussian_budget(cfg, 25, 0.01), 8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_mapping_budget.py -q
```

Expected: fails with `ModuleNotFoundError` for `scripts.gaussian.mapping_budget`.

- [ ] **Step 3: Implement helper**

Create `scripts/gaussian/mapping_budget.py` with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_mapping_budget.py -q
```

Expected: all tests pass.

### Task 2: Integrate Budget Into Gaussian Mapping

**Files:**
- Modify: `scripts/gaussian/gaussian_model.py`
- Test: `tests/test_mapping_budget.py`

- [ ] **Step 1: Add integration-focused failing test**

Append this test to `MappingBudgetTests`:

```python
    def test_uncovered_ratio_uses_accumulation_threshold(self):
        cfg = {
            "mapping_budget": {
                "enabled": True,
                "accum_threshold": 0.6,
                "new_gaussian_budgets": [100, 80, 40, 20],
                "min_new_gaussians": 8,
            }
        }
        budget = select_new_gaussian_budget(cfg, 0, uncovered_ratio=0.50)
        self.assertEqual(budget, 50)
```

- [ ] **Step 2: Run test to verify it fails if helper does not support ratio cleanly**

Run:

```bash
python -m pytest tests/test_mapping_budget.py -q
```

Expected: pass if Task 1 helper already handles explicit ratios. If it passes, continue; this test documents the integration contract.

- [ ] **Step 3: Modify `GaussianModel.add_new_frame`**

In `scripts/gaussian/gaussian_model.py`:

```python
from gaussian.mapping_budget import select_new_gaussian_budget
```

Replace the fixed `40000` call with:

```python
        mapping_budget_cfg = self.cfg.get("mapping_budget", {})
        accum_threshold = float(mapping_budget_cfg.get("accum_threshold", self.cfg["adc_args"]["accum_thresh"]))
        uncovered_ratio = float((pred_accum < accum_threshold).float().mean().item())
        gaussian_count_before = int(self._xyz.shape[0])
        new_gaussian_budget = select_new_gaussian_budget(
            self.cfg,
            current_gaussians=gaussian_count_before,
            uncovered_ratio=uncovered_ratio,
        )

        new_added_pc, new_added_pc_color, unnorm_rots = get_pointcloud(
            self.tfer,
            new_added_c2w,
            new_added_color.permute(2, 0, 1),
            new_added_depth.permute(2, 0, 1),
            pred_accum,
            new_gaussian_budget,
        )
```

After `num_pts = new_added_pc.shape[0]`, add:

```python
        if mapping_budget_cfg.get("log", True):
            print(
                "[mapping_budget] "
                f"gaussians_before={gaussian_count_before} "
                f"uncovered_ratio={uncovered_ratio:.4f} "
                f"requested={new_gaussian_budget} "
                f"added={num_pts} "
                f"gaussians_after={gaussian_count_before + num_pts}"
            )
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_mapping_budget.py tests/test_jetson_runtime_overrides.py tests/test_runtime_budget.py -q
```

Expected: all tests pass.

### Task 3: Add Config Defaults for Jetson Hotel Wrapper Path

**Files:**
- Modify: `configs/rtg/hotel.yaml`
- Modify: `configs/hierarchical/smallcity.yaml`

- [ ] **Step 1: Add disabled defaults**

Add this block to both configs:

```yaml
mapping_budget:
  enabled: False
  fixed_new_gaussians: 40000
  accum_threshold: 0.98
  gaussian_thresholds: [150000, 300000, 600000]
  new_gaussian_budgets: [40000, 25000, 12000, 6000]
  min_new_gaussians: 2000
  log: True
```

- [ ] **Step 2: Run override/config tests**

Run:

```bash
python -m pytest tests/test_mapping_budget.py tests/test_run_jetson_hotel_script.py -q
```

Expected: all tests pass.
