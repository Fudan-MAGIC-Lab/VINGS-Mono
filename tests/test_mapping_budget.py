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

    def test_uncovered_ratio_is_clamped(self):
        cfg = {
            "mapping_budget": {
                "enabled": True,
                "new_gaussian_budgets": [100, 80, 40, 20],
                "min_new_gaussians": 8,
            }
        }

        self.assertEqual(select_new_gaussian_budget(cfg, 0, 2.0), 100)
        self.assertEqual(select_new_gaussian_budget(cfg, 0, -1.0), 8)


if __name__ == "__main__":
    unittest.main()
