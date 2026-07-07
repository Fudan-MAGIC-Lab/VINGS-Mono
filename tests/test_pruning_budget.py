import unittest

from scripts.gaussian.pruning_budget import cap_prune_mask_by_score, select_pruning_budget


class PruningBudgetTests(unittest.TestCase):
    def test_disabled_policy_preserves_original_interval_and_thresholds(self):
        budget = select_pruning_budget({}, current_gaussians=500_000, time_idx=3)

        self.assertFalse(budget["enabled"])
        self.assertTrue(budget["should_prune"])
        self.assertEqual(budget["interval"], 4)
        self.assertEqual(budget["low_threshold"], 0.05)
        self.assertEqual(budget["high_threshold"], 0.8)
        self.assertEqual(budget["max_prune_ratio"], 1.0)

    def test_enabled_policy_becomes_more_aggressive_as_map_grows(self):
        cfg = {"pruning_budget": {"enabled": True}}

        early = select_pruning_budget(cfg, current_gaussians=100_000, time_idx=0)
        mid = select_pruning_budget(cfg, current_gaussians=200_000, time_idx=2)
        late = select_pruning_budget(cfg, current_gaussians=400_000, time_idx=1)
        pressure = select_pruning_budget(cfg, current_gaussians=700_000, time_idx=0)

        self.assertEqual(early["stage"], 0)
        self.assertEqual(early["interval"], 4)
        self.assertFalse(early["should_prune"])

        self.assertEqual(mid["stage"], 1)
        self.assertEqual(mid["interval"], 3)
        self.assertTrue(mid["should_prune"])

        self.assertEqual(late["stage"], 2)
        self.assertEqual(late["interval"], 2)
        self.assertTrue(late["should_prune"])

        self.assertEqual(pressure["stage"], 3)
        self.assertEqual(pressure["interval"], 1)
        self.assertTrue(pressure["should_prune"])
        self.assertLess(pressure["low_threshold"], early["low_threshold"])
        self.assertLess(pressure["high_threshold"], early["high_threshold"])
        self.assertGreater(pressure["max_prune_ratio"], early["max_prune_ratio"])

    def test_pruning_thresholds_default_to_mapping_budget_thresholds(self):
        cfg = {
            "mapping_budget": {"gaussian_thresholds": [10, 20, 30]},
            "pruning_budget": {"enabled": True},
        }

        self.assertEqual(select_pruning_budget(cfg, 9, 0)["stage"], 0)
        self.assertEqual(select_pruning_budget(cfg, 12, 0)["stage"], 1)
        self.assertEqual(select_pruning_budget(cfg, 22, 0)["stage"], 2)
        self.assertEqual(select_pruning_budget(cfg, 32, 0)["stage"], 3)

    def test_cap_prune_mask_keeps_lowest_score_candidates(self):
        torch = __import__("torch")
        prune_mask = torch.tensor([True, True, False, True, True])
        scores = torch.tensor([0.4, 0.1, 0.0, 0.3, 0.2])

        capped = cap_prune_mask_by_score(prune_mask, scores, max_prune_count=2)

        self.assertEqual(capped.tolist(), [False, True, False, False, True])


if __name__ == "__main__":
    unittest.main()
