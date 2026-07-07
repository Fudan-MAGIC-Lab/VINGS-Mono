import unittest

import torch

from scripts.gaussian.pixel_budget import make_pixel_mask, select_pixel_budget


class PixelBudgetTests(unittest.TestCase):
    def test_disabled_budget_keeps_all_pixels(self):
        budget = select_pixel_budget({}, current_gaussians=500_000)

        self.assertFalse(budget["enabled"])
        self.assertEqual(budget["stage"], 0)
        self.assertEqual(budget["keep_ratio"], 1.0)

    def test_enabled_budget_uses_gaussian_count_schedule(self):
        cfg = {"pixel_budget": {"enabled": True}}

        self.assertEqual(select_pixel_budget(cfg, 100_000)["keep_ratio"], 1.0)
        self.assertEqual(select_pixel_budget(cfg, 200_000)["keep_ratio"], 0.75)
        self.assertEqual(select_pixel_budget(cfg, 400_000)["keep_ratio"], 0.5)
        self.assertEqual(select_pixel_budget(cfg, 700_000)["keep_ratio"], 0.5)

    def test_pixel_thresholds_default_to_mapping_budget_thresholds(self):
        cfg = {
            "mapping_budget": {"gaussian_thresholds": [10, 20, 30]},
            "pixel_budget": {"enabled": True},
        }

        self.assertEqual(select_pixel_budget(cfg, 9)["stage"], 0)
        self.assertEqual(select_pixel_budget(cfg, 12)["stage"], 1)
        self.assertEqual(select_pixel_budget(cfg, 22)["stage"], 2)
        self.assertEqual(select_pixel_budget(cfg, 32)["stage"], 3)

    def test_make_pixel_mask_keeps_expected_number_of_pixels(self):
        generator = torch.Generator(device="cpu").manual_seed(7)

        mask = make_pixel_mask(10, 10, keep_ratio=0.5, device="cpu", generator=generator)

        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(int(mask.sum().item()), 50)

    def test_make_pixel_mask_clamps_ratio(self):
        empty = make_pixel_mask(4, 4, keep_ratio=-1.0, device="cpu")
        full = make_pixel_mask(4, 4, keep_ratio=2.0, device="cpu")

        self.assertEqual(int(empty.sum().item()), 1)
        self.assertEqual(int(full.sum().item()), 16)


if __name__ == "__main__":
    unittest.main()
