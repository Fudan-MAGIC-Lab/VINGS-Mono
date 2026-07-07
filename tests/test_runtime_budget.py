import importlib.util
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "runtime_budget.py"


def _load_runtime_budget_module():
    spec = importlib.util.spec_from_file_location("runtime_budget", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RuntimeBudgetControllerTests(unittest.TestCase):
    def test_base_stage_keeps_low_overhead_budget(self):
        module = _load_runtime_budget_module()
        controller = module.RuntimeBudgetController(
            {
                "training_args": {"iters": 30},
                "runtime_budget": {"enabled": True},
            },
            dataset_length=405,
        )

        budget = controller.evaluate(frame_idx=9, keyframe_id=12, gaussian_count=50_000)

        self.assertEqual(budget["stage"], 0)
        self.assertEqual(budget["mapper_iters"], 30)
        self.assertEqual(budget["vis_interval"], 10)
        self.assertEqual(budget["loop_interval"], 6)
        self.assertTrue(budget["should_run_vis"])
        self.assertTrue(budget["should_run_loop"])

    def test_mid_stage_reduces_frequency_and_mapper_iters(self):
        module = _load_runtime_budget_module()
        controller = module.RuntimeBudgetController(
            {
                "training_args": {"iters": 30},
                "runtime_budget": {"enabled": True},
            },
            dataset_length=405,
        )

        budget = controller.evaluate(frame_idx=150, keyframe_id=121, gaussian_count=160_000)

        self.assertEqual(budget["stage"], 1)
        self.assertEqual(budget["mapper_iters"], 20)
        self.assertEqual(budget["vis_interval"], 20)
        self.assertEqual(budget["loop_interval"], 9)

    def test_late_stage_pushes_to_most_conservative_budget(self):
        module = _load_runtime_budget_module()
        controller = module.RuntimeBudgetController(
            {
                "training_args": {"iters": 30},
                "runtime_budget": {"enabled": True},
            },
            dataset_length=405,
        )

        budget = controller.evaluate(frame_idx=330, keyframe_id=240, gaussian_count=320_000)

        self.assertEqual(budget["stage"], 2)
        self.assertEqual(budget["mapper_iters"], 12)
        self.assertEqual(budget["vis_interval"], 30)
        self.assertEqual(budget["loop_interval"], 12)
        self.assertFalse(budget["should_run_vis"])
        self.assertTrue(budget["should_run_loop"])

    def test_runtime_gaussian_thresholds_default_to_mapping_budget_thresholds(self):
        module = _load_runtime_budget_module()
        controller = module.RuntimeBudgetController(
            {
                "training_args": {"iters": 30},
                "mapping_budget": {"gaussian_thresholds": [10, 20, 30]},
                "runtime_budget": {"enabled": True},
            },
            dataset_length=1000,
        )

        mid = controller.evaluate(frame_idx=0, keyframe_id=0, gaussian_count=12)
        late = controller.evaluate(frame_idx=0, keyframe_id=0, gaussian_count=22)

        self.assertEqual(mid["stage"], 1)
        self.assertEqual(late["stage"], 2)


if __name__ == "__main__":
    unittest.main()
