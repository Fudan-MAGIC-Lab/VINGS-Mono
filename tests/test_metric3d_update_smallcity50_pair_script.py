import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "reports/tensorrt_fp16/run_metric3d_update_smallcity50_pair.sh"


class Metric3DUpdateSmallCity50PairScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = SCRIPT.read_text()

    def test_uses_new_immutable_paths_and_never_deletes_results(self):
        self.assertIn("metric3d_update_smallcity50_online_20260712_try1", self.content)
        self.assertIn("smallcity_metric3d_update_online_20260712_try1", self.content)
        self.assertIn('if [ -e "$EVIDENCE_DIR" ]', self.content)
        self.assertIn('if [ -e "$output_parent" ]', self.content)
        self.assertNotIn("rm -rf", self.content)
        self.assertNotIn("rm -f", self.content)

    def test_preserves_balanced_fast_smallcity50_arguments(self):
        for text in (
            "data/smallcity_subset_50/small_city",
            "--frontend-save-buffer 64",
            "--training-iters 10",
            "--metric-depth-scale 0.75",
            "--enable-mapping-budget",
            "--enable-jetson-pruning",
            "--enable-pixel-budget",
            "--enable-metric-depth-schedule",
            "--enable-jetson-motion-gate",
            "--motion-gate-backend vpi_cpp",
            "--motion-gate-threshold 12.0",
        ):
            self.assertIn(text, self.content)

    def test_candidate_enables_both_strict_tensorrt_backends_once(self):
        for text in (
            "--metric-depth-backend tensorrt",
            "--metric-depth-engine engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan",
            "--droid-update-backend tensorrt",
            "--droid-update-engine engines/tensorrt/droid/droid_update_core_43x77_deltafp32.plan",
            "--tensorrt-strict",
        ):
            self.assertEqual(self.content.count(text), 1)

    def test_verifies_both_actual_backends_and_no_fallback(self):
        for text in (
            'metric_status = metadata["metric_depth"]',
            'update_status = metadata["droid_update"]',
            'status["actual_backend"] != expected_backend',
            'status["fallback_reason"] is not None',
            'update_status["precision"] != "fp16_delta_fp32"',
        ):
            self.assertIn(text, self.content)

    def test_captures_environment_resources_evaluation_and_failures(self):
        for text in (
            "git status --short",
            "git submodule status --recursive",
            "python -m pip freeze",
            "pgrep -af '[s]cripts/run.py|[t]egrastats'",
            "tegrastats --interval 1000",
            "evaluate_smallcity_run.py",
            "compare_tracking_runs.py",
            "PYTORCH_STATUS=$?",
            "TENSORRT_STATUS=$?",
            'exit "$PYTORCH_STATUS"',
            'exit "$TENSORRT_STATUS"',
        ):
            self.assertIn(text, self.content)


if __name__ == "__main__":
    unittest.main()
