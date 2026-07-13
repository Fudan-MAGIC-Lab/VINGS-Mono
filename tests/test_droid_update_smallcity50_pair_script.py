import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "reports/tensorrt_fp16/run_droid_update_smallcity50_pair.sh"


class DroidUpdateSmallCity50PairScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = SCRIPT.read_text()

    def test_uses_unique_names_and_refuses_overwrite(self):
        self.assertIn("set -uo pipefail", self.content)
        self.assertIn(
            "jetson_smallcity_gt50_pytorch_droid_update_online_t10", self.content
        )
        self.assertIn(
            "jetson_smallcity_gt50_tensorrt_droid_update_online_t10", self.content
        )
        self.assertIn('if [ -e "$EVIDENCE_DIR" ]', self.content)
        self.assertIn('if [ -e "$output_parent" ]', self.content)
        self.assertNotIn("rm -rf", self.content)
        self.assertNotIn("rm -f", self.content)

    def test_retry_uses_new_paths_without_overwriting_failed_evidence(self):
        self.assertIn("droid_update_smallcity50_online_20260712_retry2", self.content)
        self.assertIn("smallcity_droid_update_online_20260712_retry2", self.content)

    def test_captures_environment_and_guards_processes(self):
        for text in (
            "git status --short",
            "git submodule status --recursive",
            "git rev-parse HEAD",
            "python -m pip freeze",
            "pgrep -af '[s]cripts/run.py|[t]egrastats'",
        ):
            self.assertIn(text, self.content)

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

    def test_candidate_adds_only_strict_update_tensorrt(self):
        for text in (
            "--droid-update-backend tensorrt",
            "--droid-update-engine engines/tensorrt/droid/droid_update_core_43x77_deltafp32.plan",
            "--tensorrt-strict",
        ):
            self.assertEqual(self.content.count(text), 1)
        self.assertNotIn("--metric-depth-backend tensorrt", self.content)
        self.assertNotIn("--droid-cnet-backend tensorrt", self.content)

    def test_verifies_actual_backend_precision_and_no_fallback(self):
        for text in (
            'status = metadata["droid_update"]',
            'expected_backend = "torch" if label == "pytorch" else "tensorrt"',
            'status["actual_backend"] != expected_backend',
            'status["fallback_reason"] is not None',
            'status["precision"] != "fp16_delta_fp32"',
        ):
            self.assertIn(text, self.content)

    def test_records_resources_evaluation_comparison_and_failures(self):
        for text in (
            "tegrastats --interval 1000",
            "trap cleanup EXIT INT TERM",
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
