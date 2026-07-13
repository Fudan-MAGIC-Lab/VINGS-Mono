import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "reports/tensorrt_fp16/run_droid_cnet_smallcity50_pair.sh"


class DroidCNetSmallCity50PairScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = SCRIPT.read_text()

    def test_uses_unique_names_and_refuses_overwrite(self):
        self.assertIn("set -uo pipefail", self.content)
        self.assertIn(
            "jetson_smallcity_gt50_pytorch_droid_cnet_online_t10", self.content
        )
        self.assertIn(
            "jetson_smallcity_gt50_tensorrt_droid_cnet_online_t10", self.content
        )
        self.assertIn('if [ -e "$EVIDENCE_DIR" ]', self.content)
        self.assertIn('if [ -e "$output_parent" ]', self.content)
        self.assertNotIn("rm -rf", self.content)
        self.assertNotIn("rm -f", self.content)

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
            "--metric-depth-mode keyframe",
            "--metric-depth-warmup 8",
            "--metric-depth-keyframe-min-interval 3",
            "--metric-depth-keyframe-force-interval 10",
            "--metric-depth-high-motion-ratio 3.0",
            "--enable-jetson-motion-gate",
            "--motion-gate-backend vpi_cpp",
            "--motion-gate-threshold 12.0",
            "--motion-gate-force-interval 8",
            "--motion-gate-resize 96,160",
            "--motion-gate-grid-size 4",
            "--motion-gate-vpi-levels 1",
            "--motion-gate-vpi-quality low",
        ):
            self.assertIn(text, self.content)

    def test_candidate_adds_only_strict_cnet_tensorrt(self):
        for text in (
            "--droid-cnet-backend tensorrt",
            "--droid-cnet-engine engines/tensorrt/droid/droid_cnet_b1_344x616_fp16.plan",
            "--tensorrt-strict",
        ):
            self.assertEqual(self.content.count(text), 1)
        self.assertNotIn("--metric-depth-backend tensorrt", self.content)
        self.assertNotIn("--metric-depth-engine", self.content)

    def test_verifies_actual_backend_and_no_fallback(self):
        for text in (
            "runtime_profile_metadata.json",
            'expected_backend = "torch" if label == "pytorch" else "tensorrt"',
            'status["actual_backend"] != expected_backend',
            'status["fallback_reason"] is not None',
            'not status["strict"]',
        ):
            self.assertIn(text, self.content)

    def test_records_logs_evaluations_comparison_and_failures(self):
        for text in (
            "tegrastats --interval 1000",
            "trap cleanup EXIT INT TERM",
            'run_${label}.log',
            'status_${label}.txt',
            'tegrastats_${label}.log',
            'run_dir_${label}.txt',
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
