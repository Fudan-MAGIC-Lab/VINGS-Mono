import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "reports/tensorrt_fp16/run_metric3d_smallcity50_pair.sh"


class Metric3DSmallCity50PairScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = SCRIPT.read_text()

    def test_uses_unique_immutable_run_names_and_refuses_overwrite(self):
        self.assertIn("set -uo pipefail", self.content)
        self.assertIn("jetson_smallcity_gt50_pytorch_metric3d_online_t10", self.content)
        self.assertIn("jetson_smallcity_gt50_tensorrt_metric3d_online_t10", self.content)
        self.assertIn('if [ -e "$EVIDENCE_DIR" ]', self.content)
        self.assertIn('if [ -e "$output_parent" ]', self.content)
        self.assertNotIn("rm -rf", self.content)
        self.assertNotIn("rm -f", self.content)

    def test_captures_environment_and_rejects_residual_processes(self):
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

    def test_candidate_adds_strict_metric3d_tensorrt_only(self):
        for text in (
            "--metric-depth-backend tensorrt",
            "--metric-depth-engine engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan",
            "--tensorrt-strict",
        ):
            self.assertEqual(self.content.count(text), 1)

    def test_records_separate_logs_status_resources_and_evaluation(self):
        for text in (
            "tegrastats --interval 1000",
            "trap cleanup EXIT INT TERM",
            'run_${label}.log',
            'status_${label}.txt',
            'tegrastats_${label}.log',
            'run_dir_${label}.txt',
            "evaluate_smallcity_run.py",
            "compare_tracking_runs.py",
        ):
            self.assertIn(text, self.content)

    def test_propagates_control_and_candidate_failures(self):
        self.assertNotIn("if ! run_variant", self.content)
        self.assertIn("PYTORCH_STATUS=$?", self.content)
        self.assertIn("TENSORRT_STATUS=$?", self.content)
        self.assertIn('exit "$PYTORCH_STATUS"', self.content)
        self.assertIn('exit "$TENSORRT_STATUS"', self.content)


if __name__ == "__main__":
    unittest.main()
