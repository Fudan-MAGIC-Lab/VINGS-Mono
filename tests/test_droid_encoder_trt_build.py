import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "reports/tensorrt_fp16/build_droid_encoders_fp16.sh"


class DroidEncoderTensorRTBuildScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = SCRIPT.read_text()

    def test_builds_fnet_and_cnet_independently_in_order(self):
        self.assertIn("build_encoder fnet", self.content)
        self.assertIn("build_encoder cnet", self.content)
        self.assertLess(
            self.content.index("build_encoder fnet"),
            self.content.index("build_encoder cnet"),
        )
        self.assertIn("FNET_STATUS=$?", self.content)
        self.assertIn('exit "$FNET_STATUS"', self.content)

    def test_can_build_one_encoder_without_requiring_the_other(self):
        self.assertIn('if [ "$#" -eq 1 ]', self.content)
        self.assertIn('build_encoder "$1"', self.content)
        self.assertIn("unsupported encoder", self.content)

    def test_uses_exact_static_artifact_and_evidence_names(self):
        for text in (
            'droid_${encoder}_b1_344x616.onnx',
            'droid_${encoder}_b1_344x616_fp16.plan',
            'droid_${encoder}_trtexec_build.log',
            'droid_${encoder}_trtexec_tegrastats.log',
            'droid_${encoder}_trtexec_status.txt',
        ):
            self.assertIn(text, self.content)

    def test_uses_guarded_fp16_build_only_command(self):
        for text in (
            "set -uo pipefail",
            "pgrep -af '[s]cripts/run.py|[t]rtexec|[t]egrastats'",
            'if [ ! -f "$onnx" ]',
            'if [ -e "$plan" ]',
            "/usr/src/tensorrt/bin/trtexec",
            '--onnx="$onnx"',
            '--saveEngine="$plan"',
            "--fp16",
            "--memPoolSize=workspace:1024MiB",
            "--buildOnly",
            "--verbose",
            "--dumpLayerInfo",
            "tegrastats --interval 1000",
            "trap cleanup EXIT INT TERM",
        ):
            self.assertIn(text, self.content)

    def test_never_overwrites_or_deletes_engine_artifacts(self):
        self.assertNotIn("rm -rf", self.content)
        self.assertNotIn('rm -f "$plan"', self.content)
        self.assertNotIn('rm -f "$onnx"', self.content)


if __name__ == "__main__":
    unittest.main()
