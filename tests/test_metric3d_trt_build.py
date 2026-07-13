import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "reports" / "tensorrt_fp16" / "build_metric3d_fp16.sh"


class Metric3DTensorRTBuildScriptTests(unittest.TestCase):
    def test_build_uses_static_metric3d_fp16_artifacts(self):
        text = BUILD_SCRIPT.read_text()

        self.assertIn(
            "engines/tensorrt/metric3d/metric3d_v2s_448x784_opset16.onnx", text
        )
        self.assertNotIn(
            "ONNX=engines/tensorrt/metric3d/metric3d_v2s_448x784.onnx", text
        )
        self.assertIn("engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan", text)
        self.assertIn("--fp16", text)
        self.assertIn("--memPoolSize=workspace:1024MiB", text)
        self.assertNotIn("droid_update", text)

    def test_build_has_dedicated_logs_and_status(self):
        text = BUILD_SCRIPT.read_text()

        self.assertIn("metric3d_opset16_trtexec_build.log", text)
        self.assertIn("metric3d_opset16_trtexec_tegrastats.log", text)
        self.assertIn("metric3d_opset16_trtexec_status.txt", text)

    def test_build_is_guarded_and_reproducible(self):
        text = BUILD_SCRIPT.read_text()

        self.assertIn("[s]cripts/run.py|[t]rtexec|[t]egrastats", text)
        self.assertIn("trap cleanup EXIT", text)
        self.assertIn("--buildOnly", text)
        self.assertIn("--verbose", text)
        self.assertIn("--dumpLayerInfo", text)


if __name__ == "__main__":
    unittest.main()
