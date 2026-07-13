import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from acceleration.verify_droid_delta_precision import verify_delta_precision_log


BUILD_SCRIPT = REPO_ROOT / "reports" / "tensorrt_fp16" / "build_droid_update_deltafp32.sh"
DELTA_LAYERS = (
    "/delta/delta.0/Conv",
    "/delta/delta.1/Relu",
    "/delta/delta.2/Conv",
)

VALID_LOG = """
Layer(CaskConvolution): /delta/delta.0/Conv + /delta/delta.1/Relu, input (Float[-1,128,43,77]) -> hidden (Float[-1,128,43,77])
Layer(CaskConvolution): /delta/delta.2/Conv, hidden (Float[-1,128,43,77]) -> delta (Float[-1,2,43,77])
Layer(CaskConvolution): /weight/weight.0/Conv + /weight/weight.1/Relu, input (Half[-1,128,43,77]) -> hidden (Half[-1,128,43,77])
Layer(CaskConvolution): /weight/weight.2/Conv + Sigmoid, hidden (Half[-1,128,43,77]) -> weight (Half[-1,2,43,77])
"""


class DeltaFp32BuildScriptTests(unittest.TestCase):
    def test_build_uses_strict_delta_fp32_constraints(self):
        text = BUILD_SCRIPT.read_text()

        self.assertIn("--fp16", text)
        self.assertIn("--precisionConstraints=obey", text)
        self.assertNotIn("--precisionConstraints=prefer", text)
        self.assertIn("--layerPrecisions=", text)
        self.assertIn("--layerOutputTypes=", text)
        for name in DELTA_LAYERS:
            self.assertEqual(text.count(f"{name}:fp32"), 1)
        self.assertEqual(text.count("$DELTA_FP32"), 2)

    def test_build_is_isolated_from_pure_fp16_artifact(self):
        text = BUILD_SCRIPT.read_text()

        self.assertIn("droid_update_core_43x77_deltafp32.plan", text)
        self.assertNotIn("droid_update_core_43x77_fp16.plan", text)
        self.assertIn("droid_update_deltafp32_trtexec_build.log", text)
        self.assertIn("droid_update_deltafp32_trtexec_tegrastats.log", text)
        self.assertIn("droid_update_deltafp32_trtexec_status.txt", text)

    def test_build_preserves_profiles_and_guards_processes(self):
        text = BUILD_SCRIPT.read_text()

        self.assertIn("--minShapes=net:1x128x43x77", text)
        self.assertIn("--optShapes=net:16x128x43x77", text)
        self.assertIn("--maxShapes=net:48x128x43x77", text)
        self.assertIn("[s]cripts/run.py|[t]rtexec|[t]egrastats", text)
        self.assertIn("trap cleanup EXIT", text)
        self.assertIn("--dumpLayerInfo", text)


class DeltaFp32PrecisionLogTests(unittest.TestCase):
    def test_accepts_separate_fp32_delta_and_fp16_weight(self):
        self.assertTrue(verify_delta_precision_log(VALID_LOG))

    def test_rejects_delta_weight_fusion(self):
        fused = VALID_LOG + (
            "Layer(CaskConvolution): /delta/delta.0/Conv || /weight/weight.0/Conv "
            "(Float[-1,128,43,77])\n"
        )

        with self.assertRaisesRegex(ValueError, "fused"):
            verify_delta_precision_log(fused)

    def test_rejects_half_delta_layer(self):
        half_delta = VALID_LOG.replace(
            "hidden (Float[-1,128,43,77]) -> delta (Float[-1,2,43,77])",
            "hidden (Half[-1,128,43,77]) -> delta (Half[-1,2,43,77])",
        )

        with self.assertRaisesRegex(ValueError, "FP32"):
            verify_delta_precision_log(half_delta)


if __name__ == "__main__":
    unittest.main()
