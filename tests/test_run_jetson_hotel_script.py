import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_jetson_hotel.sh"


class RunJetsonHotelScriptTests(unittest.TestCase):
    def test_hotel_launcher_exists(self):
        self.assertTrue(SCRIPT_PATH.is_file(), f"Missing launcher script: {SCRIPT_PATH}")

    def test_hotel_launcher_uses_validated_defaults(self):
        content = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("configs/rtg/hotel.yaml", content)
        self.assertIn('loop_onnx_provider="${VINGS_LOOP_ONNX_PROVIDER:-cpu}"', content)
        self.assertIn('frontend_image_size="${VINGS_FRONTEND_IMAGE_SIZE:-256,448}"', content)
        self.assertIn('training_iters="${VINGS_TRAINING_ITERS:-30}"', content)
        self.assertIn('adaptive_runtime="${VINGS_ADAPTIVE_RUNTIME:-1}"', content)
        self.assertIn('lightglue_weight_dir="${VINGS_LIGHTGLUE_DIR:-$shared_root/ckpts/lightglue}"', content)
        self.assertIn('--loop-onnx-provider "$loop_onnx_provider"', content)
        self.assertIn('--frontend-image-size "$frontend_image_size"', content)
        self.assertIn('--training-iters "$training_iters"', content)
        self.assertIn('adaptive_runtime_args+=(--adaptive-runtime)', content)


if __name__ == "__main__":
    unittest.main()
