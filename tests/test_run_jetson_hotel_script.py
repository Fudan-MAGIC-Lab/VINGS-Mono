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
        self.assertIn('frontend_iters1="${VINGS_FRONTEND_ITERS1:-3}"', content)
        self.assertIn('frontend_iters2="${VINGS_FRONTEND_ITERS2:-1}"', content)
        self.assertIn('frontend_adaptive_iters="${VINGS_FRONTEND_ADAPTIVE_ITERS:-0}"', content)
        self.assertIn('frontend_iters1_high="${VINGS_FRONTEND_ITERS1_HIGH:-3}"', content)
        self.assertIn('frontend_iters2_high="${VINGS_FRONTEND_ITERS2_HIGH:-1}"', content)
        self.assertIn('frontend_iters_high_motion_ratio="${VINGS_FRONTEND_ITERS_HIGH_MOTION_RATIO:-2.0}"', content)
        self.assertIn('frontend_iters_force_interval="${VINGS_FRONTEND_ITERS_FORCE_INTERVAL:-0}"', content)
        self.assertIn('training_iters="${VINGS_TRAINING_ITERS:-30}"', content)
        self.assertIn('adaptive_runtime="${VINGS_ADAPTIVE_RUNTIME:-1}"', content)
        self.assertIn('mapping_budget="${VINGS_MAPPING_BUDGET:-0}"', content)
        self.assertIn('pruning_budget="${VINGS_PRUNING_BUDGET:-0}"', content)
        self.assertIn('pixel_budget="${VINGS_PIXEL_BUDGET:-0}"', content)
        self.assertIn('profile_runtime="${VINGS_PROFILE_RUNTIME:-0}"', content)
        self.assertIn('metric_depth_schedule="${VINGS_METRIC_DEPTH_SCHEDULE:-0}"', content)
        self.assertIn('metric_depth_warmup="${VINGS_METRIC_DEPTH_WARMUP:-30}"', content)
        self.assertIn('metric_depth_interval="${VINGS_METRIC_DEPTH_INTERVAL:-5}"', content)
        self.assertIn('metric_depth_mode="${VINGS_METRIC_DEPTH_MODE:-keyframe}"', content)
        self.assertIn('metric_depth_keyframe_min_interval="${VINGS_METRIC_DEPTH_KEYFRAME_MIN_INTERVAL:-3}"', content)
        self.assertIn('metric_depth_keyframe_force_interval="${VINGS_METRIC_DEPTH_KEYFRAME_FORCE_INTERVAL:-10}"', content)
        self.assertIn('metric_depth_high_motion_ratio="${VINGS_METRIC_DEPTH_HIGH_MOTION_RATIO:-3.0}"', content)
        self.assertIn('metric_depth_scale="${VINGS_METRIC_DEPTH_SCALE:-0.75}"', content)
        self.assertIn('lightglue_weight_dir="${VINGS_LIGHTGLUE_DIR:-$shared_root/ckpts/lightglue}"', content)
        self.assertIn('--loop-onnx-provider "$loop_onnx_provider"', content)
        self.assertIn('--frontend-image-size "$frontend_image_size"', content)
        self.assertIn('--frontend-iters1 "$frontend_iters1"', content)
        self.assertIn('--frontend-iters2 "$frontend_iters2"', content)
        self.assertIn('frontend_adaptive_iters_args+=(--frontend-adaptive-iters)', content)
        self.assertIn('frontend_adaptive_iters_args+=(--frontend-iters1-high "$frontend_iters1_high")', content)
        self.assertIn('frontend_adaptive_iters_args+=(--frontend-iters2-high "$frontend_iters2_high")', content)
        self.assertIn('frontend_adaptive_iters_args+=(--frontend-iters-high-motion-ratio "$frontend_iters_high_motion_ratio")', content)
        self.assertIn('frontend_adaptive_iters_args+=(--frontend-iters-force-interval "$frontend_iters_force_interval")', content)
        self.assertIn('--training-iters "$training_iters"', content)
        self.assertIn('adaptive_runtime_args+=(--adaptive-runtime)', content)
        self.assertIn('mapping_budget_args+=(--enable-mapping-budget)', content)
        self.assertIn('pruning_budget_args+=(--enable-jetson-pruning)', content)
        self.assertIn('pixel_budget_args+=(--enable-pixel-budget)', content)
        self.assertIn('profile_runtime_args+=(--profile-runtime)', content)
        self.assertIn('metric_depth_schedule_args+=(--enable-metric-depth-schedule)', content)
        self.assertIn('metric_depth_schedule_args+=(--metric-depth-warmup "$metric_depth_warmup")', content)
        self.assertIn('metric_depth_schedule_args+=(--metric-depth-interval "$metric_depth_interval")', content)
        self.assertIn('metric_depth_schedule_args+=(--metric-depth-mode "$metric_depth_mode")', content)
        self.assertIn('metric_depth_schedule_args+=(--metric-depth-keyframe-min-interval "$metric_depth_keyframe_min_interval")', content)
        self.assertIn('metric_depth_schedule_args+=(--metric-depth-keyframe-force-interval "$metric_depth_keyframe_force_interval")', content)
        self.assertIn('metric_depth_schedule_args+=(--metric-depth-high-motion-ratio "$metric_depth_high_motion_ratio")', content)
        self.assertIn('--metric-depth-scale "$metric_depth_scale"', content)


if __name__ == "__main__":
    unittest.main()
