import pathlib
import io
import runpy
import sys
import types
import unittest
from contextlib import redirect_stderr


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_run_globals(argv):
    original_argv = sys.argv[:]
    stubbed_modules = {}

    def stub_module(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        stubbed_modules[name] = sys.modules.get(name)
        sys.modules[name] = module
        return module

    class DummyDBAFusion:
        pass

    class DummyGaussianModel:
        pass

    stub_module("lietorch", SE3=object)
    frontend_pkg = stub_module("frontend")
    frontend_dbaf = stub_module("frontend.dbaf", DBAFusion=DummyDBAFusion)
    frontend_pkg.dbaf = frontend_dbaf

    gaussian_pkg = stub_module("gaussian")
    gaussian_general_utils = stub_module(
        "gaussian.general_utils",
        load_config=lambda path: {"mode": "vo", "dataset": {"module": "dummy_dataset_module"}},
        get_name=lambda cfg: "dummy",
    )
    gaussian_model = stub_module("gaussian.gaussian_model", GaussianModel=DummyGaussianModel)
    gaussian_pkg.general_utils = gaussian_general_utils
    gaussian_pkg.gaussian_model = gaussian_model

    stub_module("dummy_dataset_module", get_dataset=lambda cfg: None)
    vings_utils_pkg = stub_module("vings_utils")
    middleware_utils = stub_module(
        "vings_utils.middleware_utils",
        judge_and_package=lambda *args, **kwargs: None,
        retrieve_to_tracker=lambda *args, **kwargs: None,
        datapacket_to_nerfslam=lambda data_packet, idx: data_packet,
    )
    vings_utils_pkg.middleware_utils = middleware_utils

    sys.argv = argv
    try:
        return runpy.run_path(str(REPO_ROOT / "scripts" / "run.py"), run_name="__test__")
    finally:
        sys.argv = original_argv
        for name, previous in stubbed_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _load_metric_globals():
    stubbed_modules = {}

    def stub_module(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        stubbed_modules[name] = sys.modules.get(name)
        sys.modules[name] = module
        return module

    class DummyMetric:
        pass

    stub_module("metric_modules", Metric=DummyMetric)
    try:
        return runpy.run_path(str(REPO_ROOT / "scripts" / "metric" / "metric_model.py"), run_name="__test__")
    finally:
        for name, previous in stubbed_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class JetsonRuntimeOverrideTests(unittest.TestCase):
    def test_run_py_applies_disable_loop_and_training_iters(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--disable-loop",
                "--training-iters",
                "30",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "training_args": {"iters": 50},
            "use_loop": True,
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertFalse(updated["use_loop"])
        self.assertEqual(updated["training_args"]["iters"], 30)

    def test_run_py_can_enable_adaptive_runtime(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--adaptive-runtime",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "training_args": {"iters": 30},
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertTrue(updated["runtime_budget"]["enabled"])

    def test_run_py_can_enable_mapping_budget(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--enable-mapping-budget",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "mapping_budget": {"enabled": False},
            "training_args": {"iters": 30},
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertTrue(updated["mapping_budget"]["enabled"])

    def test_run_py_can_enable_jetson_pruning_budget(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--enable-jetson-pruning",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "pruning_budget": {"enabled": False},
            "training_args": {"iters": 30},
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertTrue(updated["pruning_budget"]["enabled"])

    def test_run_py_can_enable_pixel_budget(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--enable-pixel-budget",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "pixel_budget": {"enabled": False},
            "training_args": {"iters": 30},
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertTrue(updated["pixel_budget"]["enabled"])

    def test_run_py_can_enable_runtime_profiling(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--profile-runtime",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "profiling": {"runtime": {"enabled": False}},
            "training_args": {"iters": 30},
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertTrue(updated["profiling"]["runtime"]["enabled"])

    def test_run_py_can_enable_sampled_dba_memory_profiling(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--profile-runtime",
                "--profile-dba-memory",
                "--profile-dba-memory-samples-per-signature",
                "3",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "profiling": {"runtime": {"enabled": False}},
            "training_args": {"iters": 30},
        }
        updated = globals_dict["apply_overrides"](cfg)

        self.assertTrue(updated["profiling"]["runtime"]["enabled"])
        self.assertTrue(updated["profiling"]["dba_memory"]["enabled"])
        self.assertEqual(
            updated["profiling"]["dba_memory"]["samples_per_signature"],
            3,
        )

    def test_run_py_rejects_dba_memory_without_runtime_profiler(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaisesRegex(SystemExit, "2"):
                _load_run_globals(
                    [
                        "run.py",
                        "dummy.yaml",
                        "--profile-dba-memory",
                    ]
                )

        self.assertIn("requires --profile-runtime", stderr.getvalue())

    def test_run_py_rejects_nonpositive_dba_memory_sample_limit(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaisesRegex(SystemExit, "2"):
                _load_run_globals(
                    [
                        "run.py",
                        "dummy.yaml",
                        "--profile-runtime",
                        "--profile-dba-memory",
                        "--profile-dba-memory-samples-per-signature",
                        "0",
                    ]
                )

        self.assertIn("must be at least 1", stderr.getvalue())

    def test_run_py_can_override_frontend_update_iterations(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--frontend-iters1",
                "3",
                "--frontend-iters2",
                "1",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {"iters1": 4, "iters2": 2},
            "device": {},
            "looper": {},
            "training_args": {"iters": 30},
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertEqual(updated["frontend"]["iters1"], 3)
        self.assertEqual(updated["frontend"]["iters2"], 1)

    def test_run_py_can_override_frontend_save_buffer_size(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--frontend-save-buffer",
                "64",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "training_args": {"iters": 30},
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertEqual(updated["frontend"]["save_buffer_size"], 64)

    def test_run_py_can_enable_adaptive_frontend_update_iterations(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--frontend-adaptive-iters",
                "--frontend-iters1-high",
                "3",
                "--frontend-iters2-high",
                "2",
                "--frontend-iters-high-motion-ratio",
                "2.0",
                "--frontend-iters-force-interval",
                "5",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "training_args": {"iters": 30},
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertTrue(updated["frontend"]["adaptive_iters_enabled"])
        self.assertEqual(updated["frontend"]["iters1_high"], 3)
        self.assertEqual(updated["frontend"]["iters2_high"], 2)
        self.assertEqual(updated["frontend"]["iters_high_motion_ratio"], 2.0)
        self.assertEqual(updated["frontend"]["iters_force_interval"], 5)

    def test_run_py_can_enable_metric_depth_schedule(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--enable-metric-depth-schedule",
                "--metric-depth-warmup",
                "12",
                "--metric-depth-interval",
                "4",
                "--metric-depth-mode",
                "keyframe",
                "--metric-depth-keyframe-min-interval",
                "3",
                "--metric-depth-keyframe-force-interval",
                "10",
                "--metric-depth-high-motion-ratio",
                "2.0",
                "--metric-depth-scale",
                "0.75",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "metric_depth_schedule": {"enabled": False},
            "training_args": {"iters": 30},
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertTrue(updated["metric_depth_schedule"]["enabled"])
        self.assertEqual(updated["metric_depth_schedule"]["warmup_frames"], 12)
        self.assertEqual(updated["metric_depth_schedule"]["interval"], 4)
        self.assertEqual(updated["metric_depth_schedule"]["mode"], "keyframe")
        self.assertEqual(updated["metric_depth_schedule"]["keyframe_min_interval"], 3)
        self.assertEqual(updated["metric_depth_schedule"]["keyframe_force_interval"], 10)
        self.assertEqual(updated["metric_depth_schedule"]["high_motion_ratio"], 2.0)
        self.assertEqual(updated["metric_depth_scale"], 0.75)

    def test_run_py_can_select_metric_depth_tensorrt(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--metric-depth-backend",
                "tensorrt",
                "--metric-depth-engine",
                "engines/tensorrt/metric3d/model.plan",
                "--tensorrt-strict",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "inference": {
                "metric3d_backend": "torch",
                "metric3d_engine": None,
                "tensorrt_strict": False,
            },
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertEqual(
            updated["inference"],
            {
                "metric3d_backend": "tensorrt",
                "metric3d_engine": "engines/tensorrt/metric3d/model.plan",
                "tensorrt_strict": True,
            },
        )

    def test_run_py_preserves_metric_depth_backend_without_cli_overrides(self):
        globals_dict = _load_run_globals(["run.py", "dummy.yaml"])
        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "inference": {
                "metric3d_backend": "tensorrt",
                "metric3d_engine": "existing.plan",
                "tensorrt_strict": True,
            },
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertEqual(updated["inference"]["metric3d_backend"], "tensorrt")
        self.assertEqual(updated["inference"]["metric3d_engine"], "existing.plan")
        self.assertTrue(updated["inference"]["tensorrt_strict"])

    def test_run_py_can_select_droid_cnet_tensorrt(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--droid-cnet-backend",
                "tensorrt",
                "--droid-cnet-engine",
                "engines/tensorrt/droid/cnet.plan",
                "--tensorrt-strict",
            ]
        )
        cfg = {
            "inference": {
                "droid_cnet_backend": "torch",
                "droid_cnet_engine": None,
                "tensorrt_strict": False,
            }
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertEqual(updated["inference"]["droid_cnet_backend"], "tensorrt")
        self.assertEqual(
            updated["inference"]["droid_cnet_engine"],
            "engines/tensorrt/droid/cnet.plan",
        )
        self.assertTrue(updated["inference"]["tensorrt_strict"])

    def test_run_py_can_select_droid_update_tensorrt(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--droid-update-backend",
                "tensorrt",
                "--droid-update-engine",
                "engines/tensorrt/droid/update.plan",
                "--tensorrt-strict",
            ]
        )
        cfg = {
            "inference": {
                "droid_update_backend": "torch",
                "droid_update_engine": None,
                "tensorrt_strict": False,
            }
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertEqual(updated["inference"]["droid_update_backend"], "tensorrt")
        self.assertEqual(
            updated["inference"]["droid_update_engine"],
            "engines/tensorrt/droid/update.plan",
        )
        self.assertTrue(updated["inference"]["tensorrt_strict"])


    def test_run_py_can_enable_eval_export(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--export-eval",
                "--export-eval-interval",
                "5",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
            "training_args": {"iters": 30},
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertTrue(updated["eval_export"]["enabled"])
        self.assertEqual(updated["eval_export"]["interval"], 5)
        self.assertNotEqual(updated.get("use_vis"), True)

    def test_metric_model_exposes_repo_root_locator(self):
        globals_dict = _load_metric_globals()

        self.assertIn("_find_repo_root", globals_dict)
        self.assertEqual(globals_dict["REPO_ROOT"], REPO_ROOT)

    def test_run_py_can_enable_jetson_motion_gate(self):
        globals_dict = _load_run_globals(
            [
                "run.py",
                "dummy.yaml",
                "--enable-jetson-motion-gate",
                "--motion-gate-backend",
                "vpi_cpp",
                "--motion-gate-threshold",
                "1.5",
                "--motion-gate-force-interval",
                "7",
                "--motion-gate-resize",
                "80,128",
                "--motion-gate-grid-size",
                "4",
                "--motion-gate-vpi-levels",
                "1",
                "--motion-gate-vpi-quality",
                "low",
            ]
        )

        cfg = {
            "dataset": {},
            "output": {},
            "frontend": {},
            "device": {},
            "looper": {},
        }

        updated = globals_dict["apply_overrides"](cfg)

        self.assertTrue(updated["jetson_motion_gate"]["enabled"])
        self.assertEqual(updated["jetson_motion_gate"]["backend"], "vpi_cpp")
        self.assertEqual(updated["jetson_motion_gate"]["threshold"], 1.5)
        self.assertEqual(updated["jetson_motion_gate"]["force_interval"], 7)
        self.assertEqual(updated["jetson_motion_gate"]["resize"], [80, 128])
        self.assertEqual(updated["jetson_motion_gate"]["grid_size"], 4)
        self.assertEqual(updated["jetson_motion_gate"]["vpi_num_levels"], 1)
        self.assertEqual(updated["jetson_motion_gate"]["vpi_quality"], "low")


if __name__ == "__main__":
    unittest.main()
