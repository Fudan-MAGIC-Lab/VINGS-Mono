import pathlib
import runpy
import sys
import types
import unittest


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

    def test_metric_model_exposes_repo_root_locator(self):
        globals_dict = _load_metric_globals()

        self.assertIn("_find_repo_root", globals_dict)
        self.assertEqual(globals_dict["REPO_ROOT"], REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
