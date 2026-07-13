import pathlib
import sys
import types
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.modules.setdefault("lietorch", types.SimpleNamespace(SE3=None, SO3=None, Sim3=None))
sys.modules.setdefault("droid_backends", types.SimpleNamespace())


class DummyCovisibleGraph:
    def __init__(self, *args, **kwargs):
        pass


sys.modules.setdefault(
    "frontend.covisible_graph",
    types.SimpleNamespace(CovisibleGraph=DummyCovisibleGraph),
)

from scripts.frontend.dbaf_frontend import DBAFusionFrontend


class DummyNet:
    update = object()


class DummyVideo:
    pass


def make_cfg(mode="vo", **frontend_overrides):
    frontend = {
        "warm_up": 8,
        "beta": 0.3,
        "frontend_nms": 2,
        "keyframe_thresh": 2.5,
        "frontend_window": 25,
        "frontend_thresh": 16.0,
        "frontend_radius": 2,
        "active_window": 8,
        "show_plot": False,
    }
    frontend.update(frontend_overrides)
    return {"mode": mode, "frontend": frontend}


class DBAFusionFrontendIterationConfigTests(unittest.TestCase):
    def test_vo_defaults_keep_existing_update_iterations(self):
        frontend = DBAFusionFrontend(DummyNet(), DummyVideo(), make_cfg("vo"))

        self.assertEqual(frontend.iters1, 4)
        self.assertEqual(frontend.iters2, 2)

    def test_frontend_config_can_override_update_iterations(self):
        frontend = DBAFusionFrontend(
            DummyNet(),
            DummyVideo(),
            make_cfg("vo", iters1=3, iters2=1),
        )

        self.assertEqual(frontend.iters1, 3)
        self.assertEqual(frontend.iters2, 1)

    def test_adaptive_iteration_config_defaults_to_disabled(self):
        frontend = DBAFusionFrontend(DummyNet(), DummyVideo(), make_cfg("vo", iters1=3, iters2=1))

        self.assertFalse(frontend.adaptive_iters_enabled)
        self.assertEqual(frontend.select_update_iters(), (3, 1))

    def test_adaptive_iterations_use_low_values_without_high_motion(self):
        video = DummyVideo()
        video.last_motion_score = 3.0
        video.last_motion_threshold = 2.5
        frontend = DBAFusionFrontend(
            DummyNet(),
            video,
            make_cfg(
                "vo",
                adaptive_iters_enabled=True,
                iters1=2,
                iters2=1,
                iters1_high=3,
                iters2_high=2,
                iters_high_motion_ratio=2.0,
            ),
        )

        self.assertEqual(frontend.select_update_iters(), (2, 1))

    def test_adaptive_iterations_use_high_values_for_high_motion(self):
        video = DummyVideo()
        video.last_motion_score = 6.0
        video.last_motion_threshold = 2.5
        frontend = DBAFusionFrontend(
            DummyNet(),
            video,
            make_cfg(
                "vo",
                adaptive_iters_enabled=True,
                iters1=2,
                iters2=1,
                iters1_high=3,
                iters2_high=2,
                iters_high_motion_ratio=2.0,
            ),
        )

        self.assertEqual(frontend.select_update_iters(), (3, 2))

    def test_adaptive_iterations_force_high_values_on_interval(self):
        video = DummyVideo()
        video.last_motion_score = 0.0
        video.last_motion_threshold = 2.5
        frontend = DBAFusionFrontend(
            DummyNet(),
            video,
            make_cfg(
                "vo",
                adaptive_iters_enabled=True,
                iters1=2,
                iters2=1,
                iters1_high=3,
                iters2_high=2,
                iters_force_interval=3,
            ),
        )
        frontend.update_call_count = 2

        self.assertEqual(frontend.select_update_iters(), (3, 2))


if __name__ == "__main__":
    unittest.main()
