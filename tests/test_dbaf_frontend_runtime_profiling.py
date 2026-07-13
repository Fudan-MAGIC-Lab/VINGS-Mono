import importlib
import pathlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock

import numpy as np
import torch

from tests.module_stubs import isolated_modules


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
with isolated_modules(
    {
        "lietorch": types.SimpleNamespace(SE3=None, SO3=None, Sim3=None),
        "droid_backends": types.SimpleNamespace(),
        "frontend.covisible_graph": types.SimpleNamespace(CovisibleGraph=object),
    },
    reload_modules=("scripts.frontend.dbaf_frontend",),
):
    _dbaf_frontend_module = importlib.import_module(
        "scripts.frontend.dbaf_frontend"
    )
    DBAFusionFrontend = _dbaf_frontend_module.DBAFusionFrontend


class RecordingProfiler:
    def __init__(self):
        self.stages = []

    @contextmanager
    def time(self, stage, frame_idx=None):
        self.stages.append((stage, frame_idx))
        yield


class FakePoseBatch:
    def __init__(self, poses):
        self.poses = poses

    def __getitem__(self, item):
        return self

    def __mul__(self, other):
        return self

    def cpu(self):
        return self

    def inv(self):
        return self

    def matrix(self):
        return np.eye(4)

    def translation(self):
        return torch.zeros(3, 3)


class FakeGraph:
    def __init__(self):
        self.corr = object()
        self.age = torch.tensor([0, 30, 40])
        self.ii = torch.tensor([0, 1, 2], dtype=torch.int64)
        self.jj = torch.tensor([1, 2, 3], dtype=torch.int64)
        self.calls = []

    def rm_factors(self, mask, store=True):
        self.calls.append(("rm_factors", store))

    def add_proximity_factors(self, *args, **kwargs):
        self.calls.append(("add_proximity_factors", args, kwargs))

    def update(self, *args, **kwargs):
        self.calls.append(("update", args, kwargs))

    def rm_keyframe(self, index):
        self.calls.append(("rm_keyframe", index))


class FakeVideo:
    def __init__(self):
        self.imu_enabled = False
        self.tstamp = torch.arange(8, dtype=torch.float32)
        self.vi_init_time = 1e9
        self.reinit = False
        self.state = None
        self.disps = torch.ones(8, dtype=torch.float32)
        self.disps_sens = torch.zeros(8, dtype=torch.float32)
        self.poses = torch.zeros(8, 7, dtype=torch.float32)
        self.Ti1c = np.eye(4)
        self.gnss_init_t1 = -1
        self.ten0 = np.zeros(3)
        self.gnss_init_time = -1.0
        self.dirty = torch.zeros(8, dtype=torch.bool)
        self.counter = types.SimpleNamespace(value=6)

    def distance(self, *args, **kwargs):
        return torch.tensor(10.0)

    @contextmanager
    def get_lock(self):
        yield


class DBAFusionFrontendRuntimeProfilingTests(unittest.TestCase):
    def make_frontend(self):
        frontend = DBAFusionFrontend.__new__(DBAFusionFrontend)
        frontend.video = FakeVideo()
        frontend.graph = FakeGraph()
        frontend.profiler = RecordingProfiler()
        frontend.profiler_frame_idx = 13
        frontend.new_frame_added = False
        frontend.count = 4
        frontend.t1 = 4
        frontend.visual_only = True
        frontend.visual_only_init = False
        frontend.high_freq_output = False
        frontend.zupt = False
        frontend.max_age = 25
        frontend.active_window = 8
        frontend.frontend_window = 4
        frontend.frontend_radius = 2
        frontend.frontend_nms = 2
        frontend.frontend_thresh = 16.0
        frontend.beta = 0.3
        frontend.iters1 = 2
        frontend.iters2 = 1
        frontend.adaptive_iters_enabled = False
        frontend.iters1_high = 2
        frontend.iters2_high = 1
        frontend.iters_high_motion_ratio = 2.0
        frontend.iters_force_interval = 0
        frontend.update_call_count = 0
        frontend.rollup = False
        frontend.refTw = np.eye(4)
        frontend.show_plot = False
        frontend.all_gt = None
        frontend.keyframe_thresh = 1.0
        frontend.vi_warmup = 12
        frontend.all_gnss = []
        return frontend

    def test_update_records_internal_frontend_stages_for_new_keyframe(self):
        frontend = self.make_frontend()

        with mock.patch.object(_dbaf_frontend_module, "SE3", FakePoseBatch):
            frontend._DBAFusionFrontend__update()

        self.assertEqual(
            frontend.profiler.stages,
            [
                ("frontend_update_bookkeeping", 13),
                ("frontend_update_imu_ingest", 13),
                ("frontend_update_manage_edges", 13),
                ("frontend_update_add_proximity_factors", 13),
                ("frontend_update_non_keyframe_prepare", 13),
                ("frontend_update_non_keyframe_graph_update", 13),
                ("frontend_update_output_pose_distance", 13),
                ("frontend_update_keyframe_decision", 13),
                ("frontend_update_keyframe_graph_update", 13),
                ("frontend_update_init_checks", 13),
                ("frontend_update_finalize_state", 13),
            ],
        )
        self.assertTrue(frontend.new_frame_added)

    def test_update_uses_adaptive_high_iteration_counts_for_high_motion(self):
        frontend = self.make_frontend()
        frontend.adaptive_iters_enabled = True
        frontend.iters1 = 1
        frontend.iters2 = 1
        frontend.iters1_high = 3
        frontend.iters2_high = 2
        frontend.iters_high_motion_ratio = 2.0
        frontend.video.last_motion_score = 6.0
        frontend.video.last_motion_threshold = 2.5

        with mock.patch.object(_dbaf_frontend_module, "SE3", FakePoseBatch):
            frontend._DBAFusionFrontend__update()

        update_calls = [call for call in frontend.graph.calls if call[0] == "update"]
        self.assertEqual(len(update_calls), 5)


if __name__ == "__main__":
    unittest.main()
