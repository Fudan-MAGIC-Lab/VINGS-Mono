import importlib
import pathlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest import mock

import torch

from tests.module_stubs import isolated_modules

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
with isolated_modules(
    {
        "lietorch": types.SimpleNamespace(SE3=None, SO3=None, Sim3=None),
        "droid_backends": types.SimpleNamespace(),
        "frontend.droid_net": types.SimpleNamespace(DroidNet=object),
    },
    reload_modules=("frontend.motion_filter",),
):
    _motion_filter_module = importlib.import_module("frontend.motion_filter")
    MotionFilter = _motion_filter_module.MotionFilter


class FakeSE3:
    @staticmethod
    def Identity(*args, **kwargs):
        return types.SimpleNamespace(data=torch.zeros(7))


class FakeVideo:
    def __init__(self, counter_value):
        self.counter = types.SimpleNamespace(value=counter_value)
        self.appended_depths = []

    def append(self, *item):
        self.appended_depths.append(item[4])
        self.counter.value += 1


class RecordingProfiler:
    def __init__(self):
        self.stages = []

    @contextmanager
    def time(self, stage, frame_idx=None):
        self.stages.append((stage, frame_idx))
        yield


class MotionFilterLazyDepthTests(unittest.TestCase):
    def make_filter(self, counter_value, motion_value):
        filt = MotionFilter.__new__(MotionFilter)
        filt.video = FakeVideo(counter_value=counter_value)
        filt.thresh = 2.5
        filt.device = "cpu"
        filt.count = 0
        filt.MEAN = torch.zeros(3, 1, 1)
        filt.STDV = torch.ones(3, 1, 1)
        filt.net = torch.zeros(1, 1, 1, 1)
        filt.inp = torch.zeros(1, 1, 1, 1)
        filt.fmap = torch.zeros(1, 1, 1, 1)
        filt._motion_value = motion_value
        filt.profiler = None
        filt.profiler_frame_idx = None
        return filt

    def patch_filter_ops(self):
        def fake_feature_encoder(filt, image):
            return torch.zeros(1, 1, 1, 1)

        def fake_context_encoder(filt, image):
            return torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)

        def fake_update(net, inp, corr):
            value = float(self.current_filter._motion_value)
            return None, torch.full((1, 1, 1, 1, 1), value), None

        patches = [
            mock.patch.object(MotionFilter, "_MotionFilter__feature_encoder", fake_feature_encoder),
            mock.patch.object(MotionFilter, "_MotionFilter__context_encoder", fake_context_encoder),
            mock.patch.object(_motion_filter_module.lietorch, "SE3", FakeSE3),
            mock.patch.object(_motion_filter_module.pops, "coords_grid", lambda *args, **kwargs: torch.zeros(1, 1, 1, 2)),
            mock.patch.object(_motion_filter_module, "CorrBlock", lambda *args, **kwargs: (lambda coords: torch.zeros(1, 1, 1, 1))),
        ]
        return patches, fake_update

    def run_track(self, filt, depth_provider):
        self.current_filter = filt
        patches, fake_update = self.patch_filter_ops()
        filt.update = fake_update
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            image = torch.zeros(1, 3, 16, 16)
            intrinsics = torch.ones(4)
            filt.track(0, image, depth_provider, intrinsics)

    def test_initial_append_realizes_lazy_depth(self):
        calls = []
        filt = self.make_filter(counter_value=0, motion_value=0.0)
        self.run_track(filt, lambda: calls.append("predict") or "depth")
        self.assertEqual(calls, ["predict"])
        self.assertEqual(filt.video.appended_depths, ["depth"])

    def test_motion_append_realizes_lazy_depth(self):
        calls = []
        filt = self.make_filter(counter_value=1, motion_value=3.0)
        self.run_track(filt, lambda: calls.append("predict") or "depth")
        self.assertEqual(calls, ["predict"])
        self.assertEqual(filt.video.appended_depths, ["depth"])

    def test_motion_append_passes_motion_context_to_lazy_depth(self):
        contexts = []
        filt = self.make_filter(counter_value=1, motion_value=6.0)

        def depth_provider(**context):
            contexts.append(context)
            return "depth"

        self.run_track(filt, depth_provider)

        self.assertEqual(filt.video.appended_depths, ["depth"])
        self.assertEqual(contexts, [{"motion_score": 6.0, "motion_threshold": 2.5}])
        self.assertEqual(filt.video.last_motion_score, 6.0)
        self.assertEqual(filt.video.last_motion_threshold, 2.5)
        self.assertTrue(filt.video.last_motion_added_keyframe)

    def test_non_keyframe_does_not_realize_lazy_depth(self):
        calls = []
        filt = self.make_filter(counter_value=1, motion_value=1.0)
        self.run_track(filt, lambda: calls.append("predict") or "depth")
        self.assertEqual(calls, [])
        self.assertEqual(filt.video.appended_depths, [])
        self.assertEqual(filt.video.last_motion_score, 1.0)
        self.assertEqual(filt.video.last_motion_threshold, 2.5)
        self.assertFalse(filt.video.last_motion_added_keyframe)

    def test_initial_frame_records_feature_context_append_and_lazy_depth_stages(self):
        profiler = RecordingProfiler()
        filt = self.make_filter(counter_value=0, motion_value=0.0)
        filt.profiler = profiler
        filt.profiler_frame_idx = 11

        self.run_track(filt, lambda: "depth")

        self.assertEqual(
            profiler.stages,
            [
                ("motion_filter_feature_encoder", 11),
                ("motion_filter_context_encoder", 11),
                ("lazy_metric_depth_inside_tracking", 11),
                ("motion_filter_append", 11),
            ],
        )

    def test_motion_keyframe_records_corr_update_context_append_and_lazy_depth_stages(self):
        profiler = RecordingProfiler()
        filt = self.make_filter(counter_value=1, motion_value=3.0)
        filt.profiler = profiler
        filt.profiler_frame_idx = 12

        self.run_track(filt, lambda: "depth")

        self.assertEqual(
            profiler.stages,
            [
                ("motion_filter_feature_encoder", 12),
                ("motion_filter_corr_update", 12),
                ("motion_filter_context_encoder", 12),
                ("lazy_metric_depth_inside_tracking", 12),
                ("motion_filter_append", 12),
            ],
        )


if __name__ == "__main__":
    unittest.main()
