import importlib
import pathlib
import sys
import types
import unittest
from contextlib import contextmanager

from tests.module_stubs import isolated_modules


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
with isolated_modules(
    {
        "lietorch": types.SimpleNamespace(SE3=None, SO3=None, Sim3=None),
        "droid_backends": types.SimpleNamespace(),
        "frontend.droid_net": types.SimpleNamespace(DroidNet=object),
        "frontend.depth_video": types.SimpleNamespace(DepthVideo=object),
        "frontend.motion_filter": types.SimpleNamespace(MotionFilter=object),
        "frontend.dbaf_frontend": types.SimpleNamespace(DBAFusionFrontend=object),
        "frontend.motion_gate": types.SimpleNamespace(JetsonMotionGate=object),
    },
    reload_modules=("frontend.dbaf",),
):
    DBAFusion = importlib.import_module("frontend.dbaf").DBAFusion


class RecordingProfiler:
    def __init__(self):
        self.stages = []
        self.metadata = {}

    @contextmanager
    def time(self, stage, frame_idx=None):
        self.stages.append((stage, frame_idx))
        yield

    def set_metadata(self, key, value):
        self.metadata[key] = value


class FakeMotionFilter:
    def __init__(self):
        self.calls = []
        self.profiler = None
        self.profiler_frame_idx = None

    def track(self, tstamp, image, depth, intrinsic):
        self.calls.append((tstamp, image, depth, intrinsic))


class FakeFrontend:
    def __init__(self):
        self.calls = 0
        self.profiler = None
        self.profiler_frame_idx = None
        self.graph = types.SimpleNamespace(profiler=None, profiler_frame_idx=None)

    def __call__(self):
        self.calls += 1


class FakeMotionGate:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def decide(self, image):
        self.calls.append(image)
        return self.decisions.pop(0)


class DBAFusionRuntimeProfilingTests(unittest.TestCase):
    def test_set_runtime_profiler_propagates_to_tracking_components(self):
        tracker = DBAFusion.__new__(DBAFusion)
        tracker.filterx = FakeMotionFilter()
        tracker.frontend = FakeFrontend()
        tracker.net = types.SimpleNamespace(
            update=types.SimpleNamespace(profiler=None, profiler_frame_idx=None)
        )
        profiler = RecordingProfiler()

        tracker.set_runtime_profiler(profiler)
        tracker.set_runtime_profiler_frame_idx(9)

        self.assertIs(tracker.profiler, profiler)
        self.assertIs(tracker.filterx.profiler, profiler)
        self.assertIs(tracker.frontend.profiler, profiler)
        self.assertIs(tracker.frontend.graph.profiler, profiler)
        self.assertIs(tracker.net.update.profiler, profiler)
        self.assertEqual(tracker.profiler_frame_idx, 9)
        self.assertEqual(tracker.filterx.profiler_frame_idx, 9)
        self.assertEqual(tracker.frontend.profiler_frame_idx, 9)
        self.assertEqual(tracker.frontend.graph.profiler_frame_idx, 9)
        self.assertEqual(tracker.net.update.profiler_frame_idx, 9)

    def test_set_runtime_profiler_records_cnet_actual_backend(self):
        tracker = DBAFusion.__new__(DBAFusion)
        tracker.filterx = FakeMotionFilter()
        tracker.frontend = FakeFrontend()
        tracker.net = types.SimpleNamespace(
            update=types.SimpleNamespace(profiler=None, profiler_frame_idx=None)
        )
        tracker.cnet_backend = None
        tracker.cnet_backend_status = {
            "requested_backend": "torch",
            "actual_backend": "torch",
            "strict": False,
            "engine_path": None,
            "engine_sha256": None,
            "fallback_reason": None,
        }
        profiler = RecordingProfiler()

        tracker.set_runtime_profiler(profiler)

        self.assertEqual(
            profiler.metadata["droid_cnet"]["actual_backend"], "torch"
        )

    def test_set_runtime_profiler_records_update_actual_backend(self):
        tracker = DBAFusion.__new__(DBAFusion)
        tracker.filterx = FakeMotionFilter()
        tracker.frontend = FakeFrontend()
        tracker.net = types.SimpleNamespace(
            update=types.SimpleNamespace(profiler=None, profiler_frame_idx=None)
        )
        tracker.update_backend = None
        tracker.update_backend_status = {
            "requested_backend": "torch",
            "actual_backend": "torch",
            "strict": False,
            "engine_path": None,
            "engine_sha256": None,
            "precision": None,
            "fallback_reason": None,
        }
        profiler = RecordingProfiler()

        tracker.set_runtime_profiler(profiler)

        self.assertEqual(
            profiler.metadata["droid_update"]["actual_backend"], "torch"
        )

    def test_set_runtime_profiler_records_dba_memory_configuration(self):
        tracker = DBAFusion.__new__(DBAFusion)
        tracker.filterx = FakeMotionFilter()
        tracker.frontend = FakeFrontend()
        tracker.net = types.SimpleNamespace(
            update=types.SimpleNamespace(
                profiler=None,
                profiler_frame_idx=None,
            )
        )
        tracker.cfg = {
            "device": {"tracker": "cuda:0"},
            "profiling": {
                "dba_memory": {
                    "enabled": True,
                    "samples_per_signature": 2,
                }
            },
        }
        profiler = RecordingProfiler()

        tracker.set_runtime_profiler(profiler)

        self.assertEqual(
            profiler.metadata["dba_memory"],
            {
                "enabled": True,
                "samples_per_signature": 2,
                "tracker_device": "cuda:0",
            },
        )

    def test_track_records_motion_filter_and_frontend_update_stages(self):
        tracker = DBAFusion.__new__(DBAFusion)
        tracker.filterx = FakeMotionFilter()
        tracker.frontend = FakeFrontend()
        tracker.motion_gate = None
        tracker.profiler = RecordingProfiler()
        tracker.profiler_frame_idx = 7

        data_packet = {
            "timestamp": 1.25,
            "rgb": "image",
            "intrinsic": "intrinsic",
            "depth": "depth",
        }

        tracker.track(data_packet)

        self.assertEqual(
            tracker.profiler.stages,
            [
                ("frontend_motion_filter", 7),
                ("frontend_dba_update", 7),
            ],
        )
        self.assertEqual(tracker.filterx.calls, [(1.25, "image", "depth", "intrinsic")])
        self.assertEqual(tracker.frontend.calls, 1)

    def test_track_skips_motion_filter_and_frontend_when_motion_gate_skips(self):
        tracker = DBAFusion.__new__(DBAFusion)
        tracker.filterx = FakeMotionFilter()
        tracker.frontend = FakeFrontend()
        tracker.motion_gate = FakeMotionGate(
            [{"skip": True, "score": 0.1, "reason": "low_motion", "backend": "opencv"}]
        )
        tracker.profiler = RecordingProfiler()
        tracker.profiler_frame_idx = 8

        data_packet = {
            "timestamp": 1.5,
            "rgb": "image",
            "intrinsic": "intrinsic",
            "depth": "depth",
        }

        tracker.track(data_packet)

        self.assertEqual(
            tracker.profiler.stages,
            [
                ("frontend_motion_gate", 8),
            ],
        )
        self.assertEqual(tracker.motion_gate.calls, ["image"])
        self.assertEqual(tracker.filterx.calls, [])
        self.assertEqual(tracker.frontend.calls, 0)


if __name__ == "__main__":
    unittest.main()
