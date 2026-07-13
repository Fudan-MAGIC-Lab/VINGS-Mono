import pathlib
import sys
import unittest
from contextlib import contextmanager

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from frontend.covisible_graph import CovisibleGraph
from frontend.droid_net import UpdateModule


class RecordingProfiler:
    def __init__(self):
        self.stages = []

    @contextmanager
    def time(self, stage, frame_idx=None):
        self.stages.append((stage, frame_idx))
        yield


class FakeCorr:
    def __call__(self, coords):
        batch, edges, height, width, _ = coords.shape
        return torch.zeros(batch, edges, 196, height, width)


class FakeUpdate:
    def __call__(self, net, inp, corr, motion, ii, jj, upsample):
        batch, edges, _, height, width = net.shape
        delta = torch.zeros(batch, edges, height, width, 2)
        weight = torch.ones_like(delta)
        damping = torch.ones(1, height, width)
        upmask = torch.zeros(batch, edges, 8 * 8 * 9, height, width)
        return net, delta, weight, damping, upmask


class FakeVideo:
    def __init__(self):
        self.disps = torch.ones(2, 2, 3)
        self.imu_enabled = False
        self.visual_only_init = False
        self.ba_calls = 0
        self.upsample_calls = 0

    def reproject(self, ii, jj):
        coords = torch.zeros(1, len(ii), 2, 3, 2)
        return coords, torch.ones_like(coords[..., 0], dtype=torch.bool)

    def ba(self, *args, **kwargs):
        self.ba_calls += 1

    def upsample(self, *args, **kwargs):
        self.upsample_calls += 1


class DroidFineProfilingTests(unittest.TestCase):
    def test_update_module_records_core_and_graph_aggregation(self):
        module = UpdateModule().eval()
        profiler = RecordingProfiler()
        module.profiler = profiler
        module.profiler_frame_idx = 9
        net = torch.randn(1, 4, 128, 3, 5)
        inp = torch.randn(1, 4, 128, 3, 5)
        corr = torch.randn(1, 4, 196, 3, 5)
        flow = torch.randn(1, 4, 4, 3, 5)

        with torch.no_grad():
            module(net, inp, corr, flow, ii=torch.tensor([0, 0, 1, 1]), upsample=True)

        self.assertEqual(
            profiler.stages,
            [
                ("covisible_graph_update_op", 9),
                ("covisible_graph_graph_agg", 9),
            ],
        )

    def test_covisible_graph_records_corr_ba_and_upsample(self):
        graph = CovisibleGraph.__new__(CovisibleGraph)
        graph.video = FakeVideo()
        graph.update_op = FakeUpdate()
        graph.coords0 = torch.zeros(2, 3, 2)
        graph.ii = torch.tensor([0])
        graph.jj = torch.tensor([0])
        graph.target = torch.zeros(1, 1, 2, 3, 2)
        graph.weight = torch.zeros_like(graph.target)
        graph.corr = FakeCorr()
        graph.net = torch.zeros(1, 1, 128, 2, 3)
        graph.inp = torch.zeros_like(graph.net)
        graph.damping = torch.ones(2, 2, 3)
        graph.upsample = True
        graph.age = torch.zeros(1, dtype=torch.long)
        graph.inac_range = 3
        graph.ii_inac = torch.empty(0, dtype=torch.long)
        graph.jj_inac = torch.empty(0, dtype=torch.long)
        graph.target_inac = torch.zeros(1, 0, 2, 3, 2)
        graph.weight_inac = torch.zeros_like(graph.target_inac)
        graph.show_oldest_disparity = False
        graph.show_flow_and_weight = False
        graph.show_covisible_graph = False
        graph.far_threshold = -1.0
        graph.mask_threshold = -1.0
        graph.profiler = RecordingProfiler()
        graph.profiler_frame_idx = 11

        graph.update(t0=1)

        self.assertEqual(
            graph.profiler.stages,
            [
                ("covisible_graph_corr", 11),
                ("covisible_graph_ba", 11),
                ("covisible_graph_upsample", 11),
            ],
        )
        self.assertEqual(graph.video.ba_calls, 1)
        self.assertEqual(graph.video.upsample_calls, 1)


if __name__ == "__main__":
    unittest.main()
