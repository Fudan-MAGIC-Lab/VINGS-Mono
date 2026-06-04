import pathlib
import sys
import types
import unittest

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.frontend.dbaf_frontend import DBAFusionFrontend


def make_video(state, imu_enabled=False, size=6):
    video = types.SimpleNamespace()
    video.counter = types.SimpleNamespace(value=size)
    video.tstamp = torch.arange(size, dtype=torch.float32)
    video.images = torch.arange(size, dtype=torch.float32)
    video.dirty = torch.arange(size, dtype=torch.float32)
    video.red = torch.arange(size, dtype=torch.float32)
    video.poses = torch.arange(size, dtype=torch.float32)
    video.disps = torch.arange(size, dtype=torch.float32)
    video.disps_sens = torch.arange(size, dtype=torch.float32)
    video.disps_up = torch.arange(size, dtype=torch.float32)
    video.intrinsics = torch.arange(size, dtype=torch.float32)
    video.fmaps = torch.arange(size, dtype=torch.float32)
    video.nets = torch.arange(size, dtype=torch.float32)
    video.inps = torch.arange(size, dtype=torch.float32)
    video.depths_cov = torch.arange(size, dtype=torch.float32)
    video.depths_cov_up = torch.arange(size, dtype=torch.float32)
    video.last_t0 = 4
    video.last_t1 = 5
    video.cur_ii = 4
    video.cur_jj = 5
    video.imu_enabled = imu_enabled
    video.cur_graph = None
    video.cur_result = None
    video.marg_factor = None
    video.state = state
    return video


def make_graph(size=4):
    graph = types.SimpleNamespace()
    graph.ii = torch.arange(size, dtype=torch.int64)
    graph.jj = torch.arange(size, dtype=torch.int64)
    graph.ii_inac = torch.tensor([0, 1, 3, 4], dtype=torch.int64)
    graph.jj_inac = torch.tensor([1, 2, 4, 5], dtype=torch.int64)
    graph.target_inac = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    graph.weight_inac = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)
    graph.ii_bad = torch.arange(size, dtype=torch.int64)
    graph.jj_bad = torch.arange(size, dtype=torch.int64)
    return graph


class RollupTests(unittest.TestCase):
    def make_frontend(self, state, mode="vo", imu_enabled=False):
        frontend = DBAFusionFrontend.__new__(DBAFusionFrontend)
        frontend.cfg = {"mode": mode}
        frontend.t1 = 6
        frontend.count = 6
        frontend.video = make_video(state=state, imu_enabled=imu_enabled)
        frontend.graph = make_graph()
        return frontend

    def test_rollup_skips_optional_state_in_vo_mode(self):
        frontend = self.make_frontend(state=None, mode="vo", imu_enabled=False)

        frontend._DBAFusionFrontend__rollup(2)

        self.assertEqual(frontend.t1, 4)
        self.assertEqual(frontend.count, 4)
        self.assertEqual(frontend.video.counter.value, 4)
        self.assertEqual(frontend.video.last_t0, 2)
        self.assertEqual(frontend.video.last_t1, 3)
        self.assertTrue(torch.equal(frontend.video.tstamp, torch.tensor([2, 3, 4, 5, 0, 1], dtype=torch.float32)))
        self.assertIsNone(frontend.video.state)

    def test_rollup_truncates_state_when_present(self):
        state = types.SimpleNamespace(
            timestamps=[0, 1, 2, 3, 4, 5],
            wTbs=["a", "b", "c", "d", "e", "f"],
            vs=[10, 11, 12, 13, 14, 15],
            bs=[20, 21, 22, 23, 24, 25],
            preintegrations=[30, 31, 32, 33, 34, 35],
            preintegrations_meas=[40, 41, 42, 43, 44, 45],
            gnss_valid=[50, 51, 52, 53, 54, 55],
            gnss_position=[60, 61, 62, 63, 64, 65],
            odo_valid=[70, 71, 72, 73, 74, 75],
            odo_vel=[80, 81, 82, 83, 84, 85],
        )
        frontend = self.make_frontend(state=state, mode="vio", imu_enabled=False)

        frontend._DBAFusionFrontend__rollup(2)

        self.assertEqual(frontend.video.state.timestamps, [2, 3, 4, 5])
        self.assertEqual(frontend.video.state.wTbs, ["c", "d", "e", "f"])
        self.assertEqual(frontend.video.state.odo_vel, [82, 83, 84, 85])


if __name__ == "__main__":
    unittest.main()
