import importlib
import pathlib
import sys
import types
from contextlib import nullcontext

import torch

from scripts.profiling.dba_memory import MemorySnapshot
from tests.module_stubs import isolated_modules


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

PROJECTIVE_OPS = types.ModuleType("frontend.geom.projective_ops")
GEOM_PACKAGE = types.ModuleType("frontend.geom")
GEOM_PACKAGE.__path__ = []
GEOM_PACKAGE.projective_ops = PROJECTIVE_OPS


with isolated_modules(
    {
        "lietorch": types.SimpleNamespace(SE3=object),
        "frontend.modules.corr": types.SimpleNamespace(
            CorrBlock=object,
            AltCorrBlock=object,
        ),
        "frontend.geom": GEOM_PACKAGE,
        "frontend.geom.projective_ops": PROJECTIVE_OPS,
        "frontend.depth_video": types.SimpleNamespace(DepthVideo=object),
    },
    reload_modules=("scripts.frontend.covisible_graph",),
):
    CovisibleGraph = importlib.import_module(
        "scripts.frontend.covisible_graph"
    ).CovisibleGraph


class DetailProfiler:
    def __init__(self):
        self.details = []

    def record_detail(self, kind, payload, frame_idx=None):
        self.details.append((kind, payload, frame_idx))

    def time(self, stage, frame_idx=None):
        return nullcontext()


class FakeMemoryBackend:
    def __init__(self):
        self.allocated = 1_000

    def synchronize(self):
        pass

    def reset_peak(self):
        pass

    def snapshot(self):
        return MemorySnapshot(
            allocated_bytes=self.allocated,
            reserved_bytes=2_000,
            peak_allocated_bytes=self.allocated,
            free_bytes=8_000,
            total_bytes=10_000,
        )


class RecordingBuilder:
    def finish(self):
        return {
            "schema_version": 1,
            "sample_ordinal": 1,
            "signature": {"active_edges": 4},
            "entry": {},
            "stages": [
                {"name": "corr_sample_complete"},
                {"name": "update_op_complete"},
                {"name": "ba_inputs_complete"},
                {"name": "ba_complete"},
                {"name": "upsample_complete"},
            ],
        }


class FakeCorr:
    def __init__(self):
        self.corr_pyramid = [torch.zeros(16, dtype=torch.float16)]

    def __call__(self, coords):
        edges, height, width = coords.shape[1:4]
        return torch.zeros(
            1, edges, 4, height, width, dtype=torch.float16
        )


class FakeUpdate:
    core_backend = None

    def __call__(self, net, inp, corr, motn, ii, jj, upsample):
        edges, height, width = motn.shape[1], motn.shape[3], motn.shape[4]
        delta = torch.zeros(1, edges, height, width, 2)
        weight = torch.ones_like(delta)
        damping = torch.ones(
            torch.unique(ii).numel(), height, width
        )
        upmask = torch.zeros(1)
        return net, delta, weight, damping, upmask


def make_graph(enabled=False, samples=1):
    graph = CovisibleGraph.__new__(CovisibleGraph)
    graph.device = "cpu"
    graph.ii = torch.tensor([1, 1, 2, 3], dtype=torch.long)
    graph.jj = torch.tensor([2, 3, 3, 4], dtype=torch.long)
    graph.ii_inac = torch.tensor([0, 1], dtype=torch.long)
    graph.jj_inac = torch.tensor([1, 2], dtype=torch.long)
    graph.ht = 43
    graph.wd = 77
    graph.upsample = True
    graph.inac_range = 3
    graph.video = types.SimpleNamespace(
        ht=344,
        wd=616,
        cfg={
            "mode": "vo",
            "profiling": {
                "dba_memory": {
                    "enabled": enabled,
                    "samples_per_signature": samples,
                }
            },
        },
    )
    graph.update_op = types.SimpleNamespace(core_backend=None)
    graph.net = torch.zeros(1, 4, 2, 2, dtype=torch.float16)
    graph.inp = graph.net[:, :, :1]
    corr_base = torch.zeros(16, dtype=torch.float16)
    graph.corr = types.SimpleNamespace(
        corr_pyramid=[corr_base, corr_base[2:8]]
    )
    graph.coords0 = torch.zeros(43, 77, 2)
    graph.target = torch.zeros(1, 4, 43, 77, 2)
    graph.weight = torch.zeros_like(graph.target)
    graph.damping = torch.zeros(8, 43, 77)
    graph.profiler = DetailProfiler()
    graph.profiler_frame_idx = 17
    graph.dba_memory_limiter = None
    graph.dba_memory_backend = None
    return graph


def test_disabled_memory_profile_does_not_create_sampler():
    graph = make_graph(enabled=False)

    graph._configure_dba_memory_sampler(backend=FakeMemoryBackend())

    assert graph.dba_memory_limiter is None
    assert graph.dba_memory_backend is None


def test_preview_signature_matches_inactive_ba_shape():
    graph = make_graph(enabled=True)

    call = graph._preview_dba_memory_signature(
        t0=2,
        t1=None,
        use_inactive=True,
    )

    assert call.active_edges == 4
    assert call.ba_edges == 6
    assert call.source_poses == 3
    assert call.pose_window == 3


def test_sampler_records_only_reserved_signatures():
    graph = make_graph(enabled=True, samples=1)
    graph._configure_dba_memory_sampler(backend=FakeMemoryBackend())
    call = graph._preview_dba_memory_signature(2, None, True)

    first = graph._begin_dba_memory_sample(call)
    second = graph._begin_dba_memory_sample(call)

    assert first is not None
    assert second is None
    assert first.finish()["entry"]["corr_resident_storage_bytes"] == 32


def test_finished_sample_records_structured_detail():
    graph = make_graph(enabled=True)

    graph._finish_dba_memory_sample(RecordingBuilder())

    kind, payload, frame_idx = graph.profiler.details[0]
    assert kind == "dba_memory_sample"
    assert frame_idx == 17
    assert [row["name"] for row in payload["stages"]] == [
        "corr_sample_complete",
        "update_op_complete",
        "ba_inputs_complete",
        "ba_complete",
        "upsample_complete",
    ]


def test_update_records_all_memory_stages_in_order():
    graph = make_graph(enabled=True, samples=1)
    graph.corr = FakeCorr()
    graph.update_op = FakeUpdate()
    graph.target_inac = torch.zeros(1, 2, 43, 77, 2)
    graph.weight_inac = torch.zeros_like(graph.target_inac)
    graph.age = torch.zeros(graph.ii.numel(), dtype=torch.long)
    graph.far_threshold = -1.0
    graph.mask_threshold = -1.0
    graph.show_oldest_disparity = False
    graph.show_flow_and_weight = False
    graph.show_covisible_graph = False
    graph.video.imu_enabled = False
    graph.video.visual_only_init = False
    graph.video.reproject = lambda ii, jj: (
        torch.zeros(1, ii.numel(), 43, 77, 2),
        None,
    )
    graph.video.ba = lambda *args, **kwargs: None
    graph.video.upsample = lambda *args, **kwargs: None
    graph._configure_dba_memory_sampler(backend=FakeMemoryBackend())

    graph.update(t0=2, t1=None, itrs=1, use_inactive=True)

    memory_rows = [
        payload
        for kind, payload, _ in graph.profiler.details
        if kind == "dba_memory_sample"
    ]
    assert len(memory_rows) == 1
    assert [row["name"] for row in memory_rows[0]["stages"]] == [
        "corr_sample_complete",
        "update_op_complete",
        "ba_inputs_complete",
        "ba_complete",
        "upsample_complete",
    ]
