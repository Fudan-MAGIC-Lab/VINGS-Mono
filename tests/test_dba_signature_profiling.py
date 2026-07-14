import importlib
import pathlib
import sys
import types

import torch

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


def make_graph():
    graph = CovisibleGraph.__new__(CovisibleGraph)
    graph.ii = torch.tensor([1, 1, 2, 3], device="cpu")
    graph.jj = torch.tensor([2, 3, 3, 4], device="cpu")
    graph.ht = 43
    graph.wd = 77
    graph.upsample = True
    graph.video = types.SimpleNamespace(
        ht=344,
        wd=616,
        cfg={"mode": "vo"},
    )
    graph.update_op = types.SimpleNamespace(core_backend=None)
    graph.net = torch.zeros(1, 4, 128, 43, 77, dtype=torch.float16)
    graph.profiler = DetailProfiler()
    graph.profiler_frame_idx = 12
    return graph


def test_build_dba_signature_uses_final_ba_edge_count():
    graph = make_graph()

    call = graph._build_dba_signature(
        t0=2,
        observed_t1=6,
        use_inactive=True,
        ba_edges=6,
    )

    assert call.active_edges == 4
    assert call.ba_edges == 6
    assert call.source_poses == 3
    assert call.pose_window == 4
    assert call.use_inactive is True
    assert call.frontend_image_size == (344, 616)
    assert call.feature_shape == (43, 77)
    assert call.backend == "torch"


def test_record_dba_signature_uses_final_ba_edge_count():
    graph = make_graph()

    call = graph._build_dba_signature(
        t0=2,
        observed_t1=6,
        use_inactive=True,
        ba_edges=6,
    )
    graph._record_dba_signature(call)

    kind, payload, frame_idx = graph.profiler.details[0]
    assert kind == "dba_signature"
    assert frame_idx == 12
    assert payload["active_edges"] == 4
    assert payload["ba_edges"] == 6
    assert payload["source_poses"] == 3
    assert payload["pose_window"] == 4
    assert payload["use_inactive"] is True
    assert payload["frontend_image_size"] == [344, 616]
    assert payload["feature_shape"] == [43, 77]
    assert payload["backend"] == "torch"
