from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "reports/cuda_graph_dba/run_phase05_memory_profiles.sh"


def test_phase05_driver_is_guarded_and_collects_memory_evidence():
    content = SCRIPT.read_text()
    for expected in (
        "set -uo pipefail",
        'if [ -e "$EVIDENCE_DIR" ]',
        'if [ -e "$output_parent" ]',
        "git status --short",
        "git submodule status --recursive",
        "python --version",
        "torch.version.cuda",
        "tegrastats --interval 1000",
        "pgrep -af '[s]cripts/run.py|[t]egrastats'",
        "--frontend-image-size 344,616",
        "--droid-update-backend torch",
        "--profile-runtime",
        "--profile-dba-memory",
        "--profile-dba-memory-samples-per-signature 2",
        '"kind": "dba_signature"',
        '"kind": "dba_memory_sample"',
    ):
        assert expected in content
    assert "rm -rf" not in content
    assert "rm -f" not in content


def test_phase05_driver_runs_only_approved_sequences_and_buffers():
    content = SCRIPT.read_text()
    assert "data/smallcity_subset_200/small_city" in content
    assert "data/smallcity_subset_50/small_city" not in content
    assert "configs/rtg/hotel.yaml" in content
    assert "--frontend-save-buffer 64" in content
    hotel_block = content.split("run_variant \\\n  hotel344", 1)[1]
    assert "--frontend-save-buffer 512" in hotel_block
    assert "--limit" not in hotel_block


def test_phase05_driver_calibrates_both_boundaries_and_plans_at_1024_mib():
    content = SCRIPT.read_text()
    assert "calibrate_dba_workspace.py" in content
    assert "corr_update_aggregation" in content
    assert "update_aggregation_only" in content
    assert "plan_dba_buckets.py" in content
    assert "--max-workspace-mb 1024" in content
    assert "--target-coverage 0.90" in content
    assert "--max-buckets 6" in content
    assert "--max-padding-ratio 0.35" in content
    assert "weighted_sample_coverage" in content
    assert "0.95" in content
