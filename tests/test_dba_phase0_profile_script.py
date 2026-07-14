from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "reports/cuda_graph_dba/run_phase0_dba_profiles.sh"


def test_phase0_driver_is_guarded_and_uses_43x77_inputs():
    content = SCRIPT.read_text()
    for expected in (
        "set -uo pipefail",
        "--frontend-image-size 344,616",
        "--profile-runtime",
        "--droid-update-backend torch",
        "data/smallcity_subset_50/small_city",
        "data/smallcity_subset_200/small_city",
        "configs/rtg/hotel.yaml",
        "runtime_profile_details.jsonl",
        "tegrastats --interval 1000",
        "git status --short",
        "git submodule status --recursive",
        "pgrep -af '[s]cripts/run.py|[t]egrastats'",
    ):
        assert expected in content
    assert "rm -rf" not in content
    assert "rm -f" not in content


def test_phase0_driver_refuses_to_overwrite_evidence():
    content = SCRIPT.read_text()
    assert 'if [ -e "$EVIDENCE_DIR" ]' in content
    assert 'if [ -e "$output_parent" ]' in content


def test_phase0_driver_includes_unprofiled_smallcity_control():
    content = SCRIPT.read_text()
    assert "cuda_graph_dba_phase0_smallcity50_control" in content
    assert 'run_dir_${label}.txt' in content
