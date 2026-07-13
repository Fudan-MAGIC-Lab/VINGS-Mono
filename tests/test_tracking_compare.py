import csv
import importlib.util
import pathlib
import tempfile
import unittest

import numpy as np
from PIL import Image


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "profiling" / "compare_tracking_runs.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compare_tracking_runs", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_rgbdnua(path, gt_color, pred_color):
    image = Image.new("RGB", (30, 20), "black")
    for x in range(10):
        for y in range(10):
            image.putpixel((x, y), gt_color)
            image.putpixel((x, y + 10), pred_color)
    image.save(path)


def make_run(root, frame_total, tracking, keyframes, pred_color):
    run_dir = root / f"run_{frame_total}"
    (run_dir / "rgbdnua").mkdir(parents=True)
    (run_dir / "droid_c2w").mkdir(parents=True)
    with (run_dir / "runtime_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["stage", "calls", "total_s", "mean_ms", "percent_of_frame_total"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "stage": "frame_total",
                "calls": "2",
                "total_s": str(frame_total),
                "mean_ms": str(frame_total * 500),
                "percent_of_frame_total": "100.0",
            }
        )
        writer.writerow(
            {
                "stage": "tracking",
                "calls": "2",
                "total_s": str(tracking),
                "mean_ms": str(tracking * 500),
                "percent_of_frame_total": str(tracking / frame_total * 100.0),
            }
        )
    (run_dir / "keyframelist.txt").write_text("\n".join(str(i) for i in range(keyframes)) + "\n")
    make_rgbdnua(run_dir / "rgbdnua" / "FrameId=001.0.png", (128, 128, 128), pred_color)
    make_rgbdnua(run_dir / "rgbdnua" / "FrameId=002.0.png", (128, 128, 128), pred_color)
    for frame in ["001", "002"]:
        pose = np.eye(4)
        pose[0, 3] = frame_total / 10.0
        np.savetxt(run_dir / "droid_c2w" / f"000{frame}.0.txt", pose)
    return run_dir


class TrackingCompareTests(unittest.TestCase):
    def test_runtime_stage_list_includes_tensorrt_boundary_profiling(self):
        module = load_module()

        for stage in (
            "metric_depth_preprocess",
            "metric_depth_model",
            "metric_depth_postprocess",
            "covisible_graph_corr",
            "covisible_graph_update_op",
            "covisible_graph_graph_agg",
            "covisible_graph_ba",
            "covisible_graph_upsample",
        ):
            self.assertIn(stage, module.RUNTIME_STAGES)

    def test_compare_runs_writes_summary_csv_and_contact_sheet(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            baseline = make_run(root, frame_total=10.0, tracking=6.0, keyframes=2, pred_color=(120, 120, 120))
            candidate = make_run(root, frame_total=8.0, tracking=4.0, keyframes=3, pred_color=(128, 128, 128))
            output_dir = root / "report"

            result = module.compare_runs(
                [("baseline", baseline), ("candidate", candidate)],
                output_dir=output_dir,
                max_contact_frames=2,
            )

            self.assertEqual(result["common_frames"], ["001.0", "002.0"])
            self.assertTrue((output_dir / "summary.md").is_file())
            self.assertTrue((output_dir / "runtime_metrics.csv").is_file())
            self.assertTrue((output_dir / "quality_metrics.csv").is_file())
            self.assertTrue((output_dir / "quality_by_frame.csv").is_file())
            self.assertTrue((output_dir / "pose_deltas.csv").is_file())
            self.assertTrue((output_dir / "rgb_quality_contact_sheet.png").is_file())

            summary = (output_dir / "summary.md").read_text()
            self.assertIn("| candidate | 8.000 | 4.000 | 3 |", summary)
            self.assertIn("Common rgbdnua frames: 2", summary)

            with (output_dir / "quality_metrics.csv").open(newline="") as handle:
                rows = {row["run"]: row for row in csv.DictReader(handle)}
            self.assertGreater(float(rows["candidate"]["mean_psnr"]), float(rows["baseline"]["mean_psnr"]))

            with (output_dir / "quality_by_frame.csv").open(newline="") as handle:
                frame_rows = list(csv.DictReader(handle))
            self.assertEqual(frame_rows[0]["frame"], "001.0")
            self.assertIn("candidate_delta_ssim_vs_baseline", frame_rows[0])

            with (output_dir / "pose_deltas.csv").open(newline="") as handle:
                pose_rows = list(csv.DictReader(handle))
            self.assertEqual(pose_rows[0]["run"], "candidate")
            self.assertEqual(pose_rows[0]["reference_run"], "baseline")
            self.assertAlmostEqual(float(pose_rows[0]["translation_delta"]), 0.2)


if __name__ == "__main__":
    unittest.main()
