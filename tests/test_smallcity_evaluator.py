import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.profiling.evaluate_smallcity_run import evaluate_run


def write_pose(path, tx):
    mat = np.eye(4)
    mat[0, 3] = tx
    np.savetxt(path, mat)


def write_rgbdnua(path, value_gt, value_render):
    tile = np.full((4, 4, 3), value_gt, dtype=np.uint8)
    render = np.full((4, 4, 3), value_render, dtype=np.uint8)
    other = np.zeros((4, 4, 3), dtype=np.uint8)
    top = np.concatenate([tile, other, other], axis=1)
    bottom = np.concatenate([render, other, other], axis=1)
    Image.fromarray(np.concatenate([top, bottom], axis=0)).save(path)


class SmallCityEvaluatorTests(unittest.TestCase):
    def test_evaluate_run_uses_manifest_pose_and_rgbdnua(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            gt = root / "gt"
            (run / "droid_c2w").mkdir(parents=True)
            (run / "rgbdnua").mkdir()
            (gt / "pose").mkdir(parents=True)

            for frame_id, pred_x, gt_x in [(0, 0.0, 1.0), (1, 1.0, 3.0), (2, 2.0, 5.0)]:
                write_pose(run / "droid_c2w" / f"{frame_id:08d}.txt", pred_x)
                write_pose(gt / "pose" / f"{frame_id:05d}.txt", gt_x)
                write_rgbdnua(run / "rgbdnua" / f"FrameId={frame_id}.png", 128, 128)

            (run / "eval_manifest.csv").write_text(
                "frame_id,pose_path,rgbdnua_path\n"
                "0,droid_c2w/00000000.txt,rgbdnua/FrameId=0.png\n"
                "1,droid_c2w/00000001.txt,rgbdnua/FrameId=1.png\n"
                "2,droid_c2w/00000002.txt,rgbdnua/FrameId=2.png\n"
            )

            metrics = evaluate_run(run, gt)

            self.assertEqual(metrics["pose_samples"], 3)
            self.assertLess(metrics["ate_sim3_rmse_m"], 1e-9)
            self.assertEqual(metrics["rgbdnua_samples"], 3)
            self.assertGreater(metrics["mean_psnr"], 90.0)
            self.assertGreater(metrics["mean_ssim"], 0.99)


if __name__ == "__main__":
    unittest.main()
