import csv
import tempfile
import unittest
from pathlib import Path

from scripts.gaussian.eval_export import append_eval_manifest, should_export_eval_frame


class EvalExportTests(unittest.TestCase):
    def test_eval_export_disabled_by_default(self):
        self.assertFalse(should_export_eval_frame({}, frame_id=10))

    def test_eval_export_enabled_exports_every_frame_by_default(self):
        self.assertTrue(should_export_eval_frame({"eval_export": {"enabled": True}}, frame_id=10))

    def test_eval_export_interval_limits_saved_frames(self):
        cfg = {"eval_export": {"enabled": True, "interval": 5}}

        self.assertTrue(should_export_eval_frame(cfg, frame_id=10))
        self.assertFalse(should_export_eval_frame(cfg, frame_id=11))

    def test_append_eval_manifest_writes_header_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp)
            cfg = {"output": {"save_dir": str(save_dir)}}

            append_eval_manifest(cfg, 5, "droid_c2w/000005.txt", "rgbdnua/FrameId=005.png")
            append_eval_manifest(cfg, 10, "droid_c2w/000010.txt", "rgbdnua/FrameId=010.png")

            with (save_dir / "eval_manifest.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row["frame_id"] for row in rows], ["5", "10"])
            self.assertEqual(rows[0]["pose_path"], "droid_c2w/000005.txt")
            self.assertEqual(rows[1]["rgbdnua_path"], "rgbdnua/FrameId=010.png")

    def test_append_eval_manifest_deduplicates_frame_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp)
            cfg = {"output": {"save_dir": str(save_dir)}}

            append_eval_manifest(cfg, 5, "droid_c2w/old.txt", "rgbdnua/old.png")
            append_eval_manifest(cfg, 5, "droid_c2w/new.txt", "rgbdnua/new.png")

            with (save_dir / "eval_manifest.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["pose_path"], "droid_c2w/new.txt")
            self.assertEqual(rows[0]["rgbdnua_path"], "rgbdnua/new.png")


if __name__ == "__main__":
    unittest.main()
