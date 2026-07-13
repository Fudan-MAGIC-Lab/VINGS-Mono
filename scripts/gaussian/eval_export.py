import csv
from pathlib import Path


def _to_frame_number(frame_id):
    if hasattr(frame_id, "item"):
        frame_id = frame_id.item()
    return int(float(frame_id))


def should_export_eval_frame(cfg, frame_id):
    export_cfg = cfg.get("eval_export", {}) or {}
    if not export_cfg.get("enabled", False):
        return False

    interval = int(export_cfg.get("interval", 1) or 1)
    if interval <= 1:
        return True

    return _to_frame_number(frame_id) % interval == 0


def append_eval_manifest(cfg, frame_id, pose_path, rgbdnua_path):
    save_dir = Path(cfg["output"]["save_dir"])
    manifest_path = save_dir / "eval_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame_key = str(_to_frame_number(frame_id))

    rows_by_frame = {}
    if manifest_path.exists():
        with manifest_path.open(newline="") as f:
            for row in csv.DictReader(f):
                rows_by_frame[row["frame_id"]] = row

    rows_by_frame[frame_key] = {
        "frame_id": frame_key,
        "pose_path": str(pose_path),
        "rgbdnua_path": str(rgbdnua_path),
    }

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "pose_path", "rgbdnua_path"])
        writer.writeheader()
        for key in sorted(rows_by_frame, key=lambda value: int(float(value))):
            writer.writerow(rows_by_frame[key])
