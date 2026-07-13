import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image


def _frame_id_from_name(path):
    match = re.search(r"(\d+(?:\.\d+)?)", path.name)
    return int(float(match.group(1))) if match else None


def _umeyama_align(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    rot = u @ s @ vt
    scale = float(np.trace(np.diag(d) @ s) / ((src_c ** 2).sum() / n))
    trans = mu_dst - scale * rot @ mu_src
    aligned = (scale * (rot @ src.T)).T + trans
    return aligned, scale


def _psnr(gt, pred):
    x = gt.astype(np.float32) / 255.0
    y = pred.astype(np.float32) / 255.0
    mse = np.mean((x - y) ** 2)
    return float("inf") if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))


def _simple_ssim(gt, pred):
    x = gt.astype(np.float64) / 255.0
    y = pred.astype(np.float64) / 255.0
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    vals = []
    for chan in range(3):
        xc = x[..., chan]
        yc = y[..., chan]
        mux = xc.mean()
        muy = yc.mean()
        vx = xc.var()
        vy = yc.var()
        cov = ((xc - mux) * (yc - muy)).mean()
        vals.append(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))
    return float(np.mean(vals))


def _load_manifest(run_dir):
    manifest = run_dir / "eval_manifest.csv"
    if manifest.exists():
        rows_by_frame = {}
        with manifest.open() as f:
            for row in csv.DictReader(f):
                rows_by_frame[int(float(row["frame_id"]))] = row
        for frame_id in sorted(rows_by_frame):
            row = rows_by_frame[frame_id]
            yield frame_id, run_dir / row["pose_path"], run_dir / row["rgbdnua_path"]
        return

    pose_by_frame = {_frame_id_from_name(p): p for p in (run_dir / "droid_c2w").glob("*.txt")}
    rgb_by_frame = {_frame_id_from_name(p): p for p in (run_dir / "rgbdnua").glob("FrameId=*.png")}
    for frame_id in sorted(set(pose_by_frame) & set(rgb_by_frame)):
        yield frame_id, pose_by_frame[frame_id], rgb_by_frame[frame_id]


def evaluate_run(run_dir, dataset_root):
    run_dir = Path(run_dir)
    dataset_root = Path(dataset_root)
    rows = list(_load_manifest(run_dir))

    pose_rows = []
    quality_rows = []
    for frame_id, pose_path, rgb_path in rows:
        gt_pose = dataset_root / "pose" / f"{frame_id:05d}.txt"
        if pose_path.exists() and gt_pose.exists():
            pred = np.loadtxt(pose_path)[:3, 3]
            gt = np.loadtxt(gt_pose)[:3, 3]
            pose_rows.append((frame_id, pred, gt))
        if rgb_path.exists():
            img = np.array(Image.open(rgb_path).convert("RGB"))
            h, w = img.shape[:2]
            tile_h = h // 2
            tile_w = w // 3 if w % 3 == 0 else w // 2
            gt_rgb = img[:tile_h, :tile_w]
            render_rgb = img[tile_h:tile_h * 2, :tile_w]
            quality_rows.append((frame_id, _psnr(gt_rgb, render_rgb), _simple_ssim(gt_rgb, render_rgb)))

    metrics = {
        "manifest_samples": len(rows),
        "pose_samples": len(pose_rows),
        "rgbdnua_samples": len(quality_rows),
    }

    if len(pose_rows) >= 3:
        pred = np.stack([row[1] for row in pose_rows])
        gt = np.stack([row[2] for row in pose_rows])
        aligned, scale = _umeyama_align(pred, gt)
        err = np.linalg.norm(aligned - gt, axis=1)
        metrics.update({
            "ate_sim3_rmse_m": float(np.sqrt(np.mean(err ** 2))),
            "ate_sim3_mean_m": float(np.mean(err)),
            "ate_sim3_median_m": float(np.median(err)),
            "ate_sim3_max_m": float(np.max(err)),
            "sim3_scale": float(scale),
            "first_pose_frame": pose_rows[0][0],
            "last_pose_frame": pose_rows[-1][0],
        })

    if quality_rows:
        psnr = np.array([row[1] for row in quality_rows], dtype=float)
        ssim = np.array([row[2] for row in quality_rows], dtype=float)
        metrics.update({
            "mean_psnr": float(psnr.mean()),
            "min_psnr": float(psnr.min()),
            "mean_ssim": float(ssim.mean()),
            "min_ssim": float(ssim.min()),
            "first_quality_frame": quality_rows[0][0],
            "last_quality_frame": quality_rows[-1][0],
        })

    return metrics


def write_summary(metrics, output_path):
    lines = ["# SmallCity Evaluation Summary", "", "| Metric | Value |", "|---|---:|"]
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            value = f"{value:.6f}"
        lines.append(f"| {key} | {value} |")
    Path(output_path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate exported SmallCity pose/render samples.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metrics = evaluate_run(Path(args.run_dir), Path(args.dataset_root))
    write_summary(metrics, Path(args.output))
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
