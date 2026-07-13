import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


RUNTIME_STAGES = [
    "frame_total",
    "tracking",
    "frontend_dba_update",
    "frontend_update_non_keyframe_graph_update",
    "frontend_update_keyframe_graph_update",
    "frontend_motion_filter",
    "mapping",
    "metric_depth",
    "metric_depth_preprocess",
    "metric_depth_model",
    "metric_depth_postprocess",
    "covisible_graph_corr",
    "covisible_graph_update_op",
    "covisible_graph_graph_agg",
    "covisible_graph_ba",
    "covisible_graph_upsample",
    "loop",
]


def _frame_key(path):
    return path.stem.split("=")[-1]


def _pose_frame_key(path):
    stem = path.stem
    try:
        return f"{float(stem):.1f}"
    except ValueError:
        return stem.lstrip("0") or stem


def _load_runtime_rows(run_dir):
    path = Path(run_dir) / "runtime_profile.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        return {row["stage"]: row for row in csv.DictReader(handle)}


def _stage_total(runtime_rows, stage):
    row = runtime_rows.get(stage)
    if row is None:
        return None
    return float(row["total_s"])


def _stage_calls(runtime_rows, stage):
    row = runtime_rows.get(stage)
    if row is None:
        return None
    return int(float(row["calls"]))


def _keyframe_count(run_dir):
    path = Path(run_dir) / "keyframelist.txt"
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _rgbdnua_files(run_dir):
    rgb_dir = Path(run_dir) / "rgbdnua"
    if not rgb_dir.is_dir():
        return {}
    return {_frame_key(path): path for path in rgb_dir.glob("FrameId=*.png")}


def _pose_files(run_dir):
    pose_dir = Path(run_dir) / "droid_c2w"
    if not pose_dir.is_dir():
        return {}
    return {_pose_frame_key(path): path for path in pose_dir.glob("*.txt")}


def _rgb_crops(path):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    half_h = h // 2
    cell_w = w // 3
    return arr[:half_h, :cell_w], arr[half_h : half_h * 2, :cell_w]


def _psnr(gt, pred):
    mse = float(np.mean((gt - pred) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return -10.0 * math.log10(mse)


def _simple_ssim(gt, pred):
    x = gt.mean(axis=2)
    y = pred.mean(axis=2)
    c1 = 0.01**2
    c2 = 0.03**2
    mux = float(x.mean())
    muy = float(y.mean())
    vx = float(x.var())
    vy = float(y.var())
    cov = float(((x - mux) * (y - muy)).mean())
    return ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2))


def _quality_for_run(files, common_frames):
    psnrs = []
    ssims = []
    for frame in common_frames:
        gt, pred = _rgb_crops(files[frame])
        psnrs.append(_psnr(gt, pred))
        ssims.append(_simple_ssim(gt, pred))
    if not psnrs:
        return {
            "frames": 0,
            "mean_psnr": 0.0,
            "min_psnr": 0.0,
            "mean_ssim": 0.0,
            "min_ssim": 0.0,
        }
    return {
        "frames": len(psnrs),
        "mean_psnr": float(np.mean(psnrs)),
        "min_psnr": float(np.min(psnrs)),
        "mean_ssim": float(np.mean(ssims)),
        "min_ssim": float(np.min(ssims)),
    }


def _quality_by_frame_rows(runs, files_by_run, common_frames):
    rows = []
    if not runs:
        return rows
    reference_label = runs[0][0]
    for frame in common_frames:
        row = {"frame": frame}
        for label, _run_dir in runs:
            gt, pred = _rgb_crops(files_by_run[label][frame])
            row[f"{label}_psnr"] = _psnr(gt, pred)
            row[f"{label}_ssim"] = _simple_ssim(gt, pred)
        for label, _run_dir in runs[1:]:
            row[f"{label}_delta_psnr_vs_{reference_label}"] = row[f"{label}_psnr"] - row[f"{reference_label}_psnr"]
            row[f"{label}_delta_ssim_vs_{reference_label}"] = row[f"{label}_ssim"] - row[f"{reference_label}_ssim"]
        rows.append(row)
    return rows


def _load_pose(path):
    return np.loadtxt(path)


def _pose_delta(reference_pose, pose):
    relative = np.linalg.inv(reference_pose) @ pose
    translation_delta = float(np.linalg.norm(relative[:3, 3]))
    cos_angle = float((np.trace(relative[:3, :3]) - 1.0) / 2.0)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    rotation_delta_deg = math.degrees(math.acos(cos_angle))
    return translation_delta, rotation_delta_deg


def _pose_delta_rows(runs, pose_files_by_run):
    rows = []
    if len(runs) < 2:
        return rows
    frame_sets = [set(pose_files_by_run[label]) for label, _run_dir in runs]
    common_frames = sorted(set.intersection(*frame_sets)) if frame_sets else []
    reference_label = runs[0][0]
    for frame in common_frames:
        reference_pose = _load_pose(pose_files_by_run[reference_label][frame])
        for label, _run_dir in runs[1:]:
            translation_delta, rotation_delta_deg = _pose_delta(reference_pose, _load_pose(pose_files_by_run[label][frame]))
            rows.append(
                {
                    "frame": frame,
                    "reference_run": reference_label,
                    "run": label,
                    "translation_delta": translation_delta,
                    "rotation_delta_deg": rotation_delta_deg,
                }
            )
    return rows


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_contact_sheet(runs, files_by_run, common_frames, output_path, max_frames):
    selected_frames = common_frames[:max_frames]
    if not selected_frames:
        return

    thumbs = []
    for label, _run_dir in runs:
        row = []
        for frame in selected_frames:
            image = Image.open(files_by_run[label][frame]).convert("RGB")
            image.thumbnail((240, 160))
            row.append((frame, image.copy()))
        thumbs.append((label, row))

    label_h = 26
    cell_w = 260
    cell_h = 190
    sheet = Image.new("RGB", (cell_w * len(selected_frames), cell_h * len(thumbs)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_idx, (label, row) in enumerate(thumbs):
        y0 = row_idx * cell_h
        draw.text((4, y0 + 4), label, fill=(0, 0, 0))
        for col_idx, (frame, image) in enumerate(row):
            x0 = col_idx * cell_w
            draw.text((x0 + 4, y0 + label_h), frame, fill=(0, 0, 0))
            sheet.paste(image, (x0 + 4, y0 + label_h + 18))
    sheet.save(output_path)


def compare_runs(runs, output_dir, max_contact_frames=12):
    runs = [(str(label), Path(run_dir)) for label, run_dir in runs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_by_run = {label: _load_runtime_rows(run_dir) for label, run_dir in runs}
    files_by_run = {label: _rgbdnua_files(run_dir) for label, run_dir in runs}
    pose_files_by_run = {label: _pose_files(run_dir) for label, run_dir in runs}
    frame_sets = [set(files) for files in files_by_run.values()]
    common_frames = sorted(set.intersection(*frame_sets)) if frame_sets else []

    runtime_rows = []
    for label, run_dir in runs:
        runtime = runtime_by_run[label]
        row = {
            "run": label,
            "path": str(run_dir),
            "keyframes": _keyframe_count(run_dir),
        }
        for stage in RUNTIME_STAGES:
            value = _stage_total(runtime, stage)
            row[stage] = "" if value is None else f"{value:.3f}"
            calls = _stage_calls(runtime, stage)
            row[f"{stage}_calls"] = "" if calls is None else str(calls)
        runtime_rows.append(row)

    quality_rows = []
    for label, _run_dir in runs:
        metrics = _quality_for_run(files_by_run[label], common_frames)
        quality_rows.append(
            {
                "run": label,
                "common_frames": metrics["frames"],
                "mean_psnr": f"{metrics['mean_psnr']:.3f}",
                "min_psnr": f"{metrics['min_psnr']:.3f}",
                "mean_ssim": f"{metrics['mean_ssim']:.4f}",
                "min_ssim": f"{metrics['min_ssim']:.4f}",
            }
        )

    quality_by_frame_rows = _quality_by_frame_rows(runs, files_by_run, common_frames)
    pose_delta_rows = _pose_delta_rows(runs, pose_files_by_run)

    runtime_fields = ["run", "path", "keyframes"]
    for stage in RUNTIME_STAGES:
        runtime_fields.extend([stage, f"{stage}_calls"])
    _write_csv(output_dir / "runtime_metrics.csv", runtime_rows, runtime_fields)
    _write_csv(
        output_dir / "quality_metrics.csv",
        quality_rows,
        ["run", "common_frames", "mean_psnr", "min_psnr", "mean_ssim", "min_ssim"],
    )
    quality_by_frame_fields = ["frame"]
    for label, _run_dir in runs:
        quality_by_frame_fields.extend([f"{label}_psnr", f"{label}_ssim"])
    if runs:
        reference_label = runs[0][0]
        for label, _run_dir in runs[1:]:
            quality_by_frame_fields.extend(
                [f"{label}_delta_psnr_vs_{reference_label}", f"{label}_delta_ssim_vs_{reference_label}"]
            )
    _write_csv(
        output_dir / "quality_by_frame.csv",
        [
            {key: (f"{value:.4f}" if isinstance(value, float) else value) for key, value in row.items()}
            for row in quality_by_frame_rows
        ],
        quality_by_frame_fields,
    )
    _write_csv(
        output_dir / "pose_deltas.csv",
        [
            {
                "frame": row["frame"],
                "reference_run": row["reference_run"],
                "run": row["run"],
                "translation_delta": f"{row['translation_delta']:.6f}",
                "rotation_delta_deg": f"{row['rotation_delta_deg']:.4f}",
            }
            for row in pose_delta_rows
        ],
        ["frame", "reference_run", "run", "translation_delta", "rotation_delta_deg"],
    )
    _make_contact_sheet(runs, files_by_run, common_frames, output_dir / "rgb_quality_contact_sheet.png", max_contact_frames)

    lines = [
        "# Tracking Run Compare",
        "",
        f"Common rgbdnua frames: {len(common_frames)}",
        "",
        "| Run | Frame Total s | Tracking s | Keyframes | DBA Update s | Non-KF Update s | KF Update s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in runtime_rows:
        lines.append(
            "| {run} | {frame_total} | {tracking} | {keyframes} | {frontend_dba_update} | "
            "{frontend_update_non_keyframe_graph_update} | {frontend_update_keyframe_graph_update} |".format(**row)
        )
    lines.extend(
        [
            "",
            "| Run | Common Frames | Mean PSNR | Min PSNR | Mean SSIM | Min SSIM |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quality_rows:
        lines.append(
            "| {run} | {common_frames} | {mean_psnr} | {min_psnr} | {mean_ssim} | {min_ssim} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- RGB metrics are quick rgbdnua crops: top-left GT RGB vs bottom-left rendered RGB.",
            "- These metrics are a sanity check, not a full SLAM trajectory evaluation.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")

    return {
        "common_frames": common_frames,
        "runtime_rows": runtime_rows,
        "quality_rows": quality_rows,
        "quality_by_frame_rows": quality_by_frame_rows,
        "pose_delta_rows": pose_delta_rows,
    }


def _parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare VINGS tracking optimization runs.")
    parser.add_argument("--run", action="append", type=_parse_run, required=True, help="Run as LABEL=OUTPUT_DIR")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison reports")
    parser.add_argument("--max-contact-frames", type=int, default=12)
    args = parser.parse_args(argv)
    compare_runs(args.run, args.output_dir, max_contact_frames=args.max_contact_frames)


if __name__ == "__main__":
    main()
