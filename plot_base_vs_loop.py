#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare base iSAM2 vs loop-closure iSAM2 against GT")
    ap.add_argument(
        "--gt",
        default="../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset/poses/00.txt",
        help="Ground-truth KITTI pose file",
    )
    ap.add_argument("--base-est", required=True, help="Base iSAM2 estimated pose file")
    ap.add_argument("--loop-est", required=True, help="Loop-closure estimated pose file")
    ap.add_argument("--output-dir", default="output/loop_eval/plots", help="Directory for plots and summary")
    ap.add_argument("--align", choices=["none", "se3", "sim3"], default="se3")
    return ap.parse_args()


def _load_pose_utils(repo_root: Path):
    eval_dir = repo_root / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    from pose_utils import (  # pylint: disable=import-outside-toplevel
        angle_from_rotation,
        apply_sim3_to_poses,
        read_kitti_poses,
        summarize_errors,
        trajectory_positions,
        umeyama_alignment,
    )

    return {
        "angle_from_rotation": angle_from_rotation,
        "apply_sim3_to_poses": apply_sim3_to_poses,
        "read_kitti_poses": read_kitti_poses,
        "summarize_errors": summarize_errors,
        "trajectory_positions": trajectory_positions,
        "umeyama_alignment": umeyama_alignment,
    }


def _align(gt: np.ndarray, est: np.ndarray, align: str, fns: dict) -> tuple[np.ndarray, float]:
    gt_p = fns["trajectory_positions"](gt)
    est_p = fns["trajectory_positions"](est)
    if align == "none":
        return est, 1.0
    if align == "se3":
        t, s = fns["umeyama_alignment"](est_p, gt_p, with_scale=False)
        return fns["apply_sim3_to_poses"](t, s, est), float(s)
    t, s = fns["umeyama_alignment"](est_p, gt_p, with_scale=True)
    return fns["apply_sim3_to_poses"](t, s, est), float(s)


def _errors(gt: np.ndarray, est: np.ndarray, fns: dict) -> tuple[np.ndarray, np.ndarray]:
    n = min(gt.shape[0], est.shape[0])
    gt = gt[:n]
    est = est[:n]

    trans = np.linalg.norm(gt[:, :3, 3] - est[:, :3, 3], axis=1)
    rot = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r = gt[i, :3, :3].T @ est[i, :3, :3]
        rot[i] = np.degrees(fns["angle_from_rotation"](r))
    return trans, rot


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fns = _load_pose_utils(Path(__file__).resolve().parent)

    gt = fns["read_kitti_poses"](args.gt)
    base = fns["read_kitti_poses"](args.base_est)
    loop = fns["read_kitti_poses"](args.loop_est)

    n = min(gt.shape[0], base.shape[0], loop.shape[0])
    gt = gt[:n]
    base = base[:n]
    loop = loop[:n]

    base_aligned, _ = _align(gt, base, args.align, fns)
    loop_aligned, _ = _align(gt, loop, args.align, fns)

    base_trans, base_rot = _errors(gt, base_aligned, fns)
    loop_trans, loop_rot = _errors(gt, loop_aligned, fns)

    base_t = fns["summarize_errors"](base_trans)
    base_r = fns["summarize_errors"](base_rot)
    loop_t = fns["summarize_errors"](loop_trans)
    loop_r = fns["summarize_errors"](loop_rot)

    gt_p = fns["trajectory_positions"](gt)
    base_p = fns["trajectory_positions"](base_aligned)
    loop_p = fns["trajectory_positions"](loop_aligned)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gt_p[:, 0], gt_p[:, 2], label="GT", linewidth=2, color="#1d3557")
    ax.plot(base_p[:, 0], base_p[:, 2], label="Base (aligned)", linewidth=1.6, color="#2a9d8f")
    ax.plot(loop_p[:, 0], loop_p[:, 2], label="Loop closure (aligned)", linewidth=1.6, color="#e76f51")
    ax.set_title("Trajectory Comparison (X-Z)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_base_vs_loop.png", dpi=170)
    plt.close(fig)

    idx = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(idx, base_trans, label="Base translation error", color="#2a9d8f")
    ax.plot(idx, loop_trans, label="Loop translation error", color="#e76f51")
    ax.set_title("Per-frame Translation Error")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Error (m)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "translation_error_base_vs_loop.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(idx, base_rot, label="Base rotation error", color="#2a9d8f")
    ax.plot(idx, loop_rot, label="Loop rotation error", color="#e76f51")
    ax.set_title("Per-frame Rotation Error")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Error (deg)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rotation_error_base_vs_loop.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = [base_t.rmse, loop_t.rmse, base_r.rmse, loop_r.rmse]
    labels = ["Base ATE t", "Loop ATE t", "Base ATE r", "Loop ATE r"]
    colors = ["#2a9d8f", "#e76f51", "#457b9d", "#f4a261"]
    ax.bar(np.arange(len(metrics)), metrics, color=colors)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("ATE RMSE Comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "ate_rmse_base_vs_loop.png", dpi=170)
    plt.close(fig)

    summary = {
        "align": args.align,
        "num_poses": int(n),
        "base": {
            "ATE_trans_rmse_m": base_t.rmse,
            "ATE_rot_rmse_deg": base_r.rmse,
        },
        "loop": {
            "ATE_trans_rmse_m": loop_t.rmse,
            "ATE_rot_rmse_deg": loop_r.rmse,
        },
        "plots": {
            "trajectory": str(out_dir / "trajectory_base_vs_loop.png"),
            "translation_error": str(out_dir / "translation_error_base_vs_loop.png"),
            "rotation_error": str(out_dir / "rotation_error_base_vs_loop.png"),
            "ate_rmse": str(out_dir / "ate_rmse_base_vs_loop.png"),
        },
    }
    (out_dir / "base_vs_loop_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
