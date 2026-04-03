#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot estimate-vs-GT trajectory differences for KITTI sequences"
    )
    ap.add_argument(
        "--dataset-root",
        default="../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset",
        help="KITTI dataset root containing poses/",
    )
    ap.add_argument(
        "--estimates-dir",
        default="output/batch_gt/estimates",
        help="Directory with files named poses_est_XX.txt",
    )
    ap.add_argument(
        "--output-dir",
        default="output/batch_gt/diff_plots",
        help="Directory to save generated plots and summary JSON",
    )
    ap.add_argument(
        "--align",
        choices=["none", "se3", "sim3"],
        default="se3",
        help="Alignment mode for comparing estimates to GT",
    )
    ap.add_argument(
        "--sequences",
        default="",
        help="Optional comma-separated sequence ids (e.g. 00,01,02). Default: auto from estimates-dir",
    )
    return ap.parse_args()


def _load_eval_pose_utils(repo_root: Path):
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


def _discover_sequences(estimates_dir: Path) -> list[str]:
    seqs: list[str] = []
    for p in sorted(estimates_dir.glob("poses_est_*.txt")):
        token = p.stem.replace("poses_est_", "")
        if token.isdigit():
            seqs.append(f"{int(token):02d}")
    return seqs


def _align_trajectory(
    gt: np.ndarray,
    est: np.ndarray,
    align: str,
    trajectory_positions,
    umeyama_alignment,
    apply_sim3_to_poses,
) -> tuple[np.ndarray, float]:
    gt_p = trajectory_positions(gt)
    est_p = trajectory_positions(est)

    if align == "none":
        return est, 1.0
    if align == "se3":
        t_align, scale = umeyama_alignment(est_p, gt_p, with_scale=False)
        return apply_sim3_to_poses(t_align, scale, est), float(scale)

    t_align, scale = umeyama_alignment(est_p, gt_p, with_scale=True)
    return apply_sim3_to_poses(t_align, scale, est), float(scale)


def _compute_frame_errors(gt: np.ndarray, est_aligned: np.ndarray, angle_from_rotation) -> tuple[np.ndarray, np.ndarray]:
    n = min(gt.shape[0], est_aligned.shape[0])
    gt = gt[:n]
    est_aligned = est_aligned[:n]

    trans_err = np.linalg.norm(gt[:, :3, 3] - est_aligned[:, :3, 3], axis=1)
    rot_deg = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r_diff = gt[i, :3, :3].T @ est_aligned[i, :3, :3]
        rot_deg[i] = np.degrees(angle_from_rotation(r_diff))

    return trans_err, rot_deg


def _plot_sequence(
    seq: str,
    gt: np.ndarray,
    est: np.ndarray,
    est_aligned: np.ndarray,
    trans_err: np.ndarray,
    rot_deg: np.ndarray,
    output_dir: Path,
    trajectory_positions,
) -> None:
    gt_p = trajectory_positions(gt)
    est_p = trajectory_positions(est)
    est_al_p = trajectory_positions(est_aligned)

    n = min(gt_p.shape[0], est_p.shape[0], est_al_p.shape[0])
    gt_p = gt_p[:n]
    est_p = est_p[:n]
    est_al_p = est_al_p[:n]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(gt_p[:, 0], gt_p[:, 2], label="GT", linewidth=2.0, color="#264653")
    axes[0].plot(est_p[:, 0], est_p[:, 2], label="Estimate (raw)", linewidth=1.6, color="#e76f51")
    axes[0].set_title(f"Seq {seq} Raw Trajectory (X-Z)")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(gt_p[:, 0], gt_p[:, 2], label="GT", linewidth=2.0, color="#264653")
    axes[1].plot(est_al_p[:, 0], est_al_p[:, 2], label="Estimate (aligned)", linewidth=1.6, color="#2a9d8f")
    axes[1].set_title(f"Seq {seq} Aligned Trajectory (X-Z)")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")
    axes[1].axis("equal")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    idx = np.arange(trans_err.shape[0])
    axes[2].plot(idx, trans_err, label="Trans error (m)", color="#457b9d", linewidth=1.8)
    axes[2].plot(idx, rot_deg, label="Rot error (deg)", color="#f4a261", linewidth=1.4)
    axes[2].set_title(f"Seq {seq} Per-frame Errors")
    axes[2].set_xlabel("Frame index")
    axes[2].set_ylabel("Error")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"seq_{seq}_comparison.png", dpi=170)
    plt.close(fig)


def _plot_summary(rows: list[dict], output_dir: Path) -> None:
    seqs = [r["seq"] for r in rows]
    trans_rmse = np.asarray([r["trans_rmse_m"] for r in rows], dtype=np.float64)
    rot_rmse = np.asarray([r["rot_rmse_deg"] for r in rows], dtype=np.float64)

    x = np.arange(len(seqs))
    w = 0.4

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, trans_rmse, width=w, label="Trans RMSE (m)", color="#2a9d8f")
    ax.bar(x + w / 2, rot_rmse, width=w, label="Rot RMSE (deg)", color="#f4a261")
    ax.set_xticks(x)
    ax.set_xticklabels(seqs)
    ax.set_title("Estimate vs GT Error Summary by Sequence")
    ax.set_xlabel("Sequence")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "summary_trans_rot_rmse.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    estimates_dir = Path(args.estimates_dir)
    output_dir = Path(args.output_dir)

    pose_utils = _load_eval_pose_utils(repo_root)

    if args.sequences.strip():
        seqs = [f"{int(s.strip()):02d}" for s in args.sequences.split(",") if s.strip()]
    else:
        seqs = _discover_sequences(estimates_dir)

    if not seqs:
        raise ValueError(f"No estimate files found in {estimates_dir}")

    rows: list[dict] = []

    for seq in seqs:
        gt_path = dataset_root / "poses" / f"{seq}.txt"
        est_path = estimates_dir / f"poses_est_{seq}.txt"

        if not gt_path.exists():
            print(f"[WARN] Missing GT for seq {seq}: {gt_path}")
            continue
        if not est_path.exists():
            print(f"[WARN] Missing estimate for seq {seq}: {est_path}")
            continue

        gt = pose_utils["read_kitti_poses"](str(gt_path))
        est = pose_utils["read_kitti_poses"](str(est_path))

        n = min(gt.shape[0], est.shape[0])
        gt = gt[:n]
        est = est[:n]

        est_aligned, scale = _align_trajectory(
            gt,
            est,
            args.align,
            pose_utils["trajectory_positions"],
            pose_utils["umeyama_alignment"],
            pose_utils["apply_sim3_to_poses"],
        )

        trans_err, rot_deg = _compute_frame_errors(gt, est_aligned, pose_utils["angle_from_rotation"])

        trans_stats = pose_utils["summarize_errors"](trans_err)
        rot_stats = pose_utils["summarize_errors"](rot_deg)

        _plot_sequence(
            seq,
            gt,
            est,
            est_aligned,
            trans_err,
            rot_deg,
            output_dir,
            pose_utils["trajectory_positions"],
        )

        row = {
            "seq": seq,
            "num_poses": int(n),
            "align": args.align,
            "scale": float(scale),
            "trans_rmse_m": trans_stats.rmse,
            "trans_mean_m": trans_stats.mean,
            "rot_rmse_deg": rot_stats.rmse,
            "rot_mean_deg": rot_stats.mean,
            "plot_path": str(output_dir / f"seq_{seq}_comparison.png"),
        }
        rows.append(row)
        print(
            f"seq={seq} | poses={n} | trans_rmse={trans_stats.rmse:.4f} m | "
            f"rot_rmse={rot_stats.rmse:.4f} deg"
        )

    if not rows:
        raise RuntimeError("No sequence plots generated. Check dataset and estimate paths.")

    rows = sorted(rows, key=lambda r: r["seq"])
    _plot_summary(rows, output_dir)

    summary = {
        "dataset_root": str(dataset_root),
        "estimates_dir": str(estimates_dir),
        "align": args.align,
        "rows": rows,
        "summary_plot": str(output_dir / "summary_trans_rot_rmse.png"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "difference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote summary JSON: {output_dir / 'difference_summary.json'}")
    print(f"Wrote summary plot: {output_dir / 'summary_trans_rot_rmse.png'}")


if __name__ == "__main__":
    main()
