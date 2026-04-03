#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

from pose_utils import (
    TrajectoryMetrics,
    angle_from_rotation,
    read_kitti_poses,
    relative_transforms,
    summarize_errors,
    trajectory_positions,
    umeyama_alignment,
    apply_sim3_to_poses,
)


def compute_ate(gt: np.ndarray, est: np.ndarray, align: str) -> Dict[str, object]:
    n = min(gt.shape[0], est.shape[0])
    gt = gt[:n]
    est = est[:n]

    gt_p = trajectory_positions(gt)
    est_p = trajectory_positions(est)

    if align == "none":
        T_align = np.eye(4)
        scale = 1.0
        est_aligned = est
    elif align == "se3":
        T_align, scale = umeyama_alignment(est_p, gt_p, with_scale=False)
        est_aligned = apply_sim3_to_poses(T_align, scale, est)
    elif align == "sim3":
        T_align, scale = umeyama_alignment(est_p, gt_p, with_scale=True)
        est_aligned = apply_sim3_to_poses(T_align, scale, est)
    else:
        raise ValueError(f"Unknown align mode: {align}")

    est_p_al = trajectory_positions(est_aligned)
    trans_err = np.linalg.norm(gt_p - est_p_al, axis=1)

    # Orientation error per pose (angle between R_gt and R_est_al)
    rot_err = []
    for i in range(n):
        R_gt = gt[i, :3, :3]
        R_est = est_aligned[i, :3, :3]
        R_diff = R_gt.T @ R_est
        rot_err.append(angle_from_rotation(R_diff))
    rot_err = np.array(rot_err, dtype=np.float64)

    ate_t = summarize_errors(trans_err)
    ate_r = summarize_errors(np.degrees(rot_err))

    return {
        "num_poses": int(n),
        "align": align,
        "scale": float(scale),
        "ATE_trans_m": ate_t.__dict__,
        "ATE_rot_deg": ate_r.__dict__,
        "T_align": T_align.tolist(),
    }


def compute_rpe(gt: np.ndarray, est: np.ndarray, align: str, deltas: List[int]) -> Dict[str, object]:
    n = min(gt.shape[0], est.shape[0])
    gt = gt[:n]
    est = est[:n]

    gt_p = trajectory_positions(gt)
    est_p = trajectory_positions(est)

    if align == "none":
        T_align = np.eye(4)
        scale = 1.0
        est_aligned = est
    elif align == "se3":
        T_align, scale = umeyama_alignment(est_p, gt_p, with_scale=False)
        est_aligned = apply_sim3_to_poses(T_align, scale, est)
    elif align == "sim3":
        T_align, scale = umeyama_alignment(est_p, gt_p, with_scale=True)
        est_aligned = apply_sim3_to_poses(T_align, scale, est)
    else:
        raise ValueError(f"Unknown align mode: {align}")

    out: Dict[str, object] = {"align": align, "scale": float(scale), "deltas": {}}

    for d in deltas:
        rel_gt = relative_transforms(gt, d)
        rel_est = relative_transforms(est_aligned, d)

        # Error transform: E = rel_gt^{-1} * rel_est
        E = np.linalg.inv(rel_gt) @ rel_est

        trans = np.linalg.norm(E[:, :3, 3], axis=1)
        rot = []
        for i in range(E.shape[0]):
            rot.append(angle_from_rotation(E[i, :3, :3]))
        rot = np.degrees(np.array(rot, dtype=np.float64))

        out["deltas"][str(d)] = {
            "count": int(E.shape[0]),
            "RPE_trans_m": summarize_errors(trans).__dict__,
            "RPE_rot_deg": summarize_errors(rot).__dict__,
        }

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute KITTI ATE/RPE for iSAM2 outputs")
    ap.add_argument(
        "--gt",
        default="",
        help=(
            "Ground-truth poses file (KITTI 3x4 per line). "
            "If you pass a sequence like '00' or '00.txt', also set --kitti-poses-dir."
        ),
    )
    ap.add_argument(
        "--kitti-poses-dir",
        default="",
        help=(
            "Directory containing KITTI GT pose files like 00.txt..10.txt, e.g. "
            "data_odometry_poses/dataset/poses"
        ),
    )
    ap.add_argument(
        "--seq",
        default="",
        help="KITTI odometry sequence id for GT resolution (e.g. 00). Used with --kitti-poses-dir.",
    )
    ap.add_argument("--est", required=True, help="Estimated poses file (KITTI 3x4 per line)")
    ap.add_argument(
        "--align",
        default="se3",
        choices=["none", "se3", "sim3"],
        help="Alignment for ATE/RPE. KITTI usually uses SE(3) alignment.",
    )
    ap.add_argument(
        "--rpe-deltas",
        default="1,10",
        help="Comma-separated frame deltas for RPE (e.g. 1,10,100)",
    )
    ap.add_argument("--out", default="", help="Optional path to write JSON metrics")

    args = ap.parse_args()

    def resolve_gt_path() -> str:
        if args.seq and args.kitti_poses_dir:
            return os.path.join(args.kitti_poses_dir, f"{args.seq}.txt")
        if not args.gt:
            return ""
        if os.path.exists(args.gt):
            return args.gt
        # Convenience: allow --gt 00 or --gt 00.txt together with --kitti-poses-dir
        base = os.path.basename(args.gt)
        m = base
        if m.endswith(".txt"):
            m = m[:-4]
        if m.isdigit() and len(m) == 2 and args.kitti_poses_dir:
            return os.path.join(args.kitti_poses_dir, f"{m}.txt")
        return args.gt

    gt_path = resolve_gt_path()
    if not gt_path or not os.path.exists(gt_path):
        hint = (
            "Could not find ground-truth poses file.\n\n"
            "Common KITTI odometry GT location (after downloading data_odometry_poses):\n"
            "  <KITTI_ROOT>/data_odometry_poses/dataset/poses/00.txt\n\n"
            "Examples:\n"
            "  python3 eval/compute_kitti_metrics.py --kitti-poses-dir <...>/poses --seq 00 --est <est.txt>\n"
            "  python3 eval/compute_kitti_metrics.py --gt <...>/poses/00.txt --est <est.txt>\n"
        )
        print(hint, file=sys.stderr)
        raise FileNotFoundError(gt_path)

    if not os.path.exists(args.est):
        raise FileNotFoundError(args.est)

    gt = read_kitti_poses(gt_path)
    est = read_kitti_poses(args.est)
    deltas = [int(x) for x in args.rpe_deltas.split(",") if x.strip()]

    metrics = {
        "ATE": compute_ate(gt, est, args.align),
        "RPE": compute_rpe(gt, est, args.align, deltas),
    }

    txt = json.dumps(metrics, indent=2)
    print(txt)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
