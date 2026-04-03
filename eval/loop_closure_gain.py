#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np

from pose_utils import (
    read_kitti_poses,
    trajectory_positions,
    umeyama_alignment,
    apply_sim3_to_poses,
)


def _natural_key(s: str) -> Tuple:
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s))


def load_snapshots(dir_path: str, pattern: str) -> List[Tuple[str, np.ndarray]]:
    """Loads snapshot trajectories from a directory.

    pattern is a regex applied to file basenames (e.g. r".*\\.txt$").
    Returns list of (filename, Ts).
    """
    rx = re.compile(pattern)
    files = [f for f in os.listdir(dir_path) if rx.match(f)]
    files.sort(key=_natural_key)
    if not files:
        raise ValueError(f"No files in {dir_path} matched pattern {pattern}")

    out = []
    for f in files:
        out.append((f, read_kitti_poses(os.path.join(dir_path, f))))
    return out


def ate_rmse_trans(gt: np.ndarray, est: np.ndarray, align: str) -> float:
    n = min(gt.shape[0], est.shape[0])
    gt = gt[:n]
    est = est[:n]
    gt_p = trajectory_positions(gt)
    est_p = trajectory_positions(est)

    if align == "none":
        est_al = est
    elif align == "se3":
        T, s = umeyama_alignment(est_p, gt_p, with_scale=False)
        est_al = apply_sim3_to_poses(T, s, est)
    elif align == "sim3":
        T, s = umeyama_alignment(est_p, gt_p, with_scale=True)
        est_al = apply_sim3_to_poses(T, s, est)
    else:
        raise ValueError(f"Unknown align mode: {align}")

    err = np.linalg.norm(trajectory_positions(est_al) - gt_p, axis=1)
    return float(np.sqrt(np.mean(err**2)))


def pose_correction_rms(prev: np.ndarray, post: np.ndarray) -> float:
    """RMS translation correction magnitude between two trajectories."""
    n = min(prev.shape[0], post.shape[0])
    dp = trajectory_positions(post[:n]) - trajectory_positions(prev[:n])
    return float(np.sqrt(np.mean(np.sum(dp**2, axis=1))))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Estimate loop-closure correction gain from saved iSAM2 trajectory snapshots. "
            "This is only meaningful if your snapshots are taken right before/after loop closures."
        )
    )
    ap.add_argument(
        "--gt",
        default="",
        help=(
            "Ground-truth poses file. If you pass a sequence like '00' or '00.txt', also set --kitti-poses-dir."
        ),
    )
    ap.add_argument(
        "--kitti-poses-dir",
        default="",
        help="Directory containing KITTI GT pose files like 00.txt..10.txt",
    )
    ap.add_argument(
        "--seq",
        default="",
        help="KITTI odometry sequence id (e.g. 00) used with --kitti-poses-dir",
    )
    ap.add_argument(
        "--snapshots-dir",
        required=True,
        help=(
            "Directory containing trajectory snapshots (each a KITTI poses file). "
            "Sorted naturally by filename."
        ),
    )
    ap.add_argument(
        "--pattern",
        default=r".*\\.txt$",
        help="Regex pattern for snapshot filenames",
    )
    ap.add_argument(
        "--events",
        default="",
        help=(
            "Optional path to a text file listing snapshot indices that correspond to 'after loop closure' states. "
            "One integer per line (0-based index into the sorted snapshot list)."
        ),
    )
    ap.add_argument(
        "--align",
        default="se3",
        choices=["none", "se3", "sim3"],
        help="Alignment used when computing ATE per snapshot",
    )
    ap.add_argument("--out", default="", help="Optional JSON output path")

    args = ap.parse_args()

    def resolve_gt_path() -> str:
        if args.seq and args.kitti_poses_dir:
            return os.path.join(args.kitti_poses_dir, f"{args.seq}.txt")
        if not args.gt:
            return ""
        if os.path.exists(args.gt):
            return args.gt
        base = os.path.basename(args.gt)
        m = base[:-4] if base.endswith(".txt") else base
        if m.isdigit() and len(m) == 2 and args.kitti_poses_dir:
            return os.path.join(args.kitti_poses_dir, f"{m}.txt")
        return args.gt

    gt_path = resolve_gt_path()
    if not gt_path or not os.path.exists(gt_path):
        hint = (
            "Could not find ground-truth poses file.\n\n"
            "Common KITTI odometry GT location (after downloading data_odometry_poses):\n"
            "  <KITTI_ROOT>/data_odometry_poses/dataset/poses/00.txt\n\n"
            "Example:\n"
            "  python3 eval/loop_closure_gain.py --kitti-poses-dir <...>/poses --seq 00 --snapshots-dir snapshots --events events.txt\n"
        )
        print(hint, file=sys.stderr)
        raise FileNotFoundError(gt_path)

    gt = read_kitti_poses(gt_path)
    snapshots = load_snapshots(args.snapshots_dir, args.pattern)

    ate_series = []
    for name, Ts in snapshots:
        ate_series.append(ate_rmse_trans(gt, Ts, args.align))

    report: Dict[str, object] = {
        "num_snapshots": len(snapshots),
        "align": args.align,
        "snapshots": [name for name, _ in snapshots],
        "ATE_rmse_trans_m": ate_series,
        "global_gain_m": float(ate_series[0] - ate_series[-1]),
    }

    if args.events:
        with open(args.events, "r", encoding="utf-8") as f:
            events = [int(line.strip()) for line in f if line.strip()]
        gains = []
        corrections = []
        for e in events:
            if e <= 0 or e >= len(snapshots):
                continue
            before = ate_series[e - 1]
            after = ate_series[e]
            gains.append({"event_snapshot_index": e, "ate_before": before, "ate_after": after, "gain_m": before - after})

            prev_T = snapshots[e - 1][1]
            post_T = snapshots[e][1]
            corrections.append({"event_snapshot_index": e, "rms_translation_correction_m": pose_correction_rms(prev_T, post_T)})

        report["events"] = gains
        report["corrections"] = corrections

    txt = json.dumps(report, indent=2)
    print(txt)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
