#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from slam.isam2_backend import Isam2PoseGraph
from slam.kitti_loader import KITTISequenceLoader
from slam.stereo_vo import StereoVisualOdometry


def write_kitti_poses(path: Path, ts: np.ndarray) -> None:
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim != 3 or ts.shape[1:] != (4, 4):
        raise ValueError("Expected poses with shape (N,4,4)")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in ts:
            row = t[:3, :4].reshape(-1)
            f.write(" ".join(f"{x:.12g}" for x in row) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stereo VO + iSAM2 baseline for KITTI odometry")
    ap.add_argument(
        "--dataset-root",
        default="../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset",
        help="Path containing KITTI odometry poses/ and sequences/",
    )
    ap.add_argument("--seq", default="00", help="KITTI sequence id, e.g., 00")
    ap.add_argument("--max-frames", type=int, default=0, help="0 means use all frames")
    ap.add_argument("--output", default="output/poses_est_00.txt", help="Output KITTI pose file")
    ap.add_argument("--metrics-out", default="output/metrics_00.json", help="Optional metrics JSON output")
    ap.add_argument(
        "--min-inliers",
        type=int,
        default=25,
        help="Minimum PnP inliers required before accepting motion estimate",
    )
    ap.add_argument(
        "--fallback-no-motion",
        action="store_true",
        help="If set, use identity motion when VO fails (keeps frame count). Otherwise fail fast.",
    )
    ap.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip ATE/RPE computation against ground truth.",
    )
    return ap.parse_args()


def _summarize(gt: np.ndarray, est: np.ndarray) -> dict:
    eval_dir = Path(__file__).resolve().parent / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    from compute_kitti_metrics import compute_ate, compute_rpe

    n = min(gt.shape[0], est.shape[0])
    deltas = [d for d in [1, 10, 100] if d < n]
    if not deltas:
        deltas = [1]

    ate = compute_ate(gt, est, align="se3")
    rpe = compute_rpe(gt, est, align="se3", deltas=deltas)
    return {"ATE": ate, "RPE": rpe}


def run_sequence(
    dataset_root: str,
    seq: str,
    max_frames: int,
    output_path: Path,
    metrics_path: Path,
    min_inliers: int,
    fallback_no_motion: bool,
    skip_metrics: bool,
) -> dict:
    loader = KITTISequenceLoader(dataset_root, seq)
    vo = StereoVisualOdometry(loader.calib, min_pnp_inliers=min_inliers)
    backend = Isam2PoseGraph()

    num_frames = loader.num_frames()
    if max_frames > 0:
        num_frames = min(num_frames, max_frames)

    left_prev, right_prev, _ = loader.read_stereo(0)

    accepted = 0
    fallback_used = 0

    for i in range(1, num_frames):
        left_curr, right_curr, _ = loader.read_stereo(i)

        estimate = vo.estimate_prev_to_curr(left_prev, right_prev, left_curr)
        if estimate is None:
            if not fallback_no_motion:
                raise RuntimeError(
                    f"VO failed at frame {i}. Re-run with --fallback-no-motion to continue."
                )
            print(f"[WARN] seq={seq} frame={i}: VO failed, using identity fallback")
            t_prev_curr = np.eye(4, dtype=np.float64)
            fallback_used += 1
        else:
            t_prev_curr = estimate.t_prev_to_curr
            accepted += 1

        backend.add_odometry(i - 1, i, t_prev_curr)

        left_prev = left_curr
        right_prev = right_curr

        if i % 100 == 0:
            print(
                f"seq={seq} processed {i}/{num_frames - 1} | "
                f"accepted={accepted} fallback={fallback_used}"
            )

    est = backend.trajectory_matrices()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_kitti_poses(output_path, est)
    print(f"Wrote estimated trajectory: {output_path} ({est.shape[0]} poses)")

    result = {
        "seq": f"{int(seq):02d}",
        "num_poses": int(est.shape[0]),
        "accepted_transitions": int(accepted),
        "fallback_transitions": int(fallback_used),
        "est_path": str(output_path),
    }

    if skip_metrics:
        return result

    gt = loader.read_ground_truth()
    if gt is None:
        print("[WARN] Ground truth not found; metrics skipped")
        return result

    metrics = _summarize(gt, est)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote metrics: {metrics_path}")

    result["metrics_path"] = str(metrics_path)
    result["metrics"] = metrics
    return result


def main() -> None:
    args = parse_args()
    run_sequence(
        dataset_root=args.dataset_root,
        seq=args.seq,
        max_frames=args.max_frames,
        output_path=Path(args.output),
        metrics_path=Path(args.metrics_out),
        min_inliers=args.min_inliers,
        fallback_no_motion=args.fallback_no_motion,
        skip_metrics=args.skip_metrics,
    )


if __name__ == "__main__":
    main()
