#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np

from slam.isam2_backend import Isam2PoseGraph, Isam2InfoWeighted, compute_confidence
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
    ap.add_argument(
        "--enable-loop-closure",
        action="store_true",
        help="Enable geometric loop-closure candidate search and add loop factors.",
    )
    ap.add_argument(
        "--loop-min-separation",
        type=int,
        default=120,
        help="Minimum frame gap between current frame and loop candidate.",
    )
    ap.add_argument(
        "--loop-search-radius-m",
        type=float,
        default=8.0,
        help="Candidate search radius in estimated trajectory space (meters).",
    )
    ap.add_argument(
        "--loop-max-candidates",
        type=int,
        default=3,
        help="Maximum loop candidates to verify per frame.",
    )
    ap.add_argument(
        "--loop-min-inliers",
        type=int,
        default=45,
        help="Minimum inliers required to accept a loop-closure measurement.",
    )
    ap.add_argument(
        "--loop-use-appearance-scan",
        action="store_true",
        help="Enable appearance-based candidate retrieval when spatial candidates are absent.",
    )
    ap.add_argument(
        "--loop-appearance-stride",
        type=int,
        default=20,
        help="Stride for historical frame scan during appearance-based retrieval.",
    )
    ap.add_argument(
        "--loop-appearance-min-matches",
        type=int,
        default=80,
        help="Minimum descriptor matches to keep an appearance-based loop candidate.",
    )
    ap.add_argument(
        "--loop-consistency-trans-m",
        type=float,
        default=10.0,
        help="Max translation disagreement (m) between candidate loop measurement and current graph estimate.",
    )
    ap.add_argument(
        "--loop-consistency-rot-deg",
        type=float,
        default=35.0,
        help="Max rotation disagreement (deg) between candidate loop measurement and current graph estimate.",
    )
    ap.add_argument(
        "--info-weighted",
        action="store_true",
        help="Use information-based weighting: scale each factor's covariance by VO confidence.",
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


def _refresh_positions_from_backend(backend: Isam2PoseGraph, positions: List[np.ndarray]) -> None:
    traj = backend.trajectory_matrices()
    positions.clear()
    for t in traj:
        positions.append(t[:3, 3].copy())


def _descriptor_for_frame(
    frame_idx: int,
    loader: KITTISequenceLoader,
    vo: StereoVisualOdometry,
    desc_cache: dict[int, np.ndarray | None],
) -> np.ndarray | None:
    if frame_idx in desc_cache:
        return desc_cache[frame_idx]

    left, _, _ = loader.read_stereo(frame_idx)
    _, desc = vo._detect(left)
    desc_cache[frame_idx] = desc
    return desc


def _try_add_loop_closures(
    curr_idx: int,
    left_curr: np.ndarray,
    loader: KITTISequenceLoader,
    vo: StereoVisualOdometry,
    backend: Isam2PoseGraph,
    est_positions: List[np.ndarray],
    added_pairs: Set[Tuple[int, int]],
    loop_min_separation: int,
    loop_search_radius_m: float,
    loop_max_candidates: int,
    loop_min_inliers: int,
    loop_use_appearance_scan: bool,
    loop_appearance_stride: int,
    loop_appearance_min_matches: int,
    loop_consistency_trans_m: float,
    loop_consistency_rot_deg: float,
    desc_cache: dict[int, np.ndarray | None],
) -> List[dict]:
    if curr_idx < loop_min_separation:
        return []

    curr_pos = est_positions[curr_idx]
    candidate_limit = curr_idx - loop_min_separation
    if candidate_limit <= 0:
        return []

    dists: List[Tuple[float, int, str, float]] = []
    for j in range(candidate_limit):
        if (j, curr_idx) in added_pairs:
            continue
        dist = float(np.linalg.norm(curr_pos - est_positions[j]))
        if dist <= loop_search_radius_m:
            dists.append((dist, j, "spatial", dist))

    if not dists and loop_use_appearance_scan:
        _, desc_curr = vo._detect(left_curr)
        if desc_curr is not None and desc_curr.shape[0] >= 16:
            stride = max(1, loop_appearance_stride)
            appearance_candidates: List[Tuple[float, int, str, float]] = []
            for j in range(0, candidate_limit, stride):
                if (j, curr_idx) in added_pairs:
                    continue
                desc_j = _descriptor_for_frame(j, loader, vo, desc_cache)
                if desc_j is None or desc_j.shape[0] < 16:
                    continue

                good = vo._knn_ratio_matches(desc_j, desc_curr)
                if len(good) < loop_appearance_min_matches:
                    continue

                # Use inverse match count as a sort key: more matches -> smaller key.
                spatial_dist = float(np.linalg.norm(curr_pos - est_positions[j]))
                appearance_candidates.append((1.0 / float(len(good)), j, "appearance", spatial_dist))

            appearance_candidates.sort(key=lambda x: x[0])
            dists.extend(appearance_candidates[: max(1, loop_max_candidates)])

    if not dists:
        return []

    dists.sort(key=lambda x: x[0])
    accepted: List[dict] = []

    for score, j, candidate_type, spatial_dist in dists[: max(1, loop_max_candidates)]:
        left_j, right_j, _ = loader.read_stereo(j)
        loop_est = vo.estimate_prev_to_curr(left_j, right_j, left_curr)
        used_fallback_2d2d = False
        if loop_est is None:
            loop_est = vo.estimate_prev_to_curr_2d2d(
                left_j,
                left_curr,
                min_inliers=max(20, loop_min_inliers // 2),
            )
            if loop_est is not None:
                # 2D-2D pose has unknown scale: use current estimated inter-node distance.
                t_norm = float(np.linalg.norm(loop_est.t_prev_to_curr[:3, 3]))
                if t_norm > 1e-9:
                    scale = max(spatial_dist, 1e-3) / t_norm
                    loop_est.t_prev_to_curr[:3, 3] *= scale
                used_fallback_2d2d = True

        if loop_est is None or loop_est.inliers < loop_min_inliers:
            continue

        # Reject loop candidates that strongly disagree with current graph geometry.
        t_src = backend.pose_matrix(j)
        t_dst = backend.pose_matrix(curr_idx)
        pred_src_dst = np.linalg.inv(t_src) @ t_dst
        meas_src_dst = loop_est.t_prev_to_curr

        trans_disagreement = float(np.linalg.norm(pred_src_dst[:3, 3] - meas_src_dst[:3, 3]))
        r_diff = pred_src_dst[:3, :3].T @ meas_src_dst[:3, :3]
        cos_theta = float((np.trace(r_diff) - 1.0) * 0.5)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        rot_disagreement_deg = float(np.degrees(np.arccos(cos_theta)))

        if (
            trans_disagreement > loop_consistency_trans_m
            or rot_disagreement_deg > loop_consistency_rot_deg
        ):
            continue

        try:
            backend.add_loop_closure(j, curr_idx, loop_est.t_prev_to_curr)
        except RuntimeError as exc:
            print(f"[WARN] loop insertion rejected at ({j},{curr_idx}): {exc}")
            continue

        added_pairs.add((j, curr_idx))
        accepted.append(
            {
                "from_idx": int(j),
                "to_idx": int(curr_idx),
                "candidate_distance_m": spatial_dist,
                "candidate_score": score,
                "candidate_type": candidate_type,
                "inliers": int(loop_est.inliers),
                "matches": int(loop_est.total_matches),
                "used_fallback_2d2d": used_fallback_2d2d,
                "consistency_trans_disagreement_m": trans_disagreement,
                "consistency_rot_disagreement_deg": rot_disagreement_deg,
            }
        )

    if accepted:
        _refresh_positions_from_backend(backend, est_positions)

    return accepted


def run_sequence(
    dataset_root: str,
    seq: str,
    max_frames: int,
    output_path: Path,
    metrics_path: Path,
    min_inliers: int,
    fallback_no_motion: bool,
    skip_metrics: bool,
    enable_loop_closure: bool = False,
    loop_min_separation: int = 120,
    loop_search_radius_m: float = 8.0,
    loop_max_candidates: int = 3,
    loop_min_inliers: int = 45,
    loop_use_appearance_scan: bool = False,
    loop_appearance_stride: int = 20,
    loop_appearance_min_matches: int = 80,
    loop_consistency_trans_m: float = 10.0,
    loop_consistency_rot_deg: float = 35.0,
    info_weighted: bool = False,
) -> dict:
    loader = KITTISequenceLoader(dataset_root, seq)
    vo = StereoVisualOdometry(loader.calib, min_pnp_inliers=min_inliers)
    backend = Isam2InfoWeighted() if info_weighted else Isam2PoseGraph()

    num_frames = loader.num_frames()
    if max_frames > 0:
        num_frames = min(num_frames, max_frames)

    left_prev, right_prev, _ = loader.read_stereo(0)

    accepted = 0
    fallback_used = 0
    confidence_log: list[float] = []
    loop_closures_added = 0
    loop_pairs: List[dict] = []
    added_pairs: Set[Tuple[int, int]] = set()
    est_positions: List[np.ndarray] = [backend.pose_matrix(0)[:3, 3].copy()]
    desc_cache: dict[int, np.ndarray | None] = {}

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
            confidence = 0.1  # minimum confidence for fallback
        else:
            t_prev_curr = estimate.t_prev_to_curr
            accepted += 1
            confidence = compute_confidence(
                estimate.inliers, estimate.total_matches, estimate.mean_reproj_error
            )

        if info_weighted:
            backend.add_odometry_weighted(i - 1, i, t_prev_curr, confidence)
            confidence_log.append(confidence)
        else:
            backend.add_odometry(i - 1, i, t_prev_curr)

        est_positions.append(backend.pose_matrix(i)[:3, 3].copy())

        if enable_loop_closure:
            accepted_loops = _try_add_loop_closures(
                curr_idx=i,
                left_curr=left_curr,
                loader=loader,
                vo=vo,
                backend=backend,
                est_positions=est_positions,
                added_pairs=added_pairs,
                loop_min_separation=loop_min_separation,
                loop_search_radius_m=loop_search_radius_m,
                loop_max_candidates=loop_max_candidates,
                loop_min_inliers=loop_min_inliers,
                loop_use_appearance_scan=loop_use_appearance_scan,
                loop_appearance_stride=loop_appearance_stride,
                loop_appearance_min_matches=loop_appearance_min_matches,
                loop_consistency_trans_m=loop_consistency_trans_m,
                loop_consistency_rot_deg=loop_consistency_rot_deg,
                desc_cache=desc_cache,
            )
            if accepted_loops:
                loop_closures_added += len(accepted_loops)
                loop_pairs.extend(accepted_loops)
                print(
                    f"[LOOP] seq={seq} frame={i}: added {len(accepted_loops)} closure(s)"
                )

        left_prev = left_curr
        right_prev = right_curr

        if i % 100 == 0:
            print(
                f"seq={seq} processed {i}/{num_frames - 1} | "
                f"accepted={accepted} fallback={fallback_used} loops={loop_closures_added}"
            )

    est = backend.trajectory_matrices()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_kitti_poses(output_path, est)
    print(f"Wrote estimated trajectory: {output_path} ({est.shape[0]} poses)")

    result = {
        "seq": f"{int(seq):02d}",
        "mode": "loop-closure" if enable_loop_closure else "base",
        "num_poses": int(est.shape[0]),
        "accepted_transitions": int(accepted),
        "fallback_transitions": int(fallback_used),
        "loop_closures_added": int(loop_closures_added),
        "loop_closure_pairs": loop_pairs,
        "est_path": str(output_path),
        "info_weighted": info_weighted,
    }

    if info_weighted and confidence_log:
        c = np.array(confidence_log)
        result["confidence_stats"] = {
            "mean": float(c.mean()),
            "std": float(c.std()),
            "min": float(c.min()),
            "max": float(c.max()),
            "median": float(np.median(c)),
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
        enable_loop_closure=args.enable_loop_closure,
        loop_min_separation=args.loop_min_separation,
        loop_search_radius_m=args.loop_search_radius_m,
        loop_max_candidates=args.loop_max_candidates,
        loop_min_inliers=args.loop_min_inliers,
        loop_use_appearance_scan=args.loop_use_appearance_scan,
        loop_appearance_stride=args.loop_appearance_stride,
        loop_appearance_min_matches=args.loop_appearance_min_matches,
        loop_consistency_trans_m=args.loop_consistency_trans_m,
        loop_consistency_rot_deg=args.loop_consistency_rot_deg,
        info_weighted=args.info_weighted,
    )


if __name__ == "__main__":
    main()
