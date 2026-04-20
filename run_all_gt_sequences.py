#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_isam2_kitti import run_sequence


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run stereo VO + iSAM2 for all KITTI sequences that have GT poses"
    )
    ap.add_argument(
        "--dataset-root",
        default="../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset",
        help="Path containing KITTI odometry poses/ and sequences/",
    )
    ap.add_argument(
        "--output-dir",
        default="output/batch_gt",
        help="Directory where per-sequence estimates/metrics and summary are written",
    )
    ap.add_argument("--max-frames", type=int, default=0, help="0 means use all frames")
    ap.add_argument(
        "--min-inliers",
        type=int,
        default=25,
        help="Minimum PnP inliers required before accepting motion estimate",
    )
    ap.add_argument(
        "--fallback-no-motion",
        action="store_true",
        help="Use identity fallback when VO fails, instead of aborting sequence run.",
    )
    ap.add_argument(
        "--sequences",
        default="",
        help="Optional comma-separated sequence ids (e.g. 00,01,02). Default: auto from poses/*.txt",
    )
    ap.add_argument(
        "--enable-loop-closure",
        action="store_true",
        help="Enable loop-closure mode during sequence runs.",
    )
    ap.add_argument(
        "--enable-confidence-weighting",
        action="store_true",
        help="Enable confidence-v1 odometry weighting during sequence runs.",
    )
    ap.add_argument(
        "--enable-info-weighting",
        action="store_true",
        help="Enable Mo-style info weighting during sequence runs.",
    )
    ap.add_argument(
        "--info-min-confidence",
        type=float,
        default=0.1,
        help="Minimum confidence clamp for info-weighted odometry factors.",
    )
    ap.add_argument(
        "--info-max-confidence",
        type=float,
        default=1.0,
        help="Maximum confidence clamp for info-weighted odometry factors.",
    )
    ap.add_argument(
        "--confidence-floor",
        type=float,
        default=0.08,
        help="Lower bound for confidence-v1 weighting.",
    )
    ap.add_argument(
        "--confidence-power",
        type=float,
        default=1.0,
        help="Exponent applied to confidence-v1 values.",
    )
    ap.add_argument(
        "--confidence-min-scale",
        type=float,
        default=0.85,
        help="Minimum odometry sigma scale for confidence-v1.",
    )
    ap.add_argument(
        "--confidence-max-scale",
        type=float,
        default=4.0,
        help="Maximum odometry sigma scale for confidence-v1.",
    )
    ap.add_argument(
        "--enable-confidence-v2",
        action="store_true",
        help="Enable confidence-v2 weighting with optional robust kernels.",
    )
    ap.add_argument(
        "--confidence-v2-floor",
        type=float,
        default=0.03,
        help="Lower bound for confidence-v2 values.",
    )
    ap.add_argument(
        "--confidence-v2-power",
        type=float,
        default=1.5,
        help="Exponent on confidence-v2 values.",
    )
    ap.add_argument(
        "--confidence-v2-odom-min-scale",
        type=float,
        default=0.7,
        help="Minimum odometry sigma scale for confidence-v2.",
    )
    ap.add_argument(
        "--confidence-v2-odom-max-scale",
        type=float,
        default=10.0,
        help="Maximum odometry sigma scale for confidence-v2.",
    )
    ap.add_argument(
        "--confidence-v2-loop-min-scale",
        type=float,
        default=0.8,
        help="Minimum loop sigma scale for confidence-v2.",
    )
    ap.add_argument(
        "--confidence-v2-loop-max-scale",
        type=float,
        default=14.0,
        help="Maximum loop sigma scale for confidence-v2.",
    )
    ap.add_argument(
        "--confidence-v2-loop-trans-tau-m",
        type=float,
        default=4.0,
        help="Translation disagreement decay (m) for confidence-v2 loop scoring.",
    )
    ap.add_argument(
        "--confidence-v2-loop-rot-tau-deg",
        type=float,
        default=12.0,
        help="Rotation disagreement decay (deg) for confidence-v2 loop scoring.",
    )
    ap.add_argument(
        "--confidence-v2-robust-kernel",
        choices=["none", "huber", "cauchy"],
        default="huber",
        help="Robust kernel used for confidence-v2 factors.",
    )
    ap.add_argument(
        "--confidence-v2-robust-k",
        type=float,
        default=1.5,
        help="Robust kernel width for confidence-v2 factors.",
    )
    ap.add_argument(
        "--degeneracy-aware",
        action="store_true",
        help="Enable Alfred-style degeneracy-aware odometry scaling (loop mode only).",
    )
    ap.add_argument(
        "--degeneracy-cond-ref",
        type=float,
        default=800.0,
        help="Reference condition number for degeneracy scaling.",
    )
    ap.add_argument(
        "--degeneracy-noise-max-mult",
        type=float,
        default=25.0,
        help="Maximum odometry sigma multiplier for degeneracy-aware mode.",
    )
    ap.add_argument(
        "--degeneracy-fallback-mult",
        type=float,
        default=None,
        help="Fallback odometry multiplier on VO failure for degeneracy-aware mode.",
    )
    ap.add_argument(
        "--loop-min-separation",
        type=int,
        default=120,
        help="Minimum frame gap between loop-linked frames.",
    )
    ap.add_argument(
        "--loop-search-radius-m",
        type=float,
        default=8.0,
        help="Maximum estimated spatial distance for loop candidate retrieval.",
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
        help="Minimum geometric inliers to accept a loop closure.",
    )
    ap.add_argument(
        "--loop-use-appearance-scan",
        action="store_true",
        help="Enable appearance-based loop-candidate retrieval fallback.",
    )
    ap.add_argument(
        "--loop-appearance-stride",
        type=int,
        default=20,
        help="Frame stride for appearance-based historical scan.",
    )
    ap.add_argument(
        "--loop-appearance-min-matches",
        type=int,
        default=80,
        help="Minimum descriptor matches to keep an appearance candidate.",
    )
    ap.add_argument(
        "--loop-consistency-trans-m",
        type=float,
        default=10.0,
        help="Max translation disagreement to accept loop closure against current estimate.",
    )
    ap.add_argument(
        "--loop-consistency-rot-deg",
        type=float,
        default=35.0,
        help="Max rotation disagreement to accept loop closure against current estimate.",
    )
    return ap.parse_args()


def discover_gt_sequences(dataset_root: Path) -> list[str]:
    poses_dir = dataset_root / "poses"
    seqs = sorted(p.stem for p in poses_dir.glob("*.txt") if p.stem.isdigit())
    return [f"{int(s):02d}" for s in seqs]


def extract_summary_row(result: dict) -> dict:
    row = {
        "seq": result["seq"],
        "mode": result.get("mode", "base"),
        "num_poses": result["num_poses"],
        "accepted_transitions": result["accepted_transitions"],
        "fallback_transitions": result["fallback_transitions"],
        "loop_closures_added": result.get("loop_closures_added", 0),
    }

    metrics = result.get("metrics")
    if metrics:
        row["ate_rmse_m"] = metrics["ATE"]["ATE_trans_m"]["rmse"]
        row["ate_rot_rmse_deg"] = metrics["ATE"]["ATE_rot_deg"]["rmse"]
        for d in ["1", "10", "100"]:
            row[f"rpe_trans_rmse_d{d}_m"] = None
            row[f"rpe_rot_rmse_d{d}_deg"] = None
        for d, vals in metrics["RPE"]["deltas"].items():
            row[f"rpe_trans_rmse_d{d}_m"] = vals["RPE_trans_m"]["rmse"]
            row[f"rpe_rot_rmse_d{d}_deg"] = vals["RPE_rot_deg"]["rmse"]
    else:
        row["ate_rmse_m"] = None
        row["ate_rot_rmse_deg"] = None

    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    headers = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            vals = []
            for h in headers:
                v = row.get(h, "")
                if v is None:
                    vals.append("")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir)

    if args.sequences.strip():
        seqs = [f"{int(s.strip()):02d}" for s in args.sequences.split(",") if s.strip()]
    else:
        seqs = discover_gt_sequences(dataset_root)

    if not seqs:
        raise ValueError(f"No GT sequences found in {dataset_root / 'poses'}")

    print(f"Running sequences: {', '.join(seqs)}")

    run_rows: list[dict] = []
    run_details: dict[str, dict] = {}

    for seq in seqs:
        print(f"\n=== Sequence {seq} ===")
        seq_start = time.time()

        est_path = output_dir / "estimates" / f"poses_est_{seq}.txt"
        metrics_path = output_dir / "metrics" / f"metrics_{seq}.json"

        result = run_sequence(
            dataset_root=str(dataset_root),
            seq=seq,
            max_frames=args.max_frames,
            output_path=est_path,
            metrics_path=metrics_path,
            min_inliers=args.min_inliers,
            fallback_no_motion=args.fallback_no_motion,
            skip_metrics=False,
            enable_loop_closure=args.enable_loop_closure,
            enable_confidence_weighting=args.enable_confidence_weighting,
            enable_info_weighting=args.enable_info_weighting,
            info_min_confidence=args.info_min_confidence,
            info_max_confidence=args.info_max_confidence,
            confidence_floor=args.confidence_floor,
            confidence_power=args.confidence_power,
            confidence_min_scale=args.confidence_min_scale,
            confidence_max_scale=args.confidence_max_scale,
            enable_confidence_v2=args.enable_confidence_v2,
            confidence_v2_floor=args.confidence_v2_floor,
            confidence_v2_power=args.confidence_v2_power,
            confidence_v2_odom_min_scale=args.confidence_v2_odom_min_scale,
            confidence_v2_odom_max_scale=args.confidence_v2_odom_max_scale,
            confidence_v2_loop_min_scale=args.confidence_v2_loop_min_scale,
            confidence_v2_loop_max_scale=args.confidence_v2_loop_max_scale,
            confidence_v2_loop_trans_tau_m=args.confidence_v2_loop_trans_tau_m,
            confidence_v2_loop_rot_tau_deg=args.confidence_v2_loop_rot_tau_deg,
            confidence_v2_robust_kernel=args.confidence_v2_robust_kernel,
            confidence_v2_robust_k=args.confidence_v2_robust_k,
            loop_min_separation=args.loop_min_separation,
            loop_search_radius_m=args.loop_search_radius_m,
            loop_max_candidates=args.loop_max_candidates,
            loop_min_inliers=args.loop_min_inliers,
            loop_use_appearance_scan=args.loop_use_appearance_scan,
            loop_appearance_stride=args.loop_appearance_stride,
            loop_appearance_min_matches=args.loop_appearance_min_matches,
            loop_consistency_trans_m=args.loop_consistency_trans_m,
            loop_consistency_rot_deg=args.loop_consistency_rot_deg,
            degeneracy_aware=args.degeneracy_aware,
            degeneracy_cond_ref=args.degeneracy_cond_ref,
            degeneracy_noise_max_mult=args.degeneracy_noise_max_mult,
            degeneracy_fallback_mult=args.degeneracy_fallback_mult,
        )

        elapsed = time.time() - seq_start
        row = extract_summary_row(result)
        row["runtime_sec"] = elapsed
        run_rows.append(row)

        run_details[seq] = {
            "runtime_sec": elapsed,
            "result": result,
        }
        print(f"Sequence {seq} complete in {elapsed:.1f}s")

    summary = {
        "dataset_root": str(dataset_root),
        "sequences": seqs,
        "rows": run_rows,
        "details": run_details,
    }

    summary_json = output_dir / "summary_metrics.json"
    summary_csv = output_dir / "summary_metrics.csv"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(summary_csv, run_rows)

    print(f"\nWrote summary JSON: {summary_json}")
    print(f"Wrote summary CSV:  {summary_csv}")


if __name__ == "__main__":
    main()
