#!/usr/bin/env python3
"""Run confidence-weighted iSAM2 on KITTI sequence 00 (all frames) and plot results.

Confidence weighting: each odometry and loop-closure factor's noise sigma is scaled
inversely with sqrt(inliers / min_inliers), capped at conf_max_scale.  High-confidence
VO estimates (many inliers) get tighter noise -> stronger influence on the factor graph.

Output: output/conf_wght_seq00_allframes/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = "/gpfs/accounts/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset"
OUTPUT_DIR = REPO_ROOT / "output" / "conf_wght_seq00_allframes"
PLOTS_DIR = OUTPUT_DIR / "plots"
EST_PATH = OUTPUT_DIR / "poses_est_00.txt"
METRICS_PATH = OUTPUT_DIR / "metrics_00.json"
SEQ = "00"

# ---------------------------------------------------------------------------
# Confidence weighting parameters
# ---------------------------------------------------------------------------
CONF_BASE_TRANS_SIGMA = 0.15   # m  — same as base at exactly min_inliers
CONF_BASE_ROT_SIGMA = 0.05     # rad — same as base at exactly min_inliers
CONF_MAX_SCALE = 4.0            # cap: noise floor = base_sigma / 4

# Loop closure parameters (same as loop baseline)
LOOP_MIN_SEPARATION = 120
LOOP_SEARCH_RADIUS_M = 8.0
LOOP_MAX_CANDIDATES = 3
LOOP_MIN_INLIERS = 45
LOOP_USE_APPEARANCE_SCAN = False
LOOP_APPEARANCE_STRIDE = 20
LOOP_APPEARANCE_MIN_MATCHES = 80
LOOP_CONSISTENCY_TRANS_M = 10.0
LOOP_CONSISTENCY_ROT_DEG = 35.0


def _repo_rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _load_eval_utils():
    eval_dir = REPO_ROOT / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from pose_utils import (  # noqa: PLC0415
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


def _align_se3(gt, est, pu):
    gt_p = pu["trajectory_positions"](gt)
    est_p = pu["trajectory_positions"](est)
    t_align, scale = pu["umeyama_alignment"](est_p, gt_p, with_scale=False)
    return pu["apply_sim3_to_poses"](t_align, scale, est)


def _per_frame_errors(gt, est_aligned, pu):
    n = min(gt.shape[0], est_aligned.shape[0])
    gt = gt[:n]
    ea = est_aligned[:n]
    trans_err = np.linalg.norm(gt[:, :3, 3] - ea[:, :3, 3], axis=1)
    rot_deg = np.zeros(n)
    for i in range(n):
        r_diff = gt[i, :3, :3].T @ ea[i, :3, :3]
        rot_deg[i] = np.degrees(pu["angle_from_rotation"](r_diff))
    return trans_err, rot_deg


def _plot_and_save(gt, est, est_aligned, trans_err, rot_deg, pu):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    gt_p = pu["trajectory_positions"](gt)
    est_p = pu["trajectory_positions"](est)
    est_al_p = pu["trajectory_positions"](est_aligned)

    n = min(gt_p.shape[0], est_p.shape[0], est_al_p.shape[0])
    gt_p = gt_p[:n]
    est_p = est_p[:n]
    est_al_p = est_al_p[:n]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Confidence-Weighted iSAM2 — Sequence 00 (all frames)", fontsize=13)

    # Raw trajectory
    axes[0].plot(gt_p[:, 0], gt_p[:, 2], label="GT", linewidth=2.0, color="#264653")
    axes[0].plot(est_p[:, 0], est_p[:, 2], label="Estimate (raw)", linewidth=1.6, color="#e76f51")
    axes[0].set_title("Seq 00 Raw Trajectory (X-Z)")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    # Aligned trajectory
    axes[1].plot(gt_p[:, 0], gt_p[:, 2], label="GT", linewidth=2.0, color="#264653")
    axes[1].plot(est_al_p[:, 0], est_al_p[:, 2], label="Estimate (aligned)", linewidth=1.6, color="#2a9d8f")
    axes[1].set_title("Seq 00 Aligned Trajectory (X-Z)")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")
    axes[1].axis("equal")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    # Per-frame errors
    idx = np.arange(trans_err.shape[0])
    axes[2].plot(idx, trans_err, label="Trans error (m)", color="#457b9d", linewidth=1.8)
    axes[2].plot(idx, rot_deg, label="Rot error (deg)", color="#f4a261", linewidth=1.4)
    axes[2].set_title("Seq 00 Per-frame Errors")
    axes[2].set_xlabel("Frame index")
    axes[2].set_ylabel("Error")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    out_png = PLOTS_DIR / "seq_00_comparison.png"
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"Saved trajectory plot: {out_png}")


def _plot_summary(trans_rmse, rot_rmse):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.array([0])
    w = 0.35
    ax.bar(x - w / 2, [trans_rmse], width=w, label="Trans RMSE (m)", color="#2a9d8f")
    ax.bar(x + w / 2, [rot_rmse], width=w, label="Rot RMSE (deg)", color="#f4a261")
    ax.set_xticks(x)
    ax.set_xticklabels(["00"])
    ax.set_title("Conf-Wght iSAM2: Estimate vs GT Error Summary")
    ax.set_xlabel("Sequence")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_png = PLOTS_DIR / "summary_trans_rot_rmse.png"
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"Saved summary plot: {out_png}")


def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Run the SLAM pipeline
    # -----------------------------------------------------------------------
    from run_isam2_kitti import run_sequence  # noqa: PLC0415

    print("=" * 60)
    print("Confidence-Weighted iSAM2 — Sequence 00 (all frames)")
    print(f"  conf_base_trans_sigma = {CONF_BASE_TRANS_SIGMA} m")
    print(f"  conf_base_rot_sigma   = {CONF_BASE_ROT_SIGMA} rad")
    print(f"  conf_max_scale        = {CONF_MAX_SCALE}")
    print("  loop closure          = disabled")
    print("=" * 60)

    result = run_sequence(
        dataset_root=DATASET_ROOT,
        seq=SEQ,
        max_frames=0,           # all frames
        output_path=EST_PATH,
        metrics_path=METRICS_PATH,
        min_inliers=25,
        fallback_no_motion=True,
        skip_metrics=False,
        enable_loop_closure=False,
        loop_min_separation=LOOP_MIN_SEPARATION,
        loop_search_radius_m=LOOP_SEARCH_RADIUS_M,
        loop_max_candidates=LOOP_MAX_CANDIDATES,
        loop_min_inliers=LOOP_MIN_INLIERS,
        loop_use_appearance_scan=LOOP_USE_APPEARANCE_SCAN,
        loop_appearance_stride=LOOP_APPEARANCE_STRIDE,
        loop_appearance_min_matches=LOOP_APPEARANCE_MIN_MATCHES,
        loop_consistency_trans_m=LOOP_CONSISTENCY_TRANS_M,
        loop_consistency_rot_deg=LOOP_CONSISTENCY_ROT_DEG,
        enable_conf_weighting=True,
        conf_base_trans_sigma=CONF_BASE_TRANS_SIGMA,
        conf_base_rot_sigma=CONF_BASE_ROT_SIGMA,
        conf_max_scale=CONF_MAX_SCALE,
    )

    print(f"\nMode: {result['mode']}")
    print(f"Poses: {result['num_poses']}")
    print(f"Accepted odometry: {result['accepted_transitions']}")
    print(f"Fallback used: {result['fallback_transitions']}")
    print(f"Loop closures: {result['loop_closures_added']}")

    # -----------------------------------------------------------------------
    # 2. Load GT and estimate, compute errors, plot
    # -----------------------------------------------------------------------
    pu = _load_eval_utils()

    gt_path = Path(DATASET_ROOT) / "poses" / f"{SEQ}.txt"
    gt = pu["read_kitti_poses"](str(gt_path))
    est = pu["read_kitti_poses"](str(EST_PATH))

    n = min(gt.shape[0], est.shape[0])
    gt = gt[:n]
    est = est[:n]

    est_aligned = _align_se3(gt, est, pu)
    trans_err, rot_deg = _per_frame_errors(gt, est_aligned, pu)

    trans_stats = pu["summarize_errors"](trans_err)
    rot_stats = pu["summarize_errors"](rot_deg)

    print(f"\nTrans RMSE: {trans_stats.rmse:.4f} m  (mean: {trans_stats.mean:.4f} m)")
    print(f"Rot  RMSE:  {rot_stats.rmse:.4f} deg (mean: {rot_stats.mean:.4f} deg)")

    _plot_and_save(gt, est, est_aligned, trans_err, rot_deg, pu)
    _plot_summary(trans_stats.rmse, rot_stats.rmse)

    # -----------------------------------------------------------------------
    # 3. Write summary JSON (same format as other algorithms)
    # -----------------------------------------------------------------------
    summary = {
        "dataset_root": DATASET_ROOT,
        "estimates_dir": _repo_rel(OUTPUT_DIR),
        "align": "se3",
        "rows": [
            {
                "seq": SEQ,
                "mode": result["mode"],
                "num_poses": int(n),
                "align": "se3",
                "scale": 1.0,
                "trans_rmse_m": trans_stats.rmse,
                "trans_mean_m": trans_stats.mean,
                "rot_rmse_deg": rot_stats.rmse,
                "rot_mean_deg": rot_stats.mean,
                "loop_closures_added": result["loop_closures_added"],
                "conf_base_trans_sigma": CONF_BASE_TRANS_SIGMA,
                "conf_base_rot_sigma": CONF_BASE_ROT_SIGMA,
                "conf_max_scale": CONF_MAX_SCALE,
                "plot_path": _repo_rel(PLOTS_DIR / "seq_00_comparison.png"),
            }
        ],
        "summary_plot": _repo_rel(PLOTS_DIR / "summary_trans_rot_rmse.png"),
    }
    summary_json = PLOTS_DIR / "difference_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary JSON: {summary_json}")

    if result.get("metrics"):
        m = result["metrics"]
        print(f"\nATE trans RMSE: {m['ATE']['ATE_trans_m']['rmse']:.4f} m")
        print(f"ATE rot  RMSE:  {m['ATE']['ATE_rot_deg']['rmse']:.4f} deg")


if __name__ == "__main__":
    main()
