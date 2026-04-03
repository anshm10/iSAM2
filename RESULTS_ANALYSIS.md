# Stereo VO + iSAM2 on KITTI Odometry: Ground Truth vs Estimate Analysis

## 1. Introduction
This report analyzes how the current stereo visual odometry (VO) + iSAM2 baseline performs against KITTI odometry ground truth.

The goals are:
- quantify trajectory error against ground truth,
- visualize where trajectories deviate,
- identify systematic failure modes,
- document the pipeline and how to reproduce the analysis.

This analysis uses the generated batch results currently in:
- `output/batch_gt_smoke/`

Important scope note:
- The current report is based on a **smoke run** with `max-frames=25` per sequence for sequences `00` to `10`.
- These results are useful for debugging trends and early validation, but they are not final benchmark numbers for full-sequence KITTI evaluation.

---

## 2. Dataset
### 2.1 Source
KITTI odometry dataset at:
- `/gpfs/accounts/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset`

### 2.2 Structure used by the code
- `sequences/<seq>/image_0/*.png`: left grayscale images
- `sequences/<seq>/image_1/*.png`: right grayscale images
- `sequences/<seq>/calib.txt`: stereo projection matrices (`P0`, `P1`, ...)
- `sequences/<seq>/times.txt`: frame timestamps
- `poses/<seq>.txt`: ground-truth trajectory (available for `00` through `10`)

### 2.3 Sequences analyzed
Ground truth sequences:
- `00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10`

---

## 3. Methodology
### 3.1 Estimation pipeline
For each sequence:
1. Load stereo pair and calibration.
2. Detect ORB features.
3. Match left-right features in previous frame to triangulate 3D points.
4. Match previous-left descriptors to current-left descriptors.
5. Estimate relative motion using `solvePnPRansac`.
6. Add odometry factor to iSAM2 incrementally.
7. Export optimized pose trajectory in KITTI pose format.

### 3.2 Metrics and alignment
Estimated trajectories are compared to ground truth with:
- `ATE` (Absolute Trajectory Error),
- `RPE` (Relative Pose Error for deltas 1 and 10 in this smoke run).

Alignment mode used in this report:
- `SE(3)` (rigid alignment, no scale correction).

### 3.3 Visualization methodology
For each sequence, the script generates:
- raw trajectory overlay (GT vs estimate, X-Z plane),
- aligned trajectory overlay (GT vs estimate, X-Z plane),
- per-frame translation and rotation error curves.

A summary chart is also generated across all sequences.

---

## 4. Code Explanation by File
### 4.1 Core estimation files
1. `slam/kitti_loader.py`
- Parses KITTI sequence folders.
- Loads stereo images, timestamps, calibration, and optional GT poses.
- Extracts intrinsic matrix and stereo baseline from `P0`/`P1`.

2. `slam/stereo_vo.py`
- Implements stereo VO front-end.
- ORB feature extraction + descriptor matching.
- Left-right disparity to 3D reconstruction.
- Temporal matching and `solvePnPRansac` for relative pose.

3. `slam/isam2_backend.py`
- Creates and updates iSAM2 factor graph.
- Adds prior on first pose.
- Adds between-factors for frame-to-frame odometry.
- Returns optimized trajectory matrices.

4. `run_isam2_kitti.py`
- Single-sequence entrypoint.
- Runs full pipeline and writes estimate.
- Computes and writes metrics against GT.
- Exposes reusable `run_sequence(...)` for batch execution.

### 4.2 Batch and plotting files
5. `run_all_gt_sequences.py`
- Auto-discovers GT sequences from `poses/*.txt`.
- Runs all GT sequences.
- Stores per-sequence estimates and per-sequence metrics.
- Writes summary JSON and CSV.

6. `plot_batch_metrics.py`
- Plots aggregate ATE/RPE/runtime/fallback charts from batch summary.

7. `plot_estimate_vs_gt.py`
- Plots detailed GT-vs-estimate overlays and per-frame error curves per sequence.
- Writes sequence-wise comparison figures and difference summary JSON.

### 4.3 Evaluation utility files
8. `eval/pose_utils.py`
- KITTI pose I/O, trajectory alignment helpers, relative transform utilities.

9. `eval/compute_kitti_metrics.py`
- Computes ATE and RPE metrics from GT and estimated trajectories.

10. `eval/loop_closure_gain.py`
- Optional loop-closure gain analysis (not used in this odometry-only baseline).

---

## 5. Detailed Analysis: Results and Graphs for All Sequences
Data source for this section:
- `output/batch_gt_smoke/summary_metrics.json`
- `output/batch_gt_smoke/diff_plots/difference_summary.json`

### 5.1 Per-sequence quantitative results (smoke run, 25 poses/sequence)

| Seq | ATE RMSE (m) | ATE Rot RMSE (deg) | RPE d1 Trans RMSE (m) | RPE d1 Rot RMSE (deg) | RPE d10 Trans RMSE (m) | RPE d10 Rot RMSE (deg) |
|---|---:|---:|---:|---:|---:|---:|
| 00 | 0.3459 | 12.9083 | 0.0811 | 0.1398 | 0.5471 | 0.9758 |
| 01 | 0.5271 | 1.3074 | 0.1010 | 0.0635 | 0.7442 | 0.2791 |
| 02 | 0.0844 | 156.8134 | 0.0295 | 0.0361 | 0.1400 | 0.1217 |
| 03 | 0.0270 | 62.6150 | 0.0261 | 0.0336 | 0.1718 | 0.1076 |
| 04 | 0.1493 | 76.2854 | 0.0413 | 0.0712 | 0.2620 | 0.3761 |
| 05 | 0.0875 | 37.3959 | 0.0205 | 0.0393 | 0.1377 | 0.1048 |
| 06 | 0.1247 | 19.6880 | 0.0359 | 0.0864 | 0.1915 | 0.4579 |
| 07 | 0.0303 | 0.6552 | 0.0085 | 0.0361 | 0.0557 | 0.1315 |
| 08 | 0.2459 | 90.7198 | 0.1897 | 0.0396 | 1.8437 | 0.2011 |
| 09 | 0.0341 | 2.9847 | 0.0149 | 0.0590 | 0.0561 | 0.2921 |
| 10 | 0.0581 | 3.8469 | 0.0261 | 0.0634 | 0.1563 | 0.5089 |

### 5.2 Aggregate summary
- Mean translational RMSE across sequences: **0.1558 m**
- Median translational RMSE across sequences: **0.0875 m**
- Best translational RMSE: **Seq 03 (0.0270 m)**
- Worst translational RMSE: **Seq 01 (0.5271 m)**

- Mean rotational RMSE across sequences: **42.2927 deg**
- Median rotational RMSE across sequences: **19.6880 deg**
- Best rotational RMSE: **Seq 07 (0.6552 deg)**
- Worst rotational RMSE: **Seq 02 (156.8134 deg)**

### 5.3 Interpretation and key observations
1. **Translation is often reasonable at short horizon**, while rotation can be unstable.
- Several sequences have low translational ATE (< 0.1 m) for first 25 frames.
- Rotational ATE is highly variable and very high on some sequences.

2. **High global rotation error with low local RPE suggests orientation drift accumulation.**
- Example: Seq 02 has low RPE d1 rotation but very high ATE rotation.
- This pattern indicates local frame-to-frame steps look plausible, but global orientation reference diverges.

3. **Sequence 08 exhibits notable translational degradation at 10-frame horizon.**
- RPE d10 translation is 1.8437 m, much higher than other sequences.
- Likely tied to weak geometry/parallax or unstable temporal matching in that segment.

4. **No fallback transitions in smoke run.**
- All transitions were accepted (24/24), so high errors are not due to identity fallback insertion.

5. **Because this is only 25 frames/sequence, conclusions are directional, not final.**
- Longer runs will better reveal drift behavior and robust sequence-level ranking.

### 5.4 Graphs
#### 5.4.1 Cross-sequence summary
- Summary Trans/Rot RMSE chart:
  ![Summary RMSE](output/batch_gt_smoke/diff_plots/summary_trans_rot_rmse.png)

- Additional aggregate charts from batch metrics:
  - ![ATE Translational RMSE](output/batch_gt_smoke/plots/ate_trans_rmse.png)
  - ![ATE Rotational RMSE](output/batch_gt_smoke/plots/ate_rot_rmse.png)
  - ![RPE Translational RMSE](output/batch_gt_smoke/plots/rpe_trans_rmse_lines.png)
  - ![Runtime](output/batch_gt_smoke/plots/runtime_sec.png)
  - ![Fallback Count](output/batch_gt_smoke/plots/fallback_count.png)

#### 5.4.2 Per-sequence comparison figures
Each figure contains raw trajectory overlay, aligned trajectory overlay, and per-frame error curves.

- Seq 00: ![Seq 00](output/batch_gt_smoke/diff_plots/seq_00_comparison.png)
- Seq 01: ![Seq 01](output/batch_gt_smoke/diff_plots/seq_01_comparison.png)
- Seq 02: ![Seq 02](output/batch_gt_smoke/diff_plots/seq_02_comparison.png)
- Seq 03: ![Seq 03](output/batch_gt_smoke/diff_plots/seq_03_comparison.png)
- Seq 04: ![Seq 04](output/batch_gt_smoke/diff_plots/seq_04_comparison.png)
- Seq 05: ![Seq 05](output/batch_gt_smoke/diff_plots/seq_05_comparison.png)
- Seq 06: ![Seq 06](output/batch_gt_smoke/diff_plots/seq_06_comparison.png)
- Seq 07: ![Seq 07](output/batch_gt_smoke/diff_plots/seq_07_comparison.png)
- Seq 08: ![Seq 08](output/batch_gt_smoke/diff_plots/seq_08_comparison.png)
- Seq 09: ![Seq 09](output/batch_gt_smoke/diff_plots/seq_09_comparison.png)
- Seq 10: ![Seq 10](output/batch_gt_smoke/diff_plots/seq_10_comparison.png)

---

## 6. How to Run
### 6.1 Run one sequence
```bash
python3 run_isam2_kitti.py \
  --dataset-root ../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset \
  --seq 00 \
  --fallback-no-motion \
  --output output/poses_est_00.txt \
  --metrics-out output/metrics_00.json
```

### 6.2 Run all GT sequences (00-10)
```bash
python3 run_all_gt_sequences.py \
  --dataset-root ../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset \
  --fallback-no-motion \
  --output-dir output/batch_gt_full
```

### 6.3 Plot aggregate batch metrics
```bash
python3 plot_batch_metrics.py \
  --summary-json output/batch_gt_full/summary_metrics.json \
  --output-dir output/batch_gt_full/plots
```

### 6.4 Plot estimate-vs-ground-truth differences
```bash
python3 plot_estimate_vs_gt.py \
  --dataset-root ../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset \
  --estimates-dir output/batch_gt_full/estimates \
  --output-dir output/batch_gt_full/diff_plots \
  --align se3
```

---

## 7. Conclusion
The current stereo VO + iSAM2 baseline is operational and produces coherent trajectory estimates across all GT sequences. On short-horizon smoke runs, translational errors are often acceptable, but rotational consistency is uneven and sequence-dependent.

Main takeaways:
1. The implementation is strong enough for end-to-end batch evaluation and visual diagnostics.
2. Rotation handling is the dominant weakness and should be the next optimization target.
3. Full-length sequence runs are required before claiming performance quality.

Recommended next steps:
1. Run full sequences (`max-frames=0`) and regenerate all plots.
2. Add orientation-focused diagnostics (yaw drift plots) and outlier statistics.
3. Tune VO thresholds and iSAM2 noise models per sequence characteristics.
4. Add loop-closure constraints to reduce accumulated orientation drift.
