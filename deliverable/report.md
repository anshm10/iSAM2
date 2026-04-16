# Information-Based Weighting for iSAM2 Pose Graph Optimization

## 1. Introduction

### 1.1 Problem Statement

In visual SLAM, a stereo visual odometry (VO) frontend produces frame-to-frame relative pose estimates that are consumed by a pose-graph backend optimizer. The standard approach assigns a fixed noise covariance to every odometry factor, treating all frames as equally reliable. In practice, VO quality varies significantly across frames due to changes in scene texture, lighting, vehicle speed, and geometric configuration. Frames with few inliers, low inlier ratios, or high reprojection error produce less reliable motion estimates, yet the optimizer weights them identically to high-quality frames.

Information-based weighting addresses this by dynamically scaling each odometry factor's covariance according to a per-frame confidence score derived from VO quality signals. The goal is to down-weight unreliable frames so that, when competing constraints exist (e.g., loop closures), the optimizer preferentially trusts high-quality measurements.

### 1.2 Context

This work is part of a stereo VO + iSAM2 pipeline evaluated on the KITTI Odometry benchmark. The pipeline uses ORB feature detection, stereo triangulation, and PnP RANSAC for frame-to-frame motion estimation, with iSAM2 as the incremental backend optimizer. Loop closure detection (spatial candidate search with geometric verification) is implemented separately and can be combined with information-based weighting.

### 1.3 Distinction: Static vs Dynamic Weighting

- **Static (confidence) weighting**: All factors of a given type share the same fixed covariance. This is the baseline approach.
- **Dynamic (information-based) weighting**: Each factor receives a covariance scaled by a per-frame confidence score computed from the actual measurement quality at that timestep. This is what we implement.

## 2. Mathematical Formulation

### 2.1 Pose-Graph Objective

The backend solves a nonlinear least-squares problem over SE(3) poses:

$$\hat{X} = \arg\min_X \sum_{(i,j) \in \mathcal{E}_o} \|r_{ij}^o(X)\|^2_{\Sigma_{ij}^{-1}} + \sum_{(p,q) \in \mathcal{E}_\ell} \|r_{pq}^\ell(X)\|^2_{\Lambda_{pq}^{-1}} + \|r_0(X)\|^2_{\Gamma^{-1}}$$

where $\mathcal{E}_o$ are odometry edges, $\mathcal{E}_\ell$ are loop closure edges, and $\Gamma$ is the prior on the first pose.

### 2.2 Information-Based Covariance Scaling

For each odometry factor at frame $k$, we compute a confidence score $c_k \in [0.1, 1.0]$ and scale the base covariance:

$$\Sigma_k' = \frac{1}{c_k} \cdot \Sigma_{\text{base}}$$

Since GTSAM parameterizes noise models by standard deviation (sigma), and $\Sigma = \text{diag}(\sigma^2)$, this translates to:

$$\sigma_k' = \frac{\sigma_{\text{base}}}{\sqrt{c_k}}$$

Low confidence ($c_k \to 0.1$) inflates the covariance by $10\times$, reducing that factor's influence. High confidence ($c_k \to 1.0$) leaves the covariance unchanged.

### 2.3 Confidence Score Computation

The confidence score aggregates three VO quality signals via geometric mean:

$$c_k = c_{\min} + (c_{\max} - c_{\min}) \cdot (s_{\text{ratio}} \cdot s_{\text{count}} \cdot s_{\text{reproj}})^{1/3}$$

Each signal is normalized to $[0, 1]$ using KITTI-calibrated ranges:

**Inlier ratio** (typical range 0.5--1.0):
$$s_{\text{ratio}} = \text{clip}\left(\frac{n_{\text{inlier}} / n_{\text{match}} - 0.5}{0.5},\ 0,\ 1\right)$$

**Absolute inlier count** (typical range 50--500):
$$s_{\text{count}} = \text{clip}\left(\frac{n_{\text{inlier}} - 50}{450},\ 0,\ 1\right)$$

**Mean reprojection error** (typical range 0.7--2.0 px):
$$s_{\text{reproj}} = \text{clip}\left(\frac{2.0 - \bar{e}_{\text{reproj}}}{1.3},\ 0,\ 1\right)$$

The geometric mean ensures balanced contribution: a single poor signal cannot be masked by two strong ones.

### 2.4 Why Geometric Mean

Arithmetic mean allows a very high inlier ratio to compensate for terrible reprojection error. Geometric mean requires all three signals to be reasonable for a high overall score. For example:

| Inlier Ratio | Inlier Count | Reproj Error | Arithmetic | Geometric |
|:---:|:---:|:---:|:---:|:---:|
| 1.0 | 1.0 | 0.0 | 0.67 | 0.0 |
| 0.8 | 0.8 | 0.8 | 0.80 | 0.80 |
| 0.5 | 0.5 | 0.5 | 0.50 | 0.50 |

## 3. Implementation

### 3.1 Modified Files

**`slam/stereo_vo.py`**: Added `mean_reproj_error` field to `MotionEstimate` dataclass. After PnP RANSAC, the mean reprojection error over inlier points is computed using `cv2.projectPoints()`.

**`slam/isam2_backend.py`**: Added two components:
- `compute_confidence(inliers, total_matches, mean_reproj_error)`: Maps VO signals to confidence score $[0.1, 1.0]$.
- `Isam2InfoWeighted(Isam2PoseGraph)`: Subclass that overrides odometry factor insertion with `add_odometry_weighted()`, which scales the noise model by $1/\sqrt{c_k}$.

**`run_isam2_kitti.py`**: Added `--info-weighted` flag. When enabled, instantiates `Isam2InfoWeighted` backend and calls `add_odometry_weighted()` with per-frame confidence.

**`run_all_gt_sequences.py`**: Passes `--info-weighted` flag through to per-sequence runs.

**`run_single_seq.sh`**: SLURM batch script for per-sequence execution with array jobs.

### 3.2 Pipeline Flow

```
For each frame k:
  1. VO frontend: extract ORB features, stereo triangulate, PnP RANSAC
  2. Compute mean reprojection error over inliers
  3. Compute confidence: c_k = f(inlier_ratio, inlier_count, reproj_error)
  4. Scale noise: sigma_k = sigma_base / sqrt(c_k)
  5. Add BetweenFactorPose3 with scaled noise to iSAM2
  6. (If loop closure enabled) Search for and add loop factors
  7. iSAM2 incremental update
```

## 4. Evaluation

### 4.1 Metrics

- **ATE (Absolute Trajectory Error)**: After SE(3) alignment of the full estimated trajectory to ground truth, compute the RMSE of per-pose position (translation) and orientation (rotation) errors. Measures global consistency.
- **RPE (Relative Pose Error)**: Error in relative pose between pairs of frames separated by $\delta$ frames. Measures local accuracy at different scales ($\delta = 1, 10, 100$).

### 4.2 Experimental Conditions

All runs use KITTI Odometry sequence 00 (4541 poses), identical VO frontend (ORB + PnP RANSAC), and SE(3)-aligned evaluation.

Three configurations tested:
1. **Base (no loops)**: Fixed odometry covariance, no loop closure.
2. **Base + Loop Closure**: Fixed odometry covariance, spatial loop closure enabled (246 loops detected).
3. **Info-Weighted + Loop Closure**: Per-frame confidence-scaled odometry covariance, spatial loop closure enabled (246 loops detected).

## 5. Results

### 5.1 ATE Summary (Seq 00)

| Method | Trans RMSE (m) | Rot RMSE (deg) | vs Base (no loops) |
|:---|---:|---:|---:|
| Base (no loops) | 32.75 | 10.49 | -- |
| Base + Loop Closure | 29.92 | 12.87 | -8.6% trans, +22.6% rot |
| Info-Weighted + Loop Closure | 30.16 | 12.96 | -7.9% trans, +23.5% rot |

### 5.2 RPE Summary (Seq 00)

| Method | RPE-1 Trans (m) | RPE-1 Rot (deg) | RPE-10 Trans (m) | RPE-10 Rot (deg) |
|:---|---:|---:|---:|---:|
| Base (no loops) | 0.0332 | 0.119 | 0.211 | 0.673 |
| Base + Loop Closure | 0.0354 | 0.201 | 0.259 | 1.540 |
| Info-Weighted + Loop Closure | 0.0356 | 0.202 | 0.262 | 1.549 |

### 5.3 Key Observations

1. **Without loop closure, information-based weighting produces identical results to baseline.** This is mathematically expected: in a pure odometry chain, the MAP estimate is the composition of measurements regardless of noise scaling. Covariance only affects the solution when competing constraints exist.

2. **With loop closure, information-based weighting (v1) produces negligible difference** (+0.8% translation, +0.7% rotation vs baseline+loops). The confidence scores on KITTI seq 00 cluster tightly (range ~0.67--0.94 with std ~0.04 on easy frames), so the covariance scaling is nearly uniform.

3. **Loop closure improves global translation but worsens rotation.** The 246 detected loop closures reduce translational ATE by 8.6% but increase rotational ATE by 22.6%. This occurs because loop closure factors with fixed noise can inject rotational distortion when the loop measurement is imperfect.

### 5.4 Comparison with Confidence V2

A separate confidence-v2 implementation on the same sequence achieves substantially better results (22.37m trans, 9.16 deg rot) by:
1. Using aggressive sigma scaling ($1/c$ instead of $1/\sqrt{c}$)
2. Applying confidence weighting to loop closure factors (not just odometry)
3. Adding Huber robust kernels to suppress outlier loop constraints

This confirms that the benefit of confidence modeling comes primarily from weighting the globally corrective loop factors, not the local odometry factors.

## 6. Plots

- `plots/seq00_trajectory_comparison.png` — XZ trajectory: ground truth vs three methods
- `plots/seq00_translation_error.png` — Per-frame translation error over time
- `plots/seq00_rotation_error.png` — Per-frame rotation error over time
- `plots/seq00_ate_summary_bars.png` — Bar chart comparing ATE across methods

## 7. File Listing

### Output
- `output/baseline_no_loops/` — Poses and metrics for base iSAM2 (no loop closure)
- `output/baseline_with_loops/` — Poses and metrics for base iSAM2 + loop closure
- `output/infoweighted_with_loops/` — Poses and metrics for info-weighted iSAM2 + loop closure
- `output/difference_summary.json` — All metrics in one JSON

### Source
- `src/isam2_backend.py` — `compute_confidence()`, `Isam2InfoWeighted` class
- `src/stereo_vo.py` — `mean_reproj_error` computation
- `src/run_isam2_kitti.py` — Main runner with `--info-weighted` flag
- `src/run_all_gt_sequences.py` — Batch runner
- `src/run_single_seq.sh` — SLURM batch script

## 8. Conclusion

Information-based weighting (v1) applied to odometry factors alone does not improve trajectory accuracy on KITTI. The VO quality on this dataset is too uniform for per-frame covariance scaling to produce meaningful differentiation. The approach is theoretically sound but requires either (a) a dataset with greater VO quality variation, or (b) extension to also weight loop closure factors with robust kernels (as demonstrated by the v2 approach).
