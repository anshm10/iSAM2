# iSAM2 + KITTI evaluation

This folder contains small Python scripts to evaluate an estimated KITTI odometry trajectory against ground truth.

## Inputs

- **Ground truth**: KITTI odometry poses file (each line is 12 floats encoding a 3x4 matrix `[R|t]`).
- **Estimate**: your iSAM2 output in the same KITTI poses format.

If your estimator outputs `x y z qx qy qz qw` (or similar), you’ll need a small exporter to KITTI pose format.

## Install

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## ATE + RPE

```bash
python3 eval/compute_kitti_metrics.py --kitti-poses-dir <KITTI_ROOT>/data_odometry_poses/dataset/poses --seq 00 --est path/to/est_00.txt --align se3 --rpe-deltas 1,10,100
```

- `--align se3` does rigid alignment of estimated positions to GT (no scale).
- `--align sim3` also estimates a global scale (sometimes used for monocular VO).

Output is JSON printed to stdout (use `--out metrics.json` to save).

## Loop-closure correction gain (optional)

Loop-closure metrics require **intermediate** trajectories, not just the final one.

Workflow:

1. During your incremental iSAM2 run, dump the current best trajectory after each update (or at least right before/after adding a loop-closure factor).
2. Put those snapshots into a directory (each snapshot is a KITTI poses file).
3. Optionally write an `events.txt` file containing the indices (0-based) of snapshots that are **after** loop closures.

Example:

```bash
python3 eval/loop_closure_gain.py --kitti-poses-dir <KITTI_ROOT>/data_odometry_poses/dataset/poses --seq 00 --snapshots-dir snapshots/ --pattern '.*\\.txt$' --events events.txt --align se3
```

Reported fields:

- `ATE_rmse_trans_m`: ATE RMSE per snapshot.
- `global_gain_m`: `ATE(first) - ATE(last)`.
- `events[*].gain_m`: per-loop closure improvement in ATE (`before - after`).
- `corrections[*].rms_translation_correction_m`: RMS magnitude of pose changes introduced by that update.

## Other useful iSAM2-relevant metrics

Consider also tracking these during runs:

- **Per-step update time** and **total runtime**.
- **Number of relinearized variables** per update and factor graph size growth.
- **Incremental consistency**: NIS/chi-squared residual statistics if you have measurement covariances.
- **Loop closure precision/recall** if you have a loop detector.
