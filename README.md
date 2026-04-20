# iSAM2 KITTI Pipeline (Unified Methods)

This repo provides a single flag-based Stereo VO + iSAM2 pipeline for KITTI odometry.

## Methods

Use `run_isam2_kitti.py` (single sequence) or `run_all_gt_sequences.py` (batch) with flags:

- `base`: odometry-only iSAM2 (default)
- `loop-closure`: add geometric loop-closure factors (`--enable-loop-closure`)
- `confidence-v1`: confidence-weighted odometry (`--enable-confidence-weighting`)
- `confidence-v2`: stronger confidence weighting + robust kernels (`--enable-confidence-v2`)
- `info-weighted`: information-weighted odometry using inlier count, match count, and reprojection error (`--enable-info-weighting`)
- `degeneracy-aware+loop`: observability-aware odometry noise scaling with loop closure (`--degeneracy-aware --enable-loop-closure`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## KITTI Dataset Layout

Expected `--dataset-root` structure:

```text
<KITTI_DATASET_ROOT>/
	poses/
		00.txt
		01.txt
		...
	sequences/
		00/
			image_0/
			image_1/
			calib.txt
			times.txt
		01/
			...
```

## Single Sequence Examples

Use `--seq 00` (or any sequence id) and choose one method configuration.

### 1) Base

```bash
python3 run_isam2_kitti.py \
	--dataset-root <KITTI_DATASET_ROOT> \
	--seq 00 \
	--fallback-no-motion \
	--output output/experiments/base/seq00_allframes/poses_est_00.txt \
	--metrics-out output/experiments/base/seq00_allframes/metrics_00.json
```

### 2) Loop Closure

```bash
python3 run_isam2_kitti.py \
	--dataset-root <KITTI_DATASET_ROOT> \
	--seq 00 \
	--fallback-no-motion \
	--enable-loop-closure \
	--output output/experiments/loop_closure/seq00_allframes/poses_est_00.txt \
	--metrics-out output/experiments/loop_closure/seq00_allframes/metrics_00.json
```

### 3) Confidence v1

```bash
python3 run_isam2_kitti.py \
	--dataset-root <KITTI_DATASET_ROOT> \
	--seq 00 \
	--fallback-no-motion \
	--enable-confidence-weighting \
	--confidence-floor 0.08 \
	--confidence-min-scale 0.85 \
	--confidence-max-scale 4.0 \
	--output output/experiments/confidence_v1/seq00_allframes/poses_est_00.txt \
	--metrics-out output/experiments/confidence_v1/seq00_allframes/metrics_00.json
```

### 4) Confidence v2

```bash
python3 run_isam2_kitti.py \
	--dataset-root <KITTI_DATASET_ROOT> \
	--seq 00 \
	--fallback-no-motion \
	--enable-confidence-v2 \
	--enable-loop-closure \
	--confidence-v2-robust-kernel huber \
	--confidence-v2-robust-k 1.5 \
	--output output/experiments/loop_closure_confidence_v2/seq00_allframes/poses_est_00.txt \
	--metrics-out output/experiments/loop_closure_confidence_v2/seq00_allframes/metrics_00.json
```

### 5) Info-Weighted

```bash
python3 run_isam2_kitti.py \
	--dataset-root <KITTI_DATASET_ROOT> \
	--seq 00 \
	--fallback-no-motion \
	--enable-loop-closure \
	--enable-info-weighting \
	--info-min-confidence 0.1 \
	--info-max-confidence 1.0 \
	--output output/experiments/info_weighted_loop/seq00_allframes/poses_est_00.txt \
	--metrics-out output/experiments/info_weighted_loop/seq00_allframes/metrics_00.json
```

### 6) Degeneracy-Aware + Loop

```bash
python3 run_isam2_kitti.py \
	--dataset-root <KITTI_DATASET_ROOT> \
	--seq 00 \
	--fallback-no-motion \
	--enable-loop-closure \
	--degeneracy-aware \
	--degeneracy-cond-ref 800 \
	--degeneracy-noise-max-mult 25 \
	--output output/experiments/degeneracy_loop/seq00_allframes/poses_est_00.txt \
	--metrics-out output/experiments/degeneracy_loop/seq00_allframes/metrics_00.json
```

## Batch Runs (All GT Sequences)

Base:

```bash
python3 run_all_gt_sequences.py \
	--dataset-root <KITTI_DATASET_ROOT> \
	--fallback-no-motion \
	--output-dir output/experiments/base/batch_gt
```

Loop closure:

```bash
python3 run_all_gt_sequences.py \
	--dataset-root <KITTI_DATASET_ROOT> \
	--fallback-no-motion \
	--enable-loop-closure \
	--output-dir output/experiments/loop_closure/batch_gt
```

Confidence v2 + loop:

```bash
python3 run_all_gt_sequences.py \
	--dataset-root <KITTI_DATASET_ROOT> \
	--fallback-no-motion \
	--enable-loop-closure \
	--enable-confidence-v2 \
	--output-dir output/experiments/loop_closure_confidence_v2/batch_gt
```

Info-weighted + loop:

```bash
python3 run_all_gt_sequences.py \
	--dataset-root <KITTI_DATASET_ROOT> \
	--fallback-no-motion \
	--enable-loop-closure \
	--enable-info-weighting \
	--output-dir output/experiments/info_weighted_loop/batch_gt
```

Degeneracy-aware + loop:

```bash
python3 run_all_gt_sequences.py \
	--dataset-root <KITTI_DATASET_ROOT> \
	--fallback-no-motion \
	--enable-loop-closure \
	--degeneracy-aware \
	--output-dir output/experiments/degeneracy_loop/batch_gt
```

## Fallback Branch Runs (If Main Outputs Do Not Reproduce)

If the unified `main` branch does not reproduce expected outputs, run the original method groups on their source branches separately:

- `loop_vs_loopconf`: methods M1-M6
- `degeneracy2.0`: methods M7-M8
- `mohammad/information-based-weighting`: method M9

Suggested workflow:

```bash
git fetch origin

# M1-M6
git checkout loop_vs_loopconf

# M7-M8
git checkout degeneracy2.0

# M9
git checkout mohammad/information-based-weighting

# return to unified implementation
git checkout main
```

Run each branch's method scripts/configs independently so the nine implementations are executed in their original branch context.

## Flag Compatibility Notes

- `--enable-confidence-v2` overrides `--enable-confidence-weighting` behavior.
- `--enable-info-weighting` cannot be combined with confidence-v1 or confidence-v2.
- `--degeneracy-aware` currently:
	- can be used with or without loop closure
	- cannot be combined with confidence-v1, confidence-v2, or info-weighting

## Output Layout

This repo currently has two output organizations:

- Legacy per-method folders (existing historical outputs under `output/`)
- Consolidated folders:
	- `output/experiments/`
	- `output/comparisons/`
	- `output/manifests/`

The consolidation copy manifest is saved at:

- `output/manifests/output_migration_manifest.json`

## Plotting and Evaluation

- Batch metric plots: `plot_batch_metrics.py`
- Estimate-vs-GT comparisons: `plot_estimate_vs_gt.py`
- Base-vs-loop comparison: `plot_base_vs_loop.py`
- Metric utilities: `eval/compute_kitti_metrics.py`, `eval/loop_closure_gain.py`

Use `--help` on each script for full options.
