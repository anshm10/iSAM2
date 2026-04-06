# iSAM2

Stereo VO + iSAM2 pipeline for KITTI odometry with three supported modes:

- `base`: odometry-only iSAM2
- `loop`: iSAM2 + loop closure factors
- `confidence`: iSAM2 with confidence-weighted odometry factors

## Single Sequence

```bash
python3 run_isam2_kitti.py \
	--dataset-root ../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset \
	--seq 00 \
	--fallback-no-motion \
	--mode base \
	--output output/poses_est_00_base.txt \
	--metrics-out output/metrics_00_base.json
```

```bash
python3 run_isam2_kitti.py \
	--dataset-root ../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset \
	--seq 00 \
	--fallback-no-motion \
	--mode loop \
	--output output/poses_est_00_loop.txt \
	--metrics-out output/metrics_00_loop.json
```

```bash
python3 run_isam2_kitti.py \
	--dataset-root ../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset \
	--seq 00 \
	--fallback-no-motion \
	--mode confidence \
	--conf-min-noise-scale 0.8 \
	--conf-max-noise-scale 2.5 \
	--output output/poses_est_00_conf.txt \
	--metrics-out output/metrics_00_conf.json
```

## Batch Run (GT Sequences)

```bash
python3 run_all_gt_sequences.py \
	--dataset-root ../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset \
	--fallback-no-motion \
	--mode base \
	--output-dir output/batch_gt_base
```

```bash
python3 run_all_gt_sequences.py \
	--dataset-root ../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset \
	--fallback-no-motion \
	--mode loop \
	--output-dir output/batch_gt_loop
```

```bash
python3 run_all_gt_sequences.py \
	--dataset-root ../../../scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset \
	--fallback-no-motion \
	--mode confidence \
	--conf-min-noise-scale 0.8 \
	--conf-max-noise-scale 2.5 \
	--output-dir output/batch_gt_conf
```
