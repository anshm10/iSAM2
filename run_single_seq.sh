#!/bin/bash
#SBATCH --job-name=isam2_%a
#SBATCH --account=rob530w26s001_class
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm_logs/seq_%a_%j.out
#SBATCH --error=slurm_logs/seq_%a_%j.err

# Usage:
#   sbatch --array=0 run_single_seq.sh baseline
#   sbatch --array=0 run_single_seq.sh infoweighted
#   sbatch --array=0-10 run_single_seq.sh baseline   (all sequences)

set -euo pipefail

SEQ=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
MODE="${1:-baseline}"

DATASET_ROOT="/scratch/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset"

if [ "$MODE" = "infoweighted" ]; then
    OUTPUT_DIR="slurm_infoweighted"
    EXTRA_FLAGS="--info-weighted"
else
    OUTPUT_DIR="slurm_baseline"
    EXTRA_FLAGS=""
fi

echo "=== Sequence ${SEQ} | Mode: ${MODE} | Job: ${SLURM_JOB_ID} ==="
echo "Start: $(date)"

module load python3.11-anaconda/2024.02
source activate slam

cd /home/mohnaqvi/568/iSAM2

python3 run_isam2_kitti.py \
    --dataset-root "${DATASET_ROOT}" \
    --seq "${SEQ}" \
    --output "output/${OUTPUT_DIR}/estimates/poses_est_${SEQ}.txt" \
    --metrics-out "output/${OUTPUT_DIR}/metrics/metrics_${SEQ}.json" \
    --min-inliers 25 \
    --fallback-no-motion \
    ${EXTRA_FLAGS}

echo "Done: $(date)"
