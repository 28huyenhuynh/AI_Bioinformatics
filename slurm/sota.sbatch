#!/bin/bash
# Reproduction runs (run_sota.py) on the FUV AI Slurm cluster, one array task per subject.
# Submit from the project folder on the login node:
#
#   mkdir -p logs
#   sbatch --array=1-27%1 slurm/sota.sbatch hartwell        # TtS + Baseline CNN, 10 splits each
#   sbatch --array=1-27%1 slurm/sota.sbatch hu              # attention CNN-RNN
#   sbatch --array=1 --time=01:00:00 slurm/sota.sbatch hartwell --models TtS --splits 1   # test
#
# Anything after the paper name is passed to run_sota.py. "%1" runs one subject at a
# time: the student QOS allows one 48 GB MIG per user. Finished folds are skipped, so
# resubmitting after a failure or timeout continues where it stopped.
#SBATCH --job-name=sota
#SBATCH --partition=student4848
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition_2g.48gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
PAPER=${1:?usage: sbatch --array=1-27%1 slurm/sota.sbatch hartwell|hu [run_sota.py options]}
shift

module load miniconda3
source activate semg
cd "$SLURM_SUBMIT_DIR"

echo "job $SLURM_JOB_ID task ${SLURM_ARRAY_TASK_ID:-} on $(hostname), paper $PAPER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -W ignore run_sota.py --paper "$PAPER" --subjects "${SLURM_ARRAY_TASK_ID:-1}" --quiet "$@"
