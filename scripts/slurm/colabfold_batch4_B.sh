#!/bin/bash
#SBATCH --job-name=cf_b4_B
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000/colabfold_batch4_B_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sangwon.jung@trillionlabs.co

module load cuda/11.8.0-fasrc01

export PATH=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin:$PATH

COLABFOLD=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/colabfold_batch
PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
OUTPUT_DIR=/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000

echo "=========================================="
echo "ColabFold: batch4 Part B (3501 samples)"
echo "Start: $(date)"
echo "=========================================="

mkdir -p $OUTPUT_DIR/colabfold_batch4

$COLABFOLD $OUTPUT_DIR/batch4_partB.fasta $OUTPUT_DIR/colabfold_batch4 \
    --num-models 1 \
    --num-recycle 3 \
    --model-type alphafold2_ptm

echo "ColabFold Part B done: $(date)"
echo "=========================================="
