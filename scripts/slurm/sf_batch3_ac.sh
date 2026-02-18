#!/bin/bash
#SBATCH --job-name=sf_b3_ac
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000/sf_batch3_ac_%j.log

module load cuda/11.8.0-fasrc01

export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/11.8.0-fasrc01/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/n/home07/sangwonjung/ProDifEvo-Refinement/reference/ml-simplefold/src/simplefold:$PYTHONPATH

echo "SimpleFold batch3 part ac: $(date)"

/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/simplefold \
    --simplefold_model simplefold_100M \
    --num_steps 500 \
    --tau 0.01 \
    --plddt \
    --fasta_path /n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000/sf_part_ac \
    --output_dir /n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000/simplefold_batch3_ac \
    --backend torch \
    --seed 42

echo "Done part ac: $(date)"
