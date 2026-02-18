#!/bin/bash
#SBATCH --job-name=cf_batch4
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000/colabfold_batch4_%j.log

module load cuda/11.8.0-fasrc01

export PATH=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin:$PATH

PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
COLABFOLD=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/colabfold_batch
OUTPUT_DIR=/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000

echo "=========================================="
echo "ColabFold: batch4 (7000 samples)"
echo "Start: $(date)"
echo "=========================================="

# Create combined fasta for batch4
cat $OUTPUT_DIR/fasta_split/batch4*.fasta > /tmp/batch4_combined_$$.fasta
N_SEQ=$(grep -c "^>" /tmp/batch4_combined_$$.fasta)
echo "Created combined fasta with $N_SEQ sequences"

mkdir -p $OUTPUT_DIR/colabfold_batch4

# Run ColabFold
$COLABFOLD /tmp/batch4_combined_$$.fasta $OUTPUT_DIR/colabfold_batch4 \
    --num-models 1 \
    --num-recycle 3 \
    --model-type alphafold2_ptm

rm /tmp/batch4_combined_$$.fasta

echo "ColabFold inference done: $(date)"

# Extract and merge pLDDT
echo "=========================================="
echo "Extract pLDDT and merge results"
echo "=========================================="

$PYTHON << 'EOF'
import os
import json
import glob

OUTPUT_DIR = '/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000'
COLABFOLD_OUT = f'{OUTPUT_DIR}/colabfold_batch4'

existing_file = f'{OUTPUT_DIR}/colabfold/plddt_results.json'
if os.path.exists(existing_file):
    with open(existing_file) as f:
        results = json.load(f)
else:
    results = {}

score_files = glob.glob(os.path.join(COLABFOLD_OUT, '*_scores_rank_001*.json'))
print(f'Found {len(score_files)} ColabFold result files')

for sf in score_files:
    with open(sf) as f:
        data = json.load(f)

    name = os.path.basename(sf).split('_scores_')[0]
    plddt = data.get('plddt', [])
    mean_plddt = sum(plddt) / len(plddt) if plddt else 0

    results[name] = {
        'mean_plddt': mean_plddt,
        'per_residue_plddt': plddt
    }

with open(existing_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f'Total ColabFold results: {len(results)}')
if results:
    mean_all = sum(r['mean_plddt'] for r in results.values()) / len(results)
    print(f'Mean pLDDT: {mean_all:.2f}')
EOF

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
