#!/bin/bash
#SBATCH --job-name=colabfold_2000
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000/colabfold_2000_%j.log

module load cuda/11.8.0-fasrc01

PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
OUTPUT_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000
FASTA_SPLIT=$OUTPUT_DIR/fasta_split
COLABFOLD_OUT=$OUTPUT_DIR/colabfold_new

mkdir -p $COLABFOLD_OUT

echo "=========================================="
echo "ColabFold inference on new 2000 samples (batch3)"
echo "Start time: $(date)"
echo "=========================================="

# Count batch3 files
N_BATCH3=$(ls $FASTA_SPLIT/batch3*.fasta 2>/dev/null | wc -l)
echo "Found $N_BATCH3 batch3 fasta files"

if [ "$N_BATCH3" -eq 0 ]; then
    echo "No batch3 files found. Generation may not have completed."
    exit 1
fi

# Create combined fasta for batch3
cat $FASTA_SPLIT/batch3*.fasta > /tmp/batch3_combined_$$.fasta
echo "Created combined fasta with $N_BATCH3 sequences"

# Run ColabFold
colabfold_batch /tmp/batch3_combined_$$.fasta $COLABFOLD_OUT \
    --num-models 1 \
    --num-recycle 3 \
    --model-type alphafold2_ptm

rm /tmp/batch3_combined_$$.fasta

echo "=========================================="
echo "Extracting pLDDT scores..."
echo "=========================================="

$PYTHON << 'EOF'
import os
import json
import glob

COLABFOLD_OUT = '/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000/colabfold_new'
OUTPUT_DIR = '/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000/colabfold'

# Load existing results
existing_results_file = os.path.join(OUTPUT_DIR, 'plddt_results.json')
if os.path.exists(existing_results_file):
    with open(existing_results_file, 'r') as f:
        results = json.load(f)
else:
    results = {}

# Get new score files
score_files = glob.glob(os.path.join(COLABFOLD_OUT, '*_scores_rank_001*.json'))
print(f'Found {len(score_files)} new result files')

for sf in score_files:
    with open(sf, 'r') as f:
        data = json.load(f)

    name = os.path.basename(sf).split('_scores_')[0]
    plddt = data.get('plddt', [])
    mean_plddt = sum(plddt) / len(plddt) if plddt else 0

    results[name] = {
        'mean_plddt': mean_plddt,
        'per_residue_plddt': plddt
    }

# Save merged results
with open(existing_results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f'Total samples in merged results: {len(results)}')
if results:
    mean_all = sum(r['mean_plddt'] for r in results.values()) / len(results)
    print(f'Mean pLDDT across all: {mean_all:.2f}')
EOF

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
