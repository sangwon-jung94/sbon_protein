#!/bin/bash
#SBATCH --job-name=cf_b4_merge
#SBATCH --partition=seas_compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000/colabfold_batch4_merge_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sangwon.jung@trillionlabs.co

PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python

echo "=========================================="
echo "Merge ColabFold batch4 results"
echo "Start: $(date)"
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

print(f'Existing results: {len(results)}')

score_files = glob.glob(os.path.join(COLABFOLD_OUT, '*_scores_rank_001*.json'))
print(f'Found {len(score_files)} ColabFold batch4 result files')

new_count = 0
for sf in score_files:
    with open(sf) as f:
        data = json.load(f)

    name = os.path.basename(sf).split('_scores_')[0]
    if name in results:
        continue

    plddt = data.get('plddt', [])
    mean_plddt = sum(plddt) / len(plddt) if plddt else 0

    results[name] = {
        'mean_plddt': mean_plddt,
        'per_residue_plddt': plddt
    }
    new_count += 1

with open(existing_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f'Added {new_count} new results')
print(f'Total ColabFold results: {len(results)}')
if results:
    mean_all = sum(r['mean_plddt'] for r in results.values()) / len(results)
    print(f'Mean pLDDT: {mean_all:.2f}')
EOF

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
