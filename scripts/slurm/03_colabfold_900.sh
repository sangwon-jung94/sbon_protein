#!/bin/bash
#SBATCH --job-name=colabfold_900
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_900/colabfold_%j.log

# Load CUDA
module load cuda/11.8.0-fasrc01

# Environment
PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
OUTPUT_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_900
FASTA_PATH=$OUTPUT_DIR/sequences.fasta
COLABFOLD_OUT=$OUTPUT_DIR/colabfold

mkdir -p $COLABFOLD_OUT

echo "Starting ColabFold inference..."
echo "Start time: $(date)"
echo "Input: $FASTA_PATH"

# Run ColabFold
$PYTHON -c "
from colabfold.batch import run, get_queries

FASTA_PATH = '$FASTA_PATH'
OUTPUT_DIR = '$COLABFOLD_OUT'

print('Getting queries...')
queries, is_complex = get_queries(FASTA_PATH)
print(f'Found {len(queries)} sequences')

print('Running ColabFold...')
run(
    queries=queries,
    result_dir=OUTPUT_DIR,
    is_complex=is_complex,
    num_models=1,
    num_recycle=3,
    model_type='alphafold2_ptm',
    msa_mode='mmseqs2_uniref_env',
    use_templates=False,
    use_amber=False,
    keep_existing_results=True,
)

print('ColabFold completed!')
"

# Extract pLDDT from results
echo "Extracting pLDDT scores..."
$PYTHON -c "
import os
import json
import glob

OUTPUT_DIR = '$COLABFOLD_OUT'

results = {}
score_files = glob.glob(os.path.join(OUTPUT_DIR, '*_scores_rank_001*.json'))
print(f'Found {len(score_files)} result files')

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

# Save summary
summary_path = os.path.join(OUTPUT_DIR, 'plddt_results.json')
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'Saved {len(results)} results to {summary_path}')
if results:
    mean_all = sum(r['mean_plddt'] for r in results.values()) / len(results)
    print(f'Mean pLDDT across all: {mean_all:.2f}')
"

echo "ColabFold complete!"
echo "End time: $(date)"
