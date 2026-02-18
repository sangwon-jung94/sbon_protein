#!/bin/bash
#SBATCH --job-name=simplefold
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000/simplefold_%j.log

module load cuda/11.8.0-fasrc01

export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/11.8.0-fasrc01/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/n/home07/sangwonjung/ProDifEvo-Refinement/reference/ml-simplefold/src/simplefold:$PYTHONPATH

PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
OUTPUT_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000
FASTA=$OUTPUT_DIR/fasta_split

echo "=========================================="
echo "SimpleFold Inference on 1000 samples"
echo "Start: $(date)"
echo "=========================================="

mkdir -p $OUTPUT_DIR/simplefold

# Run SimpleFold with pLDDT prediction
# Using simplefold_100M for faster inference
/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/simplefold \
    --simplefold_model simplefold_100M \
    --num_steps 500 \
    --tau 0.01 \
    --plddt \
    --fasta_path $FASTA \
    --output_dir $OUTPUT_DIR/simplefold \
    --backend torch \
    --seed 42

echo "SimpleFold inference done: $(date)"

echo "=========================================="
echo "Extract pLDDT results"
echo "=========================================="

$PYTHON << 'EOF'
import os
import json
import glob

OUTPUT_DIR = '/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000'
SF_DIR = f'{OUTPUT_DIR}/simplefold'

results = {}

# SimpleFold outputs PDB files with pLDDT in B-factor column
for pdb_file in glob.glob(f'{SF_DIR}/*.pdb'):
    name = os.path.basename(pdb_file).replace('.pdb', '')

    # Parse B-factors (pLDDT values) from PDB
    plddt_values = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                # B-factor is in columns 61-66
                try:
                    bfactor = float(line[60:66].strip())
                    plddt_values.append(bfactor)
                except:
                    pass

    if plddt_values:
        results[name] = {
            'mean_plddt': sum(plddt_values) / len(plddt_values),
            'per_residue_plddt': plddt_values
        }

# Save results
with open(f'{SF_DIR}/plddt_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Processed {len(results)} SimpleFold structures")
if results:
    mean_all = sum(r['mean_plddt'] for r in results.values()) / len(results)
    print(f"Mean pLDDT (SimpleFold): {mean_all:.2f}")
EOF

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
