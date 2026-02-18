#!/bin/bash
#SBATCH --job-name=esmfold_900
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_900/esmfold_%j.log
#SBATCH --dependency=singleton

# Load CUDA
module load cuda/11.8.0-fasrc01

# Environment
PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
OUTPUT_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_900
FASTA_PATH=$OUTPUT_DIR/sequences.fasta
ESMFOLD_OUT=$OUTPUT_DIR/esmfold

mkdir -p $ESMFOLD_OUT

echo "Starting ESMFold inference..."
echo "Start time: $(date)"
echo "Input: $FASTA_PATH"

# Run ESMFold
$PYTHON -c "
import sys
sys.path.insert(0, '/n/home07/sangwonjung/ProDifEvo-Refinement/reference')

import os
import json
import torch
import esm

FASTA_PATH = '$FASTA_PATH'
OUTPUT_DIR = '$ESMFOLD_OUT'

# Parse FASTA
sequences = {}
with open(FASTA_PATH, 'r') as f:
    current_name = None
    current_seq = ''
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if current_name:
                sequences[current_name] = current_seq
            current_name = line[1:]
            current_seq = ''
        else:
            current_seq += line
    if current_name:
        sequences[current_name] = current_seq

print(f'Loaded {len(sequences)} sequences')

# Load ESMFold
print('Loading ESMFold model...')
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Run inference
results = {}
for i, (name, seq) in enumerate(sequences.items()):
    print(f'Processing {i+1}/{len(sequences)}: {name} (len={len(seq)})')

    with torch.no_grad():
        output = model.infer_pdb(seq)

    # Save PDB
    pdb_path = os.path.join(OUTPUT_DIR, f'{name}.pdb')
    with open(pdb_path, 'w') as f:
        f.write(output)

    # Extract pLDDT from B-factors
    plddt_values = []
    for line in output.split('\n'):
        if line.startswith('ATOM') and ' CA ' in line:
            try:
                bfactor = float(line[60:66].strip())
                plddt_values.append(bfactor)
            except:
                pass

    mean_plddt = sum(plddt_values) / len(plddt_values) if plddt_values else 0
    results[name] = {
        'sequence': seq,
        'length': len(seq),
        'mean_plddt': mean_plddt,
        'per_residue_plddt': plddt_values
    }
    print(f'  pLDDT: {mean_plddt:.2f}')

# Save results
results_path = os.path.join(OUTPUT_DIR, 'plddt_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'Results saved to {results_path}')
print(f'Mean pLDDT across all: {sum(r[\"mean_plddt\"] for r in results.values())/len(results):.2f}')
"

echo "ESMFold complete!"
echo "End time: $(date)"
