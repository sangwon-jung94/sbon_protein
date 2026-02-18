#!/bin/bash
#SBATCH --job-name=esmfold_2000
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000/esmfold_2000_%j.log

module load cuda/11.8.0-fasrc01

PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
OUTPUT_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000
FASTA_SPLIT=$OUTPUT_DIR/fasta_split

echo "=========================================="
echo "ESMFold Inference on new 2000 samples (batch3)"
echo "Start: $(date)"
echo "=========================================="

# Count batch3 files
N_BATCH3=$(ls $FASTA_SPLIT/batch3*.fasta 2>/dev/null | wc -l)
echo "Found $N_BATCH3 batch3 fasta files"

if [ "$N_BATCH3" -eq 0 ]; then
    echo "No batch3 files found. Generation may not have completed."
    exit 1
fi

mkdir -p $OUTPUT_DIR/esmfold_new

$PYTHON << 'EOF'
import os
import json
import glob
import torch
import esm

OUTPUT_DIR = '/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000'
FASTA_SPLIT = f'{OUTPUT_DIR}/fasta_split'
ESMFOLD_OUT = f'{OUTPUT_DIR}/esmfold_new'

# Get batch3 fasta files only
fasta_files = sorted(glob.glob(f'{FASTA_SPLIT}/batch3*.fasta'))
print(f"Processing {len(fasta_files)} batch3 sequences")

# Load ESMFold
print("Loading ESMFold model...")
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

results = {}

for i, fasta_file in enumerate(fasta_files):
    name = os.path.basename(fasta_file).replace('.fasta', '')

    # Skip if already processed
    if os.path.exists(f'{ESMFOLD_OUT}/{name}.pdb'):
        print(f"Skipping {name} (already processed)")
        continue

    # Read sequence
    with open(fasta_file) as f:
        lines = f.readlines()
        seq = ''.join(line.strip() for line in lines[1:])

    print(f"[{i+1}/{len(fasta_files)}] Processing {name} (len={len(seq)})")

    try:
        with torch.no_grad():
            output = model.infer_pdb(seq)

        # Save PDB
        with open(f'{ESMFOLD_OUT}/{name}.pdb', 'w') as f:
            f.write(output)

        # Extract pLDDT from output
        plddt_values = []
        for line in output.split('\n'):
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
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
            print(f"  pLDDT: {results[name]['mean_plddt']:.2f}")
    except Exception as e:
        print(f"  Error: {e}")
        continue

# Save new results
with open(f'{ESMFOLD_OUT}/plddt_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nProcessed {len(results)} new sequences")

# Merge with existing results
existing_file = f'{OUTPUT_DIR}/esmfold/plddt_results.json'
if os.path.exists(existing_file):
    with open(existing_file) as f:
        existing = json.load(f)
    existing.update(results)
    with open(existing_file, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Merged with existing. Total: {len(existing)}")
EOF

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
