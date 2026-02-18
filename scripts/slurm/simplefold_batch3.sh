#!/bin/bash
#SBATCH --job-name=sf_batch3
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000/simplefold_batch3_%j.log

module load cuda/11.8.0-fasrc01

export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/11.8.0-fasrc01/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/n/home07/sangwonjung/ProDifEvo-Refinement/reference/ml-simplefold/src/simplefold:$PYTHONPATH

PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
OUTPUT_DIR=/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000
FASTA_SPLIT=$OUTPUT_DIR/fasta_split

echo "=========================================="
echo "SimpleFold Inference on batch3 (2000 samples)"
echo "Start: $(date)"
echo "=========================================="

# Create temp dir with only batch3 fasta files
BATCH3_DIR=$(mktemp -d)
cp $FASTA_SPLIT/batch3*.fasta $BATCH3_DIR/
N_SEQ=$(ls $BATCH3_DIR/*.fasta | wc -l)
echo "Copied $N_SEQ batch3 fasta files to $BATCH3_DIR"

mkdir -p $OUTPUT_DIR/simplefold_batch3

# Match original run_simplefold.sh exactly
/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/simplefold \
    --simplefold_model simplefold_100M \
    --num_steps 500 \
    --tau 0.01 \
    --plddt \
    --fasta_path $BATCH3_DIR \
    --output_dir $OUTPUT_DIR/simplefold_batch3 \
    --backend torch \
    --seed 42

rm -rf $BATCH3_DIR

echo "SimpleFold inference done: $(date)"

echo "=========================================="
echo "Extract pLDDT and merge results"
echo "=========================================="

$PYTHON << 'EOF'
import os
import json
import glob

OUTPUT_DIR = '/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000'
SF_DIR = f'{OUTPUT_DIR}/simplefold_batch3'

new_results = {}

for pdb_file in glob.glob(f'{SF_DIR}/**/*.pdb', recursive=True):
    name = os.path.basename(pdb_file).replace('.pdb', '')

    plddt_values = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                try:
                    bfactor = float(line[60:66].strip())
                    plddt_values.append(bfactor)
                except:
                    pass

    if plddt_values:
        new_results[name] = {
            'mean_plddt': sum(plddt_values) / len(plddt_values),
            'per_residue_plddt': plddt_values
        }

print(f"Processed {len(new_results)} batch3 SimpleFold structures")

# Merge with existing results
existing_file = f'{OUTPUT_DIR}/simplefold/plddt_results.json'
if os.path.exists(existing_file):
    with open(existing_file) as f:
        results = json.load(f)
    results.update(new_results)
else:
    results = new_results

with open(existing_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Total SimpleFold results: {len(results)}")
if results:
    mean_all = sum(r['mean_plddt'] for r in results.values()) / len(results)
    print(f"Mean pLDDT: {mean_all:.2f}")
EOF

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
