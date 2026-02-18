#!/bin/bash
#SBATCH --job-name=simplefold_2000
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000/simplefold_2000_%j.log

module load cuda/11.8.0-fasrc01

export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/11.8.0-fasrc01/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/n/home07/sangwonjung/ProDifEvo-Refinement/reference/ml-simplefold/src/simplefold:$PYTHONPATH

PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
OUTPUT_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000
FASTA_SPLIT=$OUTPUT_DIR/fasta_split

echo "=========================================="
echo "SimpleFold Inference on new 2000 samples"
echo "Start: $(date)"
echo "=========================================="

# Create temp directory with only batch3 fasta files
TMP_FASTA=/tmp/simplefold_batch3_$$
mkdir -p $TMP_FASTA
cp $FASTA_SPLIT/batch3*.fasta $TMP_FASTA/

echo "Found $(ls $TMP_FASTA/*.fasta | wc -l) fasta files to process"

mkdir -p $OUTPUT_DIR/simplefold_new

# Run SimpleFold with pLDDT prediction
/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/simplefold \
    --simplefold_model simplefold_100M \
    --num_steps 500 \
    --tau 0.01 \
    --plddt \
    --fasta_path $TMP_FASTA \
    --output_dir $OUTPUT_DIR/simplefold_new \
    --backend torch \
    --seed 42

echo "SimpleFold inference done: $(date)"

echo "=========================================="
echo "Extract and merge pLDDT results"
echo "=========================================="

$PYTHON << 'EOF'
import os
import json
import glob

SF_DIR = '/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000/simplefold_new/predictions_simplefold_100M'
OUTPUT_DIR = '/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000/simplefold'

# Load existing results
existing_results_file = os.path.join(OUTPUT_DIR, 'plddt_results.json')
if os.path.exists(existing_results_file):
    with open(existing_results_file, 'r') as f:
        results = json.load(f)
else:
    results = {}

# Process new CIF files
for cif_file in glob.glob(f'{SF_DIR}/*.cif'):
    name = os.path.basename(cif_file).replace('_sampled_0.cif', '')

    plddt_values = []
    in_metric_section = False

    with open(cif_file) as f:
        content = f.read()

    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '_ma_qa_metric_local.metric_value' in line:
            in_metric_section = True
            continue
        if in_metric_section:
            if line.startswith('#') or line.startswith('_') or line.startswith('loop_'):
                break
            parts = line.split()
            if len(parts) >= 7:
                try:
                    plddt = float(parts[-1])
                    plddt_values.append(plddt)
                except:
                    pass

    if plddt_values:
        results[name] = {
            'mean_plddt': sum(plddt_values) / len(plddt_values),
            'per_residue_plddt': plddt_values
        }

# Save merged results
with open(existing_results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Total samples in merged results: {len(results)}")
if results:
    mean_all = sum(r['mean_plddt'] for r in results.values()) / len(results)
    print(f"Mean pLDDT (SimpleFold): {mean_all:.2f}")
EOF

# Cleanup
rm -rf $TMP_FASTA

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
