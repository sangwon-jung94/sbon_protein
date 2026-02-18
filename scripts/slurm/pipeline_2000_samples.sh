#!/bin/bash
#SBATCH --job-name=pipeline_2000
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000/pipeline_2000_%j.log

module load cuda/11.8.0-fasrc01

export PYTHONPATH=/n/home07/sangwonjung/ProDifEvo-Refinement/reference/la-proteina:$PYTHONPATH
export PATH=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin:$PATH

PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
COLABFOLD=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/colabfold_batch
SHARED_DIR=/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung
OUTPUT_DIR=$SHARED_DIR/laproteina_1000
GEN_DIR=$SHARED_DIR/inference_2000samples

echo "=========================================="
echo "PIPELINE: Process 2000 + ESMFold + ColabFold"
echo "Start: $(date)"
echo "=========================================="

###########################################
# STEP 1: Process generated 2000 samples
###########################################
echo ""
echo "=========================================="
echo "STEP 1: Process generated 2000 samples"
echo "=========================================="

$PYTHON << EOF
import os
import glob
import shutil

GEN_DIR = '$GEN_DIR'
OUTPUT_DIR = '$OUTPUT_DIR'

pdb_files = sorted(glob.glob(os.path.join(GEN_DIR, '**/*.pdb'), recursive=True))
print(f"Found {len(pdb_files)} generated PDB files")

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

from Bio import PDB
parser = PDB.PDBParser(QUIET=True)

os.makedirs(f'{OUTPUT_DIR}/structures', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/fasta_split', exist_ok=True)

new_sequences = []

for i, pdb_file in enumerate(pdb_files):
    new_name = f"batch3_job_0_n_100_id_{i}"

    # Check if already processed
    if os.path.exists(f'{OUTPUT_DIR}/fasta_split/{new_name}.fasta'):
        continue

    dst_pdb = os.path.join(OUTPUT_DIR, 'structures', f'{new_name}.pdb')
    shutil.copy(pdb_file, dst_pdb)

    structure = parser.get_structure(new_name, pdb_file)
    seq = ''
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in three_to_one:
                    seq += three_to_one[residue.get_resname()]

    fasta_path = os.path.join(OUTPUT_DIR, 'fasta_split', f'{new_name}.fasta')
    with open(fasta_path, 'w') as f:
        f.write(f'>{new_name}\n{seq}\n')

    new_sequences.append((new_name, seq))

    if (i + 1) % 200 == 0:
        print(f"Processed {i + 1}/{len(pdb_files)} files")

if new_sequences:
    with open(f'{OUTPUT_DIR}/sequences.fasta', 'a') as f:
        for name, seq in new_sequences:
            f.write(f'>{name}\n{seq}\n')
    print(f"Added {len(new_sequences)} new samples")
else:
    print("All samples already processed")
EOF

echo "Processing done: $(date)"

###########################################
# STEP 2: ESMFold pLDDT
###########################################
echo ""
echo "=========================================="
echo "STEP 2: ESMFold Inference"
echo "=========================================="

$PYTHON << EOF
import os
import json
import glob
import torch
import esm

OUTPUT_DIR = '$OUTPUT_DIR'
FASTA_SPLIT = f'{OUTPUT_DIR}/fasta_split'

fasta_files = sorted(glob.glob(f'{FASTA_SPLIT}/batch3*.fasta'))
print(f"Found {len(fasta_files)} batch3 sequences for ESMFold")

if len(fasta_files) == 0:
    print("No batch3 files found!")
    exit(1)

print("Loading ESMFold model...")
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

os.makedirs(f'{OUTPUT_DIR}/esmfold', exist_ok=True)
existing_file = f'{OUTPUT_DIR}/esmfold/plddt_results.json'
if os.path.exists(existing_file):
    with open(existing_file) as f:
        results = json.load(f)
else:
    results = {}

for i, fasta_file in enumerate(fasta_files):
    name = os.path.basename(fasta_file).replace('.fasta', '')

    if name in results:
        continue

    with open(fasta_file) as f:
        lines = f.readlines()
        seq = ''.join(line.strip() for line in lines[1:])

    print(f"[{i+1}/{len(fasta_files)}] {name} (len={len(seq)})")

    try:
        with torch.no_grad():
            output = model.infer_pdb(seq)

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

        # Save periodically
        if (i + 1) % 100 == 0:
            with open(existing_file, 'w') as f:
                json.dump(results, f, indent=2)
    except Exception as e:
        print(f"  Error: {e}")

with open(existing_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Total ESMFold results: {len(results)}")
EOF

echo "ESMFold done: $(date)"

###########################################
# STEP 3: ColabFold pLDDT
###########################################
echo ""
echo "=========================================="
echo "STEP 3: ColabFold Inference"
echo "=========================================="

mkdir -p $OUTPUT_DIR/colabfold_batch3

# Create combined fasta for batch3
cat $OUTPUT_DIR/fasta_split/batch3*.fasta > /tmp/batch3_combined_$$.fasta
N_SEQ=$(grep -c "^>" /tmp/batch3_combined_$$.fasta)
echo "Created combined fasta with $N_SEQ sequences"

# Run ColabFold
$COLABFOLD /tmp/batch3_combined_$$.fasta $OUTPUT_DIR/colabfold_batch3 \
    --num-models 1 \
    --num-recycle 3 \
    --model-type alphafold2_ptm

rm /tmp/batch3_combined_$$.fasta

# Extract and merge pLDDT
$PYTHON << EOF
import os
import json
import glob

OUTPUT_DIR = '$OUTPUT_DIR'
COLABFOLD_OUT = f'{OUTPUT_DIR}/colabfold_batch3'

os.makedirs(f'{OUTPUT_DIR}/colabfold', exist_ok=True)
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

echo "ColabFold done: $(date)"

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE: $(date)"
echo "=========================================="
