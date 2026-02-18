#!/bin/bash
#SBATCH --job-name=laproteina_2000more
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000/generation_2000more_%j.log

# Load CUDA
module load cuda/11.8.0-fasrc01

# Environment
PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
LAPROTEINA_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/reference/la-proteina
OUTPUT_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000

cd $LAPROTEINA_DIR

echo "=========================================="
echo "La Proteina: Generate 2000 more samples"
echo "Start time: $(date)"
echo "=========================================="

# Run generation using proteinfoundation/generate.py
$PYTHON proteinfoundation/generate.py --config_name inference_2000samples

echo "Generation complete, now renaming and moving files..."

# Python script to rename and merge with existing dataset
$PYTHON << 'EOF'
import os
import glob
import shutil

LAPROTEINA_DIR = '/n/home07/sangwonjung/ProDifEvo-Refinement/reference/la-proteina'
OUTPUT_DIR = '/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000'

# Find generated PDBs - generate.py saves to ./inference/{config_name}
gen_dir = os.path.join(LAPROTEINA_DIR, 'inference/inference_2000samples')
pdb_pattern = os.path.join(gen_dir, '**/*.pdb')
pdb_files = sorted(glob.glob(pdb_pattern, recursive=True))

print(f"Found {len(pdb_files)} generated PDB files")

# Starting index is 1000 (after existing batch1 100 + batch2 900 = 1000)
start_idx = 1000

# Three-letter to one-letter amino acid mapping
three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

from Bio import PDB
parser = PDB.PDBParser(QUIET=True)

new_sequences = []
os.makedirs(f'{OUTPUT_DIR}/structures', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/fasta_split', exist_ok=True)

for i, pdb_file in enumerate(pdb_files):
    new_idx = start_idx + i
    # Naming convention: batch3_job_0_n_100_id_X (batch3 for new samples)
    batch_num = 3
    id_num = i
    new_name = f"batch{batch_num}_job_0_n_100_id_{id_num}"

    # Copy PDB to structures directory
    dst_pdb = os.path.join(OUTPUT_DIR, 'structures', f'{new_name}.pdb')
    shutil.copy(pdb_file, dst_pdb)

    # Extract sequence
    structure = parser.get_structure(new_name, pdb_file)
    seq = ''
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in three_to_one:
                    seq += three_to_one[residue.get_resname()]

    # Save individual fasta file
    fasta_path = os.path.join(OUTPUT_DIR, 'fasta_split', f'{new_name}.fasta')
    with open(fasta_path, 'w') as f:
        f.write(f'>{new_name}\n{seq}\n')

    new_sequences.append((new_name, seq))

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(pdb_files)} files")

# Append to main sequences.fasta
with open(f'{OUTPUT_DIR}/sequences.fasta', 'a') as f:
    for name, seq in new_sequences:
        f.write(f'>{name}\n{seq}\n')

print(f"\nAdded {len(new_sequences)} new samples to dataset")
print(f"Total samples now: {start_idx + len(new_sequences)}")
EOF

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
