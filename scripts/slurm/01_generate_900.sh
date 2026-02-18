#!/bin/bash
#SBATCH --job-name=laproteina_900
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_900/generation_%j.log

# Load CUDA
module load cuda/11.8.0-fasrc01

# Environment
PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
LAPROTEINA_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/reference/la-proteina
OUTPUT_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_900

cd $LAPROTEINA_DIR

echo "Starting La Proteina generation of 900 samples..."
echo "Start time: $(date)"

# Run generation
$PYTHON inference.py --config-name inference_900samples

# Move generated files to output directory
echo "Moving generated files..."
if [ -d "outputs/laproteina_900samples" ]; then
    cp -r outputs/laproteina_900samples/* $OUTPUT_DIR/
fi

# Extract sequences to FASTA
echo "Extracting sequences to FASTA..."
$PYTHON -c "
import os
import glob
from Bio import PDB

pdb_dir = '$OUTPUT_DIR/pdbs'
fasta_path = '$OUTPUT_DIR/sequences.fasta'

if not os.path.exists(pdb_dir):
    pdb_dir = '$OUTPUT_DIR'

pdb_files = sorted(glob.glob(os.path.join(pdb_dir, '*.pdb')))
print(f'Found {len(pdb_files)} PDB files')

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

parser = PDB.PDBParser(QUIET=True)
with open(fasta_path, 'w') as f:
    for pdb_file in pdb_files:
        name = os.path.basename(pdb_file).replace('.pdb', '')
        structure = parser.get_structure(name, pdb_file)
        seq = ''
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in three_to_one:
                        seq += three_to_one[residue.get_resname()]
        f.write(f'>{name}\n{seq}\n')

print(f'Wrote sequences to {fasta_path}')
"

echo "Generation complete!"
echo "End time: $(date)"
