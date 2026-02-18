#!/bin/bash
#SBATCH --job-name=cf_template2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000/colabfold_template2_%j.log

module load cuda/11.8.0-fasrc01

PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
OUTPUT_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000
TEMPLATE_DIR=$OUTPUT_DIR/templates_cif

mkdir -p $TEMPLATE_DIR $OUTPUT_DIR/colabfold_template

echo "=========================================="
echo "STEP 1: Convert PDB to CIF and prepare"
echo "=========================================="
echo "Start: $(date)"

$PYTHON << 'EOF'
import os
import glob
from Bio import PDB
from Bio.PDB import MMCIFIO

OUTPUT_DIR = '/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000'
TEMPLATE_DIR = f'{OUTPUT_DIR}/templates_cif'
LAPROTEINA_100 = '/n/home07/sangwonjung/ProDifEvo-Refinement/reference/la-proteina/inference/inference_100samples'
LAPROTEINA_900 = '/n/home07/sangwonjung/ProDifEvo-Refinement/reference/la-proteina/inference/inference_900samples'

three_to_one = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}
parser = PDB.PDBParser(QUIET=True)
cif_io = MMCIFIO()

# Collect all PDBs
all_pdbs = []
for pdb_file in glob.glob(f'{LAPROTEINA_100}/*/*.pdb'):
    name = os.path.basename(pdb_file).replace('.pdb', '')
    all_pdbs.append(('batch1', name, pdb_file))
for pdb_file in glob.glob(f'{LAPROTEINA_900}/*/*.pdb'):
    name = os.path.basename(pdb_file).replace('.pdb', '')
    all_pdbs.append(('batch2', name, pdb_file))

print(f"Total PDBs: {len(all_pdbs)}")

# Create FASTA and convert to CIF
with open(f'{OUTPUT_DIR}/sequences_for_template.fasta', 'w') as fasta:
    for i, (batch, name, pdb_file) in enumerate(all_pdbs):
        unique_name = f"{batch}_{name}"

        # Parse structure
        structure = parser.get_structure(name, pdb_file)
        seq = ''.join(three_to_one.get(r.get_resname(), '') for m in structure for c in m for r in c)

        # Write FASTA
        fasta.write(f">{unique_name}\n{seq}\n")

        # Convert PDB to CIF
        cif_path = f"{TEMPLATE_DIR}/{unique_name}.cif"
        cif_io.set_structure(structure)
        cif_io.save(cif_path)

        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(all_pdbs)}")

print(f"Created FASTA and CIF templates for {len(all_pdbs)} samples")
EOF

echo "Conversion done: $(date)"

echo "=========================================="
echo "STEP 2: Run ColabFold with CIF templates"
echo "=========================================="
echo "Start: $(date)"

$PYTHON << 'EOF'
from colabfold.batch import run, get_queries

OUTPUT_DIR = '/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000'
FASTA = f'{OUTPUT_DIR}/sequences_for_template.fasta'
RESULT_DIR = f'{OUTPUT_DIR}/colabfold_template'
TEMPLATE_DIR = f'{OUTPUT_DIR}/templates_cif'

queries, is_complex = get_queries(FASTA)
print(f"Running ColabFold with CIF templates on {len(queries)} sequences...")

run(
    queries=queries,
    result_dir=RESULT_DIR,
    is_complex=is_complex,
    num_models=1,
    num_recycles=3,
    model_type='alphafold2_ptm',
    msa_mode='mmseqs2_uniref_env',
    use_templates=True,
    custom_template_path=TEMPLATE_DIR,
    use_amber=False,
    keep_existing_results=True,
)

print("ColabFold with templates completed!")
EOF

echo "ColabFold done: $(date)"

echo "=========================================="
echo "STEP 3: Extract pLDDT results"
echo "=========================================="

$PYTHON << 'EOF'
import os
import json
import glob

OUTPUT_DIR = '/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000'
CF_DIR = f'{OUTPUT_DIR}/colabfold_template'

results = {}
for sf in glob.glob(f'{CF_DIR}/*_scores_rank_001*.json'):
    with open(sf) as f:
        data = json.load(f)
    name = os.path.basename(sf).split('_scores_')[0]
    plddt = data.get('plddt', [])
    results[name] = {'mean_plddt': sum(plddt)/len(plddt) if plddt else 0, 'per_residue_plddt': plddt}

with open(f'{CF_DIR}/plddt_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} results")
if results:
    mean_all = sum(r['mean_plddt'] for r in results.values()) / len(results)
    print(f"Mean pLDDT (with template): {mean_all:.2f}")
EOF

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
