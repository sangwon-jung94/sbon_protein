#!/bin/bash
#SBATCH --job-name=cf_template
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000/colabfold_template_%j.log

module load cuda/11.8.0-fasrc01

PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
LAPROTEINA_100=/n/home07/sangwonjung/ProDifEvo-Refinement/reference/la-proteina/inference/inference_100samples
LAPROTEINA_900=/n/home07/sangwonjung/ProDifEvo-Refinement/reference/la-proteina/inference/inference_900samples
OUTPUT_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000
TEMPLATE_DIR=$OUTPUT_DIR/templates

mkdir -p $TEMPLATE_DIR $OUTPUT_DIR/colabfold_template

echo "=========================================="
echo "STEP 1: Prepare templates and FASTA"
echo "=========================================="
echo "Start: $(date)"

$PYTHON << 'EOF'
import os
import glob
import shutil
from Bio import PDB

OUTPUT_DIR = '/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000'
TEMPLATE_DIR = f'{OUTPUT_DIR}/templates'
LAPROTEINA_100 = '/n/home07/sangwonjung/ProDifEvo-Refinement/reference/la-proteina/inference/inference_100samples'
LAPROTEINA_900 = '/n/home07/sangwonjung/ProDifEvo-Refinement/reference/la-proteina/inference/inference_900samples'

three_to_one = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}
parser = PDB.PDBParser(QUIET=True)

# Collect all PDBs with unique prefixes
all_pdbs = []

# Batch 1 (100 samples)
for pdb_file in glob.glob(f'{LAPROTEINA_100}/*/*.pdb'):
    name = os.path.basename(pdb_file).replace('.pdb', '')
    all_pdbs.append(('batch1', name, pdb_file))

# Batch 2 (900 samples)
for pdb_file in glob.glob(f'{LAPROTEINA_900}/*/*.pdb'):
    name = os.path.basename(pdb_file).replace('.pdb', '')
    all_pdbs.append(('batch2', name, pdb_file))

print(f"Total PDBs: {len(all_pdbs)}")

# Create FASTA and copy templates
with open(f'{OUTPUT_DIR}/sequences_for_template.fasta', 'w') as fasta:
    for batch, name, pdb_file in all_pdbs:
        unique_name = f"{batch}_{name}"

        # Extract sequence
        structure = parser.get_structure(name, pdb_file)
        seq = ''.join(three_to_one.get(r.get_resname(), '') for m in structure for c in m for r in c)

        # Write FASTA
        fasta.write(f">{unique_name}\n{seq}\n")

        # Copy PDB to template dir with matching name
        # ColabFold expects template in format: <query_name>_<hit_name>.pdb
        # We'll use query_name as the template name
        template_subdir = f"{TEMPLATE_DIR}/{unique_name}"
        os.makedirs(template_subdir, exist_ok=True)
        shutil.copy(pdb_file, f"{template_subdir}/{unique_name}.pdb")

print(f"Created FASTA and templates for {len(all_pdbs)} samples")
EOF

echo "Preparation done: $(date)"

echo "=========================================="
echo "STEP 2: Run ColabFold with templates"
echo "=========================================="
echo "Start: $(date)"

$PYTHON << 'EOF'
from colabfold.batch import run, get_queries
import os

OUTPUT_DIR = '/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_1000'
FASTA = f'{OUTPUT_DIR}/sequences_for_template.fasta'
RESULT_DIR = f'{OUTPUT_DIR}/colabfold_template'
TEMPLATE_DIR = f'{OUTPUT_DIR}/templates'

queries, is_complex = get_queries(FASTA)
print(f"Running ColabFold with templates on {len(queries)} sequences...")

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
