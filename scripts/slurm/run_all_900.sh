#!/bin/bash
#SBATCH --job-name=laproteina_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_900/all_%j.log

# Load CUDA
module load cuda/11.8.0-fasrc01

PYTHON=/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/bin/python
LAPROTEINA_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/reference/la-proteina
OUTPUT_DIR=/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_900

mkdir -p $OUTPUT_DIR/esmfold $OUTPUT_DIR/colabfold

echo "=========================================="
echo "STEP 1: La Proteina Generation (900 samples)"
echo "=========================================="
echo "Start: $(date)"

cd $LAPROTEINA_DIR
$PYTHON proteinfoundation/generate.py --config_name inference_900samples

# Extract sequences to FASTA from La Proteina output
$PYTHON -c "
import os, glob
from Bio import PDB

# La Proteina outputs to inference/inference_<config_name>/job_X/
pdb_dir = '$LAPROTEINA_DIR/inference/inference_900samples'
pdb_files = sorted(glob.glob(os.path.join(pdb_dir, '*/*.pdb')))
print(f'Found {len(pdb_files)} PDB files')

three_to_one = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}
parser = PDB.PDBParser(QUIET=True)

with open('$OUTPUT_DIR/sequences.fasta', 'w') as f:
    for pdb_file in pdb_files:
        name = os.path.basename(pdb_file).replace('.pdb', '')
        structure = parser.get_structure(name, pdb_file)
        seq = ''.join(three_to_one.get(r.get_resname(),'') for m in structure for c in m for r in c)
        f.write(f'>{name}\n{seq}\n')
print('FASTA saved')
"

echo "Generation done: $(date)"

echo "=========================================="
echo "STEP 2: ESMFold Inference"
echo "=========================================="
echo "Start: $(date)"

$PYTHON -c "
import sys
sys.path.insert(0, '/n/home07/sangwonjung/ProDifEvo-Refinement/reference')
import os, json, torch, esm

FASTA = '$OUTPUT_DIR/sequences.fasta'
OUT = '$OUTPUT_DIR/esmfold'

seqs = {}
with open(FASTA) as f:
    name, seq = None, ''
    for line in f:
        if line.startswith('>'):
            if name: seqs[name] = seq
            name, seq = line[1:].strip(), ''
        else: seq += line.strip()
    if name: seqs[name] = seq

print(f'Loaded {len(seqs)} sequences')
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

results = {}
for i, (name, seq) in enumerate(seqs.items()):
    print(f'{i+1}/{len(seqs)}: {name}')
    with torch.no_grad():
        output = model.infer_pdb(seq)
    with open(f'{OUT}/{name}.pdb', 'w') as f: f.write(output)
    plddt = [float(l[60:66]) for l in output.split('\n') if l.startswith('ATOM') and ' CA ' in l]
    results[name] = {'mean_plddt': sum(plddt)/len(plddt) if plddt else 0, 'per_residue_plddt': plddt}

with open(f'{OUT}/plddt_results.json', 'w') as f: json.dump(results, f, indent=2)
print(f'Mean pLDDT: {sum(r[\"mean_plddt\"] for r in results.values())/len(results):.2f}')
"

echo "ESMFold done: $(date)"

echo "=========================================="
echo "STEP 3: ColabFold Inference"
echo "=========================================="
echo "Start: $(date)"

$PYTHON -c "
from colabfold.batch import run, get_queries

queries, is_complex = get_queries('$OUTPUT_DIR/sequences.fasta')
print(f'Running ColabFold on {len(queries)} sequences...')

run(
    queries=queries,
    result_dir='$OUTPUT_DIR/colabfold',
    is_complex=is_complex,
    num_models=1,
    num_recycle=3,
    model_type='alphafold2_ptm',
    msa_mode='mmseqs2_uniref_env',
    use_templates=False,
    use_amber=False,
    keep_existing_results=True,
)
"

# Extract ColabFold pLDDT
$PYTHON -c "
import os, json, glob

OUT = '$OUTPUT_DIR/colabfold'
results = {}
for sf in glob.glob(f'{OUT}/*_scores_rank_001*.json'):
    with open(sf) as f: data = json.load(f)
    name = os.path.basename(sf).split('_scores_')[0]
    plddt = data.get('plddt', [])
    results[name] = {'mean_plddt': sum(plddt)/len(plddt) if plddt else 0, 'per_residue_plddt': plddt}

with open(f'{OUT}/plddt_results.json', 'w') as f: json.dump(results, f, indent=2)
print(f'ColabFold results: {len(results)} samples, mean pLDDT: {sum(r[\"mean_plddt\"] for r in results.values())/len(results):.2f}')
"

echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
