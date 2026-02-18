#!/usr/bin/env python
"""Run ESMFold on La Proteina generated sequences and calculate pLDDT."""

import sys
# Add reference directory to path (contains openfold package)
sys.path.insert(0, "/n/home07/sangwonjung/ProDifEvo-Refinement/reference")

import os
import json
import torch
import esm
from tqdm import tqdm

# Paths
FASTA_PATH = "/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_100/sequences.fasta"
OUTPUT_DIR = "/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_100/esmfold"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parse FASTA
def parse_fasta(fasta_path):
    sequences = []
    with open(fasta_path) as f:
        name, seq = None, ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    sequences.append((name, seq))
                name = line[1:]
                seq = ""
            else:
                seq += line
        if name:
            sequences.append((name, seq))
    return sequences

print("Loading ESMFold...")
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

print("Parsing sequences...")
sequences = parse_fasta(FASTA_PATH)
print(f"Found {len(sequences)} sequences")

results = []
for name, seq in tqdm(sequences, desc="Folding"):
    with torch.no_grad():
        output = model.infer_pdb(seq)
    
    # Save PDB
    pdb_path = os.path.join(OUTPUT_DIR, f"{name}.pdb")
    with open(pdb_path, 'w') as f:
        f.write(output)
    
    # Extract pLDDT from B-factor column
    plddt_values = []
    for line in output.split('\n'):
        if line.startswith('ATOM') and ' CA ' in line:
            plddt = float(line[60:66].strip())
            plddt_values.append(plddt)
    
    avg_plddt = sum(plddt_values) / len(plddt_values) if plddt_values else 0
    results.append({
        "name": name,
        "sequence": seq,
        "avg_plddt": avg_plddt,
        "min_plddt": min(plddt_values) if plddt_values else 0,
        "max_plddt": max(plddt_values) if plddt_values else 0,
        "per_residue_plddt": plddt_values
    })
    
    print(f"  {name}: pLDDT = {avg_plddt:.2f}")

# Save results
results_path = os.path.join(OUTPUT_DIR, "plddt_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

# Summary
plddts = [r["avg_plddt"] for r in results]
print(f"\n=== Summary ===")
print(f"Total samples: {len(results)}")
print(f"Mean pLDDT: {sum(plddts)/len(plddts):.2f}")
print(f"Min pLDDT: {min(plddts):.2f}")
print(f"Max pLDDT: {max(plddts):.2f}")
print(f"Results saved to: {results_path}")
