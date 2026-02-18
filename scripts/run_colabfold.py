#!/usr/bin/env python
"""Run ColabFold on La Proteina generated sequences and calculate pLDDT."""

import os
import json
from colabfold.batch import run, get_queries

# Paths
FASTA_PATH = "/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_100/sequences.fasta"
OUTPUT_DIR = "/n/home07/sangwonjung/ProDifEvo-Refinement/datasets/laproteina_100/colabfold"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Running ColabFold...")
print(f"Input: {FASTA_PATH}")
print(f"Output: {OUTPUT_DIR}")

# Get queries
queries, is_complex = get_queries(FASTA_PATH)
print(f"Found {len(queries)} sequences, is_complex={is_complex}")

# Run ColabFold
run(
    queries=queries,
    result_dir=OUTPUT_DIR,
    is_complex=is_complex,
    num_models=1,  # Use only 1 model for speed
    num_recycle=3,
    model_type="alphafold2_ptm",
    msa_mode="mmseqs2_uniref_env",
    use_templates=False,
    use_amber=False,
    keep_existing_results=False,
)

print("\nColabFold completed!")
print(f"Results saved to: {OUTPUT_DIR}")
