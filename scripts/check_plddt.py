#!/usr/bin/env python
"""Check pLDDT of generated sequences using ESMFold."""

import sys
sys.path.insert(0, "/n/home07/sangwonjung/ProDifEvo-Refinement/reference/openfold")

import torch
import esm

# Generated sequences from La Proteina
sequences = [
    ("sample_0", "MNDENNYTRNMNTYPNNERTENNNEYPENHHHHNGPNLPRNRLLNSDEYNHDPNHNENPGPDHRNQNNNNYTYDTDHRRNPMQNLNEEEPDNNNRPCPNC"),
    ("sample_1", "MEDEDENRTNRPRDNENPREDEEPPNRDDRNNHPNMDFELDGNAANRNEENDQATYNEEMGRDTPRNHGTGPIDNPNDGMNCPDCNNMGNDNCPPPNPPP"),
]

print("Loading ESMFold...")
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

print("\nFolding sequences and calculating pLDDT...\n")

for name, seq in sequences:
    print(f"=== {name} (length {len(seq)}) ===")
    
    with torch.no_grad():
        output = model.infer_pdb(seq)
    
    # Extract pLDDT from output (stored in B-factor column)
    plddt_values = []
    for line in output.split('\n'):
        if line.startswith('ATOM') and ' CA ' in line:
            # B-factor is columns 61-66
            plddt = float(line[60:66].strip())
            plddt_values.append(plddt)
    
    avg_plddt = sum(plddt_values) / len(plddt_values)
    min_plddt = min(plddt_values)
    max_plddt = max(plddt_values)
    
    print(f"  Average pLDDT: {avg_plddt:.2f}")
    print(f"  Min pLDDT: {min_plddt:.2f}")
    print(f"  Max pLDDT: {max_plddt:.2f}")
    print(f"  Sequence: {seq[:30]}...")
    print()

print("Done!")
