"""Extract pLDDT from SimpleFold batch3 CIF files and merge with existing results."""
import os
import json
import glob

OUTPUT_DIR = '/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000'
SF_DIR = f'{OUTPUT_DIR}/simplefold_batch3'

new_results = {}

# SimpleFold outputs CIF files, not PDB
for cif_file in glob.glob(f'{SF_DIR}/**/*.cif', recursive=True):
    name = os.path.basename(cif_file).replace('_sampled_0.cif', '')

    plddt_values = []
    with open(cif_file) as f:
        for line in f:
            if line.startswith('ATOM'):
                fields = line.split()
                # CIF format: atom_id is field[3], B_iso_or_equiv is field[-2]
                atom_name = fields[3]  # label_atom_id
                if atom_name == 'CA':
                    try:
                        bfactor = float(fields[-2])  # B_iso_or_equiv
                        plddt_values.append(bfactor)
                    except:
                        pass

    if plddt_values:
        new_results[name] = {
            'mean_plddt': sum(plddt_values) / len(plddt_values),
            'per_residue_plddt': plddt_values
        }

print(f"Processed {len(new_results)} batch3 SimpleFold structures")

# Merge with existing results
existing_file = f'{OUTPUT_DIR}/simplefold/plddt_results.json'
if os.path.exists(existing_file):
    with open(existing_file) as f:
        results = json.load(f)
    results.update(new_results)
else:
    results = new_results

with open(existing_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Total SimpleFold results: {len(results)}")
if results:
    mean_all = sum(r['mean_plddt'] for r in results.values()) / len(results)
    print(f"Mean pLDDT: {mean_all:.2f}")
