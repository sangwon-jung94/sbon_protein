# ProDifEvo-Refinement: Reward Hacking Analysis Summary

## Project Goal
Investigate **reward hacking** in Best-of-N protein structure selection: when a cheap proxy model is used to select "best" generated proteins, does it actually select good structures (as judged by an expensive oracle)?

---

## Experimental Setup

### Protein Generation
- **Generator**: La Proteina (NVIDIA latent diffusion model)
- **Config**: `LD1_ucond_notri_512.ckpt`, unconditional, 50 diffusion steps, 100 residues
- **Total generated**: 10,000 sequences across 4 batches

| Batch | Count | Notes |
|-------|-------|-------|
| batch1 | 100 | Initial pilot |
| batch2 | 900 | Extended to 1,000 total |
| batch3 | 2,000 | Extended to 3,000 total |
| batch4 | 7,000 | Extended to 10,000 total |

### Structure Prediction Models (pLDDT scoring)

| Model | Role | Speed | Description |
|-------|------|-------|-------------|
| **ColabFold (AlphaFold2)** | Oracle | ~50s/seq (MSA server rate-limited) | Gold standard, uses MSA |
| **ESMFold** | Proxy | ~1-2s/seq | Single-sequence, no MSA |
| **SimpleFold** | Proxy | ~7s/seq (500 diffusion steps) | Diffusion-based folding |

### Best-of-N Protocol
1. From pool of N_total sequences, randomly sample N sequences (N = 10, 20, ..., 100)
2. Select best sequence by proxy pLDDT (ESMFold or SimpleFold)
3. Report the **ColabFold pLDDT** of the proxy-selected sequence
4. Compare against: (a) best-by-ColabFold (upper bound), (b) random selection (baseline)
5. Repeat 500 trials per N, report mean ± std

---

## Data Completion Status

| Model | batch1 (100) | batch2 (900) | batch3 (2000) | batch4 (7000) |
|-------|:---:|:---:|:---:|:---:|
| ESMFold | 100 | 900 | 2,000 | **0** (not run) |
| ColabFold | 100 | 900 | 2,000 | **2,166/7,000** (in progress) |
| SimpleFold | 100 | 900 | 2,000 | **0** (not run) |

- **ColabFold batch4**: 2,166 completed out of 7,000. Job running via public MSA server (~588 seq / 48h). Estimated ~10 more days.
- **ESMFold batch4**: Not yet run. Can be completed in ~1-2 hours on 1 GPU.
- **SimpleFold batch4**: Not yet run. ~3-4 hours on 4 GPUs.

---

## Key Results (3,000 samples: batch1+2+3)

### Proxy-Oracle Correlations
| Proxy → Oracle | Pearson Correlation |
|----------------|:---:|
| ESMFold → ColabFold | **0.669** |
| SimpleFold → ColabFold | **0.100** |

### Population Statistics
| Model | Mean pLDDT |
|-------|:---:|
| ESMFold | 42.56 |
| ColabFold | 48.83 |
| SimpleFold | 38.11 |

### Best-of-N Reward Hacking Gap (Y-axis = ColabFold pLDDT)

| N | ESMFold selector (CF) | CF best (upper bound) | ESM gap | SimpleFold selector (CF) | SF gap |
|---|:---:|:---:|:---:|:---:|:---:|
| 10 | 53.88 | 57.36 | 3.47 | 49.74 | 7.62 |
| 50 | 57.92 | 63.57 | 5.65 | 49.00 | 14.57 |
| 100 | 60.47 | 66.91 | **6.44** | 47.59 | **19.31** |

- **ESMFold as selector**: Gap@100 = 6.44. Proxy selection improves oracle value well above random (~49). Effective proxy.
- **SimpleFold as selector**: Gap@100 = 19.31. Proxy selection makes oracle value **worse than random**. Complete reward hacking.

### Key Finding: Batch3 ESMFold Outliers
- 5 sequences where ESMFold drastically overestimates (gap > 15): e.g., ESM=74.8 vs CF=46.5
- These are low-complexity / repetitive sequences
- After filtering ESM ≤ 52, batch differences disappear
- Does NOT affect overall conclusion: ESMFold is still a good proxy on average

---

## Generated Analysis Artifacts

### Main Plots (in `datasets/`)
| File | Description |
|------|-------------|
| `reward_hacking_combined_3000.png` | ESMFold vs SimpleFold as selector, Y=ColabFold pLDDT, ±1σ shading |
| `reward_hacking_esm_vs_sf_3000.png` | Side-by-side proxy vs oracle comparison |
| `reward_hacking_3000samples.png` | ESMFold-only reward hacking (3000 samples) |
| `reward_hacking_distributions_3000.png` | Distribution of selected CF pLDDT across 500 trials |
| `reward_hacking_single_trial_3000.png` | Single trial pool histogram with selections marked |
| `single_trials/seed_0.png` ~ `seed_9.png` | 10 individual trial histograms |
| `reward_hacking_10trials_esm.png` | 10 trials × 10 N values for ESMFold |
| `reward_hacking_10trials_sf.png` | 10 trials × 10 N values for SimpleFold |
| `reward_hacking_1000_vs_3000.png` | 1000 vs 3000 sample comparison |
| `reward_hacking_batch_comparison.png` | Batch-level analysis |
| `esm_vs_cf_scatter.png` | ESMFold vs ColabFold scatter |
| `top20_scatter.png` | Top 20% correlation analysis |

### JSON Data (in `datasets/`)
| File | Description |
|------|-------------|
| `reward_hacking_combined_3000.json` | Full numerical results for the main analysis |
| `reward_hacking_esm_vs_sf_3000.json` | ESM vs SF comparison data |
| `reward_hacking_3000samples.json` | 3000 sample results |
| `reward_hacking_1000samples.json` | 1000 sample results |

---

## Data Locations (Harvard FASRC Cluster)

### Shared Storage (`/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000/`)
```
fasta_split/           # Individual .fasta per sequence (10,000 files)
structures/            # Generated PDB structures
sequences.fasta        # All sequences combined
esmfold/
  plddt_results.json   # 3,000 entries (batch1-3)
colabfold/
  plddt_results.json   # 3,588 entries (batch1-3 + 588 batch4)
colabfold_batch4/      # Raw ColabFold outputs (2,166 completed)
simplefold/
  plddt_results.json   # 3,000 entries (batch1-3)
batch4_partA.fasta     # 3,499 sequences (for ColabFold job A)
batch4_partB.fasta     # 3,501 sequences (for ColabFold job B)
batch4_partA_remaining.fasta  # 2,911 unprocessed from part A
```

### Project Repo (`/n/home07/sangwonjung/ProDifEvo-Refinement/`)
```
datasets/              # Symlink → shared storage + analysis outputs (PNG/JSON)
scripts/slurm/         # All SLURM job scripts
scripts/               # Analysis scripts
config/                # EvoDiff model configs
topk_hedge/            # Main Python package
reference/             # La Proteina, SimpleFold repos (symlink → shared)
```

### Conda Environment
- Path: `/n/holylabs/LABS/calmon_lab/Lab/envs/topk_hedge/`
- Python 3.10, torch 2.2.0+cu118

---

## Pending / TODO

1. **ColabFold batch4**: 4,834 sequences remaining (~10 days on public MSA server)
   - Alternative: Setup local MMseqs2 DB (~2.5TB) for faster processing
2. **ESMFold batch4**: 7,000 sequences, ~1-2 hours on GPU
3. **SimpleFold batch4**: 7,000 sequences, ~3-4 hours on 4 GPUs
4. **10,000 sample analysis**: Re-run reward hacking analysis with full 10K pool
5. **Learned proxy**: Compare `model/reward_model_ckpt.pth` as a fourth proxy

---

## Scripts Reference

### Generation
- `scripts/slurm/05_generate_7000_more.sh` — La Proteina 7000 samples (batch4)

### Scoring
- `scripts/slurm/pipeline_2000_samples.sh` — ESMFold + ColabFold pipeline (batch3)
- `scripts/slurm/colabfold_batch4_A.sh` / `_B.sh` — ColabFold batch4 (split)
- `scripts/slurm/colabfold_batch4_A_continue.sh` — ColabFold batch4 A remaining
- `scripts/slurm/colabfold_batch4_merge.sh` — Merge ColabFold batch4 results
- `scripts/slurm/simplefold_batch3.sh` — SimpleFold batch3
- `scripts/slurm/sf_batch3_aa.sh` ~ `_ad.sh` — SimpleFold batch3 parallel jobs

### Analysis
- `scripts/reward_hacking_3000.py` — Main 3000-sample analysis
- `scripts/extract_simplefold_batch3_plddt.py` — CIF → pLDDT extraction
