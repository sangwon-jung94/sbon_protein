# ProDifEvo-Refinement

## Project Overview
Reward hacking diagnostics and Top-K-from-N tuning for protein design.
Compares proxy reward (fast, cheap) vs oracle reward (slow, expensive) when selecting top-K candidates from N generated proteins.

## Tech Stack
- Python 3.9, PyTorch
- Protein generators: EvoDiff (discrete diffusion), La Proteina (NVIDIA latent diffusion)
- Folding oracles: ESMFold, ColabFold (AF2), SimpleFold
- Proxy: learned pLDDT predictor (`model/reward_model_ckpt.pth`)
- Configs: JSON files in `config/` (38M, 640M, MSA, MSA-600M EvoDiff variants)
- Infrastructure: SLURM (Harvard FAS RC cluster), W&B for logging

## Project Structure
```
topk_hedge/          # Main package
  generators/        # EvoDiff, La Proteina wrappers
  rewards/
    proxy/           # Learned proxy reward
    oracle/          # ESMFold oracle
  selection/         # Top-K selector
  metrics/           # Spearman, Kendall, overlap@k, regret
  experiments/       # Sweep runner, analysis
  data/              # Candidate / CandidateBatch dataclasses
  utils/
config/              # EvoDiff model configs (JSON)
scripts/             # Entrypoint scripts & SLURM jobs
  slurm/             # SLURM batch scripts
datasets/            # Generated data, reward hacking results (JSON/PNG)
artifacts/           # Model checkpoints (SimpleFold, pLDDT)
model/               # Reward model checkpoint
reference/           # Reference sequences (soybean proteome)
```

## Key Commands
```bash
# Demo pipeline (generate → score → metrics)
python scripts/run_demo.py

# SLURM jobs are in scripts/slurm/
# Pattern: generate → fold (ESMFold/ColabFold/SimpleFold) → analyze
```

## Conventions
- All generators inherit from `topk_hedge.generators.base.Generator`
- All reward functions inherit from `topk_hedge.rewards.base.RewardFunction`
- `Candidate` and `CandidateBatch` are the core data types (`topk_hedge.data.candidate`)
- Metrics return floats; higher = better for correlation/overlap, lower = better for regret
- Language: code and comments in English, conversation in Korean is OK
