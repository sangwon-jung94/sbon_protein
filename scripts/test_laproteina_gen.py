#!/usr/bin/env python
"""
Test script for La Proteina actual generation.
Generates a few short sequences to verify the full pipeline works.
"""

import os
import sys

# Add la-proteina to path
LAPROTEINA_PATH = "/n/home07/sangwonjung/ProDifEvo-Refinement/reference/la-proteina"
sys.path.insert(0, LAPROTEINA_PATH)
os.chdir(LAPROTEINA_PATH)

import torch
import hydra
from omegaconf import OmegaConf
import lightning as L

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Import La Proteina modules
from proteinfoundation.proteina import Proteina
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.datasets.gen_dataset import GenDataset
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb

print("\nLoading config...")
config_path = os.path.join(LAPROTEINA_PATH, "configs")
with hydra.initialize_config_dir(config_dir=config_path, version_base=hydra.__version__):
    cfg = hydra.compose(config_name="inference_ucond_notri")

# Modify config for minimal test
OmegaConf.set_struct(cfg, False)
cfg.generation.dataset.nlens_cfg.nres_lens = [100]  # Only generate length 100
cfg.generation.dataset.nsamples = 2  # Only 2 samples
cfg.generation.dataset.max_nsamples_per_batch = 2
OmegaConf.set_struct(cfg, True)

print(f"Config: {cfg.run_name_}")
print(f"Generating {cfg.generation.dataset.nsamples} samples of length {cfg.generation.dataset.nlens_cfg.nres_lens}")

# Set seed
L.seed_everything(42)

# Paths
ae_ckpt_path = os.path.join(LAPROTEINA_PATH, "checkpoints_laproteina/AE1_ucond_512.ckpt")
ld_ckpt_path = os.path.join(LAPROTEINA_PATH, "checkpoints_laproteina/LD1_ucond_notri_512.ckpt")

# Load autoencoder first
print("\nLoading autoencoder...")
autoencoder = AutoEncoder.load_from_checkpoint(ae_ckpt_path, map_location="cuda", strict=False)
autoencoder = autoencoder.eval().to("cuda")
print(f"Autoencoder loaded from {ae_ckpt_path}")

# Load diffusion model with autoencoder_ckpt_path override
print("\nLoading diffusion model...")
proteina = Proteina.load_from_checkpoint(
    ld_ckpt_path,
    map_location="cuda",
    strict=False,
    autoencoder_ckpt_path=ae_ckpt_path,  # Override the hardcoded path
)
proteina = proteina.eval().to("cuda")
print(f"Proteina loaded from {ld_ckpt_path}")

# Create dataset
print("\nCreating generation dataset...")
gen_dataset = GenDataset(
    nlens_cfg=cfg.generation.dataset.nlens_cfg,  # Keep as OmegaConf
    nsamples=cfg.generation.dataset.nsamples,
    max_nsamples_per_batch=cfg.generation.dataset.max_nsamples_per_batch,
)
print(f"Dataset size: {len(gen_dataset)}")

# Generate
print("\nGenerating samples...")
from torch.utils.data import DataLoader

dataloader = DataLoader(
    gen_dataset,
    batch_size=cfg.generation.dataset.max_nsamples_per_batch,
    shuffle=False,
)

output_dir = "/n/home07/sangwonjung/ProDifEvo-Refinement/outputs/laproteina_test"
os.makedirs(output_dir, exist_ok=True)

all_sequences = []

with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx + 1}...")

        # Move batch to device
        batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}

        # Generate using Proteina's sample method
        samples = proteina.sample(
            batch,
            autoencoder=autoencoder,
            nsteps=cfg.generation.args.nsteps,
            bb_ca_sample_args=cfg.generation.model.bb_ca,
            local_latents_sample_args=cfg.generation.model.local_latents,
            self_cond=cfg.generation.args.self_cond,
        )

        # Extract sequences
        for i, sample in enumerate(samples):
            seq = sample.get("sequence", "")
            all_sequences.append(seq)
            print(f"  Sample {i+1}: {seq[:50]}... (length {len(seq)})")

            # Save PDB if available
            if "atom37" in sample:
                pdb_path = os.path.join(output_dir, f"sample_{batch_idx}_{i}.pdb")
                write_prot_to_pdb(sample, pdb_path)
                print(f"    Saved to {pdb_path}")

print("\n" + "="*50)
print("La Proteina generation test complete!")
print(f"Generated {len(all_sequences)} sequences")
print("="*50)
