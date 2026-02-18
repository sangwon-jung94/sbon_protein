#!/usr/bin/env python
"""
Test script for La Proteina generation.

Runs a minimal generation to verify the setup works correctly.
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

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Check checkpoints
ckpt_dir = os.path.join(LAPROTEINA_PATH, "checkpoints_laproteina")
ld1_path = os.path.join(ckpt_dir, "LD1_ucond_notri_512.ckpt")
ae1_path = os.path.join(ckpt_dir, "AE1_ucond_512.ckpt")

print(f"\nCheckpoint LD1 exists: {os.path.exists(ld1_path)}")
print(f"Checkpoint AE1 exists: {os.path.exists(ae1_path)}")

# Try importing la-proteina modules
try:
    from proteinfoundation.proteina import Proteina
    from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
    print("\nLa Proteina modules imported successfully!")
except ImportError as e:
    print(f"\nFailed to import La Proteina modules: {e}")
    sys.exit(1)

# Try loading config
try:
    config_path = os.path.join(LAPROTEINA_PATH, "configs")
    with hydra.initialize_config_dir(config_dir=config_path, version_base=hydra.__version__):
        cfg = hydra.compose(config_name="inference_ucond_notri")
    print(f"\nConfig loaded successfully!")
    print(f"Run name: {cfg.get('run_name_', 'N/A')}")
    print(f"Checkpoint: {cfg.get('ckpt_name', 'N/A')}")
except Exception as e:
    print(f"\nFailed to load config: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("La Proteina setup verification complete!")
print("="*50)
