#!/usr/bin/env python3
"""
Reward hacking analysis using 3000 samples.
Proxy: ESMFold pLDDT
Oracle: ColabFold pLDDT
"""
import json
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "/n/holylabs/LABS/calmon_lab/Lab/datasets/sangwonjung/laproteina_1000"
OUT_DIR = "/n/home07/sangwonjung/ProDifEvo-Refinement/datasets"

# Load data
with open(f"{DATA_DIR}/esmfold/plddt_results.json") as f:
    esm_data = json.load(f)
with open(f"{DATA_DIR}/colabfold/plddt_results.json") as f:
    cf_data = json.load(f)

# Match keys
common_keys = sorted(set(esm_data.keys()) & set(cf_data.keys()))
print(f"Common samples: {len(common_keys)}")

esm_plddt = np.array([esm_data[k]["mean_plddt"] for k in common_keys])
cf_plddt = np.array([cf_data[k]["mean_plddt"] for k in common_keys])

print(f"ESMFold mean: {esm_plddt.mean():.2f}, std: {esm_plddt.std():.2f}")
print(f"ColabFold mean: {cf_plddt.mean():.2f}, std: {cf_plddt.std():.2f}")
print(f"Correlation: {np.corrcoef(esm_plddt, cf_plddt)[0,1]:.4f}")

# Best-of-n analysis
ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_trials = 200
seed = 42
rng = np.random.default_rng(seed)

N_total = len(common_keys)
results = {}

for n in ns:
    esm_selected_cf = []
    oracle_cf = []

    for _ in range(n_trials):
        idx = rng.choice(N_total, size=n, replace=False)

        # Proxy selects: best ESMFold pLDDT
        best_by_esm = idx[np.argmax(esm_plddt[idx])]
        esm_selected_cf.append(cf_plddt[best_by_esm])

        # Oracle selects: best ColabFold pLDDT
        best_by_cf = idx[np.argmax(cf_plddt[idx])]
        oracle_cf.append(cf_plddt[best_by_cf])

    esm_selected_cf = np.array(esm_selected_cf)
    oracle_cf = np.array(oracle_cf)
    gap = oracle_cf - esm_selected_cf

    results[str(n)] = {
        "esm_selected_cf_mean": float(esm_selected_cf.mean()),
        "esm_selected_cf_std": float(esm_selected_cf.std()),
        "oracle_cf_mean": float(oracle_cf.mean()),
        "oracle_cf_std": float(oracle_cf.std()),
        "gap_mean": float(gap.mean()),
        "gap_std": float(gap.std()),
    }
    print(f"N={n:3d}  proxy_sel={esm_selected_cf.mean():.2f}  oracle={oracle_cf.mean():.2f}  gap={gap.mean():.2f}")

# Save JSON
out_json = f"{OUT_DIR}/reward_hacking_3000samples.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

x = np.array(ns)
proxy_mean = np.array([results[str(n)]["esm_selected_cf_mean"] for n in ns])
proxy_std = np.array([results[str(n)]["esm_selected_cf_std"] for n in ns])
oracle_mean = np.array([results[str(n)]["oracle_cf_mean"] for n in ns])
oracle_std = np.array([results[str(n)]["oracle_cf_std"] for n in ns])
gap_mean = np.array([results[str(n)]["gap_mean"] for n in ns])

ax.errorbar(x, proxy_mean, yerr=proxy_std, marker="o", capsize=3,
            label="ESMFold-selected (proxy)", color="tab:blue")
ax.errorbar(x, oracle_mean, yerr=oracle_std, marker="s", capsize=3,
            label="ColabFold-selected (oracle)", color="tab:orange")
ax.fill_between(x, proxy_mean, oracle_mean, alpha=0.15, color="red",
                label=f"Reward hacking gap")

ax.set_xlabel("N (pool size)", fontsize=12)
ax.set_ylabel("ColabFold pLDDT (true)", fontsize=12)
ax.set_title(f"Best-of-N Reward Hacking (3000 samples, {n_trials} trials)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

out_png = f"{OUT_DIR}/reward_hacking_3000samples.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Saved: {out_png}")
