# TopK-Hedge (Protein Design): Reward Hacking + Top-K-from-N Tuning

This repo implements **reward-hacking diagnostics** and a **HedgeTune-style tuner** for a protein-design pipeline where we must select **Top-K candidates from N generated designs** under a **slow/expensive “true reward”** (e.g., AlphaFold2/3 pLDDT) and a **fast proxy reward** (e.g., a lightweight pLDDT predictor).

> **Core question:** If we optimize/select using a fast proxy reward, do we get **reward hacking**—i.e., selections that look good under the proxy but are systematically worse under the true oracle?  
> **Core algorithmic goal:** Given a fixed downstream budget **K** (how many designs we can afford to evaluate with the true oracle / wet lab), how large should we set **N** (how many candidates to generate + score with the proxy) to maximize the quality of the final **Top-K** under the true oracle?

---

## 1) Problem Setting

### Pipeline (standard two-stage filtering)
We consider protein design as a two-stage selection problem:

1. **Generator** produces candidates:
   - A sequence-structure co-design generator outputs `x_i = (seq_i, struct_i)` for `i = 1..N`.
2. **Fast proxy reward** scores all N candidates quickly:
   - `\hat r_i = \hat R(x_i)` (e.g., a fast pLDDT estimator).
3. **Selection** chooses **K** candidates based on proxy:
   - `S_proxy = TopK(\hat r_1.. \hat r_N, K)`
4. **Slow true oracle** evaluates only the selected K (in deployment):
   - `r_i = R(x_i)` for `i in S_proxy` (e.g., AF2/AF3 pLDDT with MSA).

> In *analysis mode* (for research/plots), we may also compute `r_i` for **all** `i=1..N` so we can quantify reward hacking precisely.

### Why Top-K-from-N (not Best-of-N)
In LLM Best-of-N, we often output **one** best answer (K=1).  
In proteins, we typically want **K candidates** for expensive validation (AF2/AF3 or wet lab). So we study **Top-K-from-N**, where K is dictated by compute or lab budget.

---

## 2) Definitions and Metrics

### Rewards
- Proxy reward: `\hat R(x)` (fast, imperfect)
- True reward: `R(x)` (slow, trusted oracle; may still be a proxy for wet lab)

### Reward hacking (operational definition)
Reward hacking is present if **selecting by proxy** consistently yields worse outcomes under the true oracle than an appropriate baseline.

We quantify it using:

**(A) Rank correlation**
- Spearman: `corr(rank(\hat r), rank(r))`
- Kendall-τ

**(B) Top-K overlap**
Let `TopK_true` be indices of top-K by true reward among all N.
- Overlap@K: `|TopK_proxy ∩ TopK_true| / K`

**(C) Regret under true reward**
- Mean true reward of selected set:
  - `μ_true(TopK_proxy) = (1/K) * Σ_{i in TopK_proxy} r_i`
- Oracle upper bound (in analysis mode):
  - `μ_true(TopK_true)`
- Regret:
  - `Regret = μ_true(TopK_true) - μ_true(TopK_proxy)`

**(D) “Proxy extremal” effect**
As N increases, proxy selection may overfit proxy quirks. Track:
- `μ_true(TopK_proxy(N))` as a function of `N`
- Does it plateau or **drop** despite proxy scores increasing?

---

## 3) Research Objective: Choosing N given K

We treat K as fixed budget (how many we can afford to validate with the true oracle).  
We want to choose **N** (how many candidates to generate + score with proxy) to maximize downstream success.

** Objective : Expected true quality of selected set): **
- Choose `N` to maximize `E[ μ_true(TopK_proxy(N)) ]`

This repo includes utilities to sweep N and compare objectives.

---

## 4) What We Implement (v0 Scope)

### v0 experiments (reward hacking diagnostics)
- Pick a generator (initially: plug-in interface; start with a minimal generator or an existing open-source generator).
- Compute proxy reward (SimpleFold-like pLDDT predictor, or another fast proxy).
- Compute true reward (ColabFold AF2; optional AF3 later).
- Run sweeps over:
  - `N ∈ {N1, N2, ...}`
  - `K ∈ {K1, K2, ...}`
  - length buckets (e.g., 50..300)
- Produce plots/tables:
  - rank correlations
  - overlap@K
  - regret curves vs N
  - `μ_true(TopK_proxy)` vs N, with confidence intervals over seeds

### v0 algorithm (HedgeTune-style, adapted)
- Implement an “N tuner” that returns `N*` given K and proxy-vs-true statistics.
- Provide both:
  - **Offline tuner**: uses a precomputed dataset of (x, \hat r, r) to estimate `N*`
  - **Online-ish tuner**: progressively increases N and stops when marginal gains saturate (budget-aware)

> Later (v1+): integrate guidance/hedging during generation instead of only post-filtering.

---

## 5) Models / Options (Pluggable)

We keep the code modular. You can swap:

### Generator (sequence-structure co-design)
- Option A: **La Proteina** (NVIDIA)  
  - Paper: `https://arxiv.org/pdf/2507.09466`  
  - Code: `https://github.com/NVIDIA-Digital-Bio/la-proteina`
- Option B: **MultiFlow**  
  - Code: `https://github.com/jasonkyuyim/multiflow`
- Option C: any generator that outputs `(sequence, structure)`.

### Proxy reward (fast)
- SimpleFold-style pLDDT predictor (sequence+structure → predicted pLDDT)
  - Code: `https://github.com/apple/ml-simplefold`
- Alternative proxies (optional): fast structure predictors (ESMFold, etc.)  
  - Note: sequence-only predictors require extra checks (structure consistency vs designed structure).

### True oracle (slow)
- AF2 via ColabFold:
  - Code: `https://github.com/sokrypton/ColabFold`
- AF3 (license needed for weights; optional later)

---

## 6) Repo Layout (proposed)
