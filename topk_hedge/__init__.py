"""
TopK-Hedge: Reward Hacking Diagnostics and Top-K-from-N Tuning for Protein Design

This package implements:
- Generators: Protein sequence/structure generators (EvoDiff, La Proteina, MultiFlow)
- Rewards: Proxy (fast) and Oracle (true) reward functions
- Selection: Top-K from N selection with HedgeTune-style N tuning
- Metrics: Reward hacking diagnostics (correlation, overlap, regret)
- Experiments: Sweep runners and analysis tools
"""

from topk_hedge.data.candidate import Candidate, CandidateBatch

__version__ = "0.1.0"
