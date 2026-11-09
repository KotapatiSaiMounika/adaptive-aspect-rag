"""All tunable knobs of Member 4's controller (use these for ablations)."""
from dataclasses import dataclass


@dataclass
class ControllerConfig:
    # --- coverage ---
    support_threshold: float = 0.38   # doc "supports" an aspect if score >= this
    coverage_target: float = 0.5     # aspect counts as covered if coverage >= this
    top_m_docs: int = 1             # coverage = noisy-OR over the best m doc scores
    reflection_weight: float = 0.4   # blend of Member 2's reflection into the score (0 = ignore)
    # --- budgets ---
    max_rounds: int = 3              # max ADDITIONAL retrieval rounds
    max_docs: int = 50               # max documents in the evidence set
    max_tokens: int = 20000          # approx. token budget of the evidence set
    # --- stagnation / repetition ---
    min_gain: float = 0.02           # min improvement of mean coverage per round
    patience: int = 1                # stop after this many consecutive low-gain rounds
    max_attempts_per_aspect: int = 2 # give up on an aspect after this many failed targeted rounds
    # --- hints for Member 3 ---
    top_k_per_query: int = 10
