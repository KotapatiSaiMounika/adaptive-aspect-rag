"""Member 4 - Aspect scoring + adaptive retrieval controller.

Usage per question:
    ctrl = AdaptiveController(aspects, ControllerConfig(), LexicalScorer())
    d = ctrl.decide(evidence)            # after the initial retrieval
    while d.action == "retrieve":
        new_docs = member3_retrieve(d.weakest_aspect, d.hints)
        evidence = merge(evidence, new_docs)
        d = ctrl.decide(evidence, reflections=..., verification=...)
    # d.action == "stop"; d.stop_reason says why; ctrl.summary() goes to Member 5
"""
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from controller.config import ControllerConfig
from controller.scoring import AspectScorer, LexicalScorer, normalize_aspects, tokenize


def estimate_tokens(text: str) -> int:
    return int(len((text or "").split()) * 1.3)


@dataclass
class Decision:
    action: str                          # "retrieve" | "stop"
    weakest_aspect: Optional[str]        # aspect id to target next (None on stop)
    weakest_aspect_text: Optional[str]
    stop_reason: Optional[str]           # sufficient | max_rounds | doc_budget | token_budget | stagnation | all_aspects_exhausted
    reason: str                          # human-readable explanation
    coverage: Dict[str, float]           # per-aspect coverage
    mean_coverage: float
    uncovered: List[str]                 # aspect ids below coverage_target
    hints: Dict = field(default_factory=dict)   # for Member 3: missing_terms, top_k, ...
    budget: Dict = field(default_factory=dict)  # rounds/docs/tokens used vs limit

    def to_dict(self):
        return asdict(self)


class AdaptiveController:
    def __init__(self, aspects, config: Optional[ControllerConfig] = None,
                 scorer: Optional[AspectScorer] = None):
        self.cfg = config or ControllerConfig()
        self.scorer = scorer or LexicalScorer()
        self.aspects = normalize_aspects(aspects)
        if not self.aspects:
            raise ValueError("Controller needs at least one aspect (from Member 2 or the dataset).")
        self._text = {a["id"]: a["text"] for a in self.aspects}
        self._cache: Dict[tuple, float] = {}     # (aspect_id, doc_id) -> content score
        self.rounds_used = 0                     # retrieval rounds ordered so far
        self.attempts: Dict[str, int] = {a["id"]: 0 for a in self.aspects}
        self.history: List[Dict] = []            # one snapshot per decide() call
        self._pending_target: Optional[str] = None
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------ scoring
    def score_matrix(self, evidence: List[Dict], reflections: Optional[Dict] = None) -> Dict[str, Dict[str, float]]:
        """{aspect_id: {doc_id: score}}. Blends the scorer with Member 2's reflection.
        reflections format: {aspect_id: {doc_id: bool|float in [0,1]}} (missing = no opinion)."""
        w = self.cfg.reflection_weight if reflections else 0.0
        mat: Dict[str, Dict[str, float]] = {}
        for aid in self._text:
            row = {}
            for d in evidence:
                key = (aid, d["doc_id"])
                if key not in self._cache:
                    self._cache[key] = float(self.scorer.score(self._text[aid], d["text"]))
                s = self._cache[key]
                r = (reflections or {}).get(aid, {}).get(d["doc_id"])
                if r is not None and w > 0:
                    s = (1 - w) * s + w * float(r)
                row[d["doc_id"]] = min(1.0, max(0.0, s))
            mat[aid] = row
        return mat

    def coverage(self, matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Noisy-OR over the best m scores: several decent docs add up, one great doc is enough."""
        cov = {}
        for aid, row in matrix.items():
            best = sorted(row.values(), reverse=True)[: self.cfg.top_m_docs]
            miss = 1.0
            for s in best:
                miss *= (1.0 - s)
            cov[aid] = round(1.0 - miss, 4)
        return cov

    # ------------------------------------------------------------ decision
    def decide(self, evidence: List[Dict], reflections: Optional[Dict] = None,
               verification: Optional[Dict] = None) -> Decision:
        """verification (from Member 5/2): {'supported': bool, 'unsupported_aspects': [aspect_ids]}.
        If the answer is not supported, those aspects are forced to be treated as uncovered."""
        cfg = self.cfg
        matrix = self.score_matrix(evidence, reflections)
        cov = self.coverage(matrix)

        flagged = []
        if verification and not verification.get("supported", True):
            flagged = [a for a in verification.get("unsupported_aspects", []) if a in cov]
            if not flagged:  # unsupported answer but no aspect named -> suspect the weakest one
                flagged = [min(cov, key=cov.get)]
            for a in flagged:
                cov[a] = min(cov[a], cfg.coverage_target - 0.01)

        mean_cov = round(sum(cov.values()) / len(cov), 4)
        uncovered = [a for a, c in cov.items() if c < cfg.coverage_target]
        n_docs = len(evidence)
        n_tokens = sum(estimate_tokens(d["text"]) for d in evidence)
        budget = {"rounds_used": self.rounds_used, "max_rounds": cfg.max_rounds,
                  "docs": n_docs, "max_docs": cfg.max_docs,
                  "tokens": n_tokens, "max_tokens": cfg.max_tokens}

        # did the previous targeted round help the aspect it targeted?
        if self._pending_target and self.history:
            prev = self.history[-1]["coverage"].get(self._pending_target, 0.0)
            if cov[self._pending_target] - prev < cfg.min_gain:
                self.attempts[self._pending_target] += 1
        self._pending_target = None

        stop_reason = None
        if not uncovered:
            stop_reason, why = "sufficient", "All aspects reached the coverage target."
        elif self.rounds_used >= cfg.max_rounds:
            stop_reason, why = "max_rounds", f"Reached max retrieval rounds ({cfg.max_rounds})."
        elif n_docs >= cfg.max_docs:
            stop_reason, why = "doc_budget", f"Evidence set has {n_docs} docs (limit {cfg.max_docs})."
        elif n_tokens >= cfg.max_tokens:
            stop_reason, why = "token_budget", f"Evidence ~{n_tokens} tokens (limit {cfg.max_tokens})."
        elif self._stagnated(mean_cov):
            stop_reason, why = "stagnation", f"Mean coverage gain < {cfg.min_gain} for {cfg.patience} round(s)."

        target = None
        if stop_reason is None:
            open_aspects = [a for a in uncovered if self.attempts[a] < cfg.max_attempts_per_aspect]
            if not open_aspects:
                stop_reason, why = "all_aspects_exhausted", "Every uncovered aspect already failed repeated targeted searches."
            else:
                # weakest = lowest coverage; tie -> fewest supporting docs
                n_support = {a: sum(s >= cfg.support_threshold for s in matrix[a].values()) for a in open_aspects}
                target = min(open_aspects, key=lambda a: (cov[a], n_support[a]))
                why = (f"Aspect '{target}' is weakest (coverage {cov[target]:.2f} < {cfg.coverage_target}); "
                       f"{len(uncovered)} aspect(s) uncovered."
                       + (" Triggered by answer verification." if target in flagged else ""))

        hints = {}
        if target:
            seen = set()
            for d in evidence:
                seen.update(tokenize(d["text"]))
            hints = {"missing_terms": [t for t in dict.fromkeys(tokenize(self._text[target])) if t not in seen],
                     "top_k": cfg.top_k_per_query, "verification_triggered": target in flagged,
                     "exclude_doc_ids": [d["doc_id"] for d in evidence]}
            self.rounds_used += 1
            self._pending_target = target
            budget["rounds_used"] = self.rounds_used

        decision = Decision(
            action="retrieve" if target else "stop",
            weakest_aspect=target, weakest_aspect_text=self._text.get(target) if target else None,
            stop_reason=stop_reason, reason=why, coverage=cov, mean_coverage=mean_cov,
            uncovered=uncovered, hints=hints, budget=budget)
        self.history.append({"round": len(self.history), "coverage": cov, "mean_coverage": mean_cov,
                             "n_docs": n_docs, "n_tokens": n_tokens, "action": decision.action,
                             "target": target, "stop_reason": stop_reason,
                             "elapsed_sec": round(time.perf_counter() - self._t0, 4)})
        return decision

    def _stagnated(self, mean_cov: float) -> bool:
        """Last `patience` retrieval rounds each improved mean coverage by < min_gain."""
        p = self.cfg.patience
        if self.rounds_used < p or len(self.history) < p:
            return False
        means = [h["mean_coverage"] for h in self.history] + [mean_cov]
        recent = means[-(p + 1):]
        return all(recent[i + 1] - recent[i] < self.cfg.min_gain for i in range(p))

    # ------------------------------------------------------------ output for Member 5
    def summary(self) -> Dict:
        last = self.history[-1] if self.history else {}
        first = self.history[0] if self.history else {}
        return {"rounds_used": self.rounds_used,
                "initial_mean_coverage": first.get("mean_coverage"),
                "final_mean_coverage": last.get("mean_coverage"),
                "final_coverage": last.get("coverage"),
                "stop_reason": last.get("stop_reason"),
                "n_docs": last.get("n_docs"), "n_tokens": last.get("n_tokens"),
                "coverage_trajectory": [h["mean_coverage"] for h in self.history],
                "attempts_per_aspect": self.attempts,
                "controller_latency_sec": last.get("elapsed_sec")}
