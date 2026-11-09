"""Summarise experiments/controller/controller_results.jsonl per domain.
   python -m controller.analyze
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

path = Path("experiments/controller/controller_results.jsonl")
rows = [json.loads(l) for l in path.open(encoding="utf-8")]
by = defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
by["ALL"] = rows

mean = lambda v: sum(v) / max(len(v), 1)
print(f"{'domain':20s} {'n':>4s} {'rounds':>6s} {'cov0':>6s} {'covN':>6s} "
      f"{'rec0':>6s} {'recN':>6s} {'docs':>6s} {'%no-retr':>8s}")
for task, rs in by.items():
    print(f"{task:20s} {len(rs):4d} {mean([r['rounds_used'] for r in rs]):6.2f} "
          f"{mean([r['initial_mean_coverage'] for r in rs]):6.3f} "
          f"{mean([r['final_mean_coverage'] for r in rs]):6.3f} "
          f"{mean([r['gold_recall_initial'] for r in rs]):6.3f} "
          f"{mean([r['gold_recall_final'] for r in rs]):6.3f} "
          f"{mean([r['n_docs'] for r in rs]):6.1f} "
          f"{100 * mean([r['rounds_used'] == 0 for r in rs]):7.1f}%")

print("\nStop reasons (ALL):", dict(Counter(r["stop_reason"] for r in rows)))