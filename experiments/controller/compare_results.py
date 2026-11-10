import json
from collections import defaultdict

path = "experiments/controller/controller_results_full_system_backup.jsonl"

with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

initial = [r["gold_recall_initial"] for r in rows]
final = [r["gold_recall_final"] for r in rows]

improved = sum(b > a for a, b in zip(initial, final))
unchanged = sum(b == a for a, b in zip(initial, final))
decreased = sum(b < a for a, b in zip(initial, final))

print("Questions:", len(rows))
print(f"Initial average recall: {sum(initial)/len(initial):.4f}")
print(f"Final average recall:   {sum(final)/len(final):.4f}")
print(f"Improvement:            {(sum(final)-sum(initial))/len(rows):.4f}")
print("Improved:", improved)
print("Unchanged:", unchanged)
print("Decreased:", decreased)

domains = defaultdict(list)
for r in rows:
    domains[r["task"]].append(r)

print("\nPer-domain results:")
for task, group in sorted(domains.items()):
    before = sum(x["gold_recall_initial"] for x in group) / len(group)
    after = sum(x["gold_recall_final"] for x in group) / len(group)
    print(f"{task:20s} n={len(group):3d} "
          f"initial={before:.4f} final={after:.4f} "
          f"gain={after-before:+.4f}")
