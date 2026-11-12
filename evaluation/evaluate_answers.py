import argparse, json, os
from data.dataset_loader import load_questions
from generation.claim_checker import judge_answer


def mean(x):
    return round(sum(x) / len(x), 3) if x else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="experiments/generation/summary.md")
    a = ap.parse_args()
    judge = os.getenv("JUDGE_MODEL") or None
    data = {fp: [json.loads(l) for l in open(fp) if l.strip()] for fp in a.files}
    common = set.intersection(*[{str(r["qid"]) for r in recs} for recs in data.values()])
    print(f"Scoring {len(common)} questions that every file contains.")
    rows = []
    for fp, recs in data.items():
        recs = [r for r in recs if str(r["qid"]) in common]
        if not recs:
            continue
        gold = {str(q["qid"]): q.get("aspects") or [] for q in load_questions(recs[0]["task"])}
        sup, util, cov, calls, abst = [], [], [], [], 0
        for r in recs:
            calls.append(r.get("n_retrieval_calls", 1))
            if r["claims"]:
                sup.append(r["report"]["support_rate"])
            else:
                abst += 1
            asp = gold.get(str(r["qid"]))
            if asp:
                j = judge_answer(r["query"], r["answer"], asp, model=judge)
                util.append(j["utility"])
                if j["covered_frac"] is not None:
                    cov.append(j["covered_frac"])
        rows.append((recs[0]["system"], recs[0]["task"], len(recs), mean(sup),
                     mean(util), mean(cov), mean(calls), f"{abst}/{len(recs)}"))
    head = ("| system | task | n | support rate | utility (1-5) | gold-aspect coverage "
            "| retrieval calls | abstained |\n|---|---|---|---|---|---|---|---|\n")
    table = head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
    print(table)
    with open(a.out, "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
