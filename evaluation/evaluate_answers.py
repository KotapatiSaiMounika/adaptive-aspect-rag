import argparse, json, os, time
from data.dataset_loader import load_questions
from generation import llm
from generation.claim_checker import judge_answer
from generation.llm import require_config

CACHE = "experiments/generation/judge_cache.jsonl"


def mean(x):
    return round(sum(x) / len(x), 3) if x else "n/a"


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                cache[r["key"]] = r["judgement"]
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="experiments/generation/summary.md")
    ap.add_argument("--pause", type=float, default=7.0, help="seconds between judge calls")
    a = ap.parse_args()
    judge = os.getenv("JUDGE_MODEL") or None
    require_config(models=[judge or os.environ.get("LLM_MODEL", "")])
    judge_name = judge or os.environ["LLM_MODEL"]

    data = {fp: [json.loads(l) for l in open(fp) if l.strip()] for fp in a.files}
    common = set.intersection(*[{str(r["qid"]) for r in recs} for recs in data.values()])
    print(f"Scoring {len(common)} questions that every file contains (judge: {judge_name}).")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    cache = load_cache(CACHE)
    rows = []
    with open(CACHE, "a", encoding="utf-8") as cf:
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
                if not asp:
                    continue
                key = f"{judge_name}|{r['system']}|{r['task']}|{r['qid']}"
                j = cache.get(key)
                if j is None:
                    try:
                        j = judge_answer(r["query"], r["answer"], asp, model=judge)
                    except Exception as e:
                        print(f"\nStopped: {type(e).__name__}: {llm.error_text(e)[:400]}")
                        print(f"{len(cache)} judgements are saved in {CACHE}.")
                        print("Run the same command again later (finished ones are skipped), "
                              "or set JUDGE_MODEL in .env to a different model.")
                        raise SystemExit(1)
                    cache[key] = j
                    cf.write(json.dumps({"key": key, "judgement": j}) + "\n")
                    cf.flush()
                    print(f"  judged {r['system']} q{r['qid']}  ({len(cache)} saved)", flush=True)
                    time.sleep(a.pause)
                util.append(j["utility"])
                if j["covered_frac"] is not None:
                    cov.append(j["covered_frac"])
            rows.append((recs[0]["system"], recs[0]["task"], len(recs), mean(sup),
                         mean(util), mean(cov), mean(calls), f"{abst}/{len(recs)}"))
    head = ("| system | task | n | support rate | utility (1-5) | gold-aspect coverage "
            "| retrieval calls | abstained |\n|---|---|---|---|---|---|---|---|\n")
    table = head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
    print(table)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
    