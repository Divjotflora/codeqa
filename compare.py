"""Paired bootstrap: is retriever A really better than B, or is it noise?

Resamples eval questions with replacement (both retrievers scored on the same
resampled questions), so per-question difficulty cancels out. Several results
files (e.g. click + jinja) are pooled into one comparison.

  python run_eval.py ... --out results_click.json
  python run_eval.py ... --out results_jinja.json
  python compare.py results_click.json results_jinja.json --a hybrid+graph --b bm25_fixed
  python compare.py results_click.json results_jinja.json --a hybrid+graph --b bm25_ast --split implicit

Comparing runs saved in different files (e.g. two embedding models): give B's files
with --b-files, one per A file, in the same order.
  python compare.py dev_wz_coderank.json --b-files dev_wz_bge.json --a hybrid+graph --b hybrid+graph
"""
import argparse
import json
import random
from pathlib import Path

ALIASES = {
    "hybrid": lambda n: n.startswith("rrf(") and "+graph" not in n and "+rerank" not in n,
    "hybrid+graph": lambda n: n.startswith("rrf(") and "+graph" in n and "+rerank" not in n,
    "hybrid+graph+rerank": lambda n: n.startswith("rrf(") and "+rerank[" in n
        and n.split("+rerank[")[1].split(",")[1].rstrip("]") == "ce",
    "hybrid+graph+rerank_rrf": lambda n: n.startswith("rrf(") and "+rerank[" in n
        and n.split("+rerank[")[1].split(",")[1].rstrip("]") == "rrf",
}


def pick(results: list[dict], key: str) -> dict:
    exact = [r for r in results if r["retriever"] == key]
    if exact:
        return exact[0]
    if key in ALIASES:
        hits = [r for r in results if ALIASES[key](r["retriever"])]
    else:  # e.g. "bm25_ast+graph" matches "bm25_ast+graph(s=10,a=0.5)"
        hits = [r for r in results if r["retriever"].split("(")[0].split("[")[0] == key
                or r["retriever"].startswith(key + "(")]
    if len(hits) != 1:
        names = ", ".join(r["retriever"] for r in results)
        raise SystemExit(f"{key!r} matched {len(hits)} retrievers; available: {names}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--b-files", nargs="+", type=Path,
                    help="take B from these files instead (one per A file, same order)")
    ap.add_argument("--metrics", default="function/recall@5,function/recall@10,function/hit@10,function/mrr")
    ap.add_argument("--split", choices=["all", "implicit", "explicit"], default="all")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = {m: [] for m in args.metrics.split(",")}
    if args.b_files and len(args.b_files) != len(args.files):
        raise SystemExit("--b-files needs exactly one file per A file")
    b_files = args.b_files or args.files
    for f, fb in zip(args.files, b_files):
        results = json.loads(f.read_text())
        results_b = json.loads(fb.read_text()) if fb != f else results
        A = {r["id"]: r for r in pick(results, args.a)["per_example"]}
        B = {r["id"]: r for r in pick(results_b, args.b)["per_example"]}
        for qid in sorted(A.keys() & B.keys()):
            ra, rb = A[qid], B[qid]
            if args.split == "implicit" and ra["mentions_gold"]:
                continue
            if args.split == "explicit" and not ra["mentions_gold"]:
                continue
            for m in pairs:
                level, key = m.split("/")
                pairs[m].append((ra[level][key], rb[level][key]))

    n = len(next(iter(pairs.values())))
    rng = random.Random(args.seed)
    idx_sets = [[rng.randrange(n) for _ in range(n)] for _ in range(args.iters)]
    b_note = f"  (from {', '.join(f.name for f in args.b_files)})" if args.b_files else ""
    print(f"A = {args.a}\nB = {args.b}{b_note}\nsplit = {args.split}, questions = {n}, "
          f"files = {', '.join(f.name for f in args.files)}\n")
    print(f"| metric | A | B | A - B | 95% CI | p (two-sided) |\n|---|---|---|---|---|---|")
    for m, ps in pairs.items():
        diffs = [a - b for a, b in ps]
        mean_a, mean_b = sum(a for a, _ in ps) / n, sum(b for _, b in ps) / n
        boots = sorted(sum(diffs[i] for i in idx) / n for idx in idx_sets)
        lo, hi = boots[int(0.025 * args.iters)], boots[int(0.975 * args.iters) - 1]
        p = min(1.0, 2 * min(sum(d <= 0 for d in boots), sum(d >= 0 for d in boots)) / args.iters)
        print(f"| {m} | {mean_a:.3f} | {mean_b:.3f} | {mean_a - mean_b:+.3f} | "
              f"[{lo:+.3f}, {hi:+.3f}] | {p:.4f} |")


if __name__ == "__main__":
    main()
