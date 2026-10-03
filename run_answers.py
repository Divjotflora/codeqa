"""Evaluate answer generation on an eval set: citation validity and whether answers
point at the right function. Resumable: re-running skips questions already in --out.

  python run_answers.py --repo ../click --index index --evalset eval/click_issues.jsonl \
      --model qwen2.5-coder:3b --limit 40 --out answers_click.jsonl
"""
import argparse
import json
import time
from pathlib import Path

from codeqa.answer import PROMPT_VERSION, Repo, answer_question
from codeqa.evalset import load
from codeqa.llm import make_llm
from codeqa.pipeline import build_system


def norm(s: str) -> str:
    return s.split("@")[0]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    cites = [c for r in rows for c in r["citations"] if c.get("kind", "path") != "name"]
    by = lambda st: sum(c["status"] == st for c in cites)
    has_cite = lambda r: any(c.get("kind", "path") != "name" for c in r["citations"])
    in_ctx = [r for r in rows if r["gold_in_context"]]
    return {
        "n": n,
        "citations_per_answer": round(len(cites) / n, 2),
        "answers_with_citation": round(sum(has_cite(r) for r in rows) / n, 3),
        "answers_saying_not_found": round(sum(r.get("not_found", False) for r in rows) / n, 3),
        "citation_valid": round(by("valid") / max(len(cites), 1), 3),
        "citation_ungrounded": round(by("ungrounded") / max(len(cites), 1), 3),
        "citation_bad": round((by("bad_range") + by("bad_file") + by("bad_label")) / max(len(cites), 1), 3),
        # retrieval ceiling: was any gold function among the snippets shown?
        "gold_in_context": round(len(in_ctx) / n, 3),
        # strict: a verified citation (path or snippet label) lands on a gold function
        "gold_cited": round(sum(r["gold_cited"] for r in rows) / n, 3),
        "gold_cited_given_in_context": round(sum(r["gold_cited"] for r in in_ctx) / max(len(in_ctx), 1), 3),
        # lenient: the answer at least names a gold function that it was shown
        "gold_named_given_in_context": round(sum(r.get("gold_named", False) for r in in_ctx) / max(len(in_ctx), 1), 3),
        # is the model choosing, or citing everything it was shown?
        "distinct_cited_per_answer": round(sum(len(cited_set(r)) for r in rows) / n, 2),
        "shown_per_answer": round(sum(len(r["shown"]) for r in rows) / n, 2),
        "cited_precision": round(mean([len(cited_set(r) & set(r["gold"])) / len(cited_set(r))
                                       for r in rows if cited_set(r)]), 3),
        # trivial baselines that use no model at all, on the same snippets
        "baseline_all_shown_precision": round(mean([len({norm(s) for s in r["shown"]} & set(r["gold"]))
                                                    / max(len(r["shown"]), 1) for r in rows]), 3),
        "baseline_top1_gold_given_in_context": round(sum(norm(r["shown"][0]) in set(r["gold"])
                                                         for r in in_ctx if r["shown"]) / max(len(in_ctx), 1), 3),
        # fairest baseline: cite exactly as many functions as the model did, but simply
        # take them in retrieval order. Beating this means the model picks better than rank.
        "baseline_matched_gold_given_in_context": round(sum(matched_hit(r) for r in in_ctx)
                                                        / max(len(in_ctx), 1), 3),
        "baseline_matched_precision": round(mean([matched_prec(r) for r in rows if cited_set(r)]), 3),
        "seconds_per_answer": round(sum(r["seconds"] for r in rows) / n, 1),
    }


def matched_hit(r: dict) -> bool:
    k = len(cited_set(r))
    return any(norm(s) in set(r["gold"]) for s in r["shown"][:k])


def matched_prec(r: dict) -> float:
    k = len(cited_set(r))
    top = {norm(s) for s in r["shown"][:k]}
    return len(top & set(r["gold"])) / k


def cited_set(r: dict) -> set[str]:
    return {norm(c["symbol"]) for c in r["citations"]
            if c.get("kind", "path") != "name" and c["status"] == "valid" and c["symbol"]}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--evalset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai", "anthropic"])
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--base-url")
    ap.add_argument("--embedder", default="st:bge-small")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-lines", type=int, default=40)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    retriever, chunks = build_system(args.index, None if args.embedder == "none" else args.embedder)
    repo = Repo(args.repo.resolve(), chunks)
    known = {norm(s) for s in repo.span}
    llm = make_llm(args.backend, args.model, args.base_url)

    done = {}
    if args.out.exists():
        for l in open(args.out, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[r["id"]] = r

    todo = []
    for ex in load(args.evalset):
        gold = {norm(s) for s in ex["gold_symbols"]} & known
        if gold:
            todo.append((ex, gold))
    todo = todo[: args.limit] if args.limit else todo

    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        for i, (ex, gold) in enumerate(todo, 1):
            if ex["id"] in done:
                continue
            a = answer_question(ex["query"], retriever, repo, llm, args.k, args.max_lines)
            row = {
                "id": ex["id"], "mentions_gold": ex["mentions_gold"], "gold": sorted(gold),
                "answer": a.text,
                "citations": [{"raw": c.raw, "file": c.file, "start": c.start, "end": c.end,
                               "status": c.status, "symbol": c.symbol, "kind": c.kind}
                              for c in a.citations],
                "shown": [s.symbol for s in a.shown],
                "gold_in_context": any(norm(s.symbol) in gold for s in a.shown),
                "gold_cited": any(norm(s) in gold for s in a.valid_symbols),
                "gold_named": any(norm(s) in gold for s in a.named_symbols),
                "not_found": a.says_not_found,
                "prompt_version": PROMPT_VERSION,
                "seconds": round(a.seconds, 1), "tokens": list(a.tokens),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[ex["id"]] = row
            el = time.time() - t0
            cs = [c for c in a.citations if c.kind != "name"]
            print(f"[{i}/{len(todo)}] {ex['id']}: {len(cs)} citations, "
                  f"{sum(c.status == 'valid' for c in cs)} valid, "
                  f"gold cited={row['gold_cited']}  ({a.seconds:.0f}s)", flush=True)

    rows = [done[ex["id"]] for ex, _ in todo if ex["id"] in done]
    out = {"all": summarize(rows),
           "implicit": summarize([r for r in rows if not r["mentions_gold"]]),
           "explicit": summarize([r for r in rows if r["mentions_gold"]])}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
