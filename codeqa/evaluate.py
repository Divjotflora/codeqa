"""Localization metrics for any retriever with `.retrieve(query, k) -> [symbol]`.

Function level: ranking with module symbols removed vs gold symbols.
Module level:   ranking mapped to each symbol's module (deduped) vs gold modules.

recall@k = mean fraction of gold items in the top k
hit@k    = share of examples with at least one gold item in the top k
mrr      = mean 1/rank of the first gold item (0 if none in the top 50)
"""
from __future__ import annotations

import statistics as st

from .chunker import Chunk

KS = (1, 5, 10, 20)


def _norm(s: str) -> str:
    return s.split("@")[0]   # drop duplicate-name suffixes, which shift between commits


def _module_of(sym: str, modules: set[str]) -> str | None:
    parts = sym.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in modules:
            return cand
    return None


def _score(ranking: list[str], gold: set[str]) -> dict:
    r = {}
    for k in KS:
        top = set(ranking[:k])
        r[f"recall@{k}"] = len(top & gold) / len(gold)
        r[f"hit@{k}"] = float(bool(top & gold))
    first = next((i for i, s in enumerate(ranking, 1) if s in gold), None)
    r["mrr"] = 1 / first if first else 0.0
    r["first_rank"] = first
    return r


def evaluate(retriever, examples: list[dict], chunks: list[Chunk], depth: int = 50) -> dict:
    known = {_norm(c.symbol) for c in chunks if c.kind != "module"}
    modules = {_norm(c.symbol) for c in chunks if c.kind == "module"}

    rows, dropped = [], 0
    for ex in examples:
        gold_f = {_norm(s) for s in ex["gold_symbols"]} & known
        gold_m = {m for m in (_module_of(_norm(s), modules) for s in gold_f) if m}
        if not gold_f:
            dropped += 1   # gold code no longer exists at the indexed commit
            continue
        ranked = [_norm(s) for s in retriever.retrieve(ex["query"], depth)]
        fn_rank = [s for s in ranked if s not in modules]
        mod_rank = []
        for s in ranked:
            m = s if s in modules else _module_of(s, modules)
            if m and m not in mod_rank:
                mod_rank.append(m)
        rows.append({
            "id": ex["id"], "mentions_gold": ex["mentions_gold"],
            "function": _score(fn_rank, gold_f), "module": _score(mod_rank, gold_m),
            "gold": sorted(gold_f), "top5": fn_rank[:5],
        })

    def agg(sub: list[dict]) -> dict:
        if not sub:
            return {"n": 0}
        out = {"n": len(sub)}
        for level in ("function", "module"):
            for key in [f"recall@{k}" for k in KS] + [f"hit@{k}" for k in KS] + ["mrr"]:
                out[f"{level}/{key}"] = round(st.mean(r[level][key] for r in sub), 4)
        return out

    return {
        "retriever": getattr(retriever, "name", type(retriever).__name__),
        "examples_total": len(examples),
        "dropped_gold_missing_at_index": dropped,
        "all": agg(rows),
        "explicit": agg([r for r in rows if r["mentions_gold"]]),
        "implicit": agg([r for r in rows if not r["mentions_gold"]]),
        "per_example": rows,
    }


def format_table(results: list[dict], split: str = "all") -> str:
    cols = ["function/recall@5", "function/recall@10", "function/hit@10", "function/mrr",
            "module/recall@1", "module/recall@5"]
    head = f"| retriever ({split}) | n | " + " | ".join(c.replace("function/", "fn ").replace("module/", "mod ") for c in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 2)]
    for r in results:
        a = r[split]
        if not a.get("n"):
            continue
        lines.append(f"| {r['retriever']} | {a['n']} | " + " | ".join(f"{a[c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)
