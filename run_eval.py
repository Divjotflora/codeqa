"""Score retrievers on an eval set against an index built by build_index.py.

  python run_eval.py --repo click --index index/ --evalset eval/click_titles.jsonl
  python run_eval.py --repo click --index index/ --evalset eval/click_issues.jsonl \\
      --embedder st:bge-small --summaries index/summaries.jsonl \\
      --retrievers bm25_fixed,bm25_ast,dense_code,dense_summary,hybrid,hybrid+graph

Retriever names:
  bm25_fixed, bm25_ast, bm25_ast+graph       always available
  dense_code                                  needs --embedder
  bm25_summary, dense_summary                 need --summaries (dense_summary also --embedder)
  hybrid        RRF of every available base retriever above except bm25_fixed
  hybrid+graph  hybrid, then graph expansion
  <any>+rerank      cross-encoder rerank of <any>'s top --rerank-top (needs --reranker)
  <any>+rerank_rrf  same, fused with <any>'s own ranking by RRF
                    e.g. hybrid+graph+rerank, hybrid+graph+rerank_rrf
"""
import argparse
import json
from pathlib import Path

from codeqa.chunker import Chunk
from codeqa.evalset import load
from codeqa.evaluate import evaluate, format_table
from codeqa.fusion import RRF, GraphExpand
from codeqa.graph import CodeGraph
from codeqa.retrievers import BM25AST, BM25Fixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--evalset", type=Path, required=True)
    ap.add_argument("--retrievers", default="bm25_fixed,bm25_ast,bm25_ast+graph")
    ap.add_argument("--embedder", help="st:bge-small | st:coderank | voyage:voyage-code-3 | hashing")
    ap.add_argument("--summaries", type=Path)
    ap.add_argument("--reranker", help="minilm | mxbai-xsmall | bge-base | jina-v2 | <hf id>")
    ap.add_argument("--rerank-top", type=int, default=50)
    ap.add_argument("--rerank-query", choices=["full", "title", "title+ids"], default="full",
                    help="what the cross-encoder sees of the query")
    ap.add_argument("--graph-seeds", type=int, default=10)
    ap.add_argument("--graph-alpha", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    args = ap.parse_args()

    chunks = [Chunk(**json.loads(l)) for l in open(args.index / "chunks.jsonl")]
    examples = load(args.evalset)
    wanted = args.retrievers.split(",")

    summaries = None
    if args.summaries:
        from codeqa.summarize import load_summaries
        summaries = load_summaries(args.summaries)
        cov = sum(c.content_hash in summaries for c in chunks) / len(chunks)
        print(f"summaries cover {cov:.1%} of chunks")
    embedder = None
    if args.embedder:
        from codeqa.embed import get_embedder
        embedder = get_embedder(args.embedder)

    graph = None
    if any("graph" in w for w in wanted):
        graph = CodeGraph.load(args.index / "graph.json")

    base: dict = {}
    if "bm25_fixed" in wanted:
        base["bm25_fixed"] = BM25Fixed(args.repo.resolve(), chunks)
    base["bm25_ast"] = BM25AST(chunks)
    if summaries:
        base["bm25_summary"] = BM25AST(chunks, summaries)
    if embedder:
        from codeqa.embed import DenseRetriever
        base["dense_code"] = DenseRetriever(chunks, embedder, "code", cache_dir=args.index)
        if summaries:
            base["dense_summary"] = DenseRetriever(chunks, embedder, "summary", summaries, cache_dir=args.index)

    scorer = cache = None
    if any("+rerank" in w for w in wanted):
        if not args.reranker:
            raise SystemExit("+rerank retrievers need --reranker")
        from codeqa.rerank import CrossEncoderScorer, ScoreCache
        scorer = CrossEncoderScorer(args.reranker)
        cache = ScoreCache(args.index / f"ce_{args.reranker.replace('/', '_')}.json")

    built: dict = {}

    def make(name: str):
        if name in built:
            return built[name]
        for suffix, mode in (("+rerank_rrf", "rrf"), ("+rerank", "ce")):
            if name.endswith(suffix):
                from codeqa.rerank import Reranker
                r = Reranker(make(name[: -len(suffix)]), chunks, scorer, mode,
                             args.rerank_top, query_mode=args.rerank_query, cache=cache)
                built[name] = r
                return r
        built[name] = r = _make(name)
        return r

    def _make(name: str):
        if name in base:
            return base[name]
        if name == "bm25_ast+graph":
            return GraphExpand(base["bm25_ast"], graph, args.graph_seeds, args.graph_alpha)
        if name.startswith("hybrid"):
            parts = [r for n, r in base.items() if n != "bm25_fixed"
                     and not (n == "bm25_ast" and "bm25_summary" in base)]
            h = RRF(parts)
            return GraphExpand(h, graph, args.graph_seeds, args.graph_alpha) if name == "hybrid+graph" else h
        raise SystemExit(f"retriever {name!r} unavailable (missing --embedder/--summaries?)")

    results = []
    for n in wanted:
        r = make(n)
        results.append(evaluate(r, examples, chunks))
        if hasattr(r, "scored_pairs"):
            cache.save()
            if r.scored_pairs:
                print(f"[rerank] {r.name}: scored {r.scored_pairs} pairs in {r.seconds:.0f}s "
                      f"({r.seconds / max(len(examples), 1):.2f}s/query)")
    args.out.write_text(json.dumps(results, indent=2))
    print(f"examples: {len(examples)}, dropped (gold gone at HEAD): "
          f"{results[0]['dropped_gold_missing_at_index']}\n")
    for split in ("all", "implicit", "explicit"):
        print(format_table(results, split), "\n")


if __name__ == "__main__":
    main()
