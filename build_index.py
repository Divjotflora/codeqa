"""Chunk a Python repo and build its code graph.

usage: python build_index.py /path/to/repo --out index/ [--max-lines 80] [--workers 8]
       python build_index.py /path/to/repo --out index/ --impact click.core.Command.invoke
"""
import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from codeqa.chunker import chunk_repo
from codeqa.graph import CodeGraph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--out", type=Path, default=Path("index"))
    ap.add_argument("--max-lines", type=int, default=80)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--impact", help="print impact analysis for this symbol after building")
    args = ap.parse_args()

    repo = args.repo.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    chunks = chunk_repo(repo, max_lines=args.max_lines)
    t1 = time.time()
    with open(args.out / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")

    graph = CodeGraph.build(repo, chunks, workers=args.workers)
    t2 = time.time()
    graph.save(args.out / "graph.json")

    kinds = Counter(c.kind for c in chunks)
    stats = {
        "repo": str(repo),
        "chunks": len(chunks),
        "chunk_kinds": dict(kinds),
        "split_functions": len({c.symbol for c in chunks if c.num_parts > 1}),
        "chunk_seconds": round(t1 - t0, 2),
        "graph_seconds": round(t2 - t1, 2),
        **graph.stats,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    if args.impact:
        print(json.dumps(graph.impact(args.impact), indent=2))


if __name__ == "__main__":
    main()
