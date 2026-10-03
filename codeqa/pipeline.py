"""Build the final retrieval system (hybrid BM25 + dense, RRF, graph expansion) from an index."""
from __future__ import annotations

import json
from pathlib import Path

from .chunker import Chunk


def load_chunks(index: Path) -> list[Chunk]:
    return [Chunk(**json.loads(l)) for l in open(index / "chunks.jsonl", encoding="utf-8")]


def build_system(index: Path, embedder: str | None = "st:bge-small", graph: bool = True,
                 seeds: int = 10, alpha: float = 0.5):
    """embedder=None gives BM25 (+ graph) only, with no model download."""
    from .fusion import RRF, GraphExpand
    from .graph import CodeGraph
    from .retrievers import BM25AST

    chunks = load_chunks(index)
    r = BM25AST(chunks)
    if embedder:
        from .embed import DenseRetriever, get_embedder
        r = RRF([r, DenseRetriever(chunks, get_embedder(embedder), "code", cache_dir=index)])
    if graph:
        r = GraphExpand(r, CodeGraph.load(index / "graph.json"), seeds, alpha)
    return r, chunks
