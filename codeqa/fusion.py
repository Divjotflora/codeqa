"""Combining retrievers.

RRF          reciprocal rank fusion: score(s) = sum_r w_r / (k + rank_r(s)).
             Rank-based, so BM25 scores and cosine similarities never need to be
             put on the same scale.
GraphExpand  take the top `seeds` results of a base retriever and let their code
             graph neighbours (callers, callees, base/sub classes, a class's own
             methods) into the ranking with score alpha / seed_rank. Fixes the
             common miss where retrieval finds the public entry point but the bug
             lives one call away.
"""
from __future__ import annotations

from collections import defaultdict

from .graph import CodeGraph


class RRF:
    def __init__(self, retrievers: list, weights: list[float] | None = None, k: int = 60, depth: int = 100):
        self.retrievers, self.k, self.depth = retrievers, k, depth
        self.weights = weights or [1.0] * len(retrievers)
        self.name = "rrf(" + ", ".join(r.name for r in retrievers) + ")"

    def scored(self, query: str) -> list[tuple[str, float]]:
        scores: dict[str, float] = defaultdict(float)
        for r, w in zip(self.retrievers, self.weights):
            for rank, s in enumerate(r.retrieve(query, self.depth), 1):
                scores[s] += w / (self.k + rank)
        return sorted(scores.items(), key=lambda x: -x[1])

    def retrieve(self, query: str, k: int = 50) -> list[str]:
        return [s for s, _ in self.scored(query)[:k]]


class GraphExpand:
    def __init__(self, base, graph: CodeGraph, seeds: int = 10, alpha: float = 0.5,
                 edge_types: tuple[str, ...] = ("calls", "called_by", "inherits", "methods"),
                 depth: int = 100, max_per_seed: int = 8):
        self.base, self.g, self.seeds, self.alpha, self.depth = base, graph, seeds, alpha, depth
        self.max_per_seed = max_per_seed
        self.edge_types = set(edge_types)
        self.name = f"{base.name}+graph(s={seeds},a={alpha})"

    def _neighbors(self, s: str) -> list[str]:
        g = self.g.g
        if s not in g:
            return []
        out = []
        if "calls" in self.edge_types:
            out += self.g.callees(s)
        if "called_by" in self.edge_types:
            out += self.g.callers(s)
        if "inherits" in self.edge_types:
            out += self.g._nbrs(s, "inherits") + self.g._nbrs(s, "inherits", reverse=True)
        if "methods" in self.edge_types and g.nodes[s]["kind"] == "class":
            out += self.g._nbrs(s, "contains")
        return [n for n in out if n in g and not g.nodes[n]["is_test"] and g.nodes[n]["kind"] != "module"]

    def retrieve(self, query: str, k: int = 50) -> list[str]:
        # Work on reciprocal ranks (1/rank) so any base retriever plugs in. A neighbour
        # of the rank-r seed earns alpha/r; neighbours of several seeds add up, so code
        # that sits between multiple hits rises. Per-seed cap stops a hub function with
        # 60 callers from flooding the list; among a seed's neighbours, ones the base
        # retriever already liked go first.
        ranking = self.base.retrieve(query, self.depth)
        base = {s: 1.0 / i for i, s in enumerate(ranking, 1)}
        scores = dict(base)
        for r, s in enumerate(ranking[: self.seeds], 1):
            nbrs = sorted(set(self._neighbors(s)) - {s}, key=lambda n: (-base.get(n, 0.0), n))
            for n in nbrs[: self.max_per_seed]:
                scores[n] = scores.get(n, 0.0) + self.alpha / r
        return [s for s, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]
