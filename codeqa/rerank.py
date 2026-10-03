"""Cross-encoder reranking of a base retriever's top candidates.

A bi-encoder (embed.py) embeds query and code separately, so it can't compare them
token by token. A cross-encoder reads (query, code) *together* and scores the pair:
much more accurate per pair, but far too slow to run over a whole repo. So it only
reorders the top `top_n` results of the base retriever; everything below keeps the
base order.

Two ways to use the scores:
  ce   rank purely by cross-encoder score
  rrf  fuse the base ranking and the cross-encoder ranking with RRF. Safer when the
       reranker was trained on web text rather than code and is only partly reliable.

What the cross-encoder sees of the query (query_mode):
  full       the whole query, cut at 1500 characters (the original setting)
  title      only the first line: for an issue, its title
  title+ids  the title plus code identifiers pulled from the body: names in
             tracebacks (`in parse_args`), backticked names, dotted/snake_case/CamelCase
             tokens. Compact, so most of the 512-token window is left for the code.
Why: a cross-encoder reads query and code in one 512-token window. A long issue
fills it, and the traceback that names the function is what gets cut off.

Scores are cached on disk per (model, query, chunk content), so trying both modes,
or re-running an eval, never scores the same pair twice.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import re

from .chunker import Chunk

_TB_FUNC = re.compile(r'File "[^"]+", line \d+, in ([A-Za-z_][\w]*)')
_BACKTICK = re.compile(r"`([^`\n]{2,80})`")
_CODEY = re.compile(r"\b(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*\b|\b[a-z]+(?:_[a-z0-9]+)+\b|\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b")
_NOISE = {"e.g", "i.e", "etc", "github.com", "http", "https"}


def condense_query(query: str, mode: str = "full", max_ids: int = 25) -> str:
    if mode == "full":
        return query[:1500]
    title = query.strip().split("\n", 1)[0].strip()
    if mode == "title":
        return title
    if mode != "title+ids":
        raise ValueError(mode)
    body = query[len(title):]
    ids: list[str] = []
    for found in (_TB_FUNC.findall(body), _BACKTICK.findall(body), _CODEY.findall(body)):
        for t in found:
            t = t.strip().strip("()")
            if t and t.lower() not in _NOISE and t not in ids and t not in title and len(t) <= 60:
                ids.append(t)
    return title if not ids else f"{title}\nmentions: {', '.join(ids[:max_ids])}"

# name -> (hf model id, trust_remote_code)
PRESETS = {
    "minilm": ("cross-encoder/ms-marco-MiniLM-L-6-v2", False),          # 22M params, fast
    "mxbai-xsmall": ("mixedbread-ai/mxbai-rerank-xsmall-v1", False),    # 70M
    "bge-base": ("BAAI/bge-reranker-base", False),                      # 278M, slower
    "jina-v2": ("jinaai/jina-reranker-v2-base-multilingual", True),     # 278M, trained incl. code search
}


class CrossEncoderScorer:
    def __init__(self, model: str = "minilm", max_length: int = 512, batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        hf_id, trust = PRESETS.get(model, (model, True))
        self.name = model
        self.model = CrossEncoder(hf_id, max_length=max_length, trust_remote_code=trust)
        self.batch_size = batch_size

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(s) for s in self.model.predict(pairs, batch_size=self.batch_size,
                                                     show_progress_bar=False)]


class ScoreCache:
    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict[str, float] = {}
        if path and path.exists():
            self.data = json.loads(path.read_text())
        self.dirty = False

    @staticmethod
    def key(query: str, doc: str) -> str:
        return hashlib.sha1(f"{query}\x00{doc}".encode()).hexdigest()[:24]

    def save(self) -> None:
        if self.path and self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data))
            self.dirty = False


class Reranker:
    def __init__(self, base, chunks: list[Chunk], scorer, mode: str = "ce", top_n: int = 50,
                 query_mode: str = "full", doc_chars: int = 2000, cache: ScoreCache | None = None,
                 rrf_k: int = 60):
        assert mode in ("ce", "rrf")
        condense_query("x", query_mode)   # validate early
        self.base, self.scorer, self.mode, self.top_n = base, scorer, mode, top_n
        self.query_mode, self.doc_chars, self.rrf_k = query_mode, doc_chars, rrf_k
        self.cache = cache or ScoreCache(None)
        qtag = "" if query_mode == "full" else f",q={query_mode}"
        self.name = f"{base.name}+rerank[{getattr(scorer, 'name', 'scorer')},{mode}{qtag}]"
        # first chunk of each symbol represents it (split functions: part 1 has the signature)
        self.doc_of: dict[str, str] = {}
        self.kind_of: dict[str, str] = {}
        for c in chunks:
            if c.symbol not in self.doc_of:
                self.doc_of[c.symbol] = f"{c.symbol}\n{c.text}"[: self.doc_chars]
                self.kind_of[c.symbol] = c.kind
        self.seconds = 0.0
        self.scored_pairs = 0

    def scores(self, query: str, symbols: list[str]) -> dict[str, float]:
        q = condense_query(query, self.query_mode)
        keys = {s: ScoreCache.key(q, self.doc_of[s]) for s in symbols}
        todo = [s for s in symbols if keys[s] not in self.cache.data]
        if todo:
            t0 = time.time()
            new = self.scorer([(q, self.doc_of[s]) for s in todo])
            self.seconds += time.time() - t0
            self.scored_pairs += len(todo)
            for s, v in zip(todo, new):
                self.cache.data[keys[s]] = v
            self.cache.dirty = True
        return {s: self.cache.data[keys[s]] for s in symbols}

    def retrieve(self, query: str, k: int = 50) -> list[str]:
        ranking = self.base.retrieve(query, max(k, self.top_n))
        head = [s for s in ranking[: self.top_n]
                if s in self.doc_of and self.kind_of[s] != "module"]
        if not head:
            return ranking[:k]
        sc = self.scores(query, head)
        if self.mode == "ce":
            reranked = sorted(head, key=lambda s: -sc[s])
        else:
            base_rank = {s: i for i, s in enumerate(head, 1)}
            ce_rank = {s: i for i, s in enumerate(sorted(head, key=lambda s: -sc[s]), 1)}
            fused = {s: 1 / (self.rrf_k + base_rank[s]) + 1 / (self.rrf_k + ce_rank[s]) for s in head}
            reranked = sorted(head, key=lambda s: -fused[s])
        seen = set(reranked)
        rest = [s for s in ranking if s not in seen]
        return (reranked + rest)[:k]
