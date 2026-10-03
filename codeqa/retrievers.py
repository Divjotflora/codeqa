"""Lexical baselines. Both return a ranked list of *symbols* so they're scored
identically; module symbols may appear and are used for module-level metrics.

  BM25AST    one document per AST chunk (what the system will actually use)
  BM25Fixed  the naive baseline: fixed line windows; a retrieved window is
             credited to the functions it overlaps (largest overlap first),
             so function-level recall is comparable, if a bit generous.
"""
from __future__ import annotations

import keyword
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .chunker import Chunk

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
STOP = set(keyword.kwlist) | {
    "self", "cls", "the", "and", "for", "that", "this", "with", "not", "are", "but", "from",
    "when", "can", "have", "has", "was", "will", "should", "would", "it", "is", "to", "of",
    "in", "on", "an", "be", "as", "at", "or", "if", "by", "we", "you", "i", "my", "me",
    "there", "which", "what", "how", "use", "used", "using", "also", "into", "than", "then",
    "str", "int", "none", "true", "false", "args", "kwargs", "return", "def", "import",
}


def tokenize(text: str) -> list[str]:
    """Identifier-aware: `get_terminal_size`/`getTerminalSize` -> the full
    identifier plus its parts, so prose ("terminal size") matches code."""
    out = []
    for ident in _IDENT.findall(text):
        low = ident.lower()
        parts = [p.lower() for piece in ident.split("_") for p in _CAMEL.findall(piece)]
        if low not in STOP and len(low) > 1:
            out.append(low)
        if len(parts) > 1:
            out.extend(p for p in parts if p not in STOP and len(p) > 1)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(self.n, 1)
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, tf in Counter(d).items():
                self.post[t].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def top(self, query: list[str], k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in self.post[t]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda x: -x[1])[:k]


class BM25AST:
    name = "bm25_ast"

    def __init__(self, chunks: list[Chunk], summaries: dict[str, str] | None = None):
        self.chunks = chunks
        if summaries:
            self.name = "bm25_ast+summary"
        docs = []
        for c in chunks:
            # symbol path is cheap, strong signal: click.core.Group.invoke -> core group invoke
            doc = tokenize(c.symbol.replace(".", " ")) * 2 + tokenize(c.text)
            if summaries:
                doc += tokenize(summaries.get(c.content_hash, ""))
            docs.append(doc)
        self.bm25 = BM25(docs)

    def retrieve(self, query: str, k: int = 50) -> list[str]:
        out, seen = [], set()
        for i, _ in self.bm25.top(tokenize(query), k * 3):
            s = self.chunks[i].symbol
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= k:
                break
        return out


class BM25Fixed:
    name = "bm25_fixed"

    def __init__(self, repo: Path, chunks: list[Chunk], window: int = 50, stride: int = 40):
        self.windows: list[tuple[str, int, int]] = []
        docs = []
        files = sorted({c.file for c in chunks})
        self.module_of = {c.file: c.symbol for c in chunks if c.kind == "module"}
        self.spans = defaultdict(list)
        for c in chunks:
            if c.kind != "module":
                self.spans[c.file].append((c.start_line, c.end_line, c.symbol, c.kind))
        for f in files:
            lines = (repo / f).read_text(errors="replace").splitlines()
            for start in range(0, max(len(lines), 1), stride):
                seg = lines[start:start + window]
                self.windows.append((f, start + 1, start + len(seg)))
                docs.append(tokenize(f.replace("/", " ")) + tokenize("\n".join(seg)))
                if start + window >= len(lines):
                    break
        self.bm25 = BM25(docs)

    def _symbols_in(self, f: str, a: int, b: int) -> list[str]:
        hits = []
        for s, e, sym, kind in self.spans[f]:
            ov = min(b, e) - max(a, s) + 1
            if ov > 0:
                # prefer functions over the class that contains them
                hits.append((kind == "class", -ov, sym))
        return [h[2] for h in sorted(hits)]

    def retrieve(self, query: str, k: int = 50) -> list[str]:
        out, seen = [], set()
        for i, _ in self.bm25.top(tokenize(query), k * 2):
            f, a, b = self.windows[i]
            for s in [self.module_of.get(f)] + self._symbols_in(f, a, b):
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
            if len(out) >= k * 3:
                break
        return out
