"""Dense retrieval over chunks.

Two "views" of each chunk can be embedded:
  code     symbol path + raw chunk text
  summary  symbol path + signature + the LLM summary (see summarize.py)

The summary view is the bridge for implicit questions: an issue saying
"pager swallows my flags" shares few tokens with `_pipepager`'s source, but a
plain-English summary of `_pipepager` will talk about pagers and flags.

Embeddings are cached on disk keyed by a hash of the exact embedded text, so
re-indexing after a commit only embeds chunks that changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from .chunker import Chunk

# name -> (hf model id, query prefix, document prefix, trust_remote_code)
PRESETS = {
    "bge-small": ("BAAI/bge-small-en-v1.5",
                  "Represent this sentence for searching relevant passages: ", "", False),
    "jina-code": ("jinaai/jina-embeddings-v2-base-code", "", "", True),
    "coderank": ("nomic-ai/CodeRankEmbed",
                 "Represent this query for searching relevant code: ", "", True),
    # natively supported architectures (no trust_remote_code): these keep working as
    # transformers evolves, unlike models that ship their own modeling code
    "codesearch": ("flax-sentence-embeddings/st-codesearch-distilroberta-base", "", "", False),
    "gte-modernbert": ("Alibaba-NLP/gte-modernbert-base", "", "", False),
}
# extra model-config overrides per preset. ModernBERT tries to torch.compile parts of the
# model by default; on CPU that can recompile for every new batch shape and crawl.
CONFIG_KWARGS = {"gte-modernbert": {"reference_compile": False}}


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


class SentenceTransformerEmbedder:
    """Local model; runs on CPU, faster with a GPU. Any HF model id works."""

    def __init__(self, model: str = "bge-small", batch_size: int = 32, max_seq_length: int | None = None):
        from sentence_transformers import SentenceTransformer

        hf_id, self.qp, self.dp, trust = PRESETS.get(model, (model, "", "", True))
        self.name = model
        kwargs = {"config_kwargs": CONFIG_KWARGS[model]} if model in CONFIG_KWARGS else {}
        self.model = SentenceTransformer(hf_id, trust_remote_code=trust, **kwargs)
        max_seq_length = max_seq_length or int(os.environ.get("CODEQA_MAX_SEQ", 512))
        self.model.max_seq_length = max_seq_length
        if max_seq_length != 512:
            self.name = f"{model}-{max_seq_length}"   # separate embedding cache per length
        self.batch_size = batch_size

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        p = self.qp if is_query else self.dp
        return _normalize(self.model.encode([p + t for t in texts], batch_size=self.batch_size,
                                            normalize_embeddings=True, show_progress_bar=len(texts) > 200))


class VoyageEmbedder:
    """Hosted code embeddings (needs VOYAGE_API_KEY)."""

    def __init__(self, model: str = "voyage-code-3", batch_size: int = 64):
        self.name, self.batch_size = model, batch_size
        self.key = os.environ["VOYAGE_API_KEY"]

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            body = json.dumps({"input": texts[i:i + self.batch_size], "model": self.name,
                               "input_type": "query" if is_query else "document"}).encode()
            req = urllib.request.Request("https://api.voyageai.com/v1/embeddings", data=body, headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())["data"]
            out.extend(d["embedding"] for d in sorted(data, key=lambda d: d["index"]))
        return _normalize(np.array(out))


class HashingEmbedder:
    """Deterministic bag-of-tokens vectors. Not a real semantic model: for
    tests and smoke runs where no model can be downloaded."""

    def __init__(self, dim: int = 1024):
        from .retrievers import tokenize
        self.name, self.dim, self.tok = f"hashing-{dim}", dim, tokenize

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self.tok(t):
                m[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        return _normalize(m)


def get_embedder(spec: str):
    """'st:bge-small', 'st:<hf id>', 'voyage:voyage-code-3', 'hashing'"""
    kind, _, arg = spec.partition(":")
    if kind == "st":
        return SentenceTransformerEmbedder(arg or "bge-small")
    if kind == "voyage":
        return VoyageEmbedder(arg or "voyage-code-3")
    if kind == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unknown embedder spec {spec!r}")


# --------------------------------------------------------------------------- #
class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.vecs: dict[str, np.ndarray] = {}
        if path.exists():
            z = np.load(path)
            self.vecs = dict(zip(z["keys"].tolist(), z["vecs"]))

    def get_many(self, texts: list[str], embedder) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest()[:20] for t in texts]
        missing = sorted({k: t for k, t in zip(keys, texts) if k not in self.vecs}.items())
        if missing:
            vecs = embedder.encode([t for _, t in missing])
            self.vecs.update({k: v for (k, _), v in zip(missing, vecs)})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.path, keys=np.array(list(self.vecs)), vecs=np.stack(list(self.vecs.values())))
        self.new = len(missing)
        return np.stack([self.vecs[k] for k in keys])


def chunk_view(c: Chunk, view: str, summaries: dict[str, str] | None, max_chars: int = 6000) -> str:
    if view == "code":
        return f"{c.symbol}\n{c.text}"[:max_chars]
    if view == "summary":
        s = (summaries or {}).get(c.content_hash, "")
        return f"{c.symbol}\n{c.signature}\n{s}"[:max_chars]
    raise ValueError(view)


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder, view: str = "code",
                 summaries: dict[str, str] | None = None, cache_dir: Path | None = None):
        if view == "summary" and not summaries:
            raise ValueError("summary view needs summaries (run summarize.py)")
        self.name = f"dense_{view}[{embedder.name}]"
        self.chunks, self.embedder = chunks, embedder
        texts = [chunk_view(c, view, summaries) for c in chunks]
        cache = EmbeddingCache((cache_dir or Path("index")) / f"emb_{embedder.name.replace('/', '_')}_{view}.npz")
        self.matrix = cache.get_many(texts, embedder)
        self.newly_embedded = cache.new

    def retrieve(self, query: str, k: int = 50) -> list[str]:
        q = self.embedder.encode([query[:4000]], is_query=True)[0]
        scores = self.matrix @ q
        out, seen = [], set()
        for i in np.argsort(-scores)[: k * 3]:
            s = self.chunks[i].symbol
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= k:
                break
        return out
