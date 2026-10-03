"""Plain-English summaries of chunks, written in the vocabulary a user filing an
issue would use. Cached by chunk content hash in summaries.jsonl, so reruns and
re-indexing only summarize new or changed chunks, and an interrupted run resumes.

Backends:
  anthropic  Claude via the Anthropic API (needs ANTHROPIC_API_KEY)
  openai     any OpenAI-compatible chat endpoint: a local Ollama server
             (http://localhost:11434/v1, free, no key), or hosted providers
             that expose one. Key, if needed, comes from OPENAI_API_KEY.
A small model is plenty for this.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .chunker import Chunk

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
PROMPT_VERSION = "v2"
SUMMARIZE_KINDS = ("function", "method", "class")   # module chunks are just imports + a
                                                    # list of names: nothing to summarize,
                                                    # and small models invent a purpose

PROMPT = """You are writing search-index entries for a Python codebase.

In at most 60 words of plain prose (no lists, no markdown), say what this {kind} does
and what user-visible behavior it is responsible for. Use the words a user would use
when reporting a bug or asking where some behavior lives. Only describe behavior you
can see in the code below; do not guess at error handling or edge cases that are not
there. If the code is trivial, one sentence is enough. Reply with the description only.

Location: `{symbol}` in {file}

```python
{text}
```"""


def load_summaries(path: Path) -> dict[str, str]:
    """content_hash -> summary"""
    if not path.exists():
        return {}
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if r.get("prompt_version") == PROMPT_VERSION:
                out[r["content_hash"]] = r["summary"]
    return out


def _anthropic_call(model: str):
    import anthropic

    client = anthropic.Anthropic(max_retries=6)

    def call(prompt: str) -> tuple[str, int, int]:
        msg = client.messages.create(model=model, max_tokens=150, temperature=0,
                                     messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        return text, msg.usage.input_tokens, msg.usage.output_tokens
    return call


def _openai_call(model: str, base_url: str, timeout: int = 300):
    url = base_url.rstrip("/") + "/chat/completions"
    key = os.environ.get("OPENAI_API_KEY", "")

    def call(prompt: str) -> tuple[str, int, int]:
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": 150,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and attempt < 4:
                    time.sleep(2 ** attempt * 5)
                    continue
                raise
        text = (data["choices"][0]["message"].get("content") or "").strip()
        u = data.get("usage") or {}
        return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    return call


def summarize_chunks(chunks: list[Chunk], out_path: Path, model: str = DEFAULT_MODEL,
                     workers: int = 8, max_chars: int = 8000, limit: int | None = None,
                     backend: str = "anthropic", base_url: str = "http://localhost:11434/v1",
                     path_prefix: str | None = None) -> dict:
    call = _anthropic_call(model) if backend == "anthropic" else _openai_call(model, base_url)
    done = load_summaries(out_path)
    todo, queued = [], set()
    for c in chunks:
        if c.kind not in SUMMARIZE_KINDS or (path_prefix and not c.file.startswith(path_prefix)):
            continue
        if c.content_hash not in done and c.content_hash not in queued:
            queued.add(c.content_hash)
            todo.append(c)
    todo = todo[:limit] if limit else todo

    lock = threading.Lock()
    usage = {"input_tokens": 0, "output_tokens": 0, "errors": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def work(c: Chunk) -> dict:
        text, tin, tout = call(PROMPT.format(kind=c.kind, symbol=c.symbol, file=c.file,
                                             text=c.text[:max_chars]))
        if not text:
            raise ValueError("empty summary")
        return {"content_hash": c.content_hash, "symbol": c.symbol, "summary": text,
                "model": model, "prompt_version": PROMPT_VERSION, "usage": [tin, tout]}

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex, open(out_path, "a") as f:
        futures = {ex.submit(work, c): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rec = fut.result()
            except Exception as e:
                usage["errors"] += 1
                print(f"[summarize] {futures[fut].symbol}: {e}")
                continue
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                usage["input_tokens"] += rec["usage"][0]
                usage["output_tokens"] += rec["usage"][1]
            if i % 100 == 0:
                print(f"[summarize] {i}/{len(todo)}")
    return {"already_cached": len(done), "summarized": len(todo) - usage["errors"],
            "seconds": round(time.time() - t0, 1), **usage}
