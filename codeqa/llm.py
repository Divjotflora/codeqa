"""Minimal LLM clients returning (text, input_tokens, output_tokens).

  ollama     native Ollama API (http://localhost:11434). Used for answering because it
             lets us raise the context window: Ollama's default context is small, and
             several code snippets plus an issue would be silently truncated.
  openai     any OpenAI-compatible endpoint (OPENAI_API_KEY if needed)
  anthropic  Claude (ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


def _post(url: str, body: dict, headers: dict, timeout: int) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(5 * 2 ** attempt)
                continue
            raise
        except urllib.error.URLError as e:
            raise RuntimeError(f"can't reach {url} ({e.reason}). Is the server running?") from e
    raise RuntimeError("unreachable")


def make_llm(backend: str, model: str, base_url: str | None = None, max_tokens: int = 400,
             num_ctx: int = 8192, temperature: float = 0.0, timeout: int = 900):
    if backend == "ollama":
        root = (base_url or "http://localhost:11434").rstrip("/")
        root = root[:-3] if root.endswith("/v1") else root

        def call(prompt: str):
            d = _post(f"{root}/api/chat", {
                "model": model, "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"num_ctx": num_ctx, "temperature": temperature, "num_predict": max_tokens},
            }, {}, timeout)
            return (d.get("message", {}).get("content") or "").strip(), \
                d.get("prompt_eval_count", 0), d.get("eval_count", 0)
        return call

    if backend == "openai":
        url = (base_url or "http://localhost:11434/v1").rstrip("/") + "/chat/completions"
        key = os.environ.get("OPENAI_API_KEY", "")

        def call(prompt: str):
            d = _post(url, {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                            "messages": [{"role": "user", "content": prompt}]},
                      {"Authorization": f"Bearer {key}"} if key else {}, timeout)
            u = d.get("usage") or {}
            return (d["choices"][0]["message"].get("content") or "").strip(), \
                u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        return call

    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic(max_retries=6)

        def call(prompt: str):
            m = client.messages.create(model=model, max_tokens=max_tokens, temperature=temperature,
                                       messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in m.content if b.type == "text").strip(), \
                m.usage.input_tokens, m.usage.output_tokens
        return call

    raise ValueError(f"unknown backend {backend!r}")
