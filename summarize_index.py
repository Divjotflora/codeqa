"""Write LLM summaries for every chunk in an index (resumable, cached by content hash).

  # free, local (install Ollama, then: ollama pull qwen2.5-coder:3b)
  python summarize_index.py --index index/ --backend openai --model qwen2.5-coder:3b --workers 2 --limit 50
  # Claude
  ANTHROPIC_API_KEY=... python summarize_index.py --index index/ --limit 50

Try --limit 50 first and read a few summaries before paying for the whole repo.
"""
import argparse
import json
from pathlib import Path

from codeqa.chunker import Chunk
from codeqa.summarize import DEFAULT_MODEL, summarize_chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--base-url", default="http://localhost:11434/v1",
                    help="OpenAI-compatible endpoint (default: local Ollama)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--path-prefix", help="only chunks under this path, e.g. src/ for a quality check")
    args = ap.parse_args()
    chunks = [Chunk(**json.loads(l)) for l in open(args.index / "chunks.jsonl")]
    stats = summarize_chunks(chunks, args.index / "summaries.jsonl", args.model, args.workers,
                             limit=args.limit, backend=args.backend, base_url=args.base_url,
                             path_prefix=args.path_prefix)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
