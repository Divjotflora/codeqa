"""Ask a question about an indexed repo and get an answer with verified citations.

  python ask.py --repo ../click --index index "Why does the progress bar not show up when output is piped?"
  python ask.py --repo ../click --index index --model qwen2.5-coder:3b --show-context "..."
"""
import argparse
import textwrap
from pathlib import Path

from codeqa.answer import Repo, answer_question
from codeqa.llm import make_llm
from codeqa.pipeline import build_system

MARK = {"valid": "ok ", "ungrounded": "?? ", "bad_range": "XX ", "bad_file": "XX ", "bad_label": "XX ",
        "named": "-- "}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai", "anthropic"])
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--base-url")
    ap.add_argument("--embedder", default="st:bge-small", help="'none' for BM25 + graph only")
    ap.add_argument("--k", type=int, default=5, help="functions shown to the model")
    ap.add_argument("--max-lines", type=int, default=40)
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()

    retriever, chunks = build_system(args.index, None if args.embedder == "none" else args.embedder)
    repo = Repo(args.repo.resolve(), chunks)
    llm = make_llm(args.backend, args.model, args.base_url)
    ans = answer_question(args.question, retriever, repo, llm, args.k, args.max_lines)

    if args.show_context:
        print("=" * 80)
        for s in ans.shown:
            print(s.text, "\n")
    print("=" * 80)
    print(textwrap.fill(ans.annotated(), 100, replace_whitespace=False))
    print("\nCitations:")
    if not ans.citations:
        print("  (none)")
    seen = set()
    for c in ans.citations:
        key = (c.kind, c.raw, c.status)
        if key in seen:      # the same label cited in the answer and again under "Sources:"
            continue
        seen.add(key)
        where = f"-> {c.symbol}" if c.symbol else ""
        print(f"  [{MARK.get(c.status, '   ')}] {c.raw:<45} {c.status:<11} {where}")
    print("\nRetrieved (top 5):", ", ".join(ans.retrieved[:5]))
    print(f"\n{ans.seconds:.1f}s, {ans.tokens[0]} prompt tokens, {ans.tokens[1]} output tokens")


if __name__ == "__main__":
    main()
