"""Ask questions about an indexed repo and get answers with verified citations.

One question:
  python ask.py --repo ../click --index index "How does click generate the shell completion script?"

Interactive (models load once, then ask as many questions as you like):
  python ask.py --repo ../click --index index
  commands:  :ctx  toggle showing the code snippets   :q  quit
"""
import argparse
import textwrap
import time
from pathlib import Path

from codeqa.answer import Repo, answer_question
from codeqa.llm import make_llm
from codeqa.pipeline import build_system

MARK = {"valid": "ok ", "ungrounded": "?? ", "bad_range": "XX ", "bad_file": "XX ", "bad_label": "XX ",
        "named": "-- "}


def show(ans, show_context: bool) -> None:
    if show_context:
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


def interactive(ask, show_context: bool) -> None:
    print("\nReady. Type a question, ':ctx' to toggle showing code snippets, ':q' to quit.")
    while True:
        try:
            q = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q in (":q", ":quit", "exit", "quit"):
            return
        if q == ":ctx":
            show_context = not show_context
            print(f"showing code snippets: {'on' if show_context else 'off'}")
            continue
        try:
            show(ask(q), show_context)
        except KeyboardInterrupt:
            print("\n(cancelled)")
        except Exception as e:      # keep the session alive (e.g. Ollama not running)
            print(f"error: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="omit to start an interactive session")
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

    t0 = time.time()
    retriever, chunks = build_system(args.index, None if args.embedder == "none" else args.embedder)
    repo = Repo(args.repo.resolve(), chunks)
    llm = make_llm(args.backend, args.model, args.base_url)

    def ask(q):
        return answer_question(q, retriever, repo, llm, args.k, args.max_lines)

    if args.question:
        show(ask(args.question), args.show_context)
    else:
        print(f"Loaded {len(chunks)} chunks from {args.index} in {time.time() - t0:.1f}s.")
        interactive(ask, args.show_context)


if __name__ == "__main__":
    main()
