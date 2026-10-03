"""Hallucinated-identifier check for saved answers (no model calls).

A valid citation proves the *location* is real, not that the sentence around it is.
This checks the names an answer puts in backticks (`ProgressBar.render_progress`,
`self.is_hidden`) against what the model could actually have seen:

  snippet    appears in the code snippets the model was shown
  question   appears in the question / issue text (the model repeated the user)
  repo       exists elsewhere in the repository, but wasn't shown (recalled or guessed)
  nowhere    exists in none of these: a hallucinated name

  python check_identifiers.py --repo ../click --index index --evalset eval/click_issues.jsonl \
      --answers answers_click.jsonl
"""
import argparse
import builtins
import json
import keyword
import re
from pathlib import Path

from codeqa.answer import Repo, build_snippets
from codeqa.evalset import load
from codeqa.pipeline import load_chunks

BACKTICK = re.compile(r"`([^`\n]{1,120})`")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SKIP = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls", "args", "kwargs"}


def names_in(text: str) -> set[str]:
    return set(IDENT.findall(text))


def answer_identifiers(answer: str) -> list[str]:
    out = []
    for m in BACKTICK.finditer(answer):
        for tok in IDENT.findall(m.group(1)):
            if len(tok) >= 3 and tok not in SKIP and tok not in out:
                out.append(tok)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--evalset", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-lines", type=int, default=40)
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    chunks = load_chunks(args.index)
    repo = Repo(args.repo.resolve(), chunks)
    repo_names: set[str] = set()
    for f in repo.files:
        repo_names |= names_in("\n".join(repo.lines(f)))
    query = {ex["id"]: ex["query"] for ex in load(args.evalset)}

    counts = {"snippet": 0, "question": 0, "repo": 0, "nowhere": 0}
    answers_with_nowhere, n, examples = 0, 0, []
    for line in open(args.answers, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        # rebuild exactly the snippets the model saw (same index, same settings)
        shown = build_snippets(repo, r["shown"], args.k, args.max_lines)
        seen = names_in("\n".join(s.text for s in shown))
        asked = names_in(query.get(r["id"], ""))
        bad = []
        for tok in answer_identifiers(r["answer"]):
            where = ("snippet" if tok in seen else "question" if tok in asked
                     else "repo" if tok in repo_names else "nowhere")
            counts[where] += 1
            if where == "nowhere":
                bad.append(tok)
        if bad:
            answers_with_nowhere += 1
            if len(examples) < args.examples:
                examples.append((r["id"], bad))

    total = sum(counts.values())
    print(json.dumps({
        "answers": n,
        "identifiers_per_answer": round(total / max(n, 1), 2),
        **{f"share_{k}": round(v / max(total, 1), 3) for k, v in counts.items()},
        "answers_with_hallucinated_identifier": round(answers_with_nowhere / max(n, 1), 3),
    }, indent=2))
    if examples:
        print("\nexamples of names found nowhere:")
        for qid, bad in examples:
            print(f"  {qid}: {', '.join(bad)}")


if __name__ == "__main__":
    main()
