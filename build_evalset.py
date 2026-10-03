"""Mine a localization eval set from a repo's merged PRs.

  # realistic: queries are the issues each PR closes (set GITHUB_TOKEN)
  python build_evalset.py click --source github --github-repo pallets/click --out eval/click_issues.jsonl
  # offline: queries are PR titles from merge commits (easier, leakier)
  python build_evalset.py click --source commits --out eval/click_titles.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from codeqa.evalset import mine, save


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path, help="local clone with FULL history (no --depth)")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--source", choices=["github", "commits"], default="commits")
    ap.add_argument("--github-repo", help="owner/name, required for --source github")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cache", type=Path, default=Path(".gh_cache"))
    args = ap.parse_args()
    if args.source == "github" and not args.github_repo:
        ap.error("--github-repo is required with --source github")

    examples, skipped = mine(args.repo, args.branch, args.source, args.github_repo,
                             args.cache, limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(examples, args.out)
    print(json.dumps({
        "examples": len(examples),
        "mentions_gold": sum(e.mentions_gold for e in examples),
        "gold_symbols_per_example": dict(sorted(Counter(len(e.gold_symbols) for e in examples).items())),
        "skipped": skipped,
    }, indent=2))


if __name__ == "__main__":
    main()
