"""Mine localization questions ("where would you fix this?") from a repo's history.

For every PR merged into the main branch (found from local git history):
  1. diff the merge against its first parent (the code before the PR),
  2. map the changed *old-side* lines to the innermost symbol containing them,
     using the same chunker as the index -> gold symbols and gold modules,
  3. take the query text from
       --source github  : the issue the PR closes ("Fixes #123"); the realistic
                          setting, a user describing a symptom (needs GITHUB_TOKEN)
       --source commits : the PR title from the merge commit; works offline but
                          titles often name the function, so it's easier/leakier.

Every example records `mentions_gold` (a gold symbol's name appears verbatim in
the query) so results can be split into explicit vs implicit questions.

Simplification: you'll evaluate against an index of the *current* HEAD, not of
each PR's base commit. Gold symbols that no longer exist at HEAD are filtered at
eval time and the coverage is reported. Per-commit indexing is a later upgrade.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunker import Chunker, module_name_for

PR_RE = re.compile(r"\(#(\d+)\)\s*$|Merge pull request #(\d+)")
LINK_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b[\s:]*"
    r"(?:(?:https?://github\.com/[\w.-]+/[\w.-]+/issues/)|#)(\d+)",
    re.IGNORECASE,
)
SKIP_TITLE_RE = re.compile(
    r"^(release|start|merge (branch|stable|remote)|stable\b|bump|update (pre-commit|requirements|dependencies)|"
    r"\[pre-commit|pre-commit|build\(deps)"
    r"|\btypos?\b|\bdocstrings?\b|^docs?\b|\bspelling\b|\bgrammar\b", re.IGNORECASE)
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


@dataclass
class Example:
    id: str
    query: str
    source: str
    pr: int
    issue: int | None
    commit: str
    base: str
    gold_modules: list[str]
    gold_symbols: list[str]          # function / method / class level
    gold_files: list[str]            # paths at the base commit (informational)
    mentions_gold: bool
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# git helpers
# --------------------------------------------------------------------------- #
def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True, errors="replace").stdout


def pr_commits(repo: Path, branch: str) -> list[dict]:
    """First-parent commits on `branch` that reference a PR (merge or squash)."""
    out = git(repo, "log", "--first-parent", branch, "--format=%H%x1f%P%x1f%s%x1f%b%x1e")
    commits = []
    for rec in out.split("\x1e"):
        rec = rec.strip()
        if not rec:
            continue
        sha, parents, subject, body = (rec.split("\x1f") + [""] * 4)[:4]
        parents = parents.split()
        m = PR_RE.search(subject)
        if not m or not parents:
            continue
        pr = int(m.group(1) or m.group(2))
        # "Merge pull request #N from user/branch": real title is the body's first line
        title = body.strip().split("\n")[0] if m.group(2) and body.strip() else PR_RE.sub("", subject).strip()
        commits.append({"sha": sha, "base": parents[0], "pr": pr, "title": title})
    return commits


def changed_old_lines(repo: Path, base: str, sha: str) -> dict[str, set[int]]:
    """{path_at_base: {old line numbers touched}} for .py files modified (not added)."""
    diff = git(repo, "diff", "-U0", "--no-renames", "--diff-filter=MD", base, sha, "--", "*.py")
    files: dict[str, set[int]] = {}
    cur = None
    for line in diff.splitlines():
        if line.startswith("--- "):
            cur = line[6:] if line.startswith("--- a/") else None
            if cur is not None:
                files.setdefault(cur, set())
        elif cur and (m := HUNK_RE.match(line)):
            start, count = int(m.group(1)), int(m.group(2) or 1)
            # pure insertion (count 0): anchor on the line it was inserted after
            files[cur].update(range(start, start + count) if count else [max(start, 1)])
    return files


def is_test_path(p: str) -> bool:
    parts = Path(p).parts
    return any(x in {"test", "tests", "testing", "docs", "examples"} for x in parts) or Path(p).name.startswith(("test_", "conftest"))


# --------------------------------------------------------------------------- #
# GitHub (cached; GITHUB_TOKEN strongly recommended: 5000 req/h vs 60)
# --------------------------------------------------------------------------- #
class GitHub:
    def __init__(self, owner_repo: str, cache: Path):
        self.base = f"https://api.github.com/repos/{owner_repo}"
        # namespace by repo so PR #123 of one repo never reads another repo's cache
        self.cache = cache / owner_repo.replace("/", "__")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.token = os.environ.get("GITHUB_TOKEN")

    def get(self, path: str) -> dict | None:
        name = path.strip("/").replace("/", "_") + ".json"
        f = self.cache / name
        if f.exists():
            return json.loads(f.read_text())
        req = urllib.request.Request(self.base + path, headers={
            "Accept": "application/vnd.github+json",
            **({"Authorization": f"Bearer {self.token}"} if self.token else {}),
        })
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read())
                f.write_text(json.dumps(data))
                return data
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    f.write_text("null")
                    return None
                if e.code in (403, 429):
                    reset = int(e.headers.get("x-ratelimit-reset", time.time() + 60))
                    wait = max(5, reset - time.time()) + 1
                    if wait > 900:
                        raise RuntimeError("GitHub rate limit hit; set GITHUB_TOKEN") from e
                    print(f"[github] rate limited, sleeping {wait:.0f}s")
                    time.sleep(wait)
                    continue
                if e.code >= 500 and attempt < 5:
                    time.sleep(5 * 2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                # flaky network: back off and retry (5, 10, 20, 40, 80 s)
                if attempt == 5:
                    raise RuntimeError(f"network keeps failing for {path}; re-run to resume "
                                       "(finished requests are cached)") from e
                wait = 5 * 2 ** attempt
                print(f"[github] network error ({e}); retrying in {wait}s")
                time.sleep(wait)
        return None


def linked_issues(pr: dict) -> list[int]:
    text = f"{pr.get('title') or ''}\n{pr.get('body') or ''}"
    return sorted({int(n) for n in LINK_RE.findall(text)})


def clean_issue_text(title: str, body: str, max_chars: int = 2000) -> str:
    body = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)       # issue-template comments
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return f"{title.strip()}\n\n{body}"[:max_chars].strip()


# --------------------------------------------------------------------------- #
# miner
# --------------------------------------------------------------------------- #
def mine(repo: Path, branch: str = "main", source: str = "commits", github_repo: str | None = None,
         cache: Path = Path(".gh_cache"), max_files: int = 4, max_symbols: int = 8,
         limit: int | None = None) -> tuple[list[Example], dict]:
    chunker = Chunker(max_lines=10**9)   # gold mapping wants whole functions
    gh = GitHub(github_repo, cache) if source == "github" else None
    skipped: dict[str, int] = {}
    seen_issues: set[int] = set()
    examples: list[Example] = []

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for c in pr_commits(repo, branch):
        if limit and len(examples) >= limit:
            break
        if SKIP_TITLE_RE.search(c["title"]):
            skip("housekeeping_title"); continue

        changed = {p: ls for p, ls in changed_old_lines(repo, c["base"], c["sha"]).items()
                   if not is_test_path(p)}
        if not changed:
            skip("no_modified_source_py"); continue
        if len(changed) > max_files:
            skip("too_many_files"); continue

        gold_syms: list[str] = []
        gold_mods: list[str] = []
        for path, lines in changed.items():
            try:
                src = git(repo, "show", f"{c['base']}:{path}").encode()
            except subprocess.CalledProcessError:
                continue
            mod = module_name_for(Path(path))
            gold_mods.append(mod)
            chunks = chunker.chunk_bytes(src, path, mod)
            spans = [(ch.start_line, ch.end_line, ch.symbol) for ch in chunks if ch.kind != "module"]
            for ln in lines:
                inner = [(b - a, s) for a, b, s in spans if a <= ln <= b]
                if inner:
                    sym = min(inner)[1]
                    if sym not in gold_syms:
                        gold_syms.append(sym)
        if not gold_syms:
            skip("module_level_only"); continue
        if len(gold_syms) > max_symbols:
            skip("too_many_symbols"); continue

        issue_no, query, src_name = None, c["title"], "pr_title"
        if gh is not None:
            pr = gh.get(f"/pulls/{c['pr']}")
            if pr is None:
                skip("pr_not_found"); continue
            issues = [i for i in linked_issues(pr) if i not in seen_issues]
            issue = None
            for n in issues:
                cand = gh.get(f"/issues/{n}")
                if cand and "pull_request" not in cand:
                    issue = cand
                    break
            if issue is None:
                skip("no_linked_issue"); continue
            issue_no = issue["number"]
            seen_issues.add(issue_no)
            query = clean_issue_text(issue["title"], issue.get("body") or "")
            src_name = "issue"

        names = {s.split("@")[0].rsplit(".", 1)[-1] for s in gold_syms}
        mentions = any(re.search(rf"\b{re.escape(n)}\b", query) for n in names if len(n) > 2)
        examples.append(Example(
            id=f"pr{c['pr']}", query=query, source=src_name, pr=c["pr"], issue=issue_no,
            commit=c["sha"], base=c["base"], gold_modules=sorted(set(gold_mods)),
            gold_symbols=gold_syms, gold_files=sorted(changed), mentions_gold=mentions,
            meta={"pr_title": c["title"]},
        ))
    return examples, skipped


def save(examples: list[Example], path: Path) -> None:
    with open(path, "w") as f:
        for e in examples:
            f.write(json.dumps(asdict(e)) + "\n")


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]
