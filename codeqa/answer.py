"""Answer a question from retrieved code, with every citation checked.

1. Take the top-k functions from the retriever and show them to the model as numbered
   snippets with real line numbers from the repo.
2. The model answers and cites `[path:start-end]`.
3. Every citation is verified against the repository:
     valid       the file exists, the lines exist, and they overlap a snippet the model
                 was actually shown
     ungrounded  real file and lines, but not from the shown snippets (the model cited
                 from memory or guessed)
     bad_range   the file exists but the lines don't
     bad_file    no such file
   Valid citations are mapped back to the function they point at, so an answer can be
   scored against the gold functions of an eval question.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import Chunk

PROMPT_VERSION = "v2"
PROMPT = """You answer questions about a Python codebase using ONLY the code snippets below.
Each snippet has a label like [S1], and each code line starts with its line number.

{snippets}

Question:
{question}

Reply in exactly this format:
Answer: 2-4 sentences in your own words saying which function(s) are responsible and why.
Put the snippet label right after each claim, like [S2]. You may also cite exact lines
as [path:start-end], for example [{example}].
Sources: the labels you relied on, for example [S2], [S4]

Rules: cite only snippets shown above. Do not repeat the question. If none of the
snippets is relevant, write "Answer: The snippets don't show this." and "Sources: none"."""

CITE_RE = re.compile(r"([A-Za-z0-9_.\-/\\]+\.py)\s*:\s*L?(\d+)(?:\s*[-\u2013]\s*L?(\d+))?")
LABEL_GROUP_RE = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]")
BACKTICK_RE = re.compile(r"`([A-Za-z_][\w.]*)(?:\(\))?`")
NOT_FOUND_RE = re.compile(r"snippets don.t show|Sources:\s*none", re.I)


@dataclass
class Snippet:
    label: str
    symbol: str
    file: str
    start: int
    end: int
    text: str


@dataclass
class Citation:
    raw: str
    file: str | None
    start: int
    end: int
    status: str = "unchecked"
    symbol: str | None = None
    kind: str = "path"      # path: [file:lines]   label: [S2]   name: `Class.method` of a shown snippet


@dataclass
class Answer:
    question: str
    text: str
    shown: list[Snippet]
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    seconds: float = 0.0
    tokens: tuple[int, int] = (0, 0)

    @property
    def valid_symbols(self) -> list[str]:
        """Symbols backed by a verified citation (path or label)."""
        out = []
        for c in self.citations:
            if c.kind != "name" and c.status == "valid" and c.symbol and c.symbol not in out:
                out.append(c.symbol)
        return out

    @property
    def named_symbols(self) -> list[str]:
        """Shown symbols the answer cites or merely names in backticks (lenient)."""
        out = list(self.valid_symbols)
        for c in self.citations:
            if c.kind == "name" and c.symbol and c.symbol not in out:
                out.append(c.symbol)
        return out

    @property
    def says_not_found(self) -> bool:
        return bool(NOT_FOUND_RE.search(self.text))

    def annotated(self) -> str:
        """Answer text with every non-valid citation visibly marked."""
        text = self.text
        for c in self.citations:
            if c.kind != "name" and c.status != "valid":
                text = text.replace(c.raw, f"{c.raw} (UNVERIFIED: {c.status})", 1)
        return text


class Repo:
    """File access + symbol lookup for one indexed repository."""

    def __init__(self, root: Path, chunks: list[Chunk]):
        self.root = root
        self._lines: dict[str, list[str]] = {}
        self.files = sorted({c.file for c in chunks})
        self.first_chunk: dict[str, Chunk] = {}
        self.span: dict[str, tuple[int, int]] = {}
        self.spans_by_file: dict[str, list[tuple[int, int, str]]] = {}
        for c in chunks:
            if c.kind == "module":
                continue
            if c.symbol not in self.first_chunk:
                self.first_chunk[c.symbol] = c
                self.span[c.symbol] = (c.start_line, c.end_line)
            else:
                a, b = self.span[c.symbol]
                self.span[c.symbol] = (min(a, c.start_line), max(b, c.end_line))
        for s, (a, b) in self.span.items():
            self.spans_by_file.setdefault(self.first_chunk[s].file, []).append((a, b, s))

    def lines(self, file: str) -> list[str]:
        if file not in self._lines:
            self._lines[file] = (self.root / file).read_text(encoding="utf-8", errors="replace").splitlines()
        return self._lines[file]

    def resolve(self, path: str) -> str | None:
        p = path.replace("\\", "/").lstrip("./")
        if p in self.files:
            return p
        hits = [f for f in self.files if f.endswith("/" + p)]   # model dropped a prefix like src/
        return hits[0] if len(hits) == 1 else None

    def symbol_at(self, file: str, start: int, end: int) -> str | None:
        best = None
        for a, b, s in self.spans_by_file.get(file, ()):
            ov = min(b, end) - max(a, start) + 1
            if ov > 0:
                key = (a <= start <= b, ov, -(b - a))   # contains start, most overlap, innermost
                if best is None or key > best[0]:
                    best = (key, s)
        return best[1] if best else None


def build_snippets(repo: Repo, symbols: list[str], k: int = 5, max_lines: int = 40) -> list[Snippet]:
    out = []
    for s in symbols:
        if s not in repo.span:
            continue   # module symbols, unknown names
        c = repo.first_chunk[s]
        a, b = repo.span[s]
        lines = repo.lines(c.file)
        b2 = min(b, a + max_lines - 1, len(lines))
        body = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(a, b2 + 1))
        if b2 < b:
            body += f"\n       ... ({b - b2} more lines not shown)"
        label = f"S{len(out) + 1}"
        out.append(Snippet(label, s, c.file, a, b2, f"[{label}] {c.file}:{a}-{b2}  ({s})\n{body}"))
        if len(out) >= k:
            break
    return out


def parse_citations(text: str) -> list[Citation]:
    """[path:start-end] citations. Labels and names are resolved in verify(), which
    knows which snippets were shown."""
    out = []
    for m in CITE_RE.finditer(text):
        start = int(m.group(2))
        end = int(m.group(3)) if m.group(3) else start
        out.append(Citation(m.group(0), m.group(1), min(start, end), max(start, end)))
    return out


def _label_citations(text: str, shown: list[Snippet]) -> list[Citation]:
    by_label = {s.label: s for s in shown}
    out = []
    for m in LABEL_GROUP_RE.finditer(text):
        for lab in re.findall(r"S\d+", m.group(0)):
            s = by_label.get(lab)
            if s is None:
                out.append(Citation(m.group(0), None, 0, 0, "bad_label", None, "label"))
            else:
                out.append(Citation(f"[{lab}]", s.file, s.start, s.end, "valid", s.symbol, "label"))
    return out


def _name_mentions(text: str, shown: list[Snippet]) -> list[Citation]:
    """`FloatConverter.to_url` / `Client.open()` naming a shown snippet's symbol."""
    out, seen = [], set()
    for m in BACKTICK_RE.finditer(text):
        name = m.group(1).strip(".")
        if not name or name in seen:
            continue
        for s in shown:
            sym = s.symbol.split("@")[0]
            if sym == name or sym.endswith("." + name):
                seen.add(name)
                out.append(Citation(m.group(0), s.file, s.start, s.end, "named", s.symbol, "name"))
                break
    return out


def verify(repo: Repo, citations: list[Citation], shown: list[Snippet], text: str = "") -> list[Citation]:
    for c in citations:
        f = repo.resolve(c.file or "")
        if f is None:
            c.status = "bad_file"
            continue
        c.file = f
        n = len(repo.lines(f))
        if not (1 <= c.start <= c.end <= n):
            c.status = "bad_range"
            continue
        grounded = any(s.file == f and min(s.end, c.end) >= max(s.start, c.start) for s in shown)
        c.status = "valid" if grounded else "ungrounded"
        c.symbol = repo.symbol_at(f, c.start, c.end)
    if text:
        citations += _label_citations(text, shown) + _name_mentions(text, shown)
    return citations


def answer_question(question: str, retriever, repo: Repo, llm, k: int = 5, max_lines: int = 40,
                    question_chars: int = 2000) -> Answer:
    ranked = retriever.retrieve(question, 50)
    shown = build_snippets(repo, ranked, k, max_lines)
    example = f"{shown[0].file}:{shown[0].start}-{min(shown[0].end, shown[0].start + 3)}" if shown \
        else "src/pkg/module.py:10-14"
    prompt = PROMPT.format(snippets="\n\n".join(s.text for s in shown),
                           question=question[:question_chars], example=example)
    t0 = time.time()
    text, tin, tout = llm(prompt)
    ans = Answer(question, text, shown, retrieved=ranked, seconds=time.time() - t0, tokens=(tin, tout))
    ans.citations = verify(repo, parse_citations(text), shown, text)
    return ans
