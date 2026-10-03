"""AST-aware chunking of a Python repository with tree-sitter.

Chunk units:
  - function / method : signature + docstring + decorators + full body
  - class             : a *skeleton* (header, docstring, class attributes,
                        method signatures) so method bodies aren't duplicated
  - module            : docstring, imports, top-level assignments, and an
                        index of the symbols the module defines

Nested functions stay inside their parent's chunk. Functions longer than
`max_lines` are split at statement boundaries; every part repeats the
signature so it still makes sense on its own.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

PY_LANGUAGE = Language(tspython.language())

SKIP_DIRS = {
    ".git", ".hg", ".tox", ".nox", ".venv", "venv", "env", "node_modules",
    "build", "dist", "__pycache__", "site-packages", ".mypy_cache", ".eggs",
}


@dataclass
class Chunk:
    id: str                  # unique; symbol, plus "#pN" for split parts
    symbol: str              # qualified name, e.g. click.core.Command.invoke
    kind: str                # module | class | function | method
    name: str
    file: str                # repo-relative POSIX path
    start_line: int          # 1-indexed, inclusive (includes decorators)
    end_line: int
    name_line: int           # line of the `def`/`class` name (for resolution)
    signature: str
    docstring: str | None
    decorators: list[str]
    parent: str | None       # enclosing class or module symbol
    text: str
    part: int = 1
    num_parts: int = 1
    content_hash: str = field(default="")

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha1(self.text.encode()).hexdigest()[:16]

    @property
    def is_test(self) -> bool:
        p = Path(self.file)
        return (
            any(part in {"test", "tests", "testing"} for part in p.parts)
            or p.name.startswith("test_")
            or self.name.startswith("test_")
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def module_name_for(rel_path: Path) -> str:
    """src/click/core.py -> click.core ; pkg/__init__.py -> pkg"""
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or rel_path.stem


def iter_python_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
            continue
        yield path


def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _line_start(node: Node, src: bytes) -> int:
    """Byte offset of the start of the line `node` starts on (keeps indentation)."""
    return src.rfind(b"\n", 0, node.start_byte) + 1


def _docstring(body: Node | None, src: bytes) -> str | None:
    if body is None or body.named_child_count == 0:
        return None
    first = body.named_children[0]
    if first.type == "expression_statement" and first.named_child_count:
        s = first.named_children[0]
        if s.type == "string":
            raw = _text(s, src)
            for q in ('"""', "'''", '"', "'"):
                i = raw.find(q)
                if i != -1 and raw.endswith(q) and len(raw) >= i + 2 * len(q):
                    return raw[i + len(q):-len(q)].strip()
            return raw
    return None


def _unwrap(node: Node) -> tuple[Node, list[Node]]:
    """decorated_definition -> (inner def/class, decorator nodes)."""
    if node.type == "decorated_definition":
        decs = [c for c in node.named_children if c.type == "decorator"]
        return node.child_by_field_name("definition"), decs
    return node, []


def _signature(defn: Node, src: bytes) -> str:
    body = defn.child_by_field_name("body")
    end = body.start_byte if body is not None else defn.end_byte
    return " ".join(src[defn.start_byte:end].decode("utf-8", "replace").split()).rstrip(":").strip()


# --------------------------------------------------------------------------- #
# chunker
# --------------------------------------------------------------------------- #
class Chunker:
    def __init__(self, max_lines: int = 80):
        self.parser = Parser(PY_LANGUAGE)
        self.max_lines = max_lines

    def parse(self, src: bytes):
        return self.parser.parse(src)

    def chunk_file(self, path: Path, root: Path) -> list[Chunk]:
        rel = path.relative_to(root)
        return self.chunk_bytes(path.read_bytes(), rel.as_posix(), module_name_for(rel))

    def chunk_bytes(self, src: bytes, rel_posix: str, mod: str) -> list[Chunk]:
        """Chunk source that may not exist on disk (e.g. a file at an old commit)."""
        tree = self.parse(src)
        chunks: list[Chunk] = []
        defined: list[str] = []

        def visit_block(block: Node, scope: str, in_class: bool) -> None:
            for child in block.named_children:
                if child.type not in ("function_definition", "class_definition", "decorated_definition"):
                    continue
                defn, decs = _unwrap(child)
                if defn is None:
                    continue
                name = _text(defn.child_by_field_name("name"), src)
                symbol = f"{scope}.{name}"
                if defn.type == "function_definition":
                    if any(_text(d, src).lstrip("@").split("(")[0].endswith("overload") for d in decs):
                        continue  # typing.overload stubs: the real implementation follows
                    kind = "method" if in_class else "function"
                    chunks.extend(self._function_chunks(child, defn, decs, src, rel_posix, symbol, name, kind, scope))
                    if scope == mod:
                        defined.append(f"def {name}")
                else:
                    chunks.append(self._class_chunk(child, defn, decs, src, rel_posix, symbol, name, scope))
                    if scope == mod:
                        defined.append(f"class {name}")
                    body = defn.child_by_field_name("body")
                    if body is not None:
                        visit_block(body, symbol, in_class=True)

        visit_block(tree.root_node, mod, in_class=False)
        chunks.insert(0, self._module_chunk(tree.root_node, src, rel_posix, mod, defined))
        return chunks

    # -- individual chunk builders ----------------------------------------- #
    def _function_chunks(self, outer, defn, decs, src, file, symbol, name, kind, parent) -> list[Chunk]:
        body = defn.child_by_field_name("body")
        start, end = outer.start_point[0] + 1, outer.end_point[0] + 1
        common = dict(
            symbol=symbol, kind=kind, name=name, file=file,
            name_line=defn.child_by_field_name("name").start_point[0] + 1,
            signature=_signature(defn, src), docstring=_docstring(body, src),
            decorators=[_text(d, src) for d in decs], parent=parent,
        )
        full_text = src[_line_start(outer, src):outer.end_byte].decode("utf-8", "replace")

        if end - start + 1 <= self.max_lines or body is None or body.named_child_count < 2:
            return [Chunk(id=symbol, start_line=start, end_line=end, text=full_text, **common)]

        # Split at statement boundaries; each part gets the header prepended.
        header = src[_line_start(outer, src):body.start_byte].decode("utf-8", "replace").rstrip()
        groups: list[list[Node]] = [[]]
        for stmt in body.named_children:
            cur = groups[-1]
            if cur and (stmt.end_point[0] - cur[0].start_point[0] + 1) > self.max_lines:
                groups.append([])
            groups[-1].append(stmt)

        n = len(groups)
        parts = []
        for i, g in enumerate(groups, 1):
            seg = src[_line_start(g[0], src):g[-1].end_byte].decode("utf-8", "replace")
            text = f"{header}\n    # ... part {i}/{n} of {name}\n{seg}"
            parts.append(Chunk(
                id=f"{symbol}#p{i}", part=i, num_parts=n,
                start_line=start if i == 1 else g[0].start_point[0] + 1,
                end_line=g[-1].end_point[0] + 1, text=text, **common,
            ))
        return parts

    def _class_chunk(self, outer, defn, decs, src, file, symbol, name, parent) -> Chunk:
        body = defn.child_by_field_name("body")
        lines = [_text(d, src) for d in decs] + [_signature(defn, src) + ":"]
        doc = _docstring(body, src)
        if doc:
            lines.append(f'    """{doc}"""')
        if body is not None:
            for c in body.named_children:
                inner, _ = _unwrap(c)
                if inner is not None and inner.type == "function_definition":
                    lines.append(f"    {_signature(inner, src)}: ...")
                elif inner is not None and inner.type == "class_definition":
                    lines.append(f"    {_signature(inner, src)}: ...")
                elif c.type == "expression_statement" and c is not body.named_children[0]:
                    lines.append("    " + _text(c, src).split("\n")[0])
        return Chunk(
            id=symbol, symbol=symbol, kind="class", name=name, file=file,
            start_line=outer.start_point[0] + 1, end_line=outer.end_point[0] + 1,
            name_line=defn.child_by_field_name("name").start_point[0] + 1,
            signature=_signature(defn, src), docstring=doc,
            decorators=[_text(d, src) for d in decs], parent=parent,
            text="\n".join(lines),
        )

    def _module_chunk(self, root: Node, src, file, mod, defined) -> Chunk:
        doc = _docstring(root, src)
        lines = [f"# module {mod} ({file})"]
        if doc:
            lines.append(f'"""{doc}"""')
        for c in root.named_children:
            if c.type in ("import_statement", "import_from_statement", "future_import_statement"):
                lines.append(_text(c, src))
            elif c.type == "expression_statement" and c.named_child_count and \
                    c.named_children[0].type == "assignment":
                lines.append(_text(c, src).split("\n")[0][:200])
        if defined:
            lines.append("# defines: " + ", ".join(defined))
        n_lines = src.count(b"\n") + 1
        return Chunk(
            id=mod, symbol=mod, kind="module", name=mod.rsplit(".", 1)[-1], file=file,
            start_line=1, end_line=n_lines, name_line=1, signature=f"module {mod}",
            docstring=doc, decorators=[], parent=None, text="\n".join(lines),
        )


def chunk_repo(root: Path, max_lines: int = 80) -> list[Chunk]:
    chunker = Chunker(max_lines=max_lines)
    out: list[Chunk] = []
    for path in iter_python_files(root):
        try:
            out.extend(chunker.chunk_file(path, root))
        except Exception as e:  # never let one bad file kill indexing
            print(f"[chunker] skipped {path}: {e}")
    _disambiguate(out)
    return out


def _disambiguate(chunks: list[Chunk]) -> None:
    """Same qualified name defined twice (property setters, @overload, if/else
    definitions, same module name in two places): suffix later ones with @line."""
    seen: dict[str, tuple[str, int]] = {}
    for c in chunks:
        key = (c.file, c.name_line)
        if c.symbol in seen and seen[c.symbol] != key:
            new = f"{c.symbol}@{c.file}:{c.name_line}" if c.kind == "module" else f"{c.symbol}@{c.name_line}"
            c.id = c.id.replace(c.symbol, new, 1)
            c.symbol = new
        else:
            seen.setdefault(c.symbol, key)
