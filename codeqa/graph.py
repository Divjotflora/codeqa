"""Code graph over the chunks produced by `chunker.py`.

Nodes are symbols (modules, classes, functions, methods). Edge types:
  contains    module -> class -> method          (structure)
  calls       caller -> callee                   (resolved with Jedi)
  inherits    subclass -> base class
  imports     module -> module
  tested_by   symbol -> test function that calls it directly

Call targets are resolved with Jedi's static inference (`goto`), so
`self.save()` and imported aliases land on the right definition instead of
on every function called `save`. Python is dynamic, so some calls can't be
resolved; those fall back to a unique-name match when exactly one symbol in
the repo has that name, and everything else is counted in `stats` so you
can report resolution coverage honestly.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import networkx as nx

from .chunker import Chunk

# --------------------------------------------------------------------------- #
# worker: parse one file, find call sites / base classes, resolve with Jedi
# --------------------------------------------------------------------------- #
_W: dict = {}


def _init_worker(repo_root: str) -> None:
    import jedi
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    root = Path(repo_root).resolve()
    _W["root"] = root
    # Put the repo itself FIRST on sys.path. Otherwise, if the package is also
    # pip-installed (common: click, requests, ...), Jedi resolves `import pkg`
    # to site-packages and every internal call looks "external".
    repo_paths = [str(p) for p in (root / "src", root) if p.is_dir()]
    env_paths = [p for p in jedi.get_default_environment().get_sys_path() if p not in repo_paths]
    _W["project"] = jedi.Project(repo_root, sys_path=repo_paths + env_paths)
    _W["parser"] = Parser(Language(tspython.language()))


def _callee_name_node(call):
    fn = call.child_by_field_name("function")
    if fn is None:
        return None
    if fn.type == "identifier":
        return fn
    if fn.type == "attribute":
        return fn.child_by_field_name("attribute")
    return None  # e.g. f()(), d[k](), lambda calls


def _resolve_file(rel: str) -> dict:
    import jedi

    root: Path = _W["root"]
    path = root / rel
    src = path.read_bytes()
    lines = src.split(b"\n")
    tree = _W["parser"].parse(src)

    sites = []  # (kind, name_node)
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type == "call":
            nn = _callee_name_node(n)
            if nn is not None:
                sites.append(("call", nn))
        elif n.type == "class_definition":
            supers = n.child_by_field_name("superclasses")
            if supers is not None:
                for arg in supers.named_children:
                    if arg.type == "identifier":
                        sites.append(("inherit", arg))
                    elif arg.type == "attribute":
                        sites.append(("inherit", arg.child_by_field_name("attribute")))
        stack.extend(n.children)

    imports = []
    for c in tree.root_node.named_children:
        if c.type == "import_statement":
            for d in c.named_children:
                target = d.child_by_field_name("name") if d.type == "aliased_import" else d
                imports.append({"module": target.text.decode(), "names": [], "level": 0})
        elif c.type == "import_from_statement":
            mod = c.child_by_field_name("module_name")
            text = mod.text.decode() if mod is not None else ""
            level = len(text) - len(text.lstrip("."))
            names = []
            for d in c.named_children[1:]:
                t = d.child_by_field_name("name") if d.type == "aliased_import" else d
                if t is not None and t.type == "dotted_name":
                    names.append(t.text.decode())
            imports.append({"module": text.lstrip("."), "names": names, "level": level})

    out = []
    try:
        script = jedi.Script(src.decode("utf-8", "replace"), path=str(path), project=_W["project"])
    except Exception:
        script = None

    for kind, node in sites:
        row, bcol = node.start_point
        col = len(lines[row][:bcol].decode("utf-8", "replace"))
        targets = []
        if script is not None:
            try:
                for d in script.goto(row + 1, col, follow_imports=True):
                    mp = d.module_path
                    in_repo = None
                    if mp is not None:
                        try:
                            in_repo = Path(mp).resolve().relative_to(root).as_posix()
                        except ValueError:
                            in_repo = None
                    targets.append({
                        "file": in_repo, "line": d.line, "name": d.name, "type": d.type,
                        "builtin": d.in_builtin_module(),
                    })
            except Exception:
                pass
        out.append({"kind": kind, "line": row + 1, "name": node.text.decode(), "targets": targets})

    return {"file": rel, "sites": out, "imports": imports}


# --------------------------------------------------------------------------- #
# graph
# --------------------------------------------------------------------------- #
class CodeGraph:
    def __init__(self, g: nx.MultiDiGraph | None = None, stats: dict | None = None):
        self.g = g if g is not None else nx.MultiDiGraph()
        self.stats = stats or {}

    # ---------------- construction ---------------- #
    @classmethod
    def build(cls, repo_root: Path, chunks: list[Chunk], workers: int | None = None) -> "CodeGraph":
        self = cls()
        g = self.g

        # one node per symbol (split chunks share a symbol)
        by_symbol: dict[str, Chunk] = {}
        for c in chunks:
            if c.symbol not in by_symbol:
                by_symbol[c.symbol] = c
                g.add_node(c.symbol, kind=c.kind, name=c.name, file=c.file,
                           start_line=c.start_line, end_line=c.end_line, is_test=c.is_test)
            else:  # extend span over all parts
                g.nodes[c.symbol]["end_line"] = max(g.nodes[c.symbol]["end_line"], c.end_line)

        for s, c in by_symbol.items():
            if c.parent and c.parent in by_symbol:
                g.add_edge(c.parent, s, key="contains", type="contains")

        # lookup tables
        def_at: dict[tuple[str, int], str] = {}
        module_of_file: dict[str, str] = {}
        spans: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        by_name: dict[str, list[str]] = defaultdict(list)
        for s, c in by_symbol.items():
            if c.kind == "module":
                module_of_file[c.file] = s
            else:
                def_at[(c.file, c.name_line)] = s
                spans[c.file].append((c.start_line, g.nodes[s]["end_line"], s))
                by_name[c.name].append(s)

        def enclosing(file: str, line: int) -> str | None:
            best, best_len = module_of_file.get(file), float("inf")
            for a, b, s in spans.get(file, ()):
                if a <= line <= b and b - a < best_len:
                    best, best_len = s, b - a
            return best

        # Resolve one file per task. Each worker holds its own Jedi caches (a few
        # hundred MB), so cap the pool; if worker processes die anyway (low memory,
        # or Windows/Microsoft Store Python multiprocessing quirks), fall back to
        # resolving everything in this process.
        files = sorted(module_of_file)
        workers = workers or min(4, max(1, (os.cpu_count() or 2) - 1))
        results = None
        if workers > 1:
            try:
                with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                         initargs=(str(repo_root),)) as ex:
                    results = list(ex.map(_resolve_file, files, chunksize=4))
            except BrokenProcessPool:
                print("[graph] worker processes crashed; falling back to a single process")
        if results is None:
            _init_worker(str(repo_root))
            results = []
            for i, f in enumerate(files, 1):
                results.append(_resolve_file(f))
                if i % 20 == 0 or i == len(files):
                    print(f"[graph] resolved {i}/{len(files)} files", flush=True)

        st = Counter()
        for res in results:
            f = res["file"]
            for site in res["sites"]:
                src_sym = enclosing(f, site["line"])
                if src_sym is None:
                    continue
                etype = "calls" if site["kind"] == "call" else "inherits"
                internal = []
                external = builtin = False
                for t in site["targets"]:
                    if t["file"] is None:
                        builtin |= bool(t["builtin"])
                        external |= not t["builtin"]
                        continue
                    if t["type"] == "module":
                        sym = module_of_file.get(t["file"])
                    else:
                        sym = def_at.get((t["file"], t["line"]))
                    if sym is not None:
                        internal.append(sym)

                if internal:
                    via, targets = "jedi", internal
                elif builtin:
                    st[f"{etype}_builtin"] += 1
                    continue
                elif external:
                    st[f"{etype}_external"] += 1
                    continue
                else:  # Jedi gave nothing usable: unique-name fallback
                    cands = by_name.get(site["name"], [])
                    if len(cands) == 1:
                        via, targets = "name_unique", cands
                    else:
                        st[f"{etype}_unresolved"] += 1
                        continue

                st[f"{etype}_{via}"] += 1
                for dst in set(targets):
                    if dst != src_sym:
                        self._add(src_sym, dst, etype, via=via, line=site["line"])

            # module-level imports
            src_mod = module_of_file[f]
            pkg = src_mod.split(".")
            if not f.endswith("__init__.py"):
                pkg = pkg[:-1]
            for imp in res["imports"]:
                base = imp["module"]
                if imp["level"]:
                    prefix = pkg[: len(pkg) - (imp["level"] - 1)] if imp["level"] > 1 else pkg
                    base = ".".join(prefix + ([base] if base else []))
                cands = [f"{base}.{n}" for n in imp["names"]] + [base]
                hit = next((m for m in cands if m in by_symbol and by_symbol[m].kind == "module"), None)
                if hit and hit != src_mod:
                    self._add(src_mod, hit, "imports")

        # tested_by: direct calls from test functions into non-test code
        for u, v, k in list(g.edges(keys=True)):
            if k == "calls" and g.nodes[u]["is_test"] and not g.nodes[v]["is_test"] \
                    and g.nodes[u]["kind"] in ("function", "method"):
                self._add(v, u, "tested_by")

        resolved = st["calls_jedi"] + st["calls_name_unique"]
        in_scope = resolved + st["calls_unresolved"]
        self.stats = {
            "symbols": g.number_of_nodes(),
            "edges": dict(Counter(k for _, _, k in g.edges(keys=True))),
            "call_sites": dict(sorted((k, v) for k, v in st.items() if k.startswith("calls"))),
            "inherit_sites": dict(sorted((k, v) for k, v in st.items() if k.startswith("inherits"))),
            # share of non-builtin, non-external calls we could pin to a repo symbol
            "internal_call_resolution_rate": round(resolved / in_scope, 4) if in_scope else None,
        }
        return self

    def _add(self, u: str, v: str, etype: str, via: str | None = None, line: int | None = None):
        if self.g.has_edge(u, v, key=etype):
            if line is not None:
                self.g.edges[u, v, etype].setdefault("lines", []).append(line)
            return
        attrs = {"type": etype}
        if via:
            attrs["via"] = via
        if line is not None:
            attrs["lines"] = [line]
        self.g.add_edge(u, v, key=etype, **attrs)

    # ---------------- queries ---------------- #
    def _nbrs(self, s: str, etype: str, reverse: bool = False) -> list[str]:
        edges = self.g.in_edges(s, keys=True) if reverse else self.g.out_edges(s, keys=True)
        return sorted({(u if reverse else v) for u, v, k in edges if k == etype})

    def callees(self, s: str) -> list[str]:
        return self._nbrs(s, "calls")

    def callers(self, s: str) -> list[str]:
        return self._nbrs(s, "calls", reverse=True)

    def tests_for(self, s: str) -> list[str]:
        return self._nbrs(s, "tested_by")

    def neighbors(self, s: str) -> list[str]:
        """1-hop expansion used at retrieval time."""
        out = set(self.callees(s)) | set(self.callers(s))
        out |= set(self._nbrs(s, "inherits")) | set(self._nbrs(s, "contains", reverse=True))
        return sorted(out - {s})

    def impact(self, s: str, max_depth: int = 3) -> dict:
        """What could break if `s` changes: transitive callers (plus subclasses
        for classes), with hop distance, and the tests that exercise them."""
        dist = {s: 0}
        q = deque([s])
        while q:
            cur = q.popleft()
            if dist[cur] >= max_depth:
                continue
            nxt = self.callers(cur) + self._nbrs(cur, "inherits", reverse=True)
            for p in nxt:
                if p not in dist:
                    dist[p] = dist[cur] + 1
                    q.append(p)
        affected = {k: d for k, d in dist.items() if k != s}
        code = {k: d for k, d in affected.items() if not self.g.nodes[k]["is_test"]}
        tests = {k: d for k, d in affected.items() if self.g.nodes[k]["is_test"]}
        for k in [s, *code]:
            for t in self.tests_for(k):
                tests.setdefault(t, dist[k] + 1)
        return {
            "symbol": s,
            "affected": sorted(code.items(), key=lambda x: (x[1], x[0])),
            "tests": sorted(tests.items(), key=lambda x: (x[1], x[0])),
        }

    # ---------------- persistence ---------------- #
    def save(self, path: Path) -> None:
        data = nx.node_link_data(self.g, edges="edges")
        data["stats"] = self.stats
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "CodeGraph":
        data = json.loads(path.read_text())
        stats = data.pop("stats", {})
        g = nx.node_link_graph(data, directed=True, multigraph=True, edges="edges")
        return cls(g, stats)
