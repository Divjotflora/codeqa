from pathlib import Path
from codeqa.chunker import Chunker

SRC = '''"""Mod doc."""
import os
from typing import overload

X = 1

class Store(Base):
    """A store."""
    limit = 10

    @property
    def size(self):
        return 1

    def save(self, item):
        def helper():
            return os.getcwd()
        return helper()

@overload
def f(x: int) -> int: ...
def f(x):
    return x
'''


def test_chunks(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    p = tmp_path / "pkg" / "mod.py"
    p.write_text(SRC)
    chunks = {c.id: c for c in Chunker().chunk_file(p, tmp_path)}

    assert set(chunks) == {"pkg.mod", "pkg.mod.Store", "pkg.mod.Store.size",
                           "pkg.mod.Store.save", "pkg.mod.f"}
    assert chunks["pkg.mod.Store.save"].kind == "method"
    assert "def helper" in chunks["pkg.mod.Store.save"].text      # nested stays inside
    assert chunks["pkg.mod.Store.size"].decorators == ["@property"]
    assert chunks["pkg.mod.Store.size"].start_line < chunks["pkg.mod.Store.size"].name_line
    cls = chunks["pkg.mod.Store"].text
    assert "def save(self, item): ..." in cls and "os.getcwd" not in cls  # skeleton only
    assert chunks["pkg.mod.f"].start_line == 22                   # overload stub skipped
    assert "# defines: class Store, def f" in chunks["pkg.mod"].text


def test_split_long_function(tmp_path: Path):
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    p = tmp_path / "big.py"
    p.write_text(f"def big():\n{body}\n")
    parts = Chunker(max_lines=50).chunk_file(p, tmp_path)[1:]
    assert len(parts) == 4 and all(c.symbol == "big.big" for c in parts)
    assert all(c.text.startswith("def big():") for c in parts)
    assert parts[-1].end_line == 201
