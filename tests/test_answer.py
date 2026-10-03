from pathlib import Path

from codeqa.answer import Repo, answer_question, build_snippets, parse_citations, verify
from codeqa.chunker import Chunker

SRC = '''import os


def load(path):
    """Read a file."""
    with open(path) as f:
        return f.read()


def save(path, text):
    with open(path, "w") as f:
        f.write(text)


def unused():
    return 1
'''


class Fixed:
    name = "fixed"

    def __init__(self, ranking):
        self.ranking = ranking

    def retrieve(self, q, k=50):
        return self.ranking[:k]


def setup(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    p = tmp_path / "src" / "pkg" / "io.py"
    p.write_text(SRC)
    chunks = Chunker().chunk_file(p, tmp_path)
    return Repo(tmp_path, chunks)


def test_parse_citations():
    cs = parse_citations("See [src/pkg/io.py:4-7] and pkg\\io.py:11 and x.py: L3 - L5.")
    assert [(c.file, c.start, c.end) for c in cs] == [("src/pkg/io.py", 4, 7), ("pkg\\io.py", 11, 11), ("x.py", 3, 5)]


def test_verify_statuses(tmp_path):
    repo = setup(tmp_path)
    shown = build_snippets(repo, ["pkg.io.load", "pkg.io.save"], k=2)
    assert shown[0].text.splitlines()[1].strip().startswith("4  def load")
    cs = verify(repo, parse_citations(
        "[src/pkg/io.py:4-7] [pkg/io.py:10-12] [src/pkg/io.py:15-16] [src/pkg/io.py:90-95] [nope.py:1-2]"), shown)
    assert [c.status for c in cs] == ["valid", "valid", "ungrounded", "bad_range", "bad_file"]
    assert cs[0].symbol == "pkg.io.load" and cs[1].file == "src/pkg/io.py" and cs[1].symbol == "pkg.io.save"


def test_answer_question_end_to_end(tmp_path):
    repo = setup(tmp_path)
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return "load reads the file [src/pkg/io.py:4-7]; also see [src/pkg/io.py:40-41].", 100, 20

    a = answer_question("where are files read?", Fixed(["pkg.io", "pkg.io.load", "pkg.io.save"]), repo, llm, k=2)
    assert [s.symbol for s in a.shown] == ["pkg.io.load", "pkg.io.save"]   # module skipped
    assert "    4  def load(path):" in prompts[0]
    assert a.valid_symbols == ["pkg.io.load"]
    assert "(UNVERIFIED: bad_range)" in a.annotated()


def test_label_and_name_citations(tmp_path):
    repo = setup(tmp_path)

    def llm(prompt):
        return ("Answer: Files are read by `load()` [S1]; writing happens in `pkg.io.save`. [S1, S3]\n"
                "Sources: [S1]"), 10, 5

    a = answer_question("q", Fixed(["pkg.io.load", "pkg.io.save"]), repo, llm, k=2)
    kinds = [(c.kind, c.status, c.symbol) for c in a.citations]
    assert ("label", "valid", "pkg.io.load") in kinds and ("label", "bad_label", None) in kinds
    assert ("name", "named", "pkg.io.save") in kinds
    assert a.valid_symbols == ["pkg.io.load"]
    assert a.named_symbols == ["pkg.io.load", "pkg.io.save"]
    assert "(UNVERIFIED: bad_label)" in a.annotated() and not a.says_not_found


def test_answer_identifiers():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ci", Path(__file__).parent.parent / "check_identifiers.py")
    ci = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci)
    ids = ci.answer_identifiers("Uses `self._is_atty` and `ProgressBar.render()`; `None` and `len` skipped, `x` too.")
    assert ids == ["_is_atty", "ProgressBar", "render"]


def test_near_miss_index():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ci", Path(__file__).parent.parent / "check_identifiers.py")
    ci = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci)
    near = ci.near_miss_index({"_resolve_context", "ParameterSource", "_nullpager"})
    assert near[ci.loose("resolve_context")] == "_resolve_context"
    assert near[ci.loose("nullPager")] == "_nullpager"
    assert ci.loose("ParameterSourceMap") not in near
