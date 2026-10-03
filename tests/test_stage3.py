import json
from types import SimpleNamespace

import networkx as nx

from codeqa.chunker import Chunk
from codeqa.embed import DenseRetriever, HashingEmbedder
from codeqa.fusion import RRF, GraphExpand
from codeqa.graph import CodeGraph


def chunk(sym, text, kind="function"):
    return Chunk(id=sym, symbol=sym, kind=kind, name=sym.rsplit(".", 1)[-1], file="m.py",
                 start_line=1, end_line=2, name_line=1, signature="", docstring=None,
                 decorators=[], parent="m", text=text)


class Fixed:
    def __init__(self, name, ranking):
        self.name, self.ranking = name, ranking

    def retrieve(self, q, k=50):
        return self.ranking[:k]


def test_rrf_rewards_agreement():
    r = RRF([Fixed("a", ["x", "y", "z"]), Fixed("b", ["y", "q", "x"])])
    assert r.retrieve("q", 2) == ["y", "x"]


def test_graph_expand_pulls_in_callee():
    g = nx.MultiDiGraph()
    for n in ["m.entry", "m.helper", "m.other", "m.hub"]:
        g.add_node(n, kind="function", is_test=False)
    g.add_edge("m.entry", "m.helper", key="calls", type="calls")
    base = Fixed("b", ["m.entry", "m.other", "m.hub"])
    out = GraphExpand(base, CodeGraph(g), seeds=1, alpha=0.6).retrieve("q", 4)
    assert out[:2] == ["m.entry", "m.helper"]   # 0.6 beats m.other's 1/2


def test_dense_cache_is_incremental(tmp_path):
    chunks = [chunk("m.parse_args", "def parse_args(argv): split argv into options"),
              chunk("m.render", "def render(page): draw html page")]
    r1 = DenseRetriever(chunks, HashingEmbedder(), cache_dir=tmp_path)
    assert r1.newly_embedded == 2 and r1.retrieve("parse the options", 1) == ["m.parse_args"]
    chunks[1] = chunk("m.render", "def render(page): draw html page, now with css")
    r2 = DenseRetriever(chunks, HashingEmbedder(), cache_dir=tmp_path)
    assert r2.newly_embedded == 1


def test_summarize_resumes(tmp_path, monkeypatch):
    import anthropic
    from codeqa import summarize

    calls = []

    class FakeClient:
        def __init__(self, **kw):
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Parses CLI flags.")],
                                   usage=SimpleNamespace(input_tokens=100, output_tokens=10))

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    chunks = [chunk("m.a", "def a(): pass"), chunk("m.b", "def b(): return 1")]
    out = tmp_path / "s.jsonl"
    assert summarize.summarize_chunks(chunks, out, workers=2)["summarized"] == 2
    assert summarize.summarize_chunks(chunks, out)["summarized"] == 0     # cached
    assert len(calls) == 2
    assert set(summarize.load_summaries(out).values()) == {"Parses CLI flags."}


def test_summarize_openai_compatible(tmp_path):
    import http.server
    import threading
    from codeqa import summarize

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/chat/completions" and body["model"] == "tiny"
            out = json.dumps({"choices": [{"message": {"content": "Renders help text."}}],
                              "usage": {"prompt_tokens": 50, "completion_tokens": 5}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = tmp_path / "s.jsonl"
    r = summarize.summarize_chunks([chunk("m.a", "def a(): pass")], out, model="tiny", workers=1,
                                   backend="openai", base_url=f"http://127.0.0.1:{srv.server_port}/v1")
    srv.shutdown()
    assert r["summarized"] == 1 and r["input_tokens"] == 50
    assert list(summarize.load_summaries(out).values()) == ["Renders help text."]


def test_reranker_reorders_head_only(tmp_path):
    from codeqa.rerank import Reranker, ScoreCache

    chunks = [chunk(f"m.f{i}", f"def f{i}(): pass") for i in range(6)]
    base = Fixed("b", [f"m.f{i}" for i in range(6)])
    calls = []

    def scorer(pairs):
        calls.append(len(pairs))
        return [float(d.split("\n")[0][-1]) for _, d in pairs]    # f5 > f4 > ...
    scorer.name = "fake"

    cache = ScoreCache(tmp_path / "ce.json")
    ce = Reranker(base, chunks, scorer, "ce", top_n=3, cache=cache)
    assert ce.retrieve("q", 6) == ["m.f2", "m.f1", "m.f0", "m.f3", "m.f4", "m.f5"]
    rrf = Reranker(base, chunks, scorer, "rrf", top_n=3, cache=cache)
    assert rrf.retrieve("q", 6)[3:] == ["m.f3", "m.f4", "m.f5"]
    assert calls == [3]                      # second reranker hit the shared cache
    cache.save()
    assert len(ScoreCache(tmp_path / "ce.json").data) == 3


def test_condense_query():
    from codeqa.rerank import condense_query

    q = ('Prompt crashes on empty input\n\nWhen I call `click.prompt` with `default=None`:\n'
         'Traceback (most recent call last):\n  File "x/termui.py", line 168, in prompt_func\n'
         'ValueError: bad value. See https://github.com/pallets/click e.g. visible_prompt_func')
    assert condense_query(q, "title") == "Prompt crashes on empty input"
    c = condense_query(q, "title+ids")
    assert c.startswith("Prompt crashes on empty input\nmentions: prompt_func, click.prompt")
    assert "visible_prompt_func" in c and "github.com" not in c and "e.g" not in c
    assert condense_query(q, "full") == q
