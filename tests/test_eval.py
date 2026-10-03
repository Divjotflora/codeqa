from codeqa.evalset import linked_issues, PR_RE
from codeqa.evaluate import _score
from codeqa.retrievers import tokenize


def test_linked_issues():
    pr = {"title": "Fix pager", "body": "Fixes #12, closes https://github.com/o/r/issues/34.\nsee #99"}
    assert linked_issues(pr) == [12, 34]          # "see #99" is not a closing keyword


def test_pr_re():
    assert PR_RE.search("Add thing (#3781)").group(1) == "3781"
    assert PR_RE.search("Merge pull request #110 from x/y").group(2) == "110"


def test_tokenize_splits_identifiers():
    t = tokenize("def get_terminal_size(self): HTTPServer")
    assert {"get_terminal_size", "terminal", "size", "httpserver", "http", "server"} <= set(t)
    assert "self" not in t and "def" not in t


def test_score():
    r = _score(["a", "b", "c", "d"], {"c", "z"})
    assert r["recall@1"] == 0 and r["recall@5"] == 0.5 and r["hit@5"] == 1 and r["mrr"] == 1 / 3
