import json

import pytest

from musubi_eval.dataset import load_documents, load_queries


def test_load_documents(tmp_path):
    p = tmp_path / "docs.jsonl"
    p.write_text(
        "\n".join([
            json.dumps({"id": "d1", "text": "hello"}),
            json.dumps({"id": "d2", "text": "world", "metadata": {"k": "v"}}),
        ])
    )
    docs = load_documents(str(p))
    assert len(docs) == 2
    assert docs[0].id == "d1"
    assert docs[0].text == "hello"
    assert docs[0].metadata is None
    assert docs[1].id == "d2"
    assert docs[1].metadata == {"k": "v"}


def test_load_documents_missing_field(tmp_path):
    p = tmp_path / "docs.jsonl"
    p.write_text(json.dumps({"id": "d1"}))
    with pytest.raises(ValueError, match="document row must include"):
        load_documents(str(p))


def test_load_documents_invalid_json(tmp_path):
    p = tmp_path / "docs.jsonl"
    p.write_text("{bad json")
    with pytest.raises(json.JSONDecodeError):
        load_documents(str(p))


def test_load_documents_blank_lines(tmp_path):
    p = tmp_path / "docs.jsonl"
    p.write_text(
        "\n" + json.dumps({"id": "d1", "text": "a"}) + "\n\n"
    )
    docs = load_documents(str(p))
    assert len(docs) == 1


def test_load_queries(tmp_path):
    p = tmp_path / "queries.jsonl"
    p.write_text(
        "\n".join([
            json.dumps({"id": "q1", "query": "hello", "positive_ids": ["d1"]}),
            json.dumps({"id": "q2", "query": "world", "positive_ids": ["d2"], "filter": {"k": "v"}}),
        ])
    )
    queries = load_queries(str(p))
    assert len(queries) == 2
    assert queries[0].id == "q1"
    assert queries[0].positive_ids == ["d1"]
    assert queries[0].filter is None
    assert queries[1].filter == {"k": "v"}


def test_load_queries_missing_field(tmp_path):
    p = tmp_path / "queries.jsonl"
    p.write_text(json.dumps({"id": "q1", "query": "hello"}))
    with pytest.raises(ValueError, match="query row must include"):
        load_queries(str(p))
