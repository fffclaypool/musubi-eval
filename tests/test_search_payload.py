from musubi_eval.application.run_scenario import _build_documents_payload, _search_payload
from musubi_eval.domain.models import Document, SearchParam


def test_search_payload_basic():
    param = SearchParam(name="p", k=10)
    result = _search_payload("hello", param, None)
    assert result == {"text": "hello", "k": 10}


def test_search_payload_with_ef_alpha():
    param = SearchParam(name="p", k=10, ef=64, alpha=0.5)
    result = _search_payload("hello", param, None)
    assert result == {"text": "hello", "k": 10, "ef": 64, "alpha": 0.5}


def test_search_payload_param_filter_only():
    param = SearchParam(name="p", k=10, filter={"category": "tech"})
    result = _search_payload("hello", param, None)
    assert result == {"text": "hello", "k": 10, "filter": {"category": "tech"}}


def test_search_payload_extra_filter_only():
    param = SearchParam(name="p", k=10)
    result = _search_payload("hello", param, {"status": "active"})
    assert result == {"text": "hello", "k": 10, "filter": {"status": "active"}}


def test_search_payload_both_filters():
    param = SearchParam(name="p", k=10, filter={"category": "tech"})
    result = _search_payload("hello", param, {"status": "active"})
    assert result == {
        "text": "hello",
        "k": 10,
        "filter": {"category": "tech", "status": "active"},
    }


def test_search_payload_both_filters_query_wins_on_conflict():
    param = SearchParam(name="p", k=10, filter={"category": "food", "lang": "en"})
    result = _search_payload("hello", param, {"category": "tech"})
    assert result == {
        "text": "hello",
        "k": 10,
        "filter": {"category": "tech", "lang": "en"},
    }


def test_build_documents_payload_basic():
    docs = [
        Document(id="d1", text="hello"),
        Document(id="d2", text="world", metadata={"k": "v"}),
    ]
    result = _build_documents_payload(docs)
    assert result == [
        {"id": "d1", "text": "hello"},
        {"id": "d2", "text": "world", "metadata": {"k": "v"}},
    ]


def test_build_documents_payload_empty():
    assert _build_documents_payload([]) == []
