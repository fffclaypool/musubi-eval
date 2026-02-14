import json

from scripts.validate_dataset import (
    check_duplicate_ids,
    check_positive_ids_reference,
    check_required_keys,
    validate,
)


def _write_dataset(tmp_path, documents, queries):
    """Write documents.jsonl and queries.jsonl to tmp_path."""
    docs_path = tmp_path / "documents.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    docs_path.write_text("\n".join(json.dumps(d) for d in documents) + "\n" if documents else "")
    queries_path.write_text("\n".join(json.dumps(q) for q in queries) + "\n" if queries else "")
    return str(tmp_path)


class TestCheckRequiredKeys:
    def test_all_present(self):
        items = [{"id": "1", "text": "hello"}]
        assert check_required_keys(items, ["id", "text"], "docs") == []

    def test_missing_key(self):
        items = [{"id": "1"}]
        errors = check_required_keys(items, ["id", "text"], "docs")
        assert len(errors) == 1
        assert "text" in errors[0]

    def test_multiple_missing(self):
        items = [{}]
        errors = check_required_keys(items, ["id", "text"], "docs")
        assert len(errors) == 2


class TestCheckDuplicateIds:
    def test_no_duplicates(self):
        items = [{"id": "1"}, {"id": "2"}]
        assert check_duplicate_ids(items, "docs") == []

    def test_has_duplicates(self):
        items = [{"id": "1"}, {"id": "1"}]
        errors = check_duplicate_ids(items, "docs")
        assert len(errors) == 1
        assert "duplicate" in errors[0]


class TestCheckPositiveIdsReference:
    def test_valid_references(self):
        queries = [{"id": "q1", "positive_ids": ["d1"]}]
        doc_ids = {"d1", "d2"}
        assert check_positive_ids_reference(queries, doc_ids) == []

    def test_broken_reference(self):
        queries = [{"id": "q1", "positive_ids": ["d_missing"]}]
        doc_ids = {"d1"}
        errors = check_positive_ids_reference(queries, doc_ids)
        assert len(errors) == 1
        assert "d_missing" in errors[0]


class TestValidate:
    def test_valid_dataset(self, tmp_path):
        documents = [
            {"id": "d1", "text": "hello world"},
            {"id": "d2", "text": "foo bar"},
        ]
        queries = [
            {"id": "q1", "query": "hello", "positive_ids": ["d1"]},
            {"id": "q2", "query": "foo", "positive_ids": ["d2"]},
        ]
        dataset_dir = _write_dataset(tmp_path, documents, queries)
        errors, stats = validate(dataset_dir)
        assert errors == []
        assert stats["documents_count"] == 2
        assert stats["queries_count"] == 2

    def test_missing_files(self, tmp_path):
        errors, stats = validate(str(tmp_path))
        assert len(errors) == 2
        assert any("documents.jsonl" in e for e in errors)
        assert any("queries.jsonl" in e for e in errors)

    def test_missing_required_keys(self, tmp_path):
        documents = [{"id": "d1"}]
        queries = [{"id": "q1", "query": "hello", "positive_ids": ["d1"]}]
        dataset_dir = _write_dataset(tmp_path, documents, queries)
        errors, _ = validate(dataset_dir)
        assert len(errors) == 1
        assert "text" in errors[0]

    def test_broken_positive_ids(self, tmp_path):
        documents = [{"id": "d1", "text": "hello"}]
        queries = [{"id": "q1", "query": "test", "positive_ids": ["d_missing"]}]
        dataset_dir = _write_dataset(tmp_path, documents, queries)
        errors, _ = validate(dataset_dir)
        assert len(errors) == 1
        assert "d_missing" in errors[0]

    def test_duplicate_ids(self, tmp_path):
        documents = [
            {"id": "d1", "text": "hello"},
            {"id": "d1", "text": "world"},
        ]
        queries = [{"id": "q1", "query": "test", "positive_ids": ["d1"]}]
        dataset_dir = _write_dataset(tmp_path, documents, queries)
        errors, _ = validate(dataset_dir)
        assert len(errors) == 1
        assert "duplicate" in errors[0]

    def test_empty_dataset(self, tmp_path):
        dataset_dir = _write_dataset(tmp_path, [], [])
        errors, stats = validate(dataset_dir)
        assert errors == []
        assert stats["documents_count"] == 0
        assert stats["queries_count"] == 0
