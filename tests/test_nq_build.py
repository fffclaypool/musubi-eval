import json

from scripts.build_nq_subset import (
    build_subset,
    extract_passage_text,
    parse_nq_example,
    write_jsonl,
)


def _make_nq_example(
    question="what is the capital of france",
    title="France",
    example_id=123,
    passage_tokens=None,
    annotation_start=0,
    annotation_end=5,
    candidate_index=0,
):
    """Build a minimal NQ example for testing."""
    if passage_tokens is None:
        passage_tokens = [
            {"token": "Paris", "html_token": False},
            {"token": "is", "html_token": False},
            {"token": "the", "html_token": False},
            {"token": "capital", "html_token": False},
            {"token": "of", "html_token": False},
            {"token": "France", "html_token": False},
            {"token": ".", "html_token": False},
        ]
    return {
        "question_text": question,
        "document_title": title,
        "document_url": f"https://en.wikipedia.org/wiki/{title}",
        "example_id": example_id,
        "document_tokens": passage_tokens,
        "long_answer_candidates": [
            {"start_token": 0, "end_token": len(passage_tokens), "top_level": True}
        ],
        "annotations": [
            {
                "long_answer": {
                    "start_token": annotation_start,
                    "end_token": annotation_end,
                    "candidate_index": candidate_index,
                },
                "short_answers": [],
            }
        ],
    }


class TestExtractPassageText:
    def test_basic(self):
        tokens = [
            {"token": "Hello", "html_token": False},
            {"token": "<p>", "html_token": True},
            {"token": "world", "html_token": False},
        ]
        result = extract_passage_text(tokens, 0, 3)
        assert result == "Hello world"

    def test_html_filtered(self):
        tokens = [
            {"token": "<div>", "html_token": True},
            {"token": "text", "html_token": False},
            {"token": "</div>", "html_token": True},
        ]
        result = extract_passage_text(tokens, 0, 3)
        assert result == "text"

    def test_range(self):
        tokens = [
            {"token": "a", "html_token": False},
            {"token": "b", "html_token": False},
            {"token": "c", "html_token": False},
        ]
        assert extract_passage_text(tokens, 1, 3) == "b c"


class TestParseNqExample:
    def test_valid_example(self):
        raw = _make_nq_example()
        result = parse_nq_example(raw)
        assert result is not None
        question, passage, page_id, title, url = result
        assert question == "what is the capital of france"
        assert "Paris" in passage
        assert title == "France"

    def test_no_question(self):
        raw = _make_nq_example(question="")
        assert parse_nq_example(raw) is None

    def test_no_annotations(self):
        raw = _make_nq_example()
        raw["annotations"] = []
        assert parse_nq_example(raw) is None

    def test_invalid_long_answer(self):
        raw = _make_nq_example(annotation_start=-1, annotation_end=-1, candidate_index=-1)
        assert parse_nq_example(raw) is None

    def test_fallback_to_second_annotation(self):
        raw = _make_nq_example()
        invalid_annotation = {
            "long_answer": {"start_token": -1, "end_token": -1, "candidate_index": -1},
            "short_answers": [],
        }
        valid_annotation = raw["annotations"][0]
        raw["annotations"] = [invalid_annotation, valid_annotation]
        result = parse_nq_example(raw)
        assert result is not None
        assert "Paris" in result[1]

    def test_all_annotations_invalid(self):
        raw = _make_nq_example()
        invalid = {
            "long_answer": {"start_token": -1, "end_token": -1, "candidate_index": -1},
            "short_answers": [],
        }
        raw["annotations"] = [invalid, invalid]
        assert parse_nq_example(raw) is None

    def test_example_id_zero(self):
        raw = _make_nq_example(example_id=0)
        result = parse_nq_example(raw)
        assert result is not None
        _, _, page_id, _, _ = result
        assert page_id == "0"

    def test_document_text_fallback(self):
        raw = {
            "question_text": "test question",
            "document_title": "Test",
            "document_url": "http://test",
            "example_id": 1,
            "document_text": "word0 word1 word2 word3 word4",
            "long_answer_candidates": [{"start_token": 0, "end_token": 5, "top_level": True}],
            "annotations": [
                {
                    "long_answer": {"start_token": 0, "end_token": 3, "candidate_index": 0},
                    "short_answers": [],
                }
            ],
        }
        result = parse_nq_example(raw)
        assert result is not None
        _, passage, _, _, _ = result
        assert passage == "word0 word1 word2"


class TestBuildSubset:
    def _write_nq_input(self, tmp_path, examples):
        path = tmp_path / "nq_input.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        return str(path)

    def test_basic(self, tmp_path):
        examples = [_make_nq_example(question=f"question {i}", example_id=i) for i in range(5)]
        input_path = self._write_nq_input(tmp_path, examples)
        docs, queries = build_subset(input_path, num_queries=3, seed=42, max_passages_per_page=20)
        assert len(docs) == 3
        assert len(queries) == 3
        assert all("id" in d for d in docs)
        assert all("text" in d for d in docs)
        assert all("id" in q for q in queries)
        assert all("query" in q for q in queries)
        assert all(len(q["positive_ids"]) == 1 for q in queries)

    def test_seed_reproducibility(self, tmp_path):
        examples = [_make_nq_example(question=f"question {i}", example_id=i) for i in range(10)]
        input_path = self._write_nq_input(tmp_path, examples)
        docs1, q1 = build_subset(input_path, num_queries=5, seed=42, max_passages_per_page=20)
        docs2, q2 = build_subset(input_path, num_queries=5, seed=42, max_passages_per_page=20)
        assert [q["query"] for q in q1] == [q["query"] for q in q2]

    def test_max_passages_per_page(self, tmp_path):
        examples = [_make_nq_example(question=f"question {i}", example_id=999) for i in range(10)]
        input_path = self._write_nq_input(tmp_path, examples)
        docs, queries = build_subset(input_path, num_queries=10, seed=42, max_passages_per_page=3)
        assert len(queries) == 3

    def test_skip_invalid(self, tmp_path):
        examples = [
            _make_nq_example(question="valid", example_id=1),
            _make_nq_example(question="", example_id=2),
            _make_nq_example(question="also valid", example_id=3),
        ]
        input_path = self._write_nq_input(tmp_path, examples)
        docs, queries = build_subset(input_path, num_queries=10, seed=42, max_passages_per_page=20)
        assert len(queries) == 2


class TestWriteJsonl:
    def test_write(self, tmp_path):
        items = [{"id": "1", "text": "hello"}, {"id": "2", "text": "world"}]
        path = tmp_path / "sub" / "test.jsonl"
        write_jsonl(path, items)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "1"
