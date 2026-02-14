#!/usr/bin/env python3
"""Extract a subset of Natural Questions and convert to musubi-eval format.

Usage:
    uv run python scripts/build_nq_subset.py --input-path /path/to/nq-train.jsonl

NQ input format (Google simplified JSONL):
    Each line is a JSON object with:
    - question_text: str
    - document_title: str
    - document_url: str
    - document_tokens: list[{token, start_byte, end_byte, html_token}]
      OR document_text: str (pre-extracted plain text)
    - long_answer_candidates: list[{start_token, end_token, top_level}]
    - annotations: list[{long_answer: {start_token, end_token, candidate_index},
                         short_answers: list[...]}]
    - example_id: int (optional)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def extract_passage_text(
    tokens: List[Dict[str, Any]],
    start_token: int,
    end_token: int,
) -> str:
    """Extract plain text from document tokens between start and end indices."""
    parts: List[str] = []
    for tok in tokens[start_token:end_token]:
        if tok.get("html_token", False):
            continue
        text = tok.get("token", "")
        if text:
            parts.append(text)
    return " ".join(parts)


def extract_from_document_text(
    document_text: str,
    start_token: int,
    end_token: int,
) -> str:
    """Fallback extraction when document_text is a plain string."""
    words = document_text.split()
    return " ".join(words[start_token:end_token])


def parse_nq_example(
    raw: Dict[str, Any],
) -> Optional[Tuple[str, str, str, str, str]]:
    """Parse a single NQ example and return (question, passage_text, page_id, title, url).

    Returns None if the example should be skipped.
    """
    question = raw.get("question_text", "").strip()
    if not question:
        return None

    annotations = raw.get("annotations", [])
    if not annotations:
        return None

    annotation = annotations[0]
    long_answer = annotation.get("long_answer", {})
    start_token = long_answer.get("start_token", -1)
    end_token = long_answer.get("end_token", -1)
    candidate_index = long_answer.get("candidate_index", -1)

    if start_token < 0 or end_token <= start_token or candidate_index < 0:
        return None

    tokens = raw.get("document_tokens")
    if tokens and isinstance(tokens, list) and len(tokens) > 0:
        passage = extract_passage_text(tokens, start_token, end_token)
    elif "document_text" in raw:
        passage = extract_from_document_text(raw["document_text"], start_token, end_token)
    else:
        return None

    passage = passage.strip()
    if not passage:
        return None

    title = raw.get("document_title", "")
    url = raw.get("document_url", "")
    example_id = raw.get("example_id", "")
    page_id = str(example_id) if example_id else url or title

    return question, passage, page_id, title, url


def build_subset(
    input_path: str,
    num_queries: int,
    seed: int,
    max_passages_per_page: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read NQ JSONL, sample, and convert to musubi-eval format."""
    candidates: List[Tuple[str, str, str, str, str]] = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            result = parse_nq_example(raw)
            if result is not None:
                candidates.append(result)

    rng = random.Random(seed)
    rng.shuffle(candidates)

    page_counts: Counter[str] = Counter()
    selected: List[Tuple[str, str, str, str, str]] = []

    for question, passage, page_id, title, url in candidates:
        if page_counts[page_id] >= max_passages_per_page:
            continue
        selected.append((question, passage, page_id, title, url))
        page_counts[page_id] += 1
        if len(selected) >= num_queries:
            break

    documents: List[Dict[str, Any]] = []
    queries: List[Dict[str, Any]] = []
    doc_id_set: set[str] = set()

    for idx, (question, passage, page_id, title, url) in enumerate(selected):
        doc_id = f"nq_doc_{idx:05d}"
        query_id = f"nq_q_{idx:05d}"

        if doc_id not in doc_id_set:
            doc_id_set.add(doc_id)
            documents.append(
                {
                    "id": doc_id,
                    "text": passage,
                    "metadata": {
                        "source": "natural_questions",
                        "page_id": page_id,
                        "title": title,
                    },
                }
            )

        queries.append(
            {
                "id": query_id,
                "query": question,
                "positive_ids": [doc_id],
            }
        )

    return documents, queries


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    """Write items as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build NQ subset for musubi-eval",
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to NQ source JSONL file",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=1000,
        help="Number of queries to extract (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--out-dir",
        default="examples/data/nq_1000",
        help="Output directory (default: examples/data/nq_1000)",
    )
    parser.add_argument(
        "--max-passages-per-page",
        type=int,
        default=20,
        help="Max passages per page_id (default: 20)",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading NQ data from: {input_path}")
    documents, queries = build_subset(
        str(input_path),
        num_queries=args.num_queries,
        seed=args.seed,
        max_passages_per_page=args.max_passages_per_page,
    )

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "documents.jsonl", documents)
    write_jsonl(out_dir / "queries.jsonl", queries)

    print(f"Generated {len(documents)} documents and {len(queries)} queries")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
