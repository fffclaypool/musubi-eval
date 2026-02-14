#!/usr/bin/env python3
"""Validate musubi-eval dataset (documents.jsonl + queries.jsonl).

Usage:
    uv run python scripts/validate_dataset.py --dataset-dir examples/data/nq_1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of dicts."""
    items: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return items


def check_required_keys(
    items: List[Dict[str, Any]],
    required: List[str],
    label: str,
) -> List[str]:
    """Check that all items have required keys. Return list of error messages."""
    errors: List[str] = []
    for idx, item in enumerate(items):
        for key in required:
            if key not in item:
                errors.append(f"{label}[{idx}]: missing required key '{key}'")
    return errors


def check_duplicate_ids(
    items: List[Dict[str, Any]],
    label: str,
) -> List[str]:
    """Check for duplicate ids. Return list of error messages."""
    errors: List[str] = []
    seen: Dict[str, int] = {}
    for idx, item in enumerate(items):
        item_id = item.get("id", "")
        if item_id in seen:
            errors.append(
                f"{label}[{idx}]: duplicate id '{item_id}' (first at index {seen[item_id]})"
            )
        else:
            seen[item_id] = idx
    return errors


def check_positive_ids_reference(
    queries: List[Dict[str, Any]],
    doc_ids: set[str],
) -> List[str]:
    """Check that all positive_ids in queries reference existing document ids."""
    errors: List[str] = []
    for idx, q in enumerate(queries):
        for pid in q.get("positive_ids", []):
            if pid not in doc_ids:
                errors.append(
                    f"queries[{idx}] (id={q.get('id', '?')}): "
                    f"positive_id '{pid}' not found in documents"
                )
    return errors


def compute_stats(
    documents: List[Dict[str, Any]],
    queries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute basic statistics for reporting."""
    query_lengths = [len(q.get("query", "")) for q in queries]
    positive_counts = [len(q.get("positive_ids", [])) for q in queries]
    doc_text_lengths = [len(d.get("text", "")) for d in documents]

    def _stats(values: List[int]) -> Dict[str, float]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": round(sum(values) / len(values), 2),
        }

    return {
        "documents_count": len(documents),
        "queries_count": len(queries),
        "query_length_chars": _stats(query_lengths),
        "positive_ids_per_query": _stats(positive_counts),
        "doc_text_length_chars": _stats(doc_text_lengths),
    }


def validate(dataset_dir: str) -> Tuple[List[str], Dict[str, Any]]:
    """Run all validations. Returns (errors, stats)."""
    dir_path = Path(dataset_dir)
    docs_path = dir_path / "documents.jsonl"
    queries_path = dir_path / "queries.jsonl"

    errors: List[str] = []

    if not docs_path.exists():
        errors.append(f"file not found: {docs_path}")
    if not queries_path.exists():
        errors.append(f"file not found: {queries_path}")
    if errors:
        return errors, {}

    documents = load_jsonl(docs_path)
    queries = load_jsonl(queries_path)

    errors.extend(check_required_keys(documents, ["id", "text"], "documents"))
    errors.extend(check_required_keys(queries, ["id", "query", "positive_ids"], "queries"))
    errors.extend(check_duplicate_ids(documents, "documents"))
    errors.extend(check_duplicate_ids(queries, "queries"))

    doc_ids = {d["id"] for d in documents if "id" in d}
    errors.extend(check_positive_ids_reference(queries, doc_ids))

    stats = compute_stats(documents, queries)
    return errors, stats


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Validate musubi-eval dataset")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing documents.jsonl and queries.jsonl",
    )
    args = parser.parse_args(argv)

    try:
        errors, stats = validate(args.dataset_dir)
    except ValueError as exc:
        print("\n=== Validation FAILED (1 errors) ===", file=sys.stderr)
        print(f"  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if stats:
        print("=== Dataset Statistics ===")
        print(f"  Documents: {stats['documents_count']}")
        print(f"  Queries:   {stats['queries_count']}")
        ql = stats["query_length_chars"]
        print(f"  Query length (chars): min={ql['min']} max={ql['max']} mean={ql['mean']}")
        pc = stats["positive_ids_per_query"]
        print(f"  Positive IDs/query:   min={pc['min']} max={pc['max']} mean={pc['mean']}")
        dl = stats["doc_text_length_chars"]
        print(f"  Doc text length:      min={dl['min']} max={dl['max']} mean={dl['mean']}")

    if errors:
        print(f"\n=== Validation FAILED ({len(errors)} errors) ===", file=sys.stderr)
        for err in errors[:20]:
            print(f"  ERROR: {err}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors", file=sys.stderr)
        sys.exit(1)

    print("\n=== Validation PASSED ===")


if __name__ == "__main__":
    main()
