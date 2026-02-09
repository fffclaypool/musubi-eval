import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from musubi_eval.domain.models import Document, Query


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def load_documents(path: str) -> List[Document]:
    rows = list(_read_jsonl(path))
    for row in rows:
        if "id" not in row or "text" not in row:
            raise ValueError("document row must include 'id' and 'text'")
    return [
        Document(
            id=str(row["id"]),
            text=str(row["text"]),
            metadata=row.get("metadata"),
        )
        for row in rows
    ]


def load_queries(path: str) -> List[Query]:
    rows = list(_read_jsonl(path))
    for row in rows:
        if "id" not in row or "query" not in row or "positive_ids" not in row:
            raise ValueError("query row must include 'id', 'query', and 'positive_ids'")
    return [
        Query(
            id=str(row["id"]),
            query=str(row["query"]),
            positive_ids=[str(x) for x in row.get("positive_ids", [])],
            filter=row.get("filter"),
        )
        for row in rows
    ]
