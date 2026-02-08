import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Document:
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Query:
    id: str
    query: str
    positive_ids: List[str]
    filter: Optional[Dict[str, Any]] = None


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def load_documents(path: str) -> List[Document]:
    docs: List[Document] = []
    for row in _read_jsonl(path):
        if "id" not in row or "text" not in row:
            raise ValueError("document row must include 'id' and 'text'")
        docs.append(
            Document(
                id=str(row["id"]),
                text=str(row["text"]),
                metadata=row.get("metadata"),
            )
        )
    return docs


def load_queries(path: str) -> List[Query]:
    queries: List[Query] = []
    for row in _read_jsonl(path):
        if "id" not in row or "query" not in row or "positive_ids" not in row:
            raise ValueError("query row must include 'id', 'query', and 'positive_ids'")
        positive_ids = [str(x) for x in row.get("positive_ids", [])]
        queries.append(
            Query(
                id=str(row["id"]),
                query=str(row["query"]),
                positive_ids=positive_ids,
                filter=row.get("filter"),
            )
        )
    return queries

