from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Query:
    id: str
    query: str
    positive_ids: List[str]
    filter: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class SearchParam:
    name: str
    k: int
    ef: Optional[int] = None
    alpha: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class QueryMetrics:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
