import math
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class QueryMetrics:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


def recall_at_k(relevant: Sequence[str], ranked: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = set(ranked[:k])
    rel = set(relevant)
    return len(topk & rel) / float(len(rel))


def mrr(relevant: Sequence[str], ranked: Sequence[str]) -> float:
    rel = set(relevant)
    for idx, doc_id in enumerate(ranked, start=1):
        if doc_id in rel:
            return 1.0 / float(idx)
    return 0.0


def ndcg_at_k(relevant: Sequence[str], ranked: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel = set(relevant)
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(k, len(rel))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_query_metrics(relevant: Sequence[str], ranked: Sequence[str], k: int) -> QueryMetrics:
    return QueryMetrics(
        recall_at_k=recall_at_k(relevant, ranked, k),
        mrr=mrr(relevant, ranked),
        ndcg_at_k=ndcg_at_k(relevant, ranked, k),
    )


def mean(values: List[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0

