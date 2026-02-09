import logging
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from musubi_eval.application.ports import DatasetReader, SearchGateway
from musubi_eval.config import ScenarioConfig
from musubi_eval.domain.metrics import compute_query_metrics, mean
from musubi_eval.domain.models import Document, Query, SearchParam
from musubi_eval.util import percentiles

logger = logging.getLogger(__name__)


class IngestionWaiter:
    def __init__(
        self,
        client: SearchGateway,
        poll_interval_sec: float,
        timeout_sec: float,
        now: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        self.client = client
        self.poll_interval_sec = poll_interval_sec
        self.timeout_sec = timeout_sec
        self.now = now
        self.sleep = sleep

    def wait_ready(self, job_id: str) -> Dict[str, Any]:
        start_time = self.now()
        while True:
            status = self.client.ingestion_get(job_id)
            state = str(status.get("state") or status.get("status") or "").lower()
            if state in {"ready", "completed", "done", "success"}:
                return status
            if state in {"failed", "error"}:
                raise RuntimeError(f"ingestion job failed: {status}")
            if self.now() - start_time > self.timeout_sec:
                raise TimeoutError("ingestion job timed out")
            self.sleep(self.poll_interval_sec)


def _build_documents_payload(documents: List[Document]) -> List[Dict[str, Any]]:
    return [
        {"id": d.id, "text": d.text, **({"metadata": d.metadata} if d.metadata is not None else {})}
        for d in documents
    ]


def _search_payload(
    query: str, param: SearchParam, extra_filter: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    base = {"text": query, "k": param.k}
    ef_part = {"ef": param.ef} if param.ef is not None else {}
    alpha_part = {"alpha": param.alpha} if param.alpha is not None else {}
    if param.filter and extra_filter:
        # Backend may not support logical operators like "$and", so merge flat filters.
        # Query-level filter wins on key conflicts.
        filter_part = {"filter": {**param.filter, **extra_filter}}
    elif param.filter:
        filter_part = {"filter": param.filter}
    elif extra_filter:
        filter_part = {"filter": extra_filter}
    else:
        filter_part = {}
    return {**base, **ef_part, **alpha_part, **filter_part}


class ScenarioRunner:
    def __init__(
        self,
        dataset_reader: DatasetReader,
        search_gateway: SearchGateway,
        now: Callable[[], float] = time.time,
        perf_counter: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.dataset_reader = dataset_reader
        self.search_gateway = search_gateway
        self.now = now
        self.perf_counter = perf_counter
        self.sleep = sleep

    def run(self, cfg: ScenarioConfig) -> Dict[str, Any]:
        documents = self.dataset_reader.load_documents(cfg.documents_path)
        queries = self.dataset_reader.load_queries(cfg.queries_path)

        waiter = IngestionWaiter(
            self.search_gateway,
            cfg.ingestion.poll_interval_sec,
            cfg.ingestion.timeout_sec,
            now=self.now,
            sleep=self.sleep,
        )

        logger.info("health check")
        self.search_gateway.health()

        logger.info("uploading %s documents", len(documents))
        self.search_gateway.documents_batch(_build_documents_payload(documents))

        logger.info("starting ingestion")
        job = self.search_gateway.ingestion_start()
        job_id_value = job.get("id") or job.get("job_id") or ""
        job_id = str(job_id_value)
        if not job_id:
            raise RuntimeError(f"cannot determine ingestion job id: {job}")

        logger.info("waiting for ingestion job %s", job_id)
        waiter.wait_ready(job_id)
        self._wait_search_ready(queries)

        runs = [self._run_param(param, queries) for param in cfg.search_params]

        return {
            "config": {
                "base_url": cfg.base_url,
                "documents": cfg.documents_path,
                "queries": cfg.queries_path,
                "search_params": [asdict(p) for p in cfg.search_params],
            },
            "runs": runs,
        }

    def _wait_search_ready(self, queries: List[Query]) -> None:
        if not queries:
            return
        warmup_payload = {"text": queries[0].query, "k": 1}
        deadline = self.now() + 60.0
        while True:
            try:
                self.search_gateway.search(warmup_payload)
                return
            except Exception as exc:
                if self.now() >= deadline:
                    raise RuntimeError("search backend is not ready after warm-up retries") from exc
                logger.warning("search warm-up retry after failure: %s", exc)
                self.sleep(1.0)

    def _run_param(self, param: SearchParam, queries: List[Query]) -> Dict[str, Any]:
        logger.info("running search param set: %s", param.name)
        per_query_items = [self._run_query(param, q) for q in queries]
        per_query = [item for item, _ in per_query_items]
        latencies = [lat for _, lat in per_query_items]
        recall_vals = [x["recall_at_k"] for x in per_query]
        mrr_vals = [x["mrr"] for x in per_query]
        ndcg_vals = [x["ndcg_at_k"] for x in per_query]
        latency_ps = percentiles(latencies, [50, 95])
        return {
            "name": param.name,
            "params": asdict(param),
            "metrics": {
                "recall_at_k": mean(recall_vals),
                "mrr": mean(mrr_vals),
                "ndcg_at_k": mean(ndcg_vals),
            },
            "latency_ms": {
                "mean": mean(latencies),
                **latency_ps,
            },
            "per_query": per_query,
        }

    def _run_query(self, param: SearchParam, q: Query) -> tuple[Dict[str, Any], float]:
        payload = _search_payload(q.query, param, q.filter)
        start_perf = self.perf_counter()
        ranked_ids = self.search_gateway.search(payload)
        latency_ms = (self.perf_counter() - start_perf) * 1000.0
        qm = compute_query_metrics(q.positive_ids, ranked_ids, param.k)
        return (
            {
                "query_id": q.id,
                "latency_ms": latency_ms,
                "recall_at_k": qm.recall_at_k,
                "mrr": qm.mrr,
                "ndcg_at_k": qm.ndcg_at_k,
            },
            latency_ms,
        )
