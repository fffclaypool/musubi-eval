import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import IngestionWaiter, MusubiClient
from .config import ScenarioConfig, SearchParam
from .dataset import load_documents, load_queries
from .metrics import compute_query_metrics, mean
from .util import percentiles, safe_json_dumps

logger = logging.getLogger(__name__)


def _extract_ids(search_response: Dict[str, Any]) -> List[str]:
    # Supports common response shapes: {results:[{id:...}]}, {documents:[...]}, {hits:[...]}, or list.
    if isinstance(search_response, list):
        items = search_response
    else:
        items = (
            search_response.get("results")
            or search_response.get("documents")
            or search_response.get("hits")
            or []
        )
    ids: List[str] = []
    for item in items:
        if isinstance(item, dict) and "id" in item:
            ids.append(str(item["id"]))
        elif isinstance(item, str):
            ids.append(item)
    return ids


def _build_documents_payload(documents) -> List[Dict[str, Any]]:
    payload = []
    for d in documents:
        row = {"id": d.id, "text": d.text}
        if d.metadata is not None:
            row["metadata"] = d.metadata
        payload.append(row)
    return payload


def _search_payload(query: str, param: SearchParam, extra_filter: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"text": query, "k": param.k}
    if param.ef is not None:
        payload["ef"] = param.ef
    if param.alpha is not None:
        payload["alpha"] = param.alpha
    combined_filter = None
    if param.filter and extra_filter:
        combined_filter = {"$and": [param.filter, extra_filter]}
    elif param.filter:
        combined_filter = param.filter
    elif extra_filter:
        combined_filter = extra_filter
    if combined_filter is not None:
        payload["filter"] = combined_filter
    return payload


def run_scenario(cfg: ScenarioConfig) -> Dict[str, Any]:
    documents = load_documents(cfg.documents_path)
    queries = load_queries(cfg.queries_path)

    client = MusubiClient(cfg.base_url, cfg.timeout_sec, cfg.retry)
    waiter = IngestionWaiter(client, cfg.ingestion.poll_interval_sec, cfg.ingestion.timeout_sec)

    logger.info("health check")
    client.health()

    logger.info("uploading %s documents", len(documents))
    client.documents_batch(_build_documents_payload(documents))

    logger.info("starting ingestion")
    job = client.ingestion_start()
    job_id = str(job.get("id") or job.get("job_id") or "")
    if not job_id:
        raise RuntimeError(f"cannot determine ingestion job id: {job}")

    logger.info("waiting for ingestion job %s", job_id)
    waiter.wait_ready(job_id)

    results: Dict[str, Any] = {
        "config": {
            "base_url": cfg.base_url,
            "documents": cfg.documents_path,
            "queries": cfg.queries_path,
            "search_params": [asdict(p) for p in cfg.search_params],
        },
        "runs": [],
    }

    for param in cfg.search_params:
        logger.info("running search param set: %s", param.name)
        per_query = []
        latencies: List[float] = []

        for q in queries:
            payload = _search_payload(q.query, param, q.filter)
            t0 = time.perf_counter()
            resp = client.search(payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            ranked_ids = _extract_ids(resp)
            qm = compute_query_metrics(q.positive_ids, ranked_ids, param.k)
            per_query.append(
                {
                    "query_id": q.id,
                    "latency_ms": latency_ms,
                    "recall_at_k": qm.recall_at_k,
                    "mrr": qm.mrr,
                    "ndcg_at_k": qm.ndcg_at_k,
                }
            )

        recall_vals = [x["recall_at_k"] for x in per_query]
        mrr_vals = [x["mrr"] for x in per_query]
        ndcg_vals = [x["ndcg_at_k"] for x in per_query]
        latency_ps = percentiles(latencies, [50, 95])

        results["runs"].append(
            {
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
        )

    return results


def save_results(cfg: ScenarioConfig, results: Dict[str, Any]) -> Dict[str, str]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}

    if cfg.output.save_json:
        json_path = out_dir / f"{cfg.output.save_prefix}_{timestamp}.json"
        json_path.write_text(safe_json_dumps(results))
        outputs["json"] = str(json_path)

    if cfg.output.save_csv:
        csv_path = out_dir / f"{cfg.output.save_prefix}_{timestamp}.csv"
        lines = [
            "name,recall_at_k,mrr,ndcg_at_k,latency_mean_ms,latency_p50_ms,latency_p95_ms",
        ]
        for run in results["runs"]:
            metrics = run["metrics"]
            latency = run["latency_ms"]
            lines.append(
                ",".join(
                    [
                        run["name"],
                        f"{metrics['recall_at_k']:.6f}",
                        f"{metrics['mrr']:.6f}",
                        f"{metrics['ndcg_at_k']:.6f}",
                        f"{latency['mean']:.3f}",
                        f"{latency.get('p50', 0.0) or 0.0:.3f}",
                        f"{latency.get('p95', 0.0) or 0.0:.3f}",
                    ]
                )
            )
        csv_path.write_text("\n".join(lines))
        outputs["csv"] = str(csv_path)

    return outputs
