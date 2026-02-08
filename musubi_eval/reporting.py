import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .config import EvidentlyConfig, MlflowConfig, ScenarioConfig

logger = logging.getLogger(__name__)


def _per_query_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for run in results.get("runs", []):
        params = run.get("params", {})
        for item in run.get("per_query", []):
            rows.append({
                "run_name": run.get("name", "run"),
                "query_id": item.get("query_id"),
                "latency_ms": item.get("latency_ms"),
                "recall_at_k": item.get("recall_at_k"),
                "mrr": item.get("mrr"),
                "ndcg_at_k": item.get("ndcg_at_k"),
                "k": params.get("k"),
                "ef": params.get("ef"),
                "alpha": params.get("alpha"),
            })
    return pd.DataFrame(rows)


def generate_evidently_report(
    results: Dict[str, Any], cfg: EvidentlyConfig, output_dir: Path
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _per_query_dataframe(results)
    if df.empty:
        logger.warning("evidently: no per-query data to report")
        return {}

    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.metrics import (
            ColumnCount,
            DatasetMissingValueCount,
            DuplicatedRowCount,
            MaxValue,
            MeanValue,
            MedianValue,
            MinValue,
            QuantileValue,
            RowCount,
            StdValue,
            UniqueValueCount,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("evidently: failed to import (%s)", exc)
        return {}

    definition = DataDefinition(
        numerical_columns=["latency_ms", "recall_at_k", "mrr", "ndcg_at_k", "k", "ef", "alpha"],
        categorical_columns=["run_name", "query_id"],
    )
    dataset = Dataset.from_pandas(df, data_definition=definition)

    report = Report(
        [
            RowCount(),
            ColumnCount(),
            DuplicatedRowCount(),
            DatasetMissingValueCount(),
            UniqueValueCount(column="run_name"),
            UniqueValueCount(column="query_id"),
            MeanValue(column="latency_ms"),
            MedianValue(column="latency_ms"),
            MinValue(column="latency_ms"),
            MaxValue(column="latency_ms"),
            StdValue(column="latency_ms"),
            QuantileValue(column="latency_ms", quantile=0.25),
            QuantileValue(column="latency_ms", quantile=0.50),
            QuantileValue(column="latency_ms", quantile=0.75),
            QuantileValue(column="latency_ms", quantile=0.95),
            QuantileValue(column="latency_ms", quantile=0.99),
            MeanValue(column="recall_at_k"),
            MedianValue(column="recall_at_k"),
            MinValue(column="recall_at_k"),
            MaxValue(column="recall_at_k"),
            StdValue(column="recall_at_k"),
            QuantileValue(column="recall_at_k", quantile=0.25),
            QuantileValue(column="recall_at_k", quantile=0.50),
            QuantileValue(column="recall_at_k", quantile=0.75),
            MeanValue(column="mrr"),
            MedianValue(column="mrr"),
            MinValue(column="mrr"),
            MaxValue(column="mrr"),
            StdValue(column="mrr"),
            QuantileValue(column="mrr", quantile=0.25),
            QuantileValue(column="mrr", quantile=0.50),
            QuantileValue(column="mrr", quantile=0.75),
            MeanValue(column="ndcg_at_k"),
            MedianValue(column="ndcg_at_k"),
            MinValue(column="ndcg_at_k"),
            MaxValue(column="ndcg_at_k"),
            StdValue(column="ndcg_at_k"),
            QuantileValue(column="ndcg_at_k", quantile=0.25),
            QuantileValue(column="ndcg_at_k", quantile=0.50),
            QuantileValue(column="ndcg_at_k", quantile=0.75),
            MeanValue(column="k"),
            MedianValue(column="k"),
            MinValue(column="k"),
            MaxValue(column="k"),
            StdValue(column="k"),
            MeanValue(column="ef"),
            MedianValue(column="ef"),
            MinValue(column="ef"),
            MaxValue(column="ef"),
            StdValue(column="ef"),
            MeanValue(column="alpha"),
            MedianValue(column="alpha"),
            MinValue(column="alpha"),
            MaxValue(column="alpha"),
            StdValue(column="alpha"),
        ]
    )

    eval_result = report.run(dataset, None)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    html_output = _save_evidently_html(cfg, eval_result, output_dir, timestamp)
    json_output = _save_evidently_json(cfg, eval_result, output_dir, timestamp)
    return {**html_output, **json_output}


def _save_evidently_html(
    cfg: EvidentlyConfig, eval_result: Any, output_dir: Path, timestamp: str
) -> Dict[str, str]:
    if not cfg.save_html:
        return {}
    html_path = output_dir / f"evidently_{timestamp}.html"
    if hasattr(eval_result, "save_html"):
        eval_result.save_html(str(html_path))
        return {"evidently_html": str(html_path)}
    logger.warning("evidently: save_html not available, skipping HTML output")
    return {}


def _save_evidently_json(
    cfg: EvidentlyConfig, eval_result: Any, output_dir: Path, timestamp: str
) -> Dict[str, str]:
    if not cfg.save_json:
        return {}
    json_path = output_dir / f"evidently_{timestamp}.json"
    if hasattr(eval_result, "save_json"):
        eval_result.save_json(str(json_path))
        return {"evidently_json": str(json_path)}
    payload = eval_result.json()
    json_payload = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(json_payload)
    return {"evidently_json": str(json_path)}


def log_mlflow(
    results: Dict[str, Any],
    cfg: MlflowConfig,
    scenario: ScenarioConfig,
    artifacts: Optional[Dict[str, str]] = None,
) -> None:
    if not cfg.enabled:
        return
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("mlflow: failed to import (%s)", exc)
        return

    if cfg.tracking_uri:
        mlflow.set_tracking_uri(cfg.tracking_uri)
    if cfg.experiment_name:
        mlflow.set_experiment(cfg.experiment_name)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    artifacts_dict = artifacts if artifacts is not None else {}

    for run in results.get("runs", []):
        run_name = f"{cfg.run_name_prefix}-{run.get('name','run')}-{timestamp}"
        with mlflow.start_run(run_name=run_name):
            params = run.get("params", {})
            base_params = {
                "base_url": scenario.base_url,
                "documents": scenario.documents_path,
                "queries": scenario.queries_path,
                "param_name": run.get("name"),
                "k": params.get("k"),
                "ef": params.get("ef"),
                "alpha": params.get("alpha"),
            }
            filter_value = params.get("filter")
            filter_param = (
                {"filter": json.dumps(filter_value, ensure_ascii=False)} if filter_value is not None else {}
            )
            params_payload = {**base_params, **filter_param}
            mlflow.log_params({k: v for k, v in params_payload.items() if v is not None})

            metrics = run.get("metrics", {})
            latency = run.get("latency_ms", {})
            mlflow.log_metrics(
                {
                    "recall_at_k": float(metrics.get("recall_at_k", 0.0)),
                    "mrr": float(metrics.get("mrr", 0.0)),
                    "ndcg_at_k": float(metrics.get("ndcg_at_k", 0.0)),
                    "latency_mean_ms": float(latency.get("mean", 0.0)),
                    "latency_p50_ms": float(latency.get("p50", 0.0) or 0.0),
                    "latency_p95_ms": float(latency.get("p95", 0.0) or 0.0),
                }
            )

            for name, path in artifacts_dict.items():
                try:
                    mlflow.log_artifact(path, artifact_path=name)
                except Exception as exc:
                    logger.warning("mlflow: failed to log artifact %s (%s)", path, exc)
