import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .config import EvidentlyConfig, MlflowConfig, ScenarioConfig

logger = logging.getLogger(__name__)


def _per_query_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for run in results.get("runs", []):
        run_name = run.get("name", "run")
        params = run.get("params", {})
        for item in run.get("per_query", []):
            rows.append(
                {
                    "run_name": run_name,
                    "query_id": item.get("query_id"),
                    "latency_ms": item.get("latency_ms"),
                    "recall_at_k": item.get("recall_at_k"),
                    "mrr": item.get("mrr"),
                    "ndcg_at_k": item.get("ndcg_at_k"),
                    "k": params.get("k"),
                    "ef": params.get("ef"),
                    "alpha": params.get("alpha"),
                }
            )
    return pd.DataFrame(rows)


def generate_evidently_report(
    results: Dict[str, Any], cfg: EvidentlyConfig, output_dir: Path
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}

    df = _per_query_dataframe(results)
    if df.empty:
        logger.warning("evidently: no per-query data to report")
        return outputs

    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.metrics import ColumnCount, MeanValue, RowCount
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("evidently: failed to import (%s)", exc)
        return outputs

    definition = DataDefinition(
        numerical_columns=["latency_ms", "recall_at_k", "mrr", "ndcg_at_k", "k", "ef", "alpha"],
        categorical_columns=["run_name", "query_id"],
    )
    dataset = Dataset.from_pandas(df, data_definition=definition)

    report = Report(
        [
            RowCount(),
            ColumnCount(),
            MeanValue(column="latency_ms"),
            MeanValue(column="recall_at_k"),
            MeanValue(column="mrr"),
            MeanValue(column="ndcg_at_k"),
        ]
    )

    eval_result = report.run(dataset, None)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if cfg.save_html:
        html_path = output_dir / f"evidently_{timestamp}.html"
        if hasattr(eval_result, "save_html"):
            eval_result.save_html(str(html_path))
            outputs["evidently_html"] = str(html_path)
        else:
            logger.warning("evidently: save_html not available, skipping HTML output")
    if cfg.save_json:
        json_path = output_dir / f"evidently_{timestamp}.json"
        if hasattr(eval_result, "save_json"):
            eval_result.save_json(str(json_path))
            outputs["evidently_json"] = str(json_path)
        else:
            payload = eval_result.json()
            if isinstance(payload, str):
                json_path.write_text(payload)
            else:
                json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            outputs["evidently_json"] = str(json_path)

    return outputs


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
    artifacts = artifacts or {}

    for run in results.get("runs", []):
        run_name = f"{cfg.run_name_prefix}-{run.get('name','run')}-{timestamp}"
        with mlflow.start_run(run_name=run_name):
            params = run.get("params", {})
            params_payload = {
                "base_url": scenario.base_url,
                "documents": scenario.documents_path,
                "queries": scenario.queries_path,
                "param_name": run.get("name"),
                "k": params.get("k"),
                "ef": params.get("ef"),
                "alpha": params.get("alpha"),
            }
            if params.get("filter") is not None:
                params_payload["filter"] = json.dumps(params.get("filter"), ensure_ascii=False)
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

            for name, path in artifacts.items():
                try:
                    mlflow.log_artifact(path, artifact_path=name)
                except Exception as exc:
                    logger.warning("mlflow: failed to log artifact %s (%s)", path, exc)
