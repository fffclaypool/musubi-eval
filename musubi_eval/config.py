from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_backoff_sec: float = 0.5
    max_backoff_sec: float = 5.0


@dataclass
class IngestionConfig:
    poll_interval_sec: float = 1.0
    timeout_sec: float = 600.0


@dataclass
class OutputConfig:
    dir: str = "outputs"
    save_json: bool = True
    save_csv: bool = True
    save_prefix: str = "run"


@dataclass
class EvidentlyConfig:
    enabled: bool = False
    output_dir: str = "outputs/evidently"
    save_html: bool = True
    save_json: bool = True


@dataclass
class MlflowConfig:
    enabled: bool = False
    tracking_uri: Optional[str] = None
    experiment_name: str = "musubi-eval"
    run_name_prefix: str = "musubi-eval"


@dataclass
class SearchParam:
    name: str
    k: int
    ef: Optional[int] = None
    alpha: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None


@dataclass
class ScenarioConfig:
    base_url: str
    documents_path: str
    queries_path: str
    search_params: List[SearchParam]
    timeout_sec: float = 30.0
    retry: RetryConfig = field(default_factory=RetryConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    evidently: EvidentlyConfig = field(default_factory=EvidentlyConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    log_level: str = "INFO"


def _require_key(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"missing required key: {key}")
    return d[key]


def load_scenario(path: str) -> ScenarioConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    base_url = _require_key(raw, "base_url")

    datasets = _require_key(raw, "datasets")
    documents_path = _require_key(datasets, "documents")
    queries_path = _require_key(datasets, "queries")

    search = _require_key(raw, "search")
    params_raw = _require_key(search, "params")
    if not isinstance(params_raw, list) or not params_raw:
        raise ValueError("search.params must be a non-empty list")

    search_params: List[SearchParam] = []
    for i, p in enumerate(params_raw, start=1):
        name = p.get("name") or f"param_{i}"
        k = _require_key(p, "k")
        search_params.append(
            SearchParam(
                name=name,
                k=int(k),
                ef=p.get("ef"),
                alpha=p.get("alpha"),
                filter=p.get("filter"),
            )
        )

    retry_raw = raw.get("retry", {})
    retry = RetryConfig(
        max_attempts=int(retry_raw.get("max_attempts", 3)),
        base_backoff_sec=float(retry_raw.get("base_backoff_sec", 0.5)),
        max_backoff_sec=float(retry_raw.get("max_backoff_sec", 5.0)),
    )

    ingestion_raw = raw.get("ingestion", {})
    ingestion = IngestionConfig(
        poll_interval_sec=float(ingestion_raw.get("poll_interval_sec", 1.0)),
        timeout_sec=float(ingestion_raw.get("timeout_sec", 600.0)),
    )

    output_raw = raw.get("output", {})
    output = OutputConfig(
        dir=str(output_raw.get("dir", "outputs")),
        save_json=bool(output_raw.get("save_json", True)),
        save_csv=bool(output_raw.get("save_csv", True)),
        save_prefix=str(output_raw.get("save_prefix", "run")),
    )

    evidently_raw = raw.get("evidently", {})
    evidently = EvidentlyConfig(
        enabled=bool(evidently_raw.get("enabled", False)),
        output_dir=str(evidently_raw.get("output_dir", "outputs/evidently")),
        save_html=bool(evidently_raw.get("save_html", True)),
        save_json=bool(evidently_raw.get("save_json", True)),
    )

    mlflow_raw = raw.get("mlflow", {})
    mlflow = MlflowConfig(
        enabled=bool(mlflow_raw.get("enabled", False)),
        tracking_uri=mlflow_raw.get("tracking_uri"),
        experiment_name=str(mlflow_raw.get("experiment_name", "musubi-eval")),
        run_name_prefix=str(mlflow_raw.get("run_name_prefix", "musubi-eval")),
    )

    return ScenarioConfig(
        base_url=str(base_url).rstrip("/"),
        documents_path=str(documents_path),
        queries_path=str(queries_path),
        search_params=search_params,
        timeout_sec=float(raw.get("timeout_sec", 30.0)),
        retry=retry,
        ingestion=ingestion,
        output=output,
        evidently=evidently,
        mlflow=mlflow,
        log_level=str(raw.get("log_level", "INFO")),
    )
