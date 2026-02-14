from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from musubi_eval.domain.models import SearchParam


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


@dataclass
class IntRangeConfig:
    low: int
    high: int
    step: int = 1


@dataclass
class FloatRangeConfig:
    low: float
    high: float
    step: float = 0.1


@dataclass
class SearchSpaceConfig:
    k: IntRangeConfig
    ef: IntRangeConfig
    alpha: FloatRangeConfig


@dataclass
class StudyConfig:
    name: str = "musubi-tuning"
    direction: str = "maximize"
    n_trials: int = 20
    timeout_sec: Optional[float] = None
    sampler_seed: Optional[int] = None


@dataclass
class ConstraintsConfig:
    max_latency_p95_ms: Optional[float] = None


@dataclass
class ObjectiveConfig:
    metric: str = "recall_at_k"
    latency_penalty: float = 0.0


@dataclass
class TuningOutputConfig:
    dir: str = "outputs"
    save_history_csv: bool = True
    save_history_json: bool = True
    save_best_yaml: bool = True


@dataclass
class TuningConfig:
    base_scenario: str
    search_space: SearchSpaceConfig
    study: StudyConfig = field(default_factory=StudyConfig)
    constraints: ConstraintsConfig = field(default_factory=ConstraintsConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    output: TuningOutputConfig = field(default_factory=TuningOutputConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    log_level: str = "INFO"


def _require_key(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"missing required key: {key}")
    return d[key]


def _build_search_param(item: Dict[str, Any], index: int) -> SearchParam:
    name = item.get("name") or f"param_{index}"
    k_value = _require_key(item, "k")
    return SearchParam(
        name=name,
        k=int(k_value),
        ef=item.get("ef"),
        alpha=item.get("alpha"),
        filter=item.get("filter"),
    )


def load_scenario(path: str) -> ScenarioConfig:
    raw_config = yaml.safe_load(Path(path).read_text()) or {}

    base_url = _require_key(raw_config, "base_url")

    datasets = _require_key(raw_config, "datasets")
    documents_path = _require_key(datasets, "documents")
    queries_path = _require_key(datasets, "queries")

    search = _require_key(raw_config, "search")
    params_raw = _require_key(search, "params")
    if not isinstance(params_raw, list) or not params_raw:
        raise ValueError("search.params must be a non-empty list")

    search_params = [_build_search_param(item, idx) for idx, item in enumerate(params_raw, start=1)]

    retry_raw = raw_config.get("retry", {})
    retry = RetryConfig(
        max_attempts=int(retry_raw.get("max_attempts", 3)),
        base_backoff_sec=float(retry_raw.get("base_backoff_sec", 0.5)),
        max_backoff_sec=float(retry_raw.get("max_backoff_sec", 5.0)),
    )

    ingestion_raw = raw_config.get("ingestion", {})
    ingestion = IngestionConfig(
        poll_interval_sec=float(ingestion_raw.get("poll_interval_sec", 1.0)),
        timeout_sec=float(ingestion_raw.get("timeout_sec", 600.0)),
    )

    output_raw = raw_config.get("output", {})
    output = OutputConfig(
        dir=str(output_raw.get("dir", "outputs")),
        save_json=bool(output_raw.get("save_json", True)),
        save_csv=bool(output_raw.get("save_csv", True)),
        save_prefix=str(output_raw.get("save_prefix", "run")),
    )

    evidently_raw = raw_config.get("evidently", {})
    evidently = EvidentlyConfig(
        enabled=bool(evidently_raw.get("enabled", False)),
        output_dir=str(evidently_raw.get("output_dir", "outputs/evidently")),
        save_html=bool(evidently_raw.get("save_html", True)),
        save_json=bool(evidently_raw.get("save_json", True)),
    )

    mlflow_raw = raw_config.get("mlflow", {})
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
        timeout_sec=float(raw_config.get("timeout_sec", 30.0)),
        retry=retry,
        ingestion=ingestion,
        output=output,
        evidently=evidently,
        mlflow=mlflow,
        log_level=str(raw_config.get("log_level", "INFO")),
    )


def _build_int_range(raw: Dict[str, Any]) -> IntRangeConfig:
    return IntRangeConfig(
        low=int(_require_key(raw, "low")),
        high=int(_require_key(raw, "high")),
        step=int(raw.get("step", 1)),
    )


def _build_float_range(raw: Dict[str, Any]) -> FloatRangeConfig:
    return FloatRangeConfig(
        low=float(_require_key(raw, "low")),
        high=float(_require_key(raw, "high")),
        step=float(raw.get("step", 0.1)),
    )


def load_tuning_config(path: str) -> TuningConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    base_scenario = _require_key(raw, "base_scenario")

    space_raw = _require_key(raw, "search_space")
    search_space = SearchSpaceConfig(
        k=_build_int_range(_require_key(space_raw, "k")),
        ef=_build_int_range(_require_key(space_raw, "ef")),
        alpha=_build_float_range(_require_key(space_raw, "alpha")),
    )

    study_raw = raw.get("study", {})
    direction = str(study_raw.get("direction", "maximize"))
    if direction not in ("maximize", "minimize"):
        raise ValueError(f"study.direction must be 'maximize' or 'minimize', got '{direction}'")
    study = StudyConfig(
        name=str(study_raw.get("name", "musubi-tuning")),
        direction=direction,
        n_trials=int(study_raw.get("n_trials", 20)),
        timeout_sec=float(study_raw["timeout_sec"]) if "timeout_sec" in study_raw else None,
        sampler_seed=int(study_raw["sampler_seed"]) if "sampler_seed" in study_raw else None,
    )

    constraints_raw = raw.get("constraints", {})
    constraints = ConstraintsConfig(
        max_latency_p95_ms=(
            float(constraints_raw["max_latency_p95_ms"])
            if "max_latency_p95_ms" in constraints_raw
            else None
        ),
    )

    obj_raw = raw.get("objective", {})
    metric = str(obj_raw.get("metric", "recall_at_k"))
    supported_metrics = {"recall_at_k", "mrr", "ndcg_at_k"}
    if metric not in supported_metrics:
        raise ValueError(
            f"objective.metric must be one of {sorted(supported_metrics)}, got '{metric}'"
        )
    objective = ObjectiveConfig(
        metric=metric,
        latency_penalty=float(obj_raw.get("latency_penalty", 0.0)),
    )

    out_raw = raw.get("output", {})
    output = TuningOutputConfig(
        dir=str(out_raw.get("dir", "outputs")),
        save_history_csv=bool(out_raw.get("save_history_csv", True)),
        save_history_json=bool(out_raw.get("save_history_json", True)),
        save_best_yaml=bool(out_raw.get("save_best_yaml", True)),
    )

    mlflow_raw = raw.get("mlflow", {})
    mlflow = MlflowConfig(
        enabled=bool(mlflow_raw.get("enabled", False)),
        tracking_uri=mlflow_raw.get("tracking_uri"),
        experiment_name=str(mlflow_raw.get("experiment_name", "musubi-tuning")),
        run_name_prefix=str(mlflow_raw.get("run_name_prefix", "musubi-tuning")),
    )

    return TuningConfig(
        base_scenario=str(base_scenario),
        search_space=search_space,
        study=study,
        constraints=constraints,
        objective=objective,
        output=output,
        mlflow=mlflow,
        log_level=str(raw.get("log_level", "INFO")),
    )
