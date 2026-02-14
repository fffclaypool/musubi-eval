import argparse
import logging
import time
from pathlib import Path

from .application.run_scenario import ScenarioRunner
from .application.tune_params import ParamTuner
from .config import load_scenario, load_tuning_config
from .infrastructure.dataset_jsonl import JsonlDatasetReader
from .infrastructure.musubi_http import MusubiHttpClient
from .infrastructure.results_filesystem import save_results
from .infrastructure.tuning_results import save_tuning_results
from .reporting import generate_evidently_report, log_mlflow
from .util import setup_logging

logger = logging.getLogger(__name__)


def _cmd_run(args: argparse.Namespace) -> None:
    cfg = load_scenario(args.config)
    setup_logging(cfg.log_level)
    logger.info("starting scenario: %s", args.config)
    runner = ScenarioRunner(
        dataset_reader=JsonlDatasetReader(),
        search_gateway=MusubiHttpClient(cfg.base_url, cfg.timeout_sec, cfg.retry),
    )
    results = runner.run(cfg)
    outputs = save_results(cfg, results)
    report_outputs = {}
    if cfg.evidently.enabled:
        report_outputs = generate_evidently_report(
            results, cfg.evidently, Path(cfg.evidently.output_dir)
        )
    if cfg.mlflow.enabled:
        log_mlflow(results, cfg.mlflow, cfg, {**outputs, **report_outputs})
    logger.info("results saved: %s", outputs)


def _cmd_tune(args: argparse.Namespace) -> None:
    tuning_cfg = load_tuning_config(args.config)
    setup_logging(tuning_cfg.log_level)
    logger.info("starting tuning: %s", args.config)

    base_cfg = load_scenario(tuning_cfg.base_scenario)
    tuner = ParamTuner(
        dataset_reader=JsonlDatasetReader(),
        search_gateway=MusubiHttpClient(base_cfg.base_url, base_cfg.timeout_sec, base_cfg.retry),
    )
    results = tuner.run(tuning_cfg)
    outputs = save_tuning_results(tuning_cfg, results)

    logger.info("best trial: #%d", results["best_trial"])
    logger.info("best params: %s", results["best_params"])
    logger.info("best score: %.6f", results["best_score"])

    best_trial_num = results["best_trial"]
    for trial in results.get("trials", []):
        if trial["number"] == best_trial_num:
            logger.info("best trial metrics: %s", trial["metrics"])
            logger.info("best trial latency: %s", trial["latency_ms"])
            break

    if tuning_cfg.mlflow.enabled:
        _log_tuning_mlflow(tuning_cfg, results, outputs)

    logger.info("tuning outputs saved: %s", outputs)


def _log_tuning_mlflow(tuning_cfg: object, results: dict, artifacts: dict) -> None:
    try:
        import mlflow
    except Exception as exc:
        logger.warning("mlflow: failed to import (%s)", exc)
        return

    mlflow_cfg = tuning_cfg.mlflow
    if mlflow_cfg.tracking_uri:
        mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    if mlflow_cfg.experiment_name:
        mlflow.set_experiment(mlflow_cfg.experiment_name)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for trial in results.get("trials", []):
        run_name = f"{mlflow_cfg.run_name_prefix}-trial{trial['number']}-{timestamp}"
        with mlflow.start_run(run_name=run_name):
            params = trial.get("params", {})
            mlflow.log_params(
                {
                    "k": params.get("k"),
                    "ef": params.get("ef"),
                    "alpha": params.get("alpha"),
                    "study_name": tuning_cfg.study.name,
                    "objective_metric": tuning_cfg.objective.metric,
                }
            )
            metrics = trial.get("metrics", {})
            latency = trial.get("latency_ms", {})
            mlflow.log_metrics(
                {
                    "recall_at_k": float(metrics.get("recall_at_k", 0.0)),
                    "mrr": float(metrics.get("mrr", 0.0)),
                    "ndcg_at_k": float(metrics.get("ndcg_at_k", 0.0)),
                    "latency_mean_ms": float(latency.get("mean", 0.0)),
                    "latency_p95_ms": float(latency.get("p95", 0.0) or 0.0),
                    "score": float(trial.get("score", 0.0)),
                    "constraint_violated": 1.0 if trial.get("constraint_violated") else 0.0,
                }
            )
            for name, path in artifacts.items():
                try:
                    mlflow.log_artifact(path, artifact_path=name)
                except Exception as exc:
                    logger.warning("mlflow: failed to log artifact %s (%s)", path, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="musubi evaluation runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run an evaluation scenario")
    run_p.add_argument("-c", "--config", required=True, help="path to scenario YAML")

    tune_p = sub.add_parser("tune", help="auto-tune search parameters with Optuna")
    tune_p.add_argument("-c", "--config", required=True, help="path to tuning YAML")

    args = parser.parse_args()

    if args.cmd == "run":
        _cmd_run(args)
    elif args.cmd == "tune":
        _cmd_tune(args)


if __name__ == "__main__":
    main()
