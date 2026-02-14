import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import optuna

from musubi_eval.application.ports import DatasetReader, SearchGateway
from musubi_eval.config import ScenarioConfig, TuningConfig, load_scenario
from musubi_eval.domain.models import SearchParam

from .run_scenario import ScenarioRunner

logger = logging.getLogger(__name__)

CONSTRAINT_PENALTY_MAXIMIZE = -1e6
CONSTRAINT_PENALTY_MINIMIZE = 1e6


def constraint_penalty(direction: str) -> float:
    if direction == "minimize":
        return CONSTRAINT_PENALTY_MINIMIZE
    return CONSTRAINT_PENALTY_MAXIMIZE


@dataclass
class TrialResult:
    number: int
    k: int
    ef: int
    alpha: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    latency_mean_ms: float
    latency_p95_ms: float
    score: float
    constraint_violated: bool
    status: str


def compute_objective_score(
    metric_value: float,
    latency_p95_ms: float,
    latency_penalty: float,
) -> float:
    return metric_value - latency_penalty * latency_p95_ms


def check_constraint(
    latency_p95_ms: float,
    max_latency_p95_ms: Optional[float],
) -> bool:
    if max_latency_p95_ms is None:
        return False
    return latency_p95_ms > max_latency_p95_ms


class ParamTuner:
    def __init__(
        self,
        dataset_reader: DatasetReader,
        search_gateway: SearchGateway,
    ) -> None:
        self.dataset_reader = dataset_reader
        self.search_gateway = search_gateway
        self.trial_results: List[TrialResult] = []
        self._runner: Optional[ScenarioRunner] = None
        self._base_cfg: Optional[ScenarioConfig] = None
        self._tuning_cfg: Optional[TuningConfig] = None
        self._ingestion_done = False

    def run(self, tuning_cfg: TuningConfig) -> Dict[str, Any]:
        self._tuning_cfg = tuning_cfg
        self._base_cfg = load_scenario(tuning_cfg.base_scenario)
        self._runner = ScenarioRunner(
            dataset_reader=self.dataset_reader,
            search_gateway=self.search_gateway,
        )
        self.trial_results = []
        self._ingestion_done = False

        sampler = optuna.samplers.TPESampler(
            seed=tuning_cfg.study.sampler_seed,
        )
        study = optuna.create_study(
            study_name=tuning_cfg.study.name,
            direction=tuning_cfg.study.direction,
            sampler=sampler,
        )

        study.optimize(
            self._objective,
            n_trials=tuning_cfg.study.n_trials,
            timeout=tuning_cfg.study.timeout_sec,
        )

        best = study.best_trial
        logger.info("best trial: #%d", best.number)
        logger.info(
            "best params: k=%s, ef=%s, alpha=%s",
            best.params["k"],
            best.params["ef"],
            best.params["alpha"],
        )
        logger.info("best score: %.6f", best.value)

        return {
            "best_trial": best.number,
            "best_params": best.params,
            "best_score": best.value,
            "trials": [self._trial_to_dict(t) for t in self.trial_results],
            "study_name": tuning_cfg.study.name,
            "n_trials": len(self.trial_results),
        }

    def _ensure_ingestion(self) -> None:
        if self._ingestion_done:
            return
        cfg = self._base_cfg
        documents = self.dataset_reader.load_documents(cfg.documents_path)
        queries = self.dataset_reader.load_queries(cfg.queries_path)
        self._runner.prepare_index(cfg, documents, queries)
        self._ingestion_done = True

    def _objective(self, trial: optuna.Trial) -> float:
        tcfg = self._tuning_cfg
        space = tcfg.search_space

        k = trial.suggest_int("k", space.k.low, space.k.high, step=space.k.step)
        ef = trial.suggest_int("ef", space.ef.low, space.ef.high, step=space.ef.step)
        alpha = trial.suggest_float(
            "alpha", space.alpha.low, space.alpha.high, step=space.alpha.step
        )

        param = SearchParam(name=f"trial_{trial.number}", k=k, ef=ef, alpha=alpha)

        self._ensure_ingestion()

        cfg = self._base_cfg
        queries = self.dataset_reader.load_queries(cfg.queries_path)
        run_result = self._runner.run_single_param(param, queries)

        metrics = run_result["metrics"]
        latency = run_result["latency_ms"]
        metric_value = float(metrics.get(tcfg.objective.metric, 0.0))
        latency_p95 = float(latency.get("p95", 0.0) or 0.0)

        violated = check_constraint(latency_p95, tcfg.constraints.max_latency_p95_ms)

        if violated:
            score = constraint_penalty(tcfg.study.direction)
        else:
            score = compute_objective_score(
                metric_value, latency_p95, tcfg.objective.latency_penalty
            )

        tr = TrialResult(
            number=trial.number,
            k=k,
            ef=ef,
            alpha=alpha,
            recall_at_k=float(metrics.get("recall_at_k", 0.0)),
            mrr=float(metrics.get("mrr", 0.0)),
            ndcg_at_k=float(metrics.get("ndcg_at_k", 0.0)),
            latency_mean_ms=float(latency.get("mean", 0.0)),
            latency_p95_ms=latency_p95,
            score=score,
            constraint_violated=violated,
            status="completed",
        )
        self.trial_results.append(tr)

        logger.info(
            "trial #%d: k=%d ef=%d alpha=%.2f %s=%.4f p95=%.1fms score=%.6f%s",
            trial.number,
            k,
            ef,
            alpha,
            tcfg.objective.metric,
            metric_value,
            latency_p95,
            score,
            " [CONSTRAINT VIOLATED]" if violated else "",
        )

        return score

    @staticmethod
    def _trial_to_dict(tr: TrialResult) -> Dict[str, Any]:
        return {
            "number": tr.number,
            "params": {"k": tr.k, "ef": tr.ef, "alpha": tr.alpha},
            "metrics": {
                "recall_at_k": tr.recall_at_k,
                "mrr": tr.mrr,
                "ndcg_at_k": tr.ndcg_at_k,
            },
            "latency_ms": {
                "mean": tr.latency_mean_ms,
                "p95": tr.latency_p95_ms,
            },
            "score": tr.score,
            "constraint_violated": tr.constraint_violated,
            "status": tr.status,
        }
