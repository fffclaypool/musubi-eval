import csv
import io
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from musubi_eval.config import TuningConfig, TuningOutputConfig
from musubi_eval.util import safe_json_dumps


def save_tuning_results(cfg: TuningConfig, results: Dict[str, Any]) -> Dict[str, str]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}
    outputs.update(_save_history_json(cfg.output, results, out_dir, timestamp))
    outputs.update(_save_history_csv(cfg.output, results, out_dir, timestamp))
    outputs.update(_save_best_yaml(cfg, results, out_dir, timestamp))
    return outputs


def _save_history_json(
    out_cfg: TuningOutputConfig,
    results: Dict[str, Any],
    out_dir: Path,
    timestamp: str,
) -> Dict[str, str]:
    if not out_cfg.save_history_json:
        return {}
    path = out_dir / f"tuning_{timestamp}.json"
    path.write_text(safe_json_dumps(results))
    return {"history_json": str(path)}


def _save_history_csv(
    out_cfg: TuningOutputConfig,
    results: Dict[str, Any],
    out_dir: Path,
    timestamp: str,
) -> Dict[str, str]:
    if not out_cfg.save_history_csv:
        return {}
    path = out_dir / f"tuning_{timestamp}.csv"
    fieldnames = [
        "trial",
        "k",
        "ef",
        "alpha",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "latency_mean_ms",
        "latency_p95_ms",
        "score",
        "constraint_violated",
        "status",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for trial in results.get("trials", []):
        params = trial.get("params", {})
        metrics = trial.get("metrics", {})
        latency = trial.get("latency_ms", {})
        writer.writerow(
            {
                "trial": trial.get("number", ""),
                "k": params.get("k", ""),
                "ef": params.get("ef", ""),
                "alpha": params.get("alpha", ""),
                "recall_at_k": f"{metrics.get('recall_at_k', 0.0):.6f}",
                "mrr": f"{metrics.get('mrr', 0.0):.6f}",
                "ndcg_at_k": f"{metrics.get('ndcg_at_k', 0.0):.6f}",
                "latency_mean_ms": f"{latency.get('mean', 0.0):.3f}",
                "latency_p95_ms": f"{latency.get('p95', 0.0):.3f}",
                "score": f"{trial.get('score', 0.0):.6f}",
                "constraint_violated": trial.get("constraint_violated", False),
                "status": trial.get("status", ""),
            }
        )
    path.write_text(buf.getvalue())
    return {"history_csv": str(path)}


def _save_best_yaml(
    cfg: TuningConfig,
    results: Dict[str, Any],
    out_dir: Path,
    timestamp: str,
) -> Dict[str, str]:
    if not cfg.output.save_best_yaml:
        return {}
    best_params = results.get("best_params", {})
    if not best_params:
        return {}

    best_yaml_path = out_dir / f"best_params_{timestamp}.yaml"
    best_yaml_path.write_text(
        yaml.dump({"best_params": best_params}, default_flow_style=False, allow_unicode=True)
    )

    scenario_path = _save_best_scenario(cfg, best_params, out_dir, timestamp)

    outputs = {"best_yaml": str(best_yaml_path)}
    if scenario_path:
        outputs["best_scenario"] = scenario_path
    return outputs


def _save_best_scenario(
    cfg: TuningConfig,
    best_params: Dict[str, Any],
    out_dir: Path,
    timestamp: str,
) -> str:
    try:
        base_raw = yaml.safe_load(Path(cfg.base_scenario).read_text()) or {}
    except Exception:
        return ""

    base_raw["search"] = {
        "params": [
            {
                "name": "tuned_best",
                "k": best_params.get("k"),
                "ef": best_params.get("ef"),
                "alpha": best_params.get("alpha"),
            }
        ]
    }
    path = out_dir / f"best_scenario_{timestamp}.yaml"
    path.write_text(yaml.dump(base_raw, default_flow_style=False, allow_unicode=True))
    return str(path)
