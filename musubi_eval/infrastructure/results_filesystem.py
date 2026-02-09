import csv
import io
import time
from pathlib import Path
from typing import Any, Dict

from musubi_eval.util import safe_json_dumps


def save_results(cfg: Any, results: Dict[str, Any]) -> Dict[str, str]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_output = _save_json(cfg, results, out_dir, timestamp)
    csv_output = _save_csv(cfg, results, out_dir, timestamp)
    return {**json_output, **csv_output}


def _save_json(cfg: Any, results: Dict[str, Any], out_dir: Path, timestamp: str) -> Dict[str, str]:
    if not cfg.output.save_json:
        return {}
    json_path = out_dir / f"{cfg.output.save_prefix}_{timestamp}.json"
    json_path.write_text(safe_json_dumps(results))
    return {"json": str(json_path)}


def _save_csv(cfg: Any, results: Dict[str, Any], out_dir: Path, timestamp: str) -> Dict[str, str]:
    if not cfg.output.save_csv:
        return {}
    csv_path = out_dir / f"{cfg.output.save_prefix}_{timestamp}.csv"
    fieldnames = [
        "name",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for run in results["runs"]:
        writer.writerow(_csv_row_dict(run))
    csv_path.write_text(buf.getvalue())
    return {"csv": str(csv_path)}


def _csv_row_dict(run: Dict[str, Any]) -> Dict[str, str]:
    metrics = run["metrics"]
    latency = run["latency_ms"]
    p50 = latency.get("p50", 0.0) or 0.0
    p95 = latency.get("p95", 0.0) or 0.0
    return {
        "name": run["name"],
        "recall_at_k": f"{metrics['recall_at_k']:.6f}",
        "mrr": f"{metrics['mrr']:.6f}",
        "ndcg_at_k": f"{metrics['ndcg_at_k']:.6f}",
        "latency_mean_ms": f"{latency['mean']:.3f}",
        "latency_p50_ms": f"{p50:.3f}",
        "latency_p95_ms": f"{p95:.3f}",
    }
