import csv
import io
import json
from dataclasses import dataclass, field

from musubi_eval.infrastructure.results_filesystem import save_results


@dataclass
class OutputConfig:
    dir: str = "outputs"
    save_json: bool = True
    save_csv: bool = True
    save_prefix: str = "run"


@dataclass
class FakeCfg:
    output: OutputConfig = field(default_factory=OutputConfig)


SAMPLE_RESULTS = {
    "runs": [
        {
            "name": "run1",
            "metrics": {"recall_at_k": 0.8, "mrr": 0.75, "ndcg_at_k": 0.9},
            "latency_ms": {"mean": 12.5, "p50": 10.0, "p95": 20.0},
        },
    ]
}


def test_save_results_json(tmp_path):
    cfg = FakeCfg(output=OutputConfig(dir=str(tmp_path), save_json=True, save_csv=False))
    outputs = save_results(cfg, SAMPLE_RESULTS)
    assert "json" in outputs
    data = json.loads(open(outputs["json"]).read())
    assert data["runs"][0]["name"] == "run1"


def test_save_results_csv(tmp_path):
    cfg = FakeCfg(output=OutputConfig(dir=str(tmp_path), save_json=False, save_csv=True))
    outputs = save_results(cfg, SAMPLE_RESULTS)
    assert "csv" in outputs
    content = open(outputs["csv"]).read()
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["name"] == "run1"
    assert float(rows[0]["recall_at_k"]) == 0.8


def test_save_results_both(tmp_path):
    cfg = FakeCfg(output=OutputConfig(dir=str(tmp_path), save_json=True, save_csv=True))
    outputs = save_results(cfg, SAMPLE_RESULTS)
    assert "json" in outputs
    assert "csv" in outputs


def test_save_results_neither(tmp_path):
    cfg = FakeCfg(output=OutputConfig(dir=str(tmp_path), save_json=False, save_csv=False))
    outputs = save_results(cfg, SAMPLE_RESULTS)
    assert outputs == {}


def test_save_csv_with_none_percentiles(tmp_path):
    results = {
        "runs": [
            {
                "name": "run1",
                "metrics": {"recall_at_k": 0.5, "mrr": 0.5, "ndcg_at_k": 0.5},
                "latency_ms": {"mean": 10.0, "p50": None, "p95": None},
            },
        ]
    }
    cfg = FakeCfg(output=OutputConfig(dir=str(tmp_path), save_json=False, save_csv=True))
    outputs = save_results(cfg, results)
    content = open(outputs["csv"]).read()
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    assert float(rows[0]["latency_p50_ms"]) == 0.0
