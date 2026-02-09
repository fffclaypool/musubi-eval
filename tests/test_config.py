import pytest

from musubi_eval.config import load_scenario


MINIMAL_YAML = """\
base_url: http://localhost:8080
datasets:
  documents: docs.jsonl
  queries: queries.jsonl
search:
  params:
    - name: default
      k: 10
"""

FULL_YAML = """\
base_url: http://localhost:8080/
datasets:
  documents: docs.jsonl
  queries: queries.jsonl
search:
  params:
    - name: p1
      k: 5
      ef: 64
      alpha: 0.7
      filter:
        category: tech
    - k: 10
timeout_sec: 60
retry:
  max_attempts: 5
  base_backoff_sec: 1.0
  max_backoff_sec: 10.0
ingestion:
  poll_interval_sec: 2.0
  timeout_sec: 300.0
output:
  dir: results
  save_json: true
  save_csv: false
  save_prefix: test
log_level: DEBUG
"""


def test_load_scenario_minimal(tmp_path):
    p = tmp_path / "scenario.yaml"
    p.write_text(MINIMAL_YAML)
    cfg = load_scenario(str(p))
    assert cfg.base_url == "http://localhost:8080"
    assert cfg.documents_path == "docs.jsonl"
    assert cfg.queries_path == "queries.jsonl"
    assert len(cfg.search_params) == 1
    assert cfg.search_params[0].name == "default"
    assert cfg.search_params[0].k == 10


def test_load_scenario_full(tmp_path):
    p = tmp_path / "scenario.yaml"
    p.write_text(FULL_YAML)
    cfg = load_scenario(str(p))
    assert cfg.base_url == "http://localhost:8080"
    assert len(cfg.search_params) == 2
    assert cfg.search_params[0].ef == 64
    assert cfg.search_params[0].alpha == 0.7
    assert cfg.search_params[0].filter == {"category": "tech"}
    assert cfg.search_params[1].name == "param_2"
    assert cfg.timeout_sec == 60.0
    assert cfg.retry.max_attempts == 5
    assert cfg.ingestion.poll_interval_sec == 2.0
    assert cfg.output.save_csv is False
    assert cfg.log_level == "DEBUG"


def test_load_scenario_missing_base_url(tmp_path):
    p = tmp_path / "scenario.yaml"
    p.write_text("datasets:\n  documents: d\n  queries: q\nsearch:\n  params:\n    - k: 5\n")
    with pytest.raises(ValueError, match="missing required key: base_url"):
        load_scenario(str(p))


def test_load_scenario_missing_datasets(tmp_path):
    p = tmp_path / "scenario.yaml"
    p.write_text("base_url: http://x\nsearch:\n  params:\n    - k: 5\n")
    with pytest.raises(ValueError, match="missing required key: datasets"):
        load_scenario(str(p))


def test_load_scenario_empty_params(tmp_path):
    p = tmp_path / "scenario.yaml"
    p.write_text(
        "base_url: http://x\ndatasets:\n  documents: d\n  queries: q\nsearch:\n  params: []\n"
    )
    with pytest.raises(ValueError, match="non-empty list"):
        load_scenario(str(p))


def test_load_scenario_missing_k(tmp_path):
    p = tmp_path / "scenario.yaml"
    p.write_text(
        "base_url: http://x\ndatasets:\n  documents: d\n  queries: q\nsearch:\n  params:\n    - name: bad\n"
    )
    with pytest.raises(ValueError, match="missing required key: k"):
        load_scenario(str(p))
