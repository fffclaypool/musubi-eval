import subprocess
import sys

import pytest
import yaml

from musubi_eval.application.tune_params import (
    CONSTRAINT_PENALTY_MAXIMIZE,
    CONSTRAINT_PENALTY_MINIMIZE,
    ParamTuner,
    check_constraint,
    compute_objective_score,
    constraint_penalty,
)
from musubi_eval.config import load_tuning_config
from musubi_eval.domain.models import Document, Query
from musubi_eval.infrastructure.tuning_results import save_tuning_results


# ---------------------------------------------------------------------------
# Fixtures: YAML templates
# ---------------------------------------------------------------------------

MINIMAL_SCENARIO_YAML = """\
base_url: http://localhost:8080
datasets:
  documents: docs.jsonl
  queries: queries.jsonl
search:
  params:
    - name: default
      k: 10
"""

MINIMAL_TUNING_YAML = """\
base_scenario: "{scenario_path}"
search_space:
  k:
    low: 5
    high: 20
    step: 5
  ef:
    low: 50
    high: 200
    step: 50
  alpha:
    low: 0.0
    high: 1.0
    step: 0.1
"""

FULL_TUNING_YAML = """\
base_scenario: "{scenario_path}"
log_level: DEBUG
study:
  name: test-study
  direction: maximize
  n_trials: 3
  timeout_sec: 60
  sampler_seed: 42
search_space:
  k:
    low: 5
    high: 10
    step: 5
  ef:
    low: 50
    high: 100
    step: 50
  alpha:
    low: 0.0
    high: 1.0
    step: 0.5
constraints:
  max_latency_p95_ms: 500.0
objective:
  metric: recall_at_k
  latency_penalty: 0.001
output:
  dir: "{out_dir}"
  save_history_csv: true
  save_history_json: true
  save_best_yaml: true
mlflow:
  enabled: false
"""


def _write_scenario(tmp_path):
    p = tmp_path / "scenario.yaml"
    p.write_text(MINIMAL_SCENARIO_YAML)
    return str(p)


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestLoadTuningConfig:
    def test_minimal(self, tmp_path):
        sp = _write_scenario(tmp_path)
        p = tmp_path / "tuning.yaml"
        p.write_text(MINIMAL_TUNING_YAML.format(scenario_path=sp))
        cfg = load_tuning_config(str(p))
        assert cfg.base_scenario == sp
        assert cfg.search_space.k.low == 5
        assert cfg.search_space.k.high == 20
        assert cfg.search_space.ef.step == 50
        assert cfg.search_space.alpha.low == 0.0
        assert cfg.study.direction == "maximize"
        assert cfg.study.n_trials == 20
        assert cfg.constraints.max_latency_p95_ms is None
        assert cfg.objective.metric == "recall_at_k"
        assert cfg.objective.latency_penalty == 0.0

    def test_full(self, tmp_path):
        sp = _write_scenario(tmp_path)
        out_dir = str(tmp_path / "out")
        p = tmp_path / "tuning.yaml"
        p.write_text(FULL_TUNING_YAML.format(scenario_path=sp, out_dir=out_dir))
        cfg = load_tuning_config(str(p))
        assert cfg.study.name == "test-study"
        assert cfg.study.sampler_seed == 42
        assert cfg.study.timeout_sec == 60.0
        assert cfg.constraints.max_latency_p95_ms == 500.0
        assert cfg.objective.latency_penalty == 0.001
        assert cfg.output.save_history_csv is True
        assert cfg.output.save_best_yaml is True
        assert cfg.log_level == "DEBUG"

    def test_missing_base_scenario(self, tmp_path):
        p = tmp_path / "tuning.yaml"
        p.write_text(
            "search_space:\n  k: {low: 1, high: 2}\n  ef: {low: 1, high: 2}\n  alpha: {low: 0, high: 1}\n"
        )
        with pytest.raises(ValueError, match="missing required key: base_scenario"):
            load_tuning_config(str(p))

    def test_missing_search_space(self, tmp_path):
        sp = _write_scenario(tmp_path)
        p = tmp_path / "tuning.yaml"
        p.write_text(f"base_scenario: {sp}\n")
        with pytest.raises(ValueError, match="missing required key: search_space"):
            load_tuning_config(str(p))

    def test_invalid_direction(self, tmp_path):
        sp = _write_scenario(tmp_path)
        p = tmp_path / "tuning.yaml"
        content = MINIMAL_TUNING_YAML.format(scenario_path=sp) + "study:\n  direction: invalid\n"
        p.write_text(content)
        with pytest.raises(ValueError, match="direction"):
            load_tuning_config(str(p))

    def test_invalid_metric(self, tmp_path):
        sp = _write_scenario(tmp_path)
        p = tmp_path / "tuning.yaml"
        content = (
            MINIMAL_TUNING_YAML.format(scenario_path=sp) + "objective:\n  metric: bad_metric\n"
        )
        p.write_text(content)
        with pytest.raises(ValueError, match="objective.metric"):
            load_tuning_config(str(p))


# ---------------------------------------------------------------------------
# Objective computation tests
# ---------------------------------------------------------------------------


class TestObjectiveScore:
    def test_no_penalty(self):
        score = compute_objective_score(0.85, 100.0, 0.0)
        assert score == pytest.approx(0.85)

    def test_with_penalty(self):
        score = compute_objective_score(0.85, 100.0, 0.001)
        assert score == pytest.approx(0.85 - 0.001 * 100.0)

    def test_high_latency_penalty(self):
        score = compute_objective_score(0.5, 1000.0, 0.001)
        assert score == pytest.approx(0.5 - 1.0)


# ---------------------------------------------------------------------------
# Constraint tests
# ---------------------------------------------------------------------------


class TestConstraint:
    def test_no_constraint(self):
        assert check_constraint(999.0, None) is False

    def test_within_constraint(self):
        assert check_constraint(100.0, 500.0) is False

    def test_exactly_at_limit(self):
        assert check_constraint(500.0, 500.0) is False

    def test_violated(self):
        assert check_constraint(501.0, 500.0) is True

    def test_penalty_maximize(self):
        assert constraint_penalty("maximize") == CONSTRAINT_PENALTY_MAXIMIZE
        assert constraint_penalty("maximize") < 0

    def test_penalty_minimize(self):
        assert constraint_penalty("minimize") == CONSTRAINT_PENALTY_MINIMIZE
        assert constraint_penalty("minimize") > 0


# ---------------------------------------------------------------------------
# Tuning results output tests
# ---------------------------------------------------------------------------


class TestSaveTuningResults:
    def _make_results(self):
        return {
            "best_trial": 0,
            "best_params": {"k": 10, "ef": 100, "alpha": 0.5},
            "best_score": 0.9,
            "trials": [
                {
                    "number": 0,
                    "params": {"k": 10, "ef": 100, "alpha": 0.5},
                    "metrics": {"recall_at_k": 0.9, "mrr": 0.8, "ndcg_at_k": 0.85},
                    "latency_ms": {"mean": 50.0, "p95": 80.0},
                    "score": 0.9,
                    "constraint_violated": False,
                    "status": "completed",
                }
            ],
            "study_name": "test",
            "n_trials": 1,
        }

    def test_save_all(self, tmp_path):
        sp = _write_scenario(tmp_path)
        from musubi_eval.config import (
            TuningConfig,
            SearchSpaceConfig,
            IntRangeConfig,
            FloatRangeConfig,
            TuningOutputConfig,
        )

        cfg = TuningConfig(
            base_scenario=sp,
            search_space=SearchSpaceConfig(
                k=IntRangeConfig(5, 20),
                ef=IntRangeConfig(50, 200),
                alpha=FloatRangeConfig(0.0, 1.0),
            ),
            output=TuningOutputConfig(dir=str(tmp_path / "out")),
        )
        results = self._make_results()
        outputs = save_tuning_results(cfg, results)

        assert "history_json" in outputs
        assert "history_csv" in outputs
        assert "best_yaml" in outputs
        assert "best_scenario" in outputs

        best_yaml = yaml.safe_load(open(outputs["best_yaml"]).read())
        assert best_yaml["best_params"]["k"] == 10

        best_scenario = yaml.safe_load(open(outputs["best_scenario"]).read())
        assert best_scenario["search"]["params"][0]["name"] == "tuned_best"
        assert best_scenario["search"]["params"][0]["k"] == 10

    def test_save_disabled(self, tmp_path):
        sp = _write_scenario(tmp_path)
        from musubi_eval.config import (
            TuningConfig,
            SearchSpaceConfig,
            IntRangeConfig,
            FloatRangeConfig,
            TuningOutputConfig,
        )

        cfg = TuningConfig(
            base_scenario=sp,
            search_space=SearchSpaceConfig(
                k=IntRangeConfig(5, 20),
                ef=IntRangeConfig(50, 200),
                alpha=FloatRangeConfig(0.0, 1.0),
            ),
            output=TuningOutputConfig(
                dir=str(tmp_path / "out"),
                save_history_csv=False,
                save_history_json=False,
                save_best_yaml=False,
            ),
        )
        outputs = save_tuning_results(cfg, self._make_results())
        assert outputs == {}


# ---------------------------------------------------------------------------
# ParamTuner integration test (with fakes)
# ---------------------------------------------------------------------------


class FakeDatasetReader:
    def __init__(self, documents, queries):
        self._documents = documents
        self._queries = queries

    def load_documents(self, path):
        return self._documents

    def load_queries(self, path):
        return self._queries


class FakeGateway:
    def __init__(self, results_by_text):
        self.results_by_text = results_by_text

    def health(self):
        return {"text": "ok"}

    def documents_batch(self, documents):
        return {"ok": True}

    def ingestion_start(self):
        return {"id": "job-1"}

    def ingestion_get(self, job_id):
        return {"state": "ready"}

    def search(self, payload):
        text = payload.get("text")
        return self.results_by_text.get(text, [])


class TestParamTuner:
    def test_basic_tuning(self, tmp_path):
        scenario_path = tmp_path / "scenario.yaml"
        scenario_path.write_text(MINIMAL_SCENARIO_YAML)

        docs_path = tmp_path / "docs.jsonl"
        queries_path = tmp_path / "queries.jsonl"
        docs_path.write_text('{"id": "d1", "text": "hello"}\n')
        queries_path.write_text('{"id": "q1", "query": "hello", "positive_ids": ["d1"]}\n')

        scenario_yaml = f"""\
base_url: http://localhost:8080
datasets:
  documents: {docs_path}
  queries: {queries_path}
search:
  params:
    - name: default
      k: 10
"""
        scenario_path.write_text(scenario_yaml)

        tuning_yaml = f"""\
base_scenario: "{scenario_path}"
study:
  name: test-tuning
  direction: maximize
  n_trials: 3
  sampler_seed: 42
search_space:
  k:
    low: 1
    high: 5
    step: 1
  ef:
    low: 10
    high: 50
    step: 10
  alpha:
    low: 0.0
    high: 1.0
    step: 0.5
output:
  dir: "{tmp_path / "out"}"
  save_history_csv: true
  save_history_json: true
  save_best_yaml: true
"""
        tuning_path = tmp_path / "tuning.yaml"
        tuning_path.write_text(tuning_yaml)

        cfg = load_tuning_config(str(tuning_path))

        documents = [Document(id="d1", text="hello")]
        queries = [Query(id="q1", query="hello", positive_ids=["d1"])]
        reader = FakeDatasetReader(documents, queries)
        gateway = FakeGateway({"hello": ["d1"]})

        tuner = ParamTuner(dataset_reader=reader, search_gateway=gateway)
        results = tuner.run(cfg)

        assert results["n_trials"] == 3
        assert "best_params" in results
        assert "k" in results["best_params"]
        assert "ef" in results["best_params"]
        assert "alpha" in results["best_params"]
        assert results["best_score"] > 0
        assert len(results["trials"]) == 3

    def test_constraint_violation(self, tmp_path):
        scenario_path = tmp_path / "scenario.yaml"
        docs_path = tmp_path / "docs.jsonl"
        queries_path = tmp_path / "queries.jsonl"
        docs_path.write_text('{"id": "d1", "text": "hello"}\n')
        queries_path.write_text('{"id": "q1", "query": "hello", "positive_ids": ["d1"]}\n')

        scenario_yaml = f"""\
base_url: http://localhost:8080
datasets:
  documents: {docs_path}
  queries: {queries_path}
search:
  params:
    - name: default
      k: 10
"""
        scenario_path.write_text(scenario_yaml)

        tuning_yaml = f"""\
base_scenario: "{scenario_path}"
study:
  name: constraint-test
  direction: maximize
  n_trials: 2
  sampler_seed: 1
search_space:
  k:
    low: 1
    high: 2
    step: 1
  ef:
    low: 10
    high: 20
    step: 10
  alpha:
    low: 0.0
    high: 1.0
    step: 0.5
constraints:
  max_latency_p95_ms: 0.0000001
output:
  dir: "{tmp_path / "out"}"
"""
        tuning_path = tmp_path / "tuning.yaml"
        tuning_path.write_text(tuning_yaml)

        cfg = load_tuning_config(str(tuning_path))
        documents = [Document(id="d1", text="hello")]
        queries = [Query(id="q1", query="hello", positive_ids=["d1"])]
        reader = FakeDatasetReader(documents, queries)
        gateway = FakeGateway({"hello": ["d1"]})

        tuner = ParamTuner(dataset_reader=reader, search_gateway=gateway)
        results = tuner.run(cfg)

        for trial in results["trials"]:
            assert trial["constraint_violated"] is True
            assert trial["score"] == CONSTRAINT_PENALTY_MAXIMIZE

    def test_constraint_violation_minimize(self, tmp_path):
        scenario_path = tmp_path / "scenario.yaml"
        docs_path = tmp_path / "docs.jsonl"
        queries_path = tmp_path / "queries.jsonl"
        docs_path.write_text('{"id": "d1", "text": "hello"}\n')
        queries_path.write_text('{"id": "q1", "query": "hello", "positive_ids": ["d1"]}\n')

        scenario_yaml = f"""\
base_url: http://localhost:8080
datasets:
  documents: {docs_path}
  queries: {queries_path}
search:
  params:
    - name: default
      k: 10
"""
        scenario_path.write_text(scenario_yaml)

        tuning_yaml = f"""\
base_scenario: "{scenario_path}"
study:
  name: constraint-minimize-test
  direction: minimize
  n_trials: 2
  sampler_seed: 1
search_space:
  k:
    low: 1
    high: 2
    step: 1
  ef:
    low: 10
    high: 20
    step: 10
  alpha:
    low: 0.0
    high: 1.0
    step: 0.5
constraints:
  max_latency_p95_ms: 0.0000001
output:
  dir: "{tmp_path / "out"}"
"""
        tuning_path = tmp_path / "tuning.yaml"
        tuning_path.write_text(tuning_yaml)

        cfg = load_tuning_config(str(tuning_path))
        documents = [Document(id="d1", text="hello")]
        queries = [Query(id="q1", query="hello", positive_ids=["d1"])]
        reader = FakeDatasetReader(documents, queries)
        gateway = FakeGateway({"hello": ["d1"]})

        tuner = ParamTuner(dataset_reader=reader, search_gateway=gateway)
        results = tuner.run(cfg)

        for trial in results["trials"]:
            assert trial["constraint_violated"] is True
            assert trial["score"] == CONSTRAINT_PENALTY_MINIMIZE


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCLITune:
    def test_tune_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "musubi_eval.cli", "tune", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "tuning YAML" in result.stdout
