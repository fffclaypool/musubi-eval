from musubi_eval.application.run_scenario import ScenarioRunner
from musubi_eval.config import ScenarioConfig
from musubi_eval.domain.models import SearchParam
from musubi_eval.domain.models import Document, Query


class DummyClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        self.t += 0.1
        return self.t

    def perf(self):
        self.t += 0.01
        return self.t

    def sleep(self, seconds: float):
        self.t += seconds


class FakeDatasetReader:
    def __init__(self, documents, queries):
        self._documents = documents
        self._queries = queries

    def load_documents(self, path: str):
        return self._documents

    def load_queries(self, path: str):
        return self._queries


class FakeGateway:
    def __init__(self, results_by_text):
        self.results_by_text = results_by_text
        self.calls = []
        self.last_documents = None
        self.last_search_payloads = []

    def health(self):
        self.calls.append("health")
        return {"text": "ok"}

    def documents_batch(self, documents):
        self.calls.append("documents_batch")
        self.last_documents = documents
        return {"ok": True}

    def ingestion_start(self):
        self.calls.append("ingestion_start")
        return {"id": "job-1"}

    def ingestion_get(self, job_id: str):
        self.calls.append(("ingestion_get", job_id))
        return {"state": "ready"}

    def search(self, payload):
        self.calls.append("search")
        self.last_search_payloads.append(payload)
        text = payload.get("text")
        return self.results_by_text.get(text, [])


class FlakyWarmupGateway(FakeGateway):
    def __init__(self, results_by_text, fail_count: int):
        super().__init__(results_by_text)
        self.fail_count = fail_count

    def search(self, payload):
        self.calls.append("search")
        self.last_search_payloads.append(payload)
        if self.fail_count > 0:
            self.fail_count -= 1
            raise RuntimeError("embed warming up")
        text = payload.get("text")
        return self.results_by_text.get(text, [])


def test_run_scenario():
    documents = [
        Document(id="d1", text="alpha", metadata={"category": "tech"}),
        Document(id="d2", text="beta", metadata={"category": "food"}),
    ]
    queries = [
        Query(id="q1", query="alpha", positive_ids=["d1"], filter={"category": "tech"}),
        Query(id="q2", query="beta", positive_ids=["d2"], filter={"category": "food"}),
    ]
    results_by_text = {"alpha": ["d1"], "beta": ["d2"]}

    reader = FakeDatasetReader(documents, queries)
    gateway = FakeGateway(results_by_text)
    clock = DummyClock()

    cfg = ScenarioConfig(
        base_url="http://example",
        documents_path="docs.jsonl",
        queries_path="queries.jsonl",
        search_params=[
            SearchParam(name="k2_default", k=2, ef=10, alpha=0.5),
            SearchParam(name="k2_food", k=2, ef=10, alpha=0.5, filter={"category": "food"}),
        ],
    )

    runner = ScenarioRunner(
        dataset_reader=reader,
        search_gateway=gateway,
        now=clock.now,
        perf_counter=clock.perf,
        sleep=clock.sleep,
    )
    results = runner.run(cfg)

    assert len(results["runs"]) == 2
    assert gateway.last_documents[0]["id"] == "d1"
    assert "text" in gateway.last_search_payloads[0]
    assert "query" not in gateway.last_search_payloads[0]

    assert all("$and" not in (p.get("filter") or {}) for p in gateway.last_search_payloads)

    for run in results["runs"]:
        assert run["metrics"]["recall_at_k"] == 1.0
        assert run["metrics"]["mrr"] == 1.0
        assert run["metrics"]["ndcg_at_k"] == 1.0


def test_run_scenario_warmup_retry():
    documents = [Document(id="d1", text="alpha")]
    queries = [Query(id="q1", query="alpha", positive_ids=["d1"])]
    results_by_text = {"alpha": ["d1"]}
    reader = FakeDatasetReader(documents, queries)
    gateway = FlakyWarmupGateway(results_by_text, fail_count=2)
    clock = DummyClock()
    cfg = ScenarioConfig(
        base_url="http://example",
        documents_path="docs.jsonl",
        queries_path="queries.jsonl",
        search_params=[SearchParam(name="k1", k=1)],
    )
    runner = ScenarioRunner(
        dataset_reader=reader,
        search_gateway=gateway,
        now=clock.now,
        perf_counter=clock.perf,
        sleep=clock.sleep,
    )
    results = runner.run(cfg)
    assert len(results["runs"]) == 1
    assert results["runs"][0]["metrics"]["recall_at_k"] == 1.0
