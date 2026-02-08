from musubi_eval.domain.metrics import compute_query_metrics, mean


def test_compute_query_metrics():
    relevant = ["d1", "d2"]
    ranked = ["d2", "d1", "d3"]
    qm = compute_query_metrics(relevant, ranked, k=2)
    assert qm.recall_at_k == 1.0
    assert qm.mrr == 1.0
    assert qm.ndcg_at_k == 1.0


def test_mean():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0, 3.0]) == 2.0
