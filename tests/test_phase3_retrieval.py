import numpy as np
import pytest

from qkef.retrieval.metrics import ndcg_at_k, obsolete_rate_at_k, precision_at_k, recall_at_k, reciprocal_rank
from qkef.retrieval.vector_index import VectorIndex


def test_vector_index_ranks_and_breaks_ties_by_identifier():
    index = VectorIndex(2)
    index.add("b", np.array([1.0, 0.0])); index.add("a", np.array([1.0, 0.0]))
    assert [item[0] for item in index.search(np.array([1.0, 0.0]), 2)] == ["a", "b"]


def test_vector_index_status_filter_and_remove():
    index = VectorIndex(2)
    index.add("active", np.array([1.0, 0.0]), status="active")
    index.add("old", np.array([1.0, 0.0]), status="archived")
    assert index.search(np.array([1.0, 0.0]), 5, status="active")[0][0] == "active"
    index.remove("active")
    assert index.search(np.array([1.0, 0.0]), 5, status="active") == []


def test_vector_index_rejects_invalid_vectors():
    with pytest.raises(ValueError):
        VectorIndex(2).add("x", np.array([0.0, 0.0]))


def test_retrieval_metrics_known_values():
    ranked, relevant = ["x", "a", "b"], {"a", "b"}
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert 0 < ndcg_at_k(ranked, relevant, 3) <= 1
    assert obsolete_rate_at_k(ranked, {"x"}, 2) == 0.5
