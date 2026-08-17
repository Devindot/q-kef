import numpy as np
import pytest

from qkef.quantum_inspired.state_encoder import QuantumInspiredStateEncoder, coherence_like_score, fidelity_like_similarity, hellinger_distance, state_entropy


def test_state_encoder_fit_transform_and_provenance():
    values = np.eye(4, dtype=float)
    encoder = QuantumInspiredStateEncoder(3).fit(values, ["train-1", "train-2", "train-3", "train-4"])
    states = encoder.transform(values)
    assert states.shape == (4, 3) and encoder.fit_ids[0] == "train-1"
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0)


def test_unfitted_state_encoder_fails():
    with pytest.raises(ValueError):
        QuantumInspiredStateEncoder(2).transform(np.eye(2))


def test_quantum_inspired_measures_have_expected_bounds():
    left = np.array([1.0, 0.0]); right = np.array([0.0, 1.0])
    assert fidelity_like_similarity(left, left) == 1.0
    assert fidelity_like_similarity(left, right) == 0.0
    assert state_entropy(left) == pytest.approx(0.0, abs=1e-9)
    assert 0 <= coherence_like_score(np.array([2**-0.5, 2**-0.5])) <= 1
    assert hellinger_distance(left, right) == pytest.approx(1.0)
