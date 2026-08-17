import numpy as np
import pytest

from qkef.embeddings.cache import cache_key, encode_with_cache
from qkef.embeddings.encoder import DeterministicLexicalEncoder


def test_fallback_is_deterministic_and_normalized():
    encoder = DeterministicLexicalEncoder(16)
    first = encoder.encode(["alpha beta", "gamma"])
    second = encoder.encode(["alpha beta", "gamma"])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)


def test_fallback_rejects_empty_text():
    with pytest.raises(ValueError):
        DeterministicLexicalEncoder().encode([" "])


def test_embedding_cache_reuses_exact_artifact(tmp_path):
    encoder = DeterministicLexicalEncoder(8)
    records = [("a", "alpha"), ("b", "beta")]
    first, index, hit1 = encode_with_cache(encoder, records, tmp_path, version="test")
    second, index2, hit2 = encode_with_cache(encoder, records, tmp_path, version="test")
    assert not hit1 and hit2 and index == index2
    np.testing.assert_array_equal(first, second)


def test_cache_key_changes_with_content():
    encoder = DeterministicLexicalEncoder(8)
    assert cache_key(encoder, [("a", "one")], "v") != cache_key(encoder, [("a", "two")], "v")
