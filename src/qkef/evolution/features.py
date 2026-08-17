"""Label-safe conventional and quantum-inspired event features."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Sequence

import numpy as np

from qkef.chunking import count_words, split_sentences
from qkef.quantum_inspired.state_encoder import (
    coherence_like_score,
    fidelity_like_similarity,
    hellinger_distance,
    state_entropy,
)


CONVENTIONAL_FEATURE_NAMES = [
    "candidate_top1_cosine", "candidate_top2_cosine", "candidate_top3_cosine",
    "candidate_mean_cosine", "candidate_std_cosine", "candidate_top1_top2_margin",
    "word_jaccard_top1", "sequence_ratio_top1", "incoming_word_count",
    "candidate_word_count", "length_ratio", "absolute_length_difference",
    "incoming_paragraph_count", "incoming_sentence_count", "incoming_numeric_count",
    "candidate_numeric_count", "shared_numeric_ratio", "differing_numeric_count",
    "incoming_percentage_count", "incoming_currency_count", "negation_count",
]
QUANTUM_FEATURE_NAMES = [
    "state_top1_fidelity", "state_top2_fidelity", "state_top3_fidelity",
    "state_mean_fidelity", "state_top1_top2_margin", "incoming_state_entropy",
    "candidate_state_entropy", "incoming_coherence_like", "candidate_coherence_like",
    "state_hellinger_distance",
]
NUMBER = re.compile(r"(?<!\w)(?:[$€£])?\d+(?:[.,]\d+)?%?")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def conventional_features(
    incoming_text: str,
    candidate_texts: Sequence[str],
    candidate_scores: Sequence[float],
) -> np.ndarray:
    scores = list(candidate_scores[:3]) + [0.0] * (3 - len(candidate_scores[:3]))
    top_text = candidate_texts[0] if candidate_texts else ""
    left, right = _tokens(incoming_text), _tokens(top_text)
    jaccard = len(left & right) / len(left | right) if left | right else 0.0
    incoming_numbers, candidate_numbers = NUMBER.findall(incoming_text), NUMBER.findall(top_text)
    shared = len(set(incoming_numbers) & set(candidate_numbers))
    shared_ratio = shared / max(1, len(set(incoming_numbers) | set(candidate_numbers)))
    incoming_words, candidate_words = count_words(incoming_text), count_words(top_text)
    values = [
        scores[0], scores[1], scores[2], float(np.mean(candidate_scores)) if candidate_scores else 0.0,
        float(np.std(candidate_scores)) if candidate_scores else 0.0, scores[0] - scores[1],
        jaccard, SequenceMatcher(None, incoming_text.lower(), top_text.lower()).ratio(),
        incoming_words, candidate_words, incoming_words / max(1, candidate_words),
        abs(incoming_words - candidate_words), incoming_text.count("\n\n") + 1,
        len(split_sentences(incoming_text)), len(incoming_numbers), len(candidate_numbers),
        shared_ratio, len(set(incoming_numbers) ^ set(candidate_numbers)),
        len(re.findall(r"\d+(?:\.\d+)?%", incoming_text)),
        len(re.findall(r"[$€£]\s?\d", incoming_text)),
        len(re.findall(r"\b(?:no|not|never|without|neither|nor)\b", incoming_text.lower())),
    ]
    value = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError("non-finite conventional features")
    return value


def quantum_features(incoming: np.ndarray, candidates: Sequence[np.ndarray]) -> np.ndarray:
    fidelities = [fidelity_like_similarity(incoming, item) for item in candidates]
    padded = fidelities[:3] + [0.0] * (3 - len(fidelities[:3]))
    top = candidates[0] if candidates else incoming
    values = [
        padded[0], padded[1], padded[2], float(np.mean(fidelities)) if fidelities else 0.0,
        padded[0] - padded[1], state_entropy(incoming), state_entropy(top),
        coherence_like_score(incoming), coherence_like_score(top), hellinger_distance(incoming, top),
    ]
    return np.asarray(values, dtype=np.float64)


def assert_feature_manifest_safe(names: Sequence[str]) -> None:
    forbidden_fragments = ("action", "label", "target", "relation", "mutation", "split", "provenance", "knowledge_id", "event_id", "source_id")
    if any(any(fragment in name.lower() for fragment in forbidden_fragments) for name in names):
        raise ValueError("feature manifest contains label/provenance leakage")
