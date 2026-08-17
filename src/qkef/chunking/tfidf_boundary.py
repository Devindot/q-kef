"""Per-unit lexical TF-IDF adjacent-boundary chunking."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from qkef.chunking.base import (
    TextSpan,
    count_words,
    create_chunk,
    sentence_spans,
    word_spans,
)
from qkef.schemas import ChunkStrategy, IngestedKnowledgeUnit, KnowledgeChunk


LEXICAL_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True)
class Atom:
    span: TextSpan
    word_count: int


def _lexical_tokens(text: str) -> list[str]:
    return LEXICAL_PATTERN.findall(text.lower())


def adjacent_tfidf_similarities(texts: list[str]) -> tuple[list[float], str | None]:
    """Return cosine similarities from deterministic per-unit TF-IDF vectors."""

    tokens = [_lexical_tokens(text) for text in texts]
    vocabulary = sorted({token for sentence in tokens for token in sentence})
    if not vocabulary:
        return [0.0] * max(0, len(texts) - 1), "empty_vocabulary"
    document_frequency = Counter(
        token for sentence in tokens for token in set(sentence)
    )
    sentence_count = len(tokens)
    idf = {
        token: math.log((1 + sentence_count) / (1 + document_frequency[token])) + 1
        for token in vocabulary
    }
    vectors: list[dict[str, float]] = []
    for sentence in tokens:
        counts = Counter(sentence)
        vectors.append({token: count * idf[token] for token, count in counts.items()})
    similarities: list[float] = []
    for left, right in zip(vectors, vectors[1:]):
        shared = set(left) & set(right)
        dot = sum(left[token] * right[token] for token in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        similarities.append(dot / (left_norm * right_norm) if left_norm and right_norm else 0.0)
    return similarities, None


def _atoms(text: str, max_words: int) -> tuple[list[Atom], int]:
    atoms: list[Atom] = []
    forced_fragments = 0
    for sentence in sentence_spans(text):
        sentence_text = text[sentence.start : sentence.end]
        words = word_spans(sentence_text)
        if len(words) <= max_words:
            atoms.append(Atom(sentence, max(1, len(words))))
            continue
        forced_fragments += max(0, math.ceil(len(words) / max_words) - 1)
        local_start = 0
        for word_index in range(max_words, len(words), max_words):
            local_end = words[word_index - 1].end
            atoms.append(
                Atom(
                    TextSpan(sentence.start + local_start, sentence.start + local_end),
                    count_words(sentence_text[local_start:local_end]),
                )
            )
            local_start = local_end
        atoms.append(
            Atom(
                TextSpan(sentence.start + local_start, sentence.end),
                count_words(sentence_text[local_start:]),
            )
        )
    return atoms, forced_fragments


def chunk_tfidf_boundary(
    unit: IngestedKnowledgeUnit,
    *,
    min_words: int,
    target_words: int,
    max_words: int,
    threshold: float,
) -> list[KnowledgeChunk]:
    if not 0 < min_words <= target_words <= max_words:
        raise ValueError("TF-IDF limits must satisfy 0 < min <= target <= max")
    if not 0 <= threshold <= 1:
        raise ValueError("TF-IDF boundary threshold must be between 0 and 1")
    base_sentences = sentence_spans(unit.model_text)
    sentence_texts = [unit.model_text[item.start : item.end] for item in base_sentences]
    similarities, fallback_reason = adjacent_tfidf_similarities(sentence_texts)
    if len(base_sentences) == 1 and count_words(unit.model_text) <= max_words:
        fallback_reason = fallback_reason or "single_sentence"
    if fallback_reason and count_words(unit.model_text) <= max_words:
        return [
            create_chunk(
                unit,
                ChunkStrategy.TFIDF_BOUNDARY,
                0,
                unit.model_text,
                sentence_count=max(1, len(base_sentences)),
                model_char_start=0,
                model_char_end=len(unit.model_text),
                chunking_metadata={
                    "boundary_reason": "fallback_identity",
                    "fallback_reason": fallback_reason,
                },
            )
        ]
    atoms, oversized_forced = _atoms(unit.model_text, max_words)
    atom_texts = [unit.model_text[item.span.start : item.span.end] for item in atoms]
    atom_similarities, atom_fallback = adjacent_tfidf_similarities(atom_texts)
    if atom_fallback:
        fallback_reason = atom_fallback
    groups: list[dict[str, object]] = []
    current_atoms = [atoms[0]]
    current_words = atoms[0].word_count
    for index, next_atom in enumerate(atoms[1:], start=1):
        similarity = atom_similarities[index - 1] if index - 1 < len(atom_similarities) else 0.0
        reason: str | None = None
        if current_words + next_atom.word_count > max_words:
            reason = "forced_size"
        elif current_words >= min_words and similarity < threshold:
            reason = "lexical_similarity"
        elif current_words >= target_words and "\n\n" in unit.model_text[
            current_atoms[-1].span.end : next_atom.span.start
        ]:
            reason = "paragraph_structure"
        if reason:
            groups.append(
                {"atoms": current_atoms, "words": current_words, "boundary_reason": reason}
            )
            current_atoms = [next_atom]
            current_words = next_atom.word_count
        else:
            current_atoms.append(next_atom)
            current_words += next_atom.word_count
    groups.append({"atoms": current_atoms, "words": current_words, "boundary_reason": "end"})
    short_tail_merged = False
    if len(groups) > 1 and int(groups[-1]["words"]) < min_words:
        combined = int(groups[-2]["words"]) + int(groups[-1]["words"])
        if combined <= max_words:
            groups[-2]["atoms"] = list(groups[-2]["atoms"]) + list(groups[-1]["atoms"])
            groups[-2]["words"] = combined
            groups[-2]["boundary_reason"] = "end"
            groups.pop()
            short_tail_merged = True
    chunks: list[KnowledgeChunk] = []
    for index, group in enumerate(groups):
        group_atoms = list(group["atoms"])
        start = group_atoms[0].span.start
        end = group_atoms[-1].span.end
        text = unit.model_text[start:end]
        chunks.append(
            create_chunk(
                unit,
                ChunkStrategy.TFIDF_BOUNDARY,
                index,
                text,
                sentence_count=len(group_atoms),
                model_char_start=start,
                model_char_end=end,
                chunking_metadata={
                    "boundary_reason": group["boundary_reason"],
                    "minimum_words": min_words,
                    "target_words": target_words,
                    "maximum_words": max_words,
                    "similarity_threshold": threshold,
                    "fallback_reason": fallback_reason,
                    "oversized_sentence_forced_boundaries": oversized_forced,
                    "short_tail_merged": short_tail_merged and index == len(groups) - 1,
                },
            )
        )
    return chunks
