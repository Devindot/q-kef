"""Deterministic overlapping word-window baseline."""

from qkef.chunking.base import create_chunk, sentence_spans, word_spans
from qkef.schemas import ChunkStrategy, IngestedKnowledgeUnit, KnowledgeChunk


def chunk_fixed_window(
    unit: IngestedKnowledgeUnit, max_words: int, overlap_words: int
) -> list[KnowledgeChunk]:
    if max_words <= 0 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("fixed-window configuration requires 0 <= overlap < max_words")
    words = word_spans(unit.model_text)
    if not words:
        raise ValueError(f"{unit.knowledge_id}: model text contains no countable words")
    chunks: list[KnowledgeChunk] = []
    step = max_words - overlap_words
    word_start = 0
    while word_start < len(words):
        word_end = min(word_start + max_words, len(words))
        char_start = words[word_start].start
        char_end = words[word_end - 1].end
        text = unit.model_text[char_start:char_end]
        chunks.append(
            create_chunk(
                unit,
                ChunkStrategy.FIXED_WINDOW,
                len(chunks),
                text,
                sentence_count=len(sentence_spans(text)),
                model_char_start=char_start,
                model_char_end=char_end,
                overlap_metadata={
                    "configured_overlap_words": overlap_words,
                    "word_start": word_start,
                    "word_end": word_end,
                    "overlap_with_previous_words": 0 if not chunks else overlap_words,
                },
                chunking_metadata={"max_words": max_words},
            )
        )
        if word_end == len(words):
            break
        word_start += step
    return chunks
