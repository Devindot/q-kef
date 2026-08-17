"""One-unit/one-chunk baseline."""

from qkef.chunking.base import create_chunk, sentence_spans
from qkef.schemas import ChunkStrategy, IngestedKnowledgeUnit, KnowledgeChunk


def chunk_identity(unit: IngestedKnowledgeUnit) -> list[KnowledgeChunk]:
    return [
        create_chunk(
            unit,
            ChunkStrategy.IDENTITY,
            0,
            unit.model_text,
            sentence_count=len(sentence_spans(unit.model_text)),
            model_char_start=0,
            model_char_end=len(unit.model_text),
            chunking_metadata={"boundary_reason": "identity"},
        )
    ]
