"""Metadata-driven removal of Phase 1 synthetic control scaffolding."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from qkef.datasets.evolution import MERGE_DELIMITER, SPLIT_DELIMITER


MERGE_SCAFFOLD = "Additional related information:"
SPLIT_MARKER = "--- KNOWLEDGE SEGMENT BOUNDARY ---"
ARCHIVE_TARGET_PATTERN = re.compile(
    r"(?:^|\n{2,})Target knowledge ID:\s*(?P<target>[^\n]+)(?=\n{2,}|$)"
)


@dataclass(frozen=True)
class SanitizationResult:
    text: str
    operations: list[str] = field(default_factory=list)
    extracted_metadata: dict[str, Any] = field(default_factory=dict)


def sanitize_model_text(
    text: str,
    construction_origin: str | None,
    config: Mapping[str, Any],
) -> SanitizationResult:
    """Remove only scaffolding authorized by Phase 1 construction provenance."""

    value = text
    operations: list[str] = []
    extracted: dict[str, Any] = {}
    if (
        construction_origin == "deterministic_merge_construction"
        and config.get("remove_merge_scaffold", True)
    ):
        scaffold = MERGE_DELIMITER
        replacement = "\n\n"
        if scaffold in value:
            value = value.replace(scaffold, replacement)
            operations.append("remove_synthetic_merge_header")
    if (
        construction_origin == "deterministic_compound_split"
        and config.get("neutralize_split_boundary", True)
    ):
        if SPLIT_DELIMITER in value:
            value = value.replace(SPLIT_DELIMITER, "\n\n")
            operations.append("neutralize_synthetic_split_boundary")
    if (
        construction_origin == "synthetic_administrative_notice"
        and config.get("remove_archive_target_id_from_text", True)
    ):
        match = ARCHIVE_TARGET_PATTERN.search(value)
        if match:
            extracted["archive_target_knowledge_id"] = match.group("target").strip()
            value = ARCHIVE_TARGET_PATTERN.sub("\n\n", value, count=1).strip()
            operations.append("remove_archive_target_identifier")
    return SanitizationResult(value, operations, extracted)


def known_synthetic_markers() -> tuple[str, ...]:
    """Return actual Phase 1 marker strings audited in model-facing text."""

    return (MERGE_SCAFFOLD, SPLIT_MARKER)
