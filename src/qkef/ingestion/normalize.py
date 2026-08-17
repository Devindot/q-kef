"""Conservative deterministic normalization for model-facing text."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping


ZERO_WIDTH_PATTERN = re.compile("[\u200b\u200c\u200d\ufeff]")
HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[\t\f\v ]+")


def normalize_model_text(
    text: str, config: Mapping[str, Any]
) -> tuple[str, list[str]]:
    """Return a conservative normalized copy plus only the operations applied."""

    value = text
    operations: list[str] = []
    unicode_form = str(config.get("unicode_form", "NFKC"))
    normalized = unicodedata.normalize(unicode_form, value)
    if normalized != value:
        value = normalized
        operations.append(f"unicode_normalization_{unicode_form.lower()}")
    if config.get("normalize_line_endings", True):
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        if normalized != value:
            value = normalized
            operations.append("normalize_line_endings")
    if config.get("remove_zero_width", True):
        normalized = ZERO_WIDTH_PATTERN.sub("", value)
        if normalized != value:
            value = normalized
            operations.append("remove_zero_width_characters")
    if config.get("normalize_horizontal_whitespace", True):
        normalized = HORIZONTAL_WHITESPACE_PATTERN.sub(" ", value)
        if normalized != value:
            value = normalized
            operations.append("normalize_horizontal_whitespace")
    max_blank_lines = int(config.get("max_blank_lines", 2))
    if max_blank_lines < 0:
        raise ValueError("max_blank_lines must be non-negative")
    maximum_newlines = max_blank_lines + 1
    normalized = re.sub(rf"\n{{{maximum_newlines + 1},}}", "\n" * maximum_newlines, value)
    if normalized != value:
        value = normalized
        operations.append("reduce_repeated_blank_lines")
    if config.get("trim_whitespace", True):
        normalized = value.strip()
        if normalized != value:
            value = normalized
            operations.append("trim_leading_trailing_whitespace")
    return value, operations
