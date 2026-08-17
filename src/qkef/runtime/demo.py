"""Demo-data boundary: benchmark metadata is never returned as model input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_demo_examples(path: Path) -> list[dict[str, Any]]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) < 6:
        raise ValueError("demo example collection is incomplete")
    return values


def runtime_input(example: dict[str, Any]) -> str:
    """Return text alone—never action/event/target metadata."""
    text = example["incoming"]["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("demo incoming text is empty")
    return text


def initial_demo_records(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for example in examples:
        existing = example.get("existing")
        if existing and not any(item["identifier"] == existing["identifier"] for item in records):
            records.append({"identifier": existing["identifier"], "text": existing["text"], "status": "ACTIVE", "provenance": existing["source_document_ids"]})
    return records
