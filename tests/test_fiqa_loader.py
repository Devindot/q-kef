"""Offline tests for BEIR-compatible FiQA loading."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from qkef.datasets.fiqa import FiqaDataError, load_fiqa_dataset


FIXTURE = Path(__file__).parent / "fixtures" / "mini_fiqa"


def test_valid_fixture_loads_without_text_mutation() -> None:
    dataset = load_fiqa_dataset(FIXTURE)

    assert len(dataset.documents) == 12
    assert len(dataset.queries) == 4
    assert len(dataset.qrels) == 12
    assert dataset.documents["d01"].text == "The sample account rate is 5% for the test period."
    assert dataset.documents["d01"].title == "Rate policy"
    assert len(dataset.documents_for_query("q01", "train")) == 4


def test_duplicate_document_id_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "fixture"
    shutil.copytree(FIXTURE, destination)
    first = (destination / "corpus.jsonl").read_text(encoding="utf-8").splitlines()[0]
    with (destination / "corpus.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(first + "\n")

    with pytest.raises(FiqaDataError, match="duplicate document ID"):
        load_fiqa_dataset(destination)


def test_malformed_jsonl_has_useful_location(tmp_path: Path) -> None:
    destination = tmp_path / "fixture"
    shutil.copytree(FIXTURE, destination)
    (destination / "queries.jsonl").write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(FiqaDataError, match=r"queries\.jsonl:1: malformed JSON"):
        load_fiqa_dataset(destination)


def test_invalid_qrel_reference_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "fixture"
    shutil.copytree(FIXTURE, destination)
    with (destination / "qrels" / "train.tsv").open("a", encoding="utf-8") as stream:
        stream.write("q01\tmissing-document\t1\n")

    with pytest.raises(FiqaDataError, match="unknown document ID"):
        load_fiqa_dataset(destination)
